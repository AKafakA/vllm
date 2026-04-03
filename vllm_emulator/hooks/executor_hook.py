"""Executor-level hook for emulator mode.

Intercepts at the executor level (above worker) to return timer-based
Futures that resolve after predicted GPU time. This preserves the
engine core's batch queue pipelining — the scheduler runs while the
Future is pending, matching real GPU/CPU overlap behavior.

This is the correct abstraction layer for online serving emulation
because it doesn't block the worker thread, allowing the engine core's
async scheduling to pipeline batches naturally.
"""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any

from vllm_emulator.oracle import BaseGpuCostOracle, create_oracle_from_profile_pack
from vllm_emulator.profile.loader import load_profile_pack

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import ModelRunnerOutput

# Reuse env var names from gpu_hook
ORACLE_ENABLED_ENV = "VLLM_EMULATOR_ENABLE_ORACLE"
ORACLE_PROFILE_PATH_ENV = "VLLM_EMULATOR_PROFILE_PACK"
ORACLE_MODE_ENV = "VLLM_EMULATOR_MODE"
STEP_OVERHEAD_ENV = "VLLM_EMULATOR_STEP_OVERHEAD_US"

EMULATOR_MODE_REALTIME = "realtime"
EMULATOR_MODE_ACCELERATED = "accelerated"
_MODE_ALIASES = {"online": EMULATOR_MODE_REALTIME, "offline": EMULATOR_MODE_ACCELERATED}


class ExecutorEmulatorHook:
    """Executor-level emulator hook using timer-based Futures.

    When enabled, intercepts execute_model() at the executor level
    and returns a Future that resolves after the predicted GPU latency.
    The engine core's batch queue sees this as a pending computation
    and pipelines scheduling of the next batch — matching real GPU/CPU
    overlap behavior.
    """

    def __init__(self):
        self._oracle: BaseGpuCostOracle | None = None
        self._enabled = False
        self._emulator_mode = EMULATOR_MODE_REALTIME
        self._step_overhead_us = 0.0
        self._pending_output = None  # For sample_tokens
        self._sample_future = None  # Future for sample_tokens to return
        self._gpu_free_time = 0.0  # When the virtual GPU becomes free

        # Fake output generation (simplified — reuses gpu_hook logic)
        self._rng = __import__("random").Random(42)
        self._vocab_size = 32000
        self._eos_token_id = 2

        self._initialize()

    def _initialize(self) -> None:
        if os.environ.get(ORACLE_ENABLED_ENV, "").lower() not in ("1", "true", "yes"):
            return

        profile_path = os.environ.get(ORACLE_PROFILE_PATH_ENV)
        if not profile_path:
            return

        mode = os.environ.get(ORACLE_MODE_ENV, EMULATOR_MODE_REALTIME).lower()
        mode = _MODE_ALIASES.get(mode, mode)
        self._emulator_mode = mode

        self._step_overhead_us = float(os.environ.get(STEP_OVERHEAD_ENV, "0"))

        try:
            profile_pack = load_profile_pack(profile_path)
            self._oracle = create_oracle_from_profile_pack(profile_pack)
            self._enabled = True
            print(f"[ExecutorEmulatorHook] Enabled: mode={self._emulator_mode}, "
                  f"overhead={self._step_overhead_us}us")
        except Exception as e:
            print(f"[ExecutorEmulatorHook] Failed to initialize: {e}")

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def should_use_oracle(self, scheduler_output: "SchedulerOutput") -> bool:
        return self._enabled and scheduler_output.total_num_scheduled_tokens > 0

    def has_pending_output(self) -> bool:
        return self._pending_output is not None

    def get_pending_output(self):
        output = self._pending_output
        self._pending_output = None
        return output

    def has_pending_future(self) -> bool:
        return self._sample_future is not None

    def get_sample_future(self) -> "Future":
        """Return the Future for sample_tokens.
        Resolves with the fake output after the predicted GPU time."""
        fut = self._sample_future
        self._sample_future = None
        return fut

    def create_delayed_future(
        self,
        scheduler_output: "SchedulerOutput",
        non_block: bool = False,
    ) -> "Future | ModelRunnerOutput | None":
        """Create a Future that resolves after predicted GPU time.

        Returns:
            If non_block: Future that resolves after predicted latency
            If blocking: sleeps then returns output directly
        """
        total_tokens = scheduler_output.total_num_scheduled_tokens
        latency_us = self._oracle.estimate_step_latency_us(total_tokens)
        latency_us += self._step_overhead_us
        latency_s = latency_us / 1e6

        # Create fake output
        fake_output = self._create_fake_output(scheduler_output)
        if fake_output is None:
            return None

        if not non_block or self._emulator_mode == EMULATOR_MODE_ACCELERATED:
            # Blocking mode or accelerated: return immediately
            if self._emulator_mode == EMULATOR_MODE_REALTIME and latency_s >= 0.001:
                time.sleep(latency_s)
            self._pending_output = fake_output
            return None  # Triggers sample_tokens path

        # Non-blocking realtime: return exec Future (None) immediately,
        # and create a sample Future that resolves with the output after
        # predicted GPU time. The engine core:
        # 1. Gets exec Future → resolves instantly with None
        # 2. Calls sample_tokens → gets sample Future (pending)
        # 3. Adds sample Future to batch queue
        # 4. Batch queue not full → schedules NEXT batch (overlap!)
        # 5. Timer fires → sample Future resolves → engine processes output
        exec_fut: Future = Future()
        exec_fut.set_result(None)  # Resolve immediately (like real kernel launch)

        sample_fut: Future = Future()

        # Chain timers: step N+1 can't complete before step N.
        # The virtual GPU is a serial resource — next step starts
        # only after the current step finishes.
        now = time.perf_counter()
        start_time = max(now, self._gpu_free_time)
        end_time = start_time + latency_s
        self._gpu_free_time = end_time
        delay = end_time - now

        def _resolve():
            sample_fut.set_result(fake_output)

        if delay >= 0.001:
            timer = threading.Timer(delay, _resolve)
            timer.daemon = True
            timer.start()
        else:
            _resolve()

        self._sample_future = sample_fut

        # Debug: log prediction
        if hasattr(self, '_debug_count'):
            self._debug_count += 1
        else:
            self._debug_count = 1
        if self._debug_count <= 10 or self._debug_count % 100 == 0:
            print(f"[ExecutorHook] step={self._debug_count} tt={total_tokens} "
                  f"latency={latency_us:.0f}us sleep={latency_s*1000:.1f}ms")

        return exec_fut

    def _create_fake_output(
        self, scheduler_output: "SchedulerOutput"
    ) -> "ModelRunnerOutput | None":
        """Create minimal fake ModelRunnerOutput."""
        from vllm.v1.outputs import ModelRunnerOutput

        import numpy as np

        req_ids = list(scheduler_output.num_scheduled_tokens.keys())
        if not req_ids:
            return None

        # Check which are prefill chunks vs decode
        new_req_ids = {req.req_id for req in scheduler_output.scheduled_new_reqs}
        prefill_chunk_ids = set()
        for req in scheduler_output.scheduled_new_reqs:
            if req.prompt_token_ids:
                scheduled = scheduler_output.num_scheduled_tokens.get(req.req_id, 0)
                if scheduled < len(req.prompt_token_ids):
                    prefill_chunk_ids.add(req.req_id)

        sampled_token_ids = []
        for req_id in req_ids:
            if req_id in prefill_chunk_ids:
                sampled_token_ids.append([])
            else:
                tok = self._rng.randrange(self._vocab_size)
                while tok == self._eos_token_id:
                    tok = self._rng.randrange(self._vocab_size)
                sampled_token_ids.append([tok])

        req_id_to_index = {rid: idx for idx, rid in enumerate(req_ids)}

        num_with_tokens = sum(1 for toks in sampled_token_ids if toks)
        if num_with_tokens > 0:
            from vllm.v1.outputs import LogprobsLists
            token_vals = [toks[0] for toks in sampled_token_ids if toks]
            logprobs = LogprobsLists(
                logprob_token_ids=np.array([[t] for t in token_vals], dtype=np.int32),
                logprobs=np.full((num_with_tokens, 1), -0.1, dtype=np.float32),
                sampled_token_ranks=np.zeros(num_with_tokens, dtype=np.int32),
            )
        else:
            logprobs = None

        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index=req_id_to_index,
            sampled_token_ids=sampled_token_ids,
            logprobs=logprobs,
            prompt_logprobs_dict={},
            pooler_output=[None] * len(req_ids),
        )


def get_executor_hook() -> ExecutorEmulatorHook | None:
    """Get or create the executor-level emulator hook."""
    if os.environ.get(ORACLE_ENABLED_ENV, "").lower() not in ("1", "true", "yes"):
        return None
    hook = ExecutorEmulatorHook()
    return hook if hook.is_enabled else None
