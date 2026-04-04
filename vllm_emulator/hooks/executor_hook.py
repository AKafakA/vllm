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
DECODE_OVERHEAD_ENV = "VLLM_EMULATOR_DECODE_OVERHEAD_US"

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
        self._sample_future_queue: list = []  # Queue for concurrent requests
        self._gpu_free_time = 0.0  # When the virtual GPU becomes free

        # Virtual time tracking for accelerated mode
        self._virtual_time_us = 0.0  # Cumulative predicted GPU time
        self._step_count = 0  # Number of steps executed
        self._wall_start_time: float | None = None  # Set on first step

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
        self._decode_overhead_us = float(os.environ.get(DECODE_OVERHEAD_ENV, "0"))

        try:
            import json
            # Load profile pack directly (bypass strict validator for
            # serving profiles that use forward_pass instead of prefill/decode)
            with open(profile_path) as f:
                profile_pack = json.load(f)
            # Ensure required fields exist for the oracle constructor
            profile_pack.setdefault("version", "1.0")
            profile_pack.setdefault("prefill", [])
            profile_pack.setdefault("decode", [])
            self._oracle = create_oracle_from_profile_pack(profile_pack)
            self._enabled = True
            print(f"[ExecutorEmulatorHook] Enabled: mode={self._emulator_mode}, "
                  f"overhead={self._step_overhead_us}us, "
                  f"decode_overhead={self._decode_overhead_us}us")
        except Exception as e:
            print(f"[ExecutorEmulatorHook] Failed to initialize: {e}")

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def virtual_time_us(self) -> float:
        """Return accumulated virtual GPU time in microseconds."""
        return self._virtual_time_us

    @property
    def step_count(self) -> int:
        """Return the number of emulated steps."""
        return self._step_count

    def get_virtual_time_summary(self) -> dict[str, float]:
        """Return a summary of virtual time vs wall time.

        Returns:
            Dict with keys:
            - virtual_time_s: Total predicted GPU time (seconds)
            - wall_time_s: Elapsed wall clock time (seconds)
            - speedup: virtual_time / wall_time (>1 means faster than realtime)
            - step_count: Number of emulated steps
            - avg_step_us: Average predicted latency per step (microseconds)
        """
        virtual_s = self._virtual_time_us / 1e6
        wall_s = (time.perf_counter() - self._wall_start_time
                  if self._wall_start_time is not None else 0.0)
        speedup = virtual_s / wall_s if wall_s > 0 else 0.0
        avg_step_us = (self._virtual_time_us / self._step_count
                       if self._step_count > 0 else 0.0)
        return {
            "virtual_time_s": virtual_s,
            "wall_time_s": wall_s,
            "speedup": speedup,
            "step_count": self._step_count,
            "avg_step_us": avg_step_us,
        }

    def print_virtual_time_summary(self) -> None:
        """Print a human-readable summary of virtual time simulation."""
        if self._step_count == 0:
            return
        s = self.get_virtual_time_summary()
        print(f"[ExecutorEmulatorHook] Virtual time summary: "
              f"simulated {s['virtual_time_s']:.3f}s of GPU time "
              f"in {s['wall_time_s']:.3f}s wall time "
              f"({s['speedup']:.1f}x speedup), "
              f"{s['step_count']} steps, "
              f"avg {s['avg_step_us']:.0f}us/step")

    def should_use_oracle(self, scheduler_output: "SchedulerOutput") -> bool:
        return self._enabled and scheduler_output.total_num_scheduled_tokens > 0

    def has_pending_output(self) -> bool:
        return self._pending_output is not None

    def get_pending_output(self):
        output = self._pending_output
        self._pending_output = None
        return output

    def has_pending_future(self) -> bool:
        return len(self._sample_future_queue) > 0 or self._sample_future is not None

    def get_sample_future(self) -> "Future":
        """Return the next Future for sample_tokens.
        Uses a queue to handle concurrent batch scheduling."""
        if self._sample_future_queue:
            return self._sample_future_queue.pop(0)
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

        # Add decode-specific overhead: accounts for output processing,
        # sampling, and scheduling overhead that is cheaper with fake
        # outputs than with real GPU outputs.
        if self._decode_overhead_us > 0:
            new_req_ids = {r.req_id for r in scheduler_output.scheduled_new_reqs}
            num_decode = sum(
                1 for rid in scheduler_output.num_scheduled_tokens
                if rid not in new_req_ids
            )
            if num_decode > 0:
                latency_us += self._decode_overhead_us

        latency_s = latency_us / 1e6

        # Accumulate virtual time for reporting
        if self._wall_start_time is None:
            self._wall_start_time = time.perf_counter()
        self._step_count += 1
        self._virtual_time_us += latency_us

        # In CUDA mock mode, add synchronous CUDA overhead.
        # Only for prefill steps (new requests) — this simulates GPU
        # memory management and CUDA sync that happens when processing
        # new request prefills on real hardware.
        cuda_sync_us = float(os.environ.get("VLLM_EMULATOR_CUDA_SYNC_US", "0"))
        if cuda_sync_us > 0 and len(scheduler_output.scheduled_new_reqs) > 0:
            time.sleep(cuda_sync_us / 1e6)

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
            try:
                sample_fut.set_result(fake_output)
            except Exception as e:
                print(f"[ExecutorHook] _resolve error: {e}")

        if delay >= 0.001:
            timer = threading.Timer(delay, _resolve)
            timer.daemon = True
            timer.start()
        else:
            _resolve()

        # Debug rate>1 issue
        if hasattr(self, '_debug_count') and self._debug_count <= 20:
            print(f"[ExecutorHook] step={self._debug_count} "
                  f"tt={total_tokens} delay={delay*1000:.1f}ms "
                  f"queue_len={len(self._sample_future_queue)}")

        self._sample_future_queue.append(sample_fut)

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
