"""Executor-level hook for emulator mode.

Intercepts execute_model() at the executor level to return timer-based
pending Futures that resolve after the profiled GPU time.

Architecture:
  - Chain timer: oracle predicts step_cycle, gpu_free_time chain sleeps
    for it, Future resolves with fake output.
  - Mode: realtime (sleep) or accelerated (no sleep)
  - Async engine: pending Future via threading.Timer (required by scheduler)
  - Sync engine: blocking sleep + direct return
"""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import Future
from typing import TYPE_CHECKING

from vllm_emulator.oracle import BaseGpuCostOracle, create_oracle_from_profile_pack
from vllm_emulator.profile.loader import load_profile_pack
from vllm_emulator.worker_prep_surrogate import WorkerPrepSurrogate

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import ModelRunnerOutput

ORACLE_ENABLED_ENV = "VLLM_EMULATOR_ENABLE_ORACLE"
ORACLE_PROFILE_PATH_ENV = "VLLM_EMULATOR_PROFILE_PACK"
ORACLE_MODE_ENV = "VLLM_EMULATOR_MODE"
PREP_SURROGATE_ENV = "VLLM_EMULATOR_PREP_SURROGATE"

EMULATOR_MODE_REALTIME = "realtime"
EMULATOR_MODE_ACCELERATED = "accelerated"
_MODE_ALIASES = {"online": EMULATOR_MODE_REALTIME}


class ExecutorEmulatorHook:
    """Executor-level emulator hook.

    Modes:
      realtime: time.sleep() for predicted latency (wall-clock accurate)
      accelerated: no sleep (fast simulation)

    Async engine (non_block=True):
      Returns pending Future via threading.Timer. Required by vLLM's
      async scheduler -- num_output_placeholders must be > 0 when the
      next batch is scheduled. Resolved Future deadlocks.

    Sync engine (non_block=False):
      Blocking sleep + direct output return. No Future needed.
    """

    def __init__(self):
        self._oracle: BaseGpuCostOracle | None = None
        self._prep_surrogate: WorkerPrepSurrogate | None = None
        self._enabled = False
        self._emulator_mode = EMULATOR_MODE_REALTIME
        self._profile_usage = "online"  # "online" or "offline"
        self._sample_future: Future | None = None
        self._gpu_free_time = 0.0  # Virtual GPU timeline for chain timer
        self._last_surrogate_time_s = 0.0  # Surrogate wall-clock from latest step

        # Fake output generation
        self._rng = __import__("random").Random(42)
        self._vocab_size = 32000
        self._eos_token_id = 2
        self._debug_count = 0

        # pyinstrument profiler (enabled by VLLM_EMULATOR_PYINSTRUMENT=<path>)
        self._profiler = None
        pyinst_path = os.environ.get("VLLM_EMULATOR_PYINSTRUMENT", "")
        if pyinst_path:
            try:
                from pyinstrument import Profiler
                self._profiler = Profiler()
                self._profiler_output = pyinst_path
                self._profiler.start()
                print(f"[ExecutorEmulatorHook] pyinstrument profiling to {pyinst_path}")
            except ImportError:
                print("[ExecutorEmulatorHook] pyinstrument not installed")

        # Per-step trace (enabled by VLLM_EMULATOR_HOOK_TRACE=<path>)
        self._trace_file = None
        trace_path = os.environ.get("VLLM_EMULATOR_HOOK_TRACE", "")
        if trace_path:
            self._trace_file = open(trace_path, "w")
            self._trace_file.write(
                "step,wall_s,tt,n_reqs,n_decode,n_new,has_prefill,"
                "oracle_us,timer_delay_us\n")

        self._initialize()

    def _initialize(self) -> None:
        if os.environ.get(ORACLE_ENABLED_ENV, "").lower() not in ("1", "true", "yes"):
            return

        profile_path = os.environ.get(ORACLE_PROFILE_PATH_ENV)
        if not profile_path:
            return

        mode = os.environ.get(ORACLE_MODE_ENV, EMULATOR_MODE_REALTIME).lower()
        mode = _MODE_ALIASES.get(mode, mode)
        if mode not in (EMULATOR_MODE_REALTIME, EMULATOR_MODE_ACCELERATED):
            print(f"[ExecutorEmulatorHook] Unknown mode '{mode}', using realtime")
            mode = EMULATOR_MODE_REALTIME
        self._emulator_mode = mode

        # Profile usage: online vs offline (auto-detected or env override)
        self._profile_usage = os.environ.get(
            "VLLM_EMULATOR_PROFILE_USAGE", "online").lower()

        try:
            profile_pack = load_profile_pack(profile_path)
            self._oracle = create_oracle_from_profile_pack(profile_pack)
            self._enabled = True

            # Read model metadata from profile pack (auto-collected)
            model_cfg = profile_pack.get("model_config", {})
            if model_cfg.get("vocab_size"):
                self._vocab_size = model_cfg["vocab_size"]

            if os.environ.get(PREP_SURROGATE_ENV, "").lower() in ("1", "true", "yes"):
                if not model_cfg:
                    raise RuntimeError(
                        f"{PREP_SURROGATE_ENV}=1 requires model_config in profile pack")
                self._prep_surrogate = WorkerPrepSurrogate(model_cfg)

            surr = "on" if self._prep_surrogate is not None else "off"
            print(f"[ExecutorEmulatorHook] Enabled: mode={self._emulator_mode}, "
                  f"usage={self._profile_usage}, surrogate={surr}")
        except Exception as e:
            print(f"[ExecutorEmulatorHook] Failed to initialize: {e}")

    # --- Public API ---

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def should_use_oracle(self, scheduler_output: "SchedulerOutput") -> bool:
        return self._enabled and scheduler_output.total_num_scheduled_tokens > 0

    def has_pending_future(self) -> bool:
        return self._sample_future is not None

    def get_sample_future(self) -> "Future":
        fut = self._sample_future
        self._sample_future = None
        return fut

    # --- Core ---

    def create_delayed_future(
        self,
        scheduler_output: "SchedulerOutput",
        non_block: bool = False,
    ) -> "Future | ModelRunnerOutput | None":
        """Create output with predicted GPU latency.

        Args:
            scheduler_output: Current batch from scheduler
            non_block: True = async engine (must return Future),
                       False = sync engine (can return output directly)

        Returns:
            Async engine: exec Future (resolved) + sample Future (pending/resolved)
            Sync engine: None (output stored internally)
        """
        total_tokens = scheduler_output.total_num_scheduled_tokens
        has_prefill = len(scheduler_output.scheduled_new_reqs) > 0

        # Count requests for oracle
        new_req_ids = {r.req_id for r in scheduler_output.scheduled_new_reqs}
        num_decode = sum(
            1 for rid in scheduler_output.num_scheduled_tokens
            if rid not in new_req_ids
        )
        num_new = len(scheduler_output.scheduled_new_reqs)
        num_total_reqs = len(scheduler_output.num_scheduled_tokens)

        # 1. Estimate latency from profile. num_new_reqs is F4's third axis;
        # oracle ignores it in 2D mode.
        latency_us = self._oracle.estimate_step_latency_us(
            total_tokens,
            has_prefill=has_prefill,
            num_requests=num_total_reqs,
            num_new_reqs=num_new,
        )
        latency_s = latency_us / 1e6

        # 1b. Optional: run CPU-side prep surrogate. It blocks the engine
        # thread for ~2-3ms. Do NOT subtract from latency_s — surrogate adds
        # real CPU work in parallel with GPU time the timer models. Its
        # wall-clock is added to gpu_free_time in _handle_async for correct
        # chain accumulation (matches commit 182a75877).
        self._last_surrogate_time_s = 0.0
        if (self._prep_surrogate is not None
                and self._emulator_mode == EMULATOR_MODE_REALTIME):
            self._last_surrogate_time_s = self._prep_surrogate.run_prep_surrogate(
                scheduler_output)

        # 2. Create fake output
        fake_output = self._create_fake_output(scheduler_output)
        if fake_output is None:
            if non_block:
                fut: Future = Future()
                fut.set_result(None)
                return fut
            return None

        # Trace: record per-step details
        if self._trace_file is not None:
            self._debug_count += 1
            wall_now = time.perf_counter()
            self._trace_file.write(
                f"{self._debug_count},{wall_now:.6f},{total_tokens},"
                f"{num_total_reqs},{num_decode},{num_new},{int(has_prefill)},"
                f"{latency_us:.0f},{latency_us:.0f}\n")
            if self._debug_count % 50 == 0:
                self._trace_file.flush()

        # 3. Dispatch based on engine type
        if not non_block:
            return self._handle_sync(latency_s, fake_output)
        else:
            return self._handle_async(latency_s, fake_output,
                                      total_tokens, scheduler_output)

    def _handle_sync(self, latency_s: float, fake_output) -> None:
        """Sync engine: blocking sleep + direct output."""
        if self._emulator_mode == EMULATOR_MODE_REALTIME and latency_s >= 0.001:
            time.sleep(latency_s)
        # Store for caller to retrieve via has_pending_output/get_pending_output
        self._pending_output = fake_output
        return None

    def _handle_async(self, latency_s: float, fake_output,
                      total_tokens: int,
                      scheduler_output: "SchedulerOutput") -> "Future":
        """Async engine: pending Future via chain timer + gpu_free_time."""
        exec_fut: Future = Future()
        exec_fut.set_result(None)

        sample_fut: Future = Future()
        if self._emulator_mode == EMULATOR_MODE_REALTIME:
            now = time.perf_counter()
            start_time = max(now, self._gpu_free_time)
            # Chain accumulation: add step_cycle + surrogate prep time.
            # Prevents the chain from absorbing the surrogate's blocking time,
            # so each step takes step_cycle + worker_prep, matching real
            # engine's CPU+GPU pipeline (matches commit 182a75877).
            end_time = start_time + latency_s + self._last_surrogate_time_s
            self._gpu_free_time = end_time
            delay = end_time - now

            if delay >= 0.001:
                timer = threading.Timer(delay,
                    lambda: sample_fut.set_result(fake_output))
                timer.daemon = True
                timer.start()
            else:
                sample_fut.set_result(fake_output)
        else:
            sample_fut.set_result(fake_output)

        self._sample_future = sample_fut

        # Debug logging
        self._debug_count += 1
        if self._debug_count <= 10 or self._debug_count % 100 == 0:
            n_reqs = len(scheduler_output.num_scheduled_tokens)
            n_new = len(scheduler_output.scheduled_new_reqs)
            print(f"[ExecutorHook] step={self._debug_count} tt={total_tokens} "
                  f"reqs={n_reqs} new={n_new} "
                  f"latency={latency_s*1000:.1f}ms")

        return exec_fut

    # --- Output helpers ---

    def has_pending_output(self) -> bool:
        return getattr(self, '_pending_output', None) is not None

    def get_pending_output(self):
        output = self._pending_output
        self._pending_output = None
        return output

    def _create_fake_output(
        self, scheduler_output: "SchedulerOutput"
    ) -> "ModelRunnerOutput | None":
        """Create minimal fake ModelRunnerOutput."""
        from vllm.v1.outputs import ModelRunnerOutput
        import numpy as np

        req_ids = list(scheduler_output.num_scheduled_tokens.keys())
        if not req_ids:
            return None

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

    def shutdown(self):
        """Clean up and save profiling data."""
        if self._trace_file is not None:
            self._trace_file.close()
            self._trace_file = None
        if self._profiler is not None:
            self._profiler.stop()
            with open(self._profiler_output, "w") as f:
                f.write(self._profiler.output_text(unicode=True, color=False))
            print(f"[ExecutorEmulatorHook] pyinstrument saved to {self._profiler_output}")
            # Also save HTML version
            html_path = self._profiler_output.replace(".txt", ".html")
            with open(html_path, "w") as f:
                f.write(self._profiler.output_html())
            print(f"[ExecutorEmulatorHook] pyinstrument HTML saved to {html_path}")


def get_executor_hook() -> ExecutorEmulatorHook | None:
    """Get or create the executor-level emulator hook."""
    if os.environ.get(ORACLE_ENABLED_ENV, "").lower() not in ("1", "true", "yes"):
        return None
    hook = ExecutorEmulatorHook()
    return hook if hook.is_enabled else None
