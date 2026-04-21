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
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING

from vllm_emulator.oracle import BaseGpuCostOracle, create_oracle_from_profile_pack
from vllm_emulator.profile.loader import load_profile_pack
from vllm_emulator.worker_prep_surrogate import WorkerPrepSurrogate

# F2: module-level single-worker pool, lazy-initialised when the first
# hook with VLLM_EMULATOR_PARALLEL_SURROGATE=1 loads. Shared across
# hook instances in the same process.
_PARALLEL_SURROGATE_POOL: ThreadPoolExecutor | None = None


def _get_parallel_pool() -> ThreadPoolExecutor:
    """Lazily create the F2 pool on first request."""
    global _PARALLEL_SURROGATE_POOL
    if _PARALLEL_SURROGATE_POOL is None:
        _PARALLEL_SURROGATE_POOL = ThreadPoolExecutor(max_workers=1)
    return _PARALLEL_SURROGATE_POOL

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import ModelRunnerOutput

ORACLE_ENABLED_ENV = "VLLM_EMULATOR_ENABLE_ORACLE"
ORACLE_PROFILE_PATH_ENV = "VLLM_EMULATOR_PROFILE_PACK"
ORACLE_MODE_ENV = "VLLM_EMULATOR_MODE"
PREP_SURROGATE_ENV = "VLLM_EMULATOR_PREP_SURROGATE"
SAMPLE_TOKENS_DELAY_ENV = "VLLM_EMULATOR_SAMPLE_TOKENS_DELAY"
PARALLEL_SURROGATE_ENV = "VLLM_EMULATOR_PARALLEL_SURROGATE"

_WARNED_NO_AVG_SAMPLE_MS = False  # module-level once-guard for F3

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
        self._sample_tokens_delay_s = 0.0  # F3: profiled sample-tokens time (default off)
        self._parallel_surrogate_enabled = False  # F2: default off, bit-identical off-path
        self._surrogate_fail_count = 0  # cumulative; logged when non-zero

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
        # Line-buffered so content survives SIGKILL (pkill -9) from cleanup
        # scripts — the prior default-buffered path lost the file content
        # on April 20 Phase 1 Q3 diagnostic. `buffering=1` is line-buffered
        # for text-mode writes.
        self._trace_file = None
        trace_path = os.environ.get("VLLM_EMULATOR_HOOK_TRACE", "")
        if trace_path:
            self._trace_file = open(trace_path, "w", buffering=1)
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

            # F3: Optional sample-tokens delay from profiled avg_sample_ms.
            if os.environ.get(SAMPLE_TOKENS_DELAY_ENV, "").lower() in ("1", "true", "yes"):
                avg_sample_ms = profile_pack.get("avg_sample_ms")
                if avg_sample_ms is not None:
                    self._sample_tokens_delay_s = float(avg_sample_ms) / 1000.0
                else:
                    global _WARNED_NO_AVG_SAMPLE_MS
                    if not _WARNED_NO_AVG_SAMPLE_MS:
                        print(f"[ExecutorEmulatorHook] {SAMPLE_TOKENS_DELAY_ENV}=1 "
                              f"but profile has no avg_sample_ms; feature is a no-op. "
                              f"Rebuild profile with --step-timing-csv to enable.")
                        _WARNED_NO_AVG_SAMPLE_MS = True

            # F2: only create parallel-surrogate state when the gate is set.
            if os.environ.get(PARALLEL_SURROGATE_ENV, "").lower() in ("1", "true", "yes"):
                self._parallel_surrogate_enabled = True
                _get_parallel_pool()

            surr = "on" if self._prep_surrogate is not None else "off"
            sdelay = ("%.3fms" % (self._sample_tokens_delay_s * 1000)
                      if self._sample_tokens_delay_s > 0 else "off")
            parallel = "on" if self._parallel_surrogate_enabled else "off"
            print(f"[ExecutorEmulatorHook] Enabled: mode={self._emulator_mode}, "
                  f"usage={self._profile_usage}, surrogate={surr}, "
                  f"sample_tokens_delay={sdelay}, parallel_surrogate={parallel}")
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

        # sum_kv: total KV-cache depth across scheduled cached (decode) reqs.
        # Passed to oracle for α-adjustment at query time (no-op when profile
        # wasn't built with alpha_kv).
        cached = scheduler_output.scheduled_cached_reqs
        sum_kv = sum(cached.num_computed_tokens) if cached.num_reqs > 0 else 0

        # 1. Estimate latency from profile. num_new_reqs is F4's third axis;
        # oracle ignores it in 2D mode.
        latency_us = self._oracle.estimate_step_latency_us(
            total_tokens,
            has_prefill=has_prefill,
            num_requests=num_total_reqs,
            num_new_reqs=num_new,
            sum_kv=sum_kv,
        )
        latency_s = latency_us / 1e6

        # 1b. Optional: run CPU-side prep surrogate.
        #
        # F2 parallel mode (VLLM_EMULATOR_PARALLEL_SURROGATE=1) only affects
        # the async engine path (non_block=True). When parallel, the
        # surrogate is dispatched to a single-worker ThreadPoolExecutor and
        # self._last_surrogate_time_s is NOT reset — its prior value serves
        # as a persistence-forecast (Hyndman & Athanasopoulos ch.5) used by
        # this step's chain accumulation. The timer callback joins the
        # surrogate Future and overwrites self._last_surrogate_time_s with
        # the actual wall-clock so the NEXT step predicts from THIS one.
        #
        # Default path (parallel off, or sync engine): synchronous surrogate,
        # bit-identical to commit 409fd8dc3.
        surr_fut: Future | None = None
        use_parallel = (self._parallel_surrogate_enabled
                        and non_block
                        and self._prep_surrogate is not None
                        and self._emulator_mode == EMULATOR_MODE_REALTIME)
        if use_parallel:
            pool = _get_parallel_pool()
            surr_fut = pool.submit(
                self._prep_surrogate.run_prep_surrogate, scheduler_output)
            # self._last_surrogate_time_s retains previous step's measurement
            # (additive identity 0.0 on the very first call).
        else:
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
                                      total_tokens, scheduler_output,
                                      surr_fut=surr_fut)

    def _join_surrogate_future(self, surr_fut) -> None:
        """Join the F2 parallel-surrogate Future and update persistence state.

        Loud-failure policy: if the surrogate raises, log the event, increment
        the fail counter, and keep the prior `_last_surrogate_time_s` — the
        persistence forecast still works but operators see the degradation
        rather than it being silently swallowed.
        """
        try:
            measured = surr_fut.result()
            if measured is not None:
                self._last_surrogate_time_s = float(measured)
        except Exception as e:
            self._surrogate_fail_count += 1
            print(
                f"[ExecutorEmulatorHook] surrogate Future raised "
                f"(fail_count={self._surrogate_fail_count}): "
                f"{type(e).__name__}: {e}; "
                f"keeping prior _last_surrogate_time_s="
                f"{self._last_surrogate_time_s:.6f}s. Persistence forecast "
                f"will degrade if this continues."
            )

    def _handle_sync(self, latency_s: float, fake_output) -> None:
        """Sync engine: blocking sleep + direct output."""
        if self._emulator_mode == EMULATOR_MODE_REALTIME and latency_s >= 0.001:
            time.sleep(latency_s)
        # Store for caller to retrieve via has_pending_output/get_pending_output
        self._pending_output = fake_output
        return None

    def _handle_async(self, latency_s: float, fake_output,
                      total_tokens: int,
                      scheduler_output: "SchedulerOutput",
                      surr_fut: Future | None = None) -> "Future":
        """Async engine: pending Future via chain timer + gpu_free_time.

        When surr_fut is not None (F2 parallel mode), the timer callback
        first joins the surrogate Future and updates
        self._last_surrogate_time_s with the measured wall-clock — this
        feeds the next step's persistence forecast.
        """
        exec_fut: Future = Future()
        exec_fut.set_result(None)

        sample_fut: Future = Future()
        if self._emulator_mode == EMULATOR_MODE_REALTIME:
            now = time.perf_counter()
            start_time = max(now, self._gpu_free_time)
            # Chain accumulation: step_cycle + surrogate + sample_tokens (F3).
            end_time = (start_time + latency_s
                        + self._last_surrogate_time_s
                        + self._sample_tokens_delay_s)
            self._gpu_free_time = end_time
            delay = end_time - now

            if delay >= 0.001:
                if surr_fut is not None:
                    def _resolve_parallel():
                        # Join surrogate Future, update persistence state,
                        # then resolve sample Future. Ordering invariant:
                        # sample Future never resolves before surrogate.
                        self._join_surrogate_future(surr_fut)
                        sample_fut.set_result(fake_output)
                    timer = threading.Timer(delay, _resolve_parallel)
                else:
                    timer = threading.Timer(delay,
                        lambda: sample_fut.set_result(fake_output))
                timer.daemon = True
                timer.start()
            else:
                # Below timer granularity: resolve immediately but still
                # honour the ordering invariant when parallel.
                if surr_fut is not None:
                    self._join_surrogate_future(surr_fut)
                sample_fut.set_result(fake_output)
        else:
            # accelerated mode: resolve instantly (still honour ordering)
            if surr_fut is not None:
                self._join_surrogate_future(surr_fut)
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
