"""Executor-level hook for emulator mode.

Intercepts execute_model() at the executor level to return timer-based
pending Futures that resolve after the profiled GPU time.

Architecture:
  - Path A (GPU host) or Path B (CPU-only): determined by platform
  - Mode: realtime (sleep) or accelerated (no sleep)
  - Usage: online or offline (determines profile section)
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

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import ModelRunnerOutput

ORACLE_ENABLED_ENV = "VLLM_EMULATOR_ENABLE_ORACLE"
ORACLE_PROFILE_PATH_ENV = "VLLM_EMULATOR_PROFILE_PACK"
ORACLE_MODE_ENV = "VLLM_EMULATOR_MODE"

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
      async scheduler — num_output_placeholders must be > 0 when the
      next batch is scheduled. Resolved Future deadlocks.

    Sync engine (non_block=False):
      Blocking sleep + direct output return. No Future needed.
    """

    def __init__(self):
        self._oracle: BaseGpuCostOracle | None = None
        self._enabled = False
        self._emulator_mode = EMULATOR_MODE_REALTIME
        self._profile_usage = "online"  # "online" or "offline"
        self._sample_future: Future | None = None
        self._gpu_free_time = 0.0  # Virtual GPU timeline (legacy, kept for capped mode)

        # Single-worker executor: mimics real GPU's async_output_thread.
        # Pending Futures are real queued work, not manual timestamps.
        # Naturally serializes (1 worker = 1 GPU), no drift bug.
        from concurrent.futures import ThreadPoolExecutor
        self._gpu_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="EmulatorGPU")

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
                "oracle_us,hybrid_overhead_us,sched_comp_us,total_latency_us,"
                "gpu_free_time,timer_delay_us\n")
            self._last_step_wall = None

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

        # Oracle mode: step_cycle (default), hybrid, or 2d
        self._oracle_mode = os.environ.get(
            "VLLM_EMULATOR_ORACLE_MODE", "step_cycle").lower()

        # Hybrid mode: per-request overhead added to step-cycle
        # Models host-side costs (output dispatch, IPC, KV bookkeeping)
        # that scale with concurrent requests and aren't in the profile.
        # Calibrate: (real_TPOT - emu_TPOT) / avg_concurrent_reqs
        self._overhead_per_req_us = float(os.environ.get(
            "VLLM_EMULATOR_OVERHEAD_PER_REQ_US", "0"))

        try:
            profile_pack = load_profile_pack(profile_path)
            self._oracle = create_oracle_from_profile_pack(profile_pack)
            self._enabled = True

            # Pipeline scheduling compensation (profile-derived).
            self._sched_compensation_us = self._compute_sched_compensation(
                profile_pack)

            # Auto-calibrate overhead_per_req from profile if not set manually
            if self._oracle_mode in ("hybrid", "2d") and self._overhead_per_req_us == 0:
                self._overhead_per_req_us = self._calibrate_overhead_per_req(
                    profile_pack)

            # GPU submission overhead (informational, used for diagnostics).
            self._submission_overhead_us = float(
                profile_pack.get("submission_overhead_us", 0))

            # Worker prep surrogate: GPU-free CPU work that replaces the
            # skipped worker.execute_model() CPU preparation. Recovers
            # ~2-3ms per step of CPU overhead that the timer absorbs.
            self._prep_surrogate = None
            if os.environ.get("VLLM_EMULATOR_PREP_SURROGATE", "1") == "1":
                try:
                    from vllm_emulator.worker_prep_surrogate import WorkerPrepSurrogate
                    model_cfg = profile_pack.get("model_config", {})
                    if not model_cfg:
                        print("[ExecutorHook] WARNING: profile pack has no "
                              "model_config. Re-profile with latest tracer "
                              "to auto-collect. Surrogate disabled.")
                    else:
                        self._prep_surrogate = WorkerPrepSurrogate(model_cfg)
                except Exception as e:
                    print(f"[ExecutorHook] Prep surrogate init failed: {e}")

            # Step cadence residual: the small per-step gap (~1ms) that
            # remains after the timer absorbs most of the submission overhead.
            # Applied AFTER future.result() in step_with_batch_queue to
            # break the concurrency feedback loop. Profiled from the
            # step_cycle trace as the median "other" overhead.
            self._step_residual_us = float(
                profile_pack.get("step_residual_us", 0))

            print(f"[ExecutorEmulatorHook] Enabled: mode={self._emulator_mode}, "
                  f"oracle={self._oracle_mode}, usage={self._profile_usage}, "
                  f"sched_comp={self._sched_compensation_us/1000:.1f}ms, "
                  f"overhead/req={self._overhead_per_req_us/1000:.2f}ms, "
                  f"submit_overhead={self._submission_overhead_us/1000:.1f}ms")
        except Exception as e:
            print(f"[ExecutorEmulatorHook] Failed to initialize: {e}")

    def _compute_sched_compensation(self, profile_pack: dict) -> float:
        """Derive scheduling compensation from avg decode step-cycle.

        On real GPU, a new request waits on average half a decode step
        before being scheduled. This is the pipelining advantage the
        emulator has over real GPU.

        Returns avg_decode_step_us / 2 (expected wait time).
        """
        decode_fwd = profile_pack.get("decode_forward_pass", [])
        if not decode_fwd:
            decode_fwd = profile_pack.get("forward_pass", [])
        if not decode_fwd:
            return 0.0

        low_tt = [e["latency_us"] for e in decode_fwd if e["total_tokens"] <= 4]
        if not low_tt:
            return 0.0

        avg_step = sum(low_tt) / len(low_tt)
        return avg_step / 2  # Half-step: average wait for mid-step arrival

    def _calibrate_overhead_per_req(self, profile_pack: dict) -> float:
        """Auto-calibrate per-request overhead from profile data.

        The overhead represents host-side costs (output dispatch, IPC,
        KV bookkeeping) that scale with the number of concurrent requests
        and are NOT captured in the step-cycle profile.

        Estimated from the gap between step-cycle at high vs low
        concurrency, normalized by request count difference.
        """
        decode_fwd = profile_pack.get("decode_forward_pass", [])
        if len(decode_fwd) < 2:
            return 0.0

        # Step-cycle at tt=1 (1 request): baseline cost
        low = [e["latency_us"] for e in decode_fwd if e["total_tokens"] <= 2]
        # Step-cycle at tt=5-10 (5-10 requests): higher concurrency
        high = [e["latency_us"] for e in decode_fwd
                if 5 <= e["total_tokens"] <= 10]

        if not low or not high:
            return 0.0

        avg_low = sum(low) / len(low)
        avg_high = sum(high) / len(high)

        # Average tt for each group
        avg_tt_low = sum(e["total_tokens"] for e in decode_fwd
                         if e["total_tokens"] <= 2) / len(low)
        avg_tt_high = sum(e["total_tokens"] for e in decode_fwd
                          if 5 <= e["total_tokens"] <= 10) / len(high)

        # Overhead per additional request
        tt_diff = avg_tt_high - avg_tt_low
        if tt_diff <= 0:
            return 0.0

        lat_diff = avg_high - avg_low
        overhead = lat_diff / tt_diff  # us per additional token/request

        # Clamp to reasonable range (0 to 500us per request)
        overhead = max(0.0, min(overhead, 500.0))
        return overhead

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

        # Count decode requests for hybrid/2d modes
        new_req_ids = {r.req_id for r in scheduler_output.scheduled_new_reqs}
        num_decode = sum(
            1 for rid in scheduler_output.num_scheduled_tokens
            if rid not in new_req_ids
        )
        num_new = len(scheduler_output.scheduled_new_reqs)
        num_total_reqs = len(scheduler_output.num_scheduled_tokens)

        # 1. Estimate latency from profile
        oracle_us = self._oracle.estimate_step_latency_us(
            total_tokens,
            has_prefill=has_prefill,
            profile_section=self._profile_usage,
            num_requests=num_total_reqs,
            oracle_mode=self._oracle_mode,
        )
        latency_us = oracle_us

        # Scheduling compensation: when prior GPU work is in flight
        sched_comp_applied_us = 0.0
        if has_prefill and self._sched_compensation_us > 0:
            now_check = time.perf_counter()
            if self._gpu_free_time > now_check:
                sched_comp_applied_us = self._sched_compensation_us
                latency_us += sched_comp_applied_us

        latency_s = latency_us / 1e6

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
            timer_delay_us = latency_us
            self._trace_file.write(
                f"{self._debug_count},{wall_now:.6f},{total_tokens},"
                f"{num_total_reqs},{num_decode},{num_new},{int(has_prefill)},"
                f"{oracle_us:.0f},0,"
                f"{sched_comp_applied_us:.0f},{latency_us:.0f},"
                f"{wall_now:.6f},{timer_delay_us:.0f}\n")
            if self._debug_count % 500 == 0:
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
        """Async engine: pending Future.

        Timer mode (VLLM_EMULATOR_TIMER_MODE):
          chain: threading.Timer + gpu_free_time chaining (default, best accuracy)
          pool:  ThreadPoolExecutor — alternative, no chaining
        """
        exec_fut: Future = Future()
        exec_fut.set_result(None)

        # Worker prep surrogate: run CPU-equivalent of worker.execute_model()
        # preparation. This blocks the engine thread for ~2-3ms, matching
        # real GPU's CPU prep time that the hook normally skips.
        # Do NOT subtract from latency_s — the surrogate adds real CPU work
        # that slows the engine loop, while the timer models the GPU compute
        # that runs in parallel on real hardware. The total step becomes:
        # surrogate(~2ms) + timer(step_cycle) which exceeds the profiled
        # step_cycle, but the chain's gpu_free_time accumulation uses the
        # full step_cycle, keeping Future resolution timing correct.
        self._last_surrogate_time_s = 0.0
        if (self._prep_surrogate is not None
                and self._emulator_mode == EMULATOR_MODE_REALTIME):
            self._last_surrogate_time_s = self._prep_surrogate.run_prep_surrogate(
                scheduler_output)

        timer_mode = os.environ.get("VLLM_EMULATOR_TIMER_MODE", "chain")

        if timer_mode == "chain":
            # Timer + gpu_free_time chain
            sample_fut: Future = Future()
            if self._emulator_mode == EMULATOR_MODE_REALTIME:
                now = time.perf_counter()

                # Get surrogate prep time for chain accumulation.
                # The surrogate already ran and blocked the engine. Add its
                # time to gpu_free_time so the chain grows at the correct
                # rate (step_cycle + worker_prep_overhead). This prevents
                # the chain from absorbing the surrogate time.
                _surr_time_s = getattr(self, '_last_surrogate_time_s', 0.0)
                _prev_gpu_free = self._gpu_free_time
                _chain_backed_up = now < self._gpu_free_time
                start_time = max(now, self._gpu_free_time)
                # Accumulate step_cycle + surrogate prep time
                end_time = start_time + latency_s + _surr_time_s
                self._gpu_free_time = end_time
                delay = end_time - now

                # Chain diagnostics (first 30 steps + every 200)
                if self._debug_count <= 30 or self._debug_count % 200 == 0:
                    _chain_lag = now - _prev_gpu_free if not _chain_backed_up else _prev_gpu_free - now
                    print(f"[ChainDiag] step={self._debug_count} tt={total_tokens} "
                          f"surr={_surr_time_s*1000:.2f}ms "
                          f"oracle={latency_s*1000:.1f}ms "
                          f"delay={delay*1000:.1f}ms "
                          f"backed_up={_chain_backed_up} "
                          f"lag={_chain_lag*1000:.1f}ms")

                if delay >= 0.001:
                    timer = threading.Timer(delay,
                        lambda: sample_fut.set_result(fake_output))
                    timer.daemon = True
                    timer.start()
                else:
                    sample_fut.set_result(fake_output)
            else:
                sample_fut.set_result(fake_output)
        else:
            # ThreadPool
            if self._emulator_mode == EMULATOR_MODE_REALTIME:
                def _gpu_step():
                    if latency_s >= 0.001:
                        time.sleep(latency_s)
                    return fake_output
                sample_fut = self._gpu_executor.submit(_gpu_step)
                self._gpu_free_time = time.perf_counter() + latency_s
            else:
                exec_fut.set_result(None)
                sample_fut = self._gpu_executor.submit(lambda: fake_output)

        self._sample_future = sample_fut

        # Debug logging
        self._debug_count += 1
        if self._debug_count <= 10 or self._debug_count % 100 == 0:
            n_reqs = len(scheduler_output.num_scheduled_tokens)
            n_new = len(scheduler_output.scheduled_new_reqs)
            print(f"[ExecutorHook] step={self._debug_count} tt={total_tokens} "
                  f"reqs={n_reqs} new={n_new} "
                  f"latency={latency_s*1000:.1f}ms mode={timer_mode}")

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
        """Clean up executor thread pool and save profiling data."""
        if hasattr(self, '_gpu_executor') and self._gpu_executor is not None:
            self._gpu_executor.shutdown(wait=False)
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
