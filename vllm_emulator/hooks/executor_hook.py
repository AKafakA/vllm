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

    # CUDA graph batch size padding (matches vLLM defaults)
    # When total_tokens doesn't match a captured size, vLLM pads up.
    # First time a new padded size is seen in a session, there's overhead
    # from CUDA graph selection/warmup. We model this as extra latency.
    CUDA_GRAPH_CAPTURE_SIZES = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128,
                                 160, 192, 224, 256, 320, 384, 448, 512,
                                 640, 768, 896, 1024]

    def __init__(self):
        self._oracle: BaseGpuCostOracle | None = None
        self._enabled = False
        self._emulator_mode = EMULATOR_MODE_REALTIME
        self._step_overhead_us = 0.0
        self._pending_output = None  # For sample_tokens
        self._sample_future = None  # Future for sample_tokens to return
        self._gpu_free_time = 0.0  # When the virtual GPU becomes free

        # CUDA graph shape warmup tracking
        # First encounter of a padded batch shape adds overhead (~80-100ms)
        # to model CUDA graph selection/warmup. This is critical for
        # rate=1 accuracy where batch shapes change frequently.
        self._seen_shapes: set[int] = set()
        self._cuda_graph_warmup_us = 0.0  # Set from env or profile

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
        self._cuda_graph_warmup_us = float(os.environ.get(
            "VLLM_EMULATOR_CUDA_GRAPH_WARMUP_US", "0"))

        try:
            profile_pack = load_profile_pack(profile_path)
            self._oracle = create_oracle_from_profile_pack(profile_pack)

            # Decode scheduling ratio: how much of the prefill sched delay
            # applies to decode steps. From profile or env, default 0.1.
            # This is hardware/software-dependent and should be profiled.
            self._decode_sched_ratio = float(
                os.environ.get("VLLM_EMULATOR_DECODE_SCHED_RATIO",
                               str(profile_pack.get("decode_sched_ratio", 0.1))))
            self._enabled = True

            # Auto-derive scheduling delay from profile:
            # On real GPU, execute_model blocks the engine thread, preventing
            # IPC processing. New requests wait ~half a step cycle before being
            # picked up. The timer approach doesn't block, giving the emulator
            # an artificial IPC advantage. We compensate by sleeping for
            # avg_step_time * sched_factor before each timer.
            sched_delay_env = os.environ.get("VLLM_EMULATOR_SCHED_DELAY_US")
            if sched_delay_env is not None:
                self._sched_delay_us = float(sched_delay_env)
            else:
                self._sched_delay_us = self._compute_sched_delay(profile_pack)

            # CUDA graph warmup: from profile or env
            if self._cuda_graph_warmup_us == 0:
                self._cuda_graph_warmup_us = float(
                    profile_pack.get("cuda_graph_warmup_us", 0))

            print(f"[ExecutorEmulatorHook] Enabled: mode={self._emulator_mode}, "
                  f"sched_delay={self._sched_delay_us:.0f}us, "
                  f"decode_sched_ratio={self._decode_sched_ratio:.2f}, "
                  f"cuda_graph_warmup={self._cuda_graph_warmup_us:.0f}us")
        except Exception as e:
            print(f"[ExecutorEmulatorHook] Failed to initialize: {e}")

    def _compute_sched_delay(self, profile_pack: dict) -> float:
        """Compute scheduling delay from profile's average decode step time.

        On real GPU with UniProcExecutor, execute_model blocks the engine
        thread for the full step duration. New requests arriving during
        this block wait in the IPC queue. Average wait ≈ step_time / 2.

        We use the decode_forward_pass profile at low total_tokens (tt=1-4)
        as representative of the step time at low concurrency (rate=1).
        A scaling factor of 0.5 converts step_time to average IPC wait.
        """
        decode_fwd = profile_pack.get("decode_forward_pass", [])
        if not decode_fwd:
            decode_fwd = profile_pack.get("forward_pass", [])

        # Use low-tt entries (typical at rate=1 with few concurrent requests)
        low_tt_latencies = [
            e["latency_us"] for e in decode_fwd
            if e["total_tokens"] <= 4
        ]
        if not low_tt_latencies:
            return 0.0

        avg_step_us = sum(low_tt_latencies) / len(low_tt_latencies)
        self._avg_step_latency_us = avg_step_us  # Store for batch-size scaling
        # Factor from profile or default. Prefill-weighted (decode gets reduced ratio).
        sched_factor = float(profile_pack.get("sched_factor", 1.5))
        return avg_step_us * sched_factor

    def _get_sched_delay_s(
        self, step_latency_us: float, has_prefill: bool
    ) -> float:
        """Compute scheduling delay for this step.

        Models IPC blocking: on real GPU, execute_model blocks the engine
        thread, preventing new request pickup from IPC queue. This delay
        mainly affects TTFT (new requests waiting to be scheduled).

        For prefill steps (new request arriving): full delay — models the
        IPC wait time before the request gets its first step.
        For decode-only steps: minimal delay — ongoing requests don't
        experience IPC wait (they're already scheduled).

        The delay scales with step latency (batch-size-aware).
        """
        if self._sched_delay_us <= 0:
            return 0.0

        # Scale by ratio of current step latency to average
        scale = 1.0
        if hasattr(self, '_avg_step_latency_us') and self._avg_step_latency_us > 0:
            scale = step_latency_us / self._avg_step_latency_us

        if has_prefill:
            # Full delay for prefill: new request waited in IPC queue
            return self._sched_delay_us * scale / 1e6
        else:
            # Reduced delay for decode: ongoing requests don't wait in IPC.
            # Ratio from profile (decode_sched_ratio), defaults to 0.1.
            return self._sched_delay_us * scale * self._decode_sched_ratio / 1e6

    def _get_padded_batch_size(self, total_tokens: int) -> int:
        """Round total_tokens up to the nearest CUDA graph capture size."""
        for size in self.CUDA_GRAPH_CAPTURE_SIZES:
            if size >= total_tokens:
                return size
        return total_tokens  # Beyond max capture size

    def _get_shape_warmup_us(self, total_tokens: int) -> float:
        """Return extra latency if this is a new batch shape.

        Models CUDA graph selection/warmup overhead for the first time
        a padded batch size is encountered in a session.
        """
        if self._cuda_graph_warmup_us <= 0:
            return 0.0
        padded = self._get_padded_batch_size(total_tokens)
        if padded in self._seen_shapes:
            return 0.0
        self._seen_shapes.add(padded)
        return self._cuda_graph_warmup_us

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
        has_prefill = len(scheduler_output.scheduled_new_reqs) > 0
        latency_us = self._oracle.estimate_step_latency_us(
            total_tokens, has_prefill=has_prefill)
        latency_us += self._step_overhead_us

        # CUDA graph shape warmup: first encounter of a new padded
        # batch size adds overhead (graph selection, cache miss, etc.)
        shape_warmup_us = self._get_shape_warmup_us(total_tokens)
        latency_us += shape_warmup_us

        # Add prefill-specific overhead
        prefill_overhead_us = float(os.environ.get(
            "VLLM_EMULATOR_PREFILL_OVERHEAD_US", "0"))
        if prefill_overhead_us > 0 and has_prefill:
            latency_us += prefill_overhead_us

        # Cold-start warmup ramp: first few prefills after server startup
        # have elevated latency due to CUDA graph compilation/caching.
        # Decays over the first N prefill steps.
        if has_prefill:
            cold_start_us = float(os.environ.get(
                "VLLM_EMULATOR_COLD_START_US", "0"))
            if cold_start_us > 0:
                prefill_count = getattr(self, '_prefill_count', 0)
                if prefill_count == 0:
                    latency_us += cold_start_us  # First: full cold start
                elif prefill_count == 1:
                    latency_us += cold_start_us * 0.3  # Second: partial warmup
                # Third+: no extra overhead (warm)
                self._prefill_count = prefill_count + 1

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

        # Create fake output — never return None for non_block mode,
        # as the engine core expects a Future, not None.
        fake_output = self._create_fake_output(scheduler_output)
        if fake_output is None:
            if non_block:
                # Return a resolved Future(None) that the engine can handle
                # via its empty-batch path
                fut: Future = Future()
                fut.set_result(None)
                return fut
            return None

        if not non_block or self._emulator_mode == EMULATOR_MODE_ACCELERATED:
            # Blocking mode or accelerated: return immediately
            if self._emulator_mode == EMULATOR_MODE_REALTIME and latency_s >= 0.001:
                time.sleep(latency_s)
            self._pending_output = fake_output
            return None  # Triggers sample_tokens path

        # Non-blocking realtime: timer-based with batch-size-aware
        # scheduling delay. The delay models the real GPU's thread-blocking
        # that prevents IPC processing during execute_model. Scales with
        # the profiled step time (larger batches = longer blocking).
        sched_delay_s = self._get_sched_delay_s(latency_us, has_prefill)
        if sched_delay_s >= 0.001:
            time.sleep(sched_delay_s)

        exec_fut: Future = Future()
        exec_fut.set_result(None)

        sample_fut: Future = Future()

        # Timer for full profiled time (GPU computation model)
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
            extra = f" shape_warmup={shape_warmup_us:.0f}us" if shape_warmup_us > 0 else ""
            print(f"[ExecutorHook] step={self._debug_count} tt={total_tokens} "
                  f"latency={latency_us:.0f}us sleep={latency_s*1000:.1f}ms{extra}")

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
