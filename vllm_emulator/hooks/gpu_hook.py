"""GPU worker hook for emulator mode cost estimation."""

from __future__ import annotations

import os
import random
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm_emulator.oracle import BaseGpuCostOracle, create_oracle_from_profile_pack
from vllm_emulator.profile.loader import load_profile_pack

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput, ModelRunnerOutput
    from vllm.v1.worker.gpu_worker import Worker

# Default EOS token ID (overridden at runtime from model config when available)
_DEFAULT_EOS_TOKEN_ID = 2
# Default vocab size (overridden at runtime from model config when available)
_DEFAULT_VOCAB_SIZE = 32000


# Environment variable to enable emulator cost oracle
ORACLE_ENABLED_ENV = "VLLM_EMULATOR_ENABLE_ORACLE"
ORACLE_PROFILE_PATH_ENV = "VLLM_EMULATOR_PROFILE_PACK"
ORACLE_MODE_ENV = "VLLM_EMULATOR_MODE"
# Legacy env var (backward compat)
ORACLE_BLOCKING_MODE_ENV = "VLLM_EMULATOR_BLOCKING_MODE"

# Emulator modes
EMULATOR_MODE_REALTIME = "realtime"      # Block for estimated latency (default)
EMULATOR_MODE_ACCELERATED = "accelerated"  # No blocking (virtual time, like REVATI)

# Backward compat mapping
_MODE_ALIASES = {"online": EMULATOR_MODE_REALTIME, "offline": EMULATOR_MODE_ACCELERATED}


class GpuWorkerHook:
    """Hook that intercepts GPU worker execution for emulator cost estimation.
    
    When enabled (via environment variables), this hook:
    1. Loads a profile pack from disk
    2. Creates a GPU cost oracle
    3. Intercepts execute_model() calls to estimate costs instead of running real inference
    """

    def __init__(self, worker: "Worker"):
        self._worker = worker
        self._oracle: BaseGpuCostOracle | None = None
        self._enabled = False
        self._emulator_mode = EMULATOR_MODE_REALTIME  # Default
        self._rng = random.Random(42)  # Deterministic fake token generation

        # Try to extract vocab_size and eos_token_id from model config
        self._vocab_size = _DEFAULT_VOCAB_SIZE
        self._eos_token_id = _DEFAULT_EOS_TOKEN_ID
        try:
            model_config = worker.vllm_config.model_config
            if hasattr(model_config, 'get_vocab_size'):
                self._vocab_size = model_config.get_vocab_size()
            if hasattr(model_config, 'hf_config'):
                eos = getattr(model_config.hf_config, 'eos_token_id', None)
                if eos is not None:
                    self._eos_token_id = eos if isinstance(eos, int) else eos[0]
        except Exception:
            pass  # Use defaults

        # Per-step overhead constant (calibrated, applied to all steps)
        self._step_overhead_us = float(
            os.environ.get("VLLM_EMULATOR_STEP_OVERHEAD_US", "0")
        )
        # Decode-only overhead: accounts for output processing, sampling,
        # and scheduling overhead that is cheaper with fake outputs than
        # real GPU outputs. Only applied to steps with decode sequences.
        self._decode_overhead_us = float(
            os.environ.get("VLLM_EMULATOR_DECODE_OVERHEAD_US", "0")
        )

        self._initialize_oracle()

    def _initialize_oracle(self) -> None:
        """Initialize oracle from profile pack if enabled."""
        if not os.environ.get(ORACLE_ENABLED_ENV, "").lower() in ("1", "true", "yes"):
            return

        profile_path = os.environ.get(ORACLE_PROFILE_PATH_ENV)
        if not profile_path:
            raise ValueError(
                f"{ORACLE_ENABLED_ENV} is set but {ORACLE_PROFILE_PATH_ENV} is not configured"
            )

        # Determine emulator mode (realtime or accelerated)
        mode = os.environ.get(ORACLE_MODE_ENV, "").lower()
        if not mode:
            # Fallback to legacy env var
            mode = os.environ.get(ORACLE_BLOCKING_MODE_ENV, EMULATOR_MODE_REALTIME).lower()
        # Apply backward compat aliases (online→realtime, offline→accelerated)
        mode = _MODE_ALIASES.get(mode, mode)
        self._emulator_mode = mode if mode == EMULATOR_MODE_ACCELERATED else EMULATOR_MODE_REALTIME

        try:
            profile_pack = load_profile_pack(profile_path)
            self._oracle = create_oracle_from_profile_pack(profile_pack)
            self._enabled = True
        except Exception as e:
            raise RuntimeError(
                f"Failed to load profile pack for oracle: {e}"
            ) from e

    @property
    def is_enabled(self) -> bool:
        """Return whether the hook is active."""
        return self._enabled

    @property
    def emulator_mode(self) -> str:
        """Return the emulator mode: 'realtime' or 'accelerated'."""
        return self._emulator_mode

    @property
    def blocking_mode(self) -> str:
        """Backward compat alias for emulator_mode."""
        return self._emulator_mode

    @property
    def should_block(self) -> bool:
        """Return whether we should block for timing simulation.

        True for realtime mode (default), False for accelerated mode.
        """
        return self._emulator_mode == EMULATOR_MODE_REALTIME

    @property
    def oracle(self) -> BaseGpuCostOracle | None:
        """Return the cost oracle if enabled."""
        return self._oracle

    def estimate_execution_cost(
        self, scheduler_output: "SchedulerOutput"
    ) -> dict[str, float]:
        """Estimate the cost of executing the scheduled work.
        
        Analyzes scheduler output and returns estimated latencies for
        prefill and decode phases.
        
        Returns:
            Dict with keys:
            - prefill_latency_us: Estimated prefill latency (microseconds)
            - decode_latency_us: Estimated decode latency per token
            - total_estimated_us: Combined estimate (batch-level forward pass)
        """
        if not self._enabled or self._oracle is None:
            return {"prefill_latency_us": 0, "decode_latency_us": 0, "total_estimated_us": 0}

        # UNIFIED estimation via total_tokens
        #
        # vLLM v1 runs a SINGLE fused forward pass over all tokens in
        # the batch.  The cost is a function of total_num_scheduled_tokens
        # regardless of how they split between prefill and decode.
        #
        # We also compute separate prefill/decode estimates for logging.

        total_tokens = scheduler_output.total_num_scheduled_tokens

        # For logging breakdown only:
        total_prefill_tokens = 0
        for req in scheduler_output.scheduled_new_reqs:
            if req.prompt_token_ids:
                total_prefill_tokens += len(req.prompt_token_ids)
        cached = scheduler_output.scheduled_cached_reqs
        num_decode_seqs = cached.num_reqs if cached.num_reqs > 0 else 0

        prefill_latency = 0.0
        if total_prefill_tokens > 0 and self._oracle._prefill_samples:
            prefill_latency = self._oracle.estimate_prefill_latency_us(
                total_prefill_tokens, batch_size=1
            )
        decode_latency = 0.0
        if num_decode_seqs > 0 and self._oracle._decode_samples:
            decode_latency = self._oracle.estimate_decode_latency_us(
                num_decode_seqs
            )

        # Unified: one forward pass for all tokens
        has_prefill = len(scheduler_output.scheduled_new_reqs) > 0
        batch_latency = self._oracle.estimate_step_latency_us(
            total_tokens, has_prefill=has_prefill)

        # Add per-step overhead constant (calibrated)
        batch_latency += self._step_overhead_us

        # Add decode-specific overhead (output processing gap)
        if self._decode_overhead_us > 0 and num_decode_seqs > 0:
            batch_latency += self._decode_overhead_us

        return {
            "prefill_latency_us": prefill_latency,
            "decode_latency_us": decode_latency,
            "total_estimated_us": batch_latency,
        }

    def should_use_oracle(self, scheduler_output: "SchedulerOutput") -> bool:
        """Determine if oracle should be used for this scheduling iteration.
        
        Currently always returns True when enabled, but can be extended to
        selectively enable oracle for certain request types.
        """
        return self._enabled and scheduler_output.total_num_scheduled_tokens > 0

    def create_fake_output(
        self, scheduler_output: "SchedulerOutput"
    ) -> "ModelRunnerOutput | None":
        """Create a fake ModelRunnerOutput based on oracle estimates.

        When the emulator oracle is enabled, this method creates a fake output
        that mimics the structure of a real model execution output, but with
        dummy token data. This allows the scheduler to continue working without
        actual GPU inference.

        Each request produces exactly 1 sampled token per step (standard
        auto-regressive decode). The token is drawn from a deterministic RNG
        seeded per-hook, excluding the EOS token so that requests run until
        max_tokens (the scheduler controls stopping, not fake EOS).

        Args:
            scheduler_output: The scheduler output containing request info.

        Returns:
            A ModelRunnerOutput with fake sampled tokens, or None if oracle
            is not enabled or no tokens were scheduled.
        """
        if not self.should_use_oracle(scheduler_output):
            return None

        # Import here to avoid circular imports
        from vllm.v1.outputs import ModelRunnerOutput, LogprobsLists

        # Get request IDs from scheduler output
        req_ids = list(scheduler_output.num_scheduled_tokens.keys())

        if not req_ids:
            return None

        # Token generation rules for async scheduler compatibility:
        # - Prefill chunk (partial prompt): 0 tokens (no placeholder allocated)
        # - Prefill complete (full prompt) or decode: 1 token
        #
        # A request is a "prefill chunk" if num_scheduled_tokens < remaining
        # prompt tokens. For new requests, we check if the scheduled tokens
        # cover the full prompt.
        new_req_ids = {}
        for req in scheduler_output.scheduled_new_reqs:
            prompt_len = len(req.prompt_token_ids) if req.prompt_token_ids else 0
            scheduled = scheduler_output.num_scheduled_tokens.get(req.req_id, 0)
            # If scheduled < prompt_len, this is a partial prefill chunk
            new_req_ids[req.req_id] = (prompt_len, scheduled)

        vocab = self._vocab_size
        eos = self._eos_token_id
        sampled_token_ids: list[list[int]] = []
        for req_id in req_ids:
            if req_id in new_req_ids:
                prompt_len, scheduled = new_req_ids[req_id]
                if scheduled < prompt_len:
                    # Partial prefill chunk: no token generated
                    sampled_token_ids.append([])
                    continue
            # Full prefill or decode: generate 1 fake token
            tok = self._rng.randrange(vocab)
            while tok == eos:
                tok = self._rng.randrange(vocab)
            sampled_token_ids.append([tok])

        # Create req_id_to_index mapping
        req_id_to_index = {rid: idx for idx, rid in enumerate(req_ids)}

        # Logprobs: 1 position per token-producing request.
        # Requests with empty sampled_token_ids (prefill chunks) are excluded.
        num_with_tokens = sum(1 for toks in sampled_token_ids if toks)
        if num_with_tokens > 0:
            token_vals = [toks[0] for toks in sampled_token_ids if toks]
            logprobs = LogprobsLists(
                logprob_token_ids=np.array(
                    [[t] for t in token_vals], dtype=np.int32
                ),
                logprobs=np.full((num_with_tokens, 1), -0.1, dtype=np.float32),
                sampled_token_ranks=np.zeros(num_with_tokens, dtype=np.int32),
            )
        else:
            logprobs = None

        # Create the fake output
        fake_output = ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index=req_id_to_index,
            sampled_token_ids=sampled_token_ids,
            logprobs=logprobs,
            prompt_logprobs_dict={},
            pooler_output=[None] * len(req_ids),
        )

        return fake_output


def install_worker_hook(worker: "Worker") -> GpuWorkerHook:
    """Install the GPU worker hook onto a worker instance.
    
    Args:
        worker: The GPU worker to hook.
        
    Returns:
        The installed hook instance.
    """
    return GpuWorkerHook(worker)
