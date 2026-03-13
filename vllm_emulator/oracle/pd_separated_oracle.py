"""PD-Separated cost oracle for disaggregated prefill/decode scheduling."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm_emulator.oracle.base import BaseGpuCostOracle


class PDSeparatedCostOracle:
    """Cost oracle that separates prefill and decode timing estimates.
    
    This wraps a base CostOracle and provides phase-specific timing methods
    for disaggregated prefill/decode (PD) scheduling scenarios.
    """

    def __init__(self, base_oracle: BaseGpuCostOracle):
        """Initialize PD-separated oracle with a base oracle.
        
        Args:
            base_oracle: The underlying cost oracle to use for estimates.
        """
        self._base = base_oracle

    @property
    def gpu_model(self) -> str:
        """Return the GPU model this oracle is calibrated for."""
        return self._base.gpu_model

    def estimate_prefill_time(self, num_tokens: int, batch_size: int) -> float:
        """Estimate prefill time in microseconds.
        
        Args:
            num_tokens: Number of tokens in the prompt.
            batch_size: Number of sequences in the prefill batch.
            
        Returns:
            Estimated prefill time in microseconds.
        """
        return self._base.estimate_prefill_latency_us(num_tokens, batch_size)

    def estimate_decode_time(self, num_tokens: int, batch_size: int) -> float:
        """Estimate decode time per token in microseconds.
        
        Args:
            num_tokens: Number of tokens to decode (typically 1 per step).
            batch_size: Number of active sequences in decode batch.
            
        Returns:
            Estimated decode time per token in microseconds.
        """
        # The base oracle returns per-token latency for active sequences
        per_token_latency = self._base.estimate_decode_latency_us(batch_size)
        return per_token_latency * num_tokens

    def estimate_prefill_latency_us(self, prompt_tokens: int, batch_size: int) -> float:
        """Alias for estimate_prefill_time for backward compatibility."""
        return self.estimate_prefill_time(prompt_tokens, batch_size)

    def estimate_decode_latency_us(self, active_seqs: int) -> float:
        """Alias for estimate_decode_time with batch_size=1."""
        return self.estimate_decode_time(1, active_seqs)


def create_pd_separated_oracle(base_oracle: BaseGpuCostOracle) -> PDSeparatedCostOracle:
    """Factory function to create a PD-separated oracle from a base oracle."""
    return PDSeparatedCostOracle(base_oracle)
