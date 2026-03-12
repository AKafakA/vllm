"""Profile-driven GPU cost oracle for vLLM emulator."""

from __future__ import annotations

from typing import Any

from .base import BaseGpuCostOracle


class ProfileGpuCostOracle(BaseGpuCostOracle):
    """GPU cost oracle that interpolates from a profile pack.
    
    Uses linear interpolation between profile pack samples to estimate
    prefill and decode latencies.
    """

    def __init__(self, profile_pack: dict[str, Any]):
        """Initialize oracle with a validated profile pack.
        
        Args:
            profile_pack: Validated profile pack dict with prefill/decode samples.
        """
        self._profile = profile_pack
        self._prefill_samples = profile_pack["prefill"]
        self._decode_samples = profile_pack["decode"]
        self._gpu_model = profile_pack["gpu_model"]

    @property
    def gpu_model(self) -> str:
        """Return the GPU model this oracle is calibrated for."""
        return self._gpu_model

    def estimate_prefill_latency_us(self, prompt_tokens: int, batch_size: int) -> float:
        """Estimate prefill latency via interpolation.
        
        For simplicity, uses batch_size=1 lookup with seq_len scaling.
        More sophisticated batching models can be added later.
        """
        samples = self._prefill_samples
        
        # Find bracketing samples for seq_len
        seq_lens = [s["seq_len"] for s in samples]
        
        if prompt_tokens <= seq_lens[0]:
            # Below minimum sample - use minimum
            return samples[0]["latency_us"]
        
        if prompt_tokens >= seq_lens[-1]:
            # Above maximum sample - use maximum
            return samples[-1]["latency_us"]
        
        # Linear interpolation between bracketing samples
        for i in range(len(seq_lens) - 1):
            if seq_lens[i] <= prompt_tokens <= seq_lens[i + 1]:
                # Interpolate
                lo, hi = samples[i], samples[i + 1]
                ratio = (prompt_tokens - lo["seq_len"]) / (hi["seq_len"] - lo["seq_len"])
                return lo["latency_us"] + ratio * (hi["latency_us"] - lo["latency_us"])
        
        # Fallback (shouldn't reach here)
        return samples[-1]["latency_us"]

    def estimate_decode_latency_us(self, active_seqs: int) -> float:
        """Estimate decode latency per token via interpolation.
        
        Returns latency in microseconds per token for the given number
        of active decode sequences.
        """
        samples = self._decode_samples
        
        # Find bracketing samples for active_seqs
        seq_counts = [s["active_seqs"] for s in samples]
        
        if active_seqs <= seq_counts[0]:
            return samples[0]["latency_us_per_token"]
        
        if active_seqs >= seq_counts[-1]:
            return samples[-1]["latency_us_per_token"]
        
        # Linear interpolation
        for i in range(len(seq_counts) - 1):
            if seq_counts[i] <= active_seqs <= seq_counts[i + 1]:
                lo, hi = samples[i], samples[i + 1]
                ratio = (active_seqs - lo["active_seqs"]) / (hi["active_seqs"] - lo["active_seqs"])
                return lo["latency_us_per_token"] + ratio * (
                    hi["latency_us_per_token"] - lo["latency_us_per_token"]
                )
        
        return samples[-1]["latency_us_per_token"]


def create_oracle_from_profile_pack(profile_pack: dict[str, Any]) -> ProfileGpuCostOracle:
    """Factory function to create an oracle from a profile pack."""
    return ProfileGpuCostOracle(profile_pack)
