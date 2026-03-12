"""Base interfaces for emulator GPU cost oracles."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseGpuCostOracle(ABC):
    """Abstract interface for estimating GPU compute costs in microseconds."""

    @abstractmethod
    def estimate_prefill_latency_us(self, prompt_tokens: int, batch_size: int) -> float:
        """Estimate prefill latency for a prefill micro-batch."""

    @abstractmethod
    def estimate_decode_latency_us(self, active_seqs: int) -> float:
        """Estimate decode latency per token step for active sequences."""
