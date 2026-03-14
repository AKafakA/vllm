"""Base interfaces for emulator GPU and offload cost oracles."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum


class TransferDirection(Enum):
    """Direction of KV data transfer."""
    CPU_TO_GPU = "cpu_to_gpu"
    GPU_TO_CPU = "gpu_to_cpu"


class BaseGpuCostOracle(ABC):
    """Abstract interface for estimating GPU compute costs in microseconds."""

    @abstractmethod
    def estimate_prefill_latency_us(self, prompt_tokens: int, batch_size: int) -> float:
        """Estimate prefill latency for a prefill micro-batch."""

    @abstractmethod
    def estimate_decode_latency_us(self, active_seqs: int) -> float:
        """Estimate decode latency per token step for active sequences."""


class BaseOffloadCostOracle(ABC):
    """Abstract interface for estimating KV offload transfer costs in microseconds."""

    @abstractmethod
    def get_lookup_latency_us(self, num_blocks: int) -> float:
        """Estimate latency for looking up offloaded blocks.
        
        Args:
            num_blocks: Number of KV blocks to lookup.
            
        Returns:
            Estimated latency in microseconds.
        """

    @abstractmethod
    def get_transfer_latency_us(
        self, 
        num_bytes: int, 
        direction: TransferDirection,
        concurrency: int = 1
    ) -> float:
        """Estimate latency for transferring KV data between CPU and GPU.
        
        Args:
            num_bytes: Number of bytes to transfer.
            direction: Transfer direction (CPU->GPU or GPU->CPU).
            concurrency: Number of concurrent transfers (affects bandwidth).
            
        Returns:
            Estimated latency in microseconds.
        """

    @abstractmethod
    def get_evict_latency_us(self, num_blocks: int) -> float:
        """Estimate latency for evicting (preparing to store) KV blocks.
        
        Args:
            num_blocks: Number of KV blocks to evict.
            
        Returns:
            Estimated latency in microseconds.
        """
