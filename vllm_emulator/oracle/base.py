"""Base interfaces for emulator GPU and offload cost oracles."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any


class TransferDirection(Enum):
    """Direction of KV data transfer."""
    CPU_TO_GPU = "cpu_to_gpu"
    GPU_TO_CPU = "gpu_to_cpu"
    GPU_TO_GPU = "gpu_to_gpu"


class NetworkTopology(Enum):
    """Network interconnect topology types."""
    NVLINK = "nvlink"
    PCIE = "pcie"
    INFINIBAND = "infiniband"


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
        """Estimate latency for looking up offloaded blocks."""

    @abstractmethod
    def get_transfer_latency_us(
        self, 
        num_bytes: int, 
        direction: TransferDirection,
        concurrency: int = 1
    ) -> float:
        """Estimate latency for transferring KV data between CPU and GPU."""

    @abstractmethod
    def get_evict_latency_us(self, num_blocks: int) -> float:
        """Estimate latency for evicting (preparing to store) KV blocks."""


class BaseNetworkCostOracle(ABC):
    """Abstract interface for estimating inter-GPU network communication costs."""

    @abstractmethod
    def get_all_reduce_latency_us(
        self,
        num_bytes: int,
        world_size: int,
        topology: NetworkTopology,
    ) -> float:
        """Estimate all-reduce latency for gradient/embedding aggregation."""

    @abstractmethod
    def get_send_latency_us(
        self,
        num_bytes: int,
        topology: NetworkTopology,
    ) -> float:
        """Estimate point-to-point send latency."""

    @abstractmethod
    def get_recv_latency_us(
        self,
        num_bytes: int,
        topology: NetworkTopology,
    ) -> float:
        """Estimate point-to-point receive latency."""

    @abstractmethod
    def get_kv_transfer_latency_us(
        self,
        num_bytes: int,
        direction: TransferDirection,
        topology: NetworkTopology,
        concurrency: int = 1,
    ) -> float:
        """Estimate KV cache transfer latency between GPUs (PD disaggregation)."""
