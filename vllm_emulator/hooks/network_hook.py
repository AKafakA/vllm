"""Network communication hook for emulator mode cost estimation."""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any

from vllm_emulator.oracle import (
    BaseNetworkCostOracle,
    NetworkTopology,
    TransferDirection,
    create_network_oracle_from_profile_pack,
)
from vllm_emulator.profile.loader import load_profile_pack

if TYPE_CHECKING:
    from vllm.distributed.communication_op import CUDACommunicator


# Environment variable to enable emulator network oracle
NETWORK_ORACLE_ENABLED_ENV = "VLLM_EMULATOR_ENABLE_NETWORK_ORACLE"
NETWORK_PROFILE_PATH_ENV = "VLLM_EMULATOR_NETWORK_PROFILE_PACK"
NETWORK_BLOCKING_MODE_ENV = "VLLM_EMULATOR_NETWORK_BLOCKING_MODE"
NETWORK_TOPOLOGY_ENV = "VLLM_EMULATOR_NETWORK_TOPOLOGY"

# Blocking modes
BLOCKING_MODE_ONLINE = "online"  # Block for estimated latency (default)
BLOCKING_MODE_OFFLINE = "offline"  # No blocking (virtual time)


class NetworkHook:
    """Hook that intercepts CUDA communication for emulator cost estimation.
    
    When enabled (via environment variables), this hook:
    1. Loads a profile pack from disk
    2. Creates a network cost oracle
    3. Intercepts all_reduce(), send(), recv() to estimate costs
    """

    def __init__(self, topology: str = "nvlink"):
        self._oracle: BaseNetworkCostOracle | None = None
        self._enabled = False
        self._blocking_mode = BLOCKING_MODE_ONLINE
        self._topology = self._parse_topology(topology)
        self._active_transfers = 0
        self._initialize_oracle()

    def _parse_topology(self, topology_str: str) -> NetworkTopology:
        """Parse topology string to enum."""
        topo = topology_str.lower()
        if topo in ("nvlink", "nvl"):
            return NetworkTopology.NVLINK
        elif topo in ("pcie", "pci"):
            return NetworkTopology.PCIE
        elif topo in ("ib", "infiniband"):
            return NetworkTopology.INFINIBAND
        else:
            return NetworkTopology.NVLINK  # Default

    def _initialize_oracle(self) -> None:
        """Initialize oracle from profile pack if enabled."""
        if not os.environ.get(NETWORK_ORACLE_ENABLED_ENV, "").lower() in ("1", "true", "yes"):
            return

        profile_path = os.environ.get(NETWORK_PROFILE_PATH_ENV)
        if not profile_path:
            raise ValueError(
                f"{NETWORK_ORACLE_ENABLED_ENV} is set but {NETWORK_PROFILE_PATH_ENV} is not configured"
            )

        # Determine blocking mode
        blocking_mode = os.environ.get(NETWORK_BLOCKING_MODE_ENV, BLOCKING_MODE_ONLINE).lower()
        self._blocking_mode = BLOCKING_MODE_OFFLINE if blocking_mode == BLOCKING_MODE_OFFLINE else BLOCKING_MODE_ONLINE

        # Determine topology
        topology_str = os.environ.get(NETWORK_TOPOLOGY_ENV, "nvlink").lower()
        self._topology = self._parse_topology(topology_str)

        try:
            profile_pack = load_profile_pack(profile_path)
            self._oracle = create_network_oracle_from_profile_pack(profile_pack)
            self._enabled = True
        except Exception as e:
            raise RuntimeError(
                f"Failed to load network profile pack for oracle: {e}"
            ) from e

    @property
    def is_enabled(self) -> bool:
        """Return whether the hook is active."""
        return self._enabled

    @property
    def blocking_mode(self) -> str:
        """Return the blocking mode: 'online' or 'offline'."""
        return self._blocking_mode

    @property
    def should_block(self) -> bool:
        """Return whether we should block for timing simulation."""
        return self._blocking_mode == BLOCKING_MODE_ONLINE

    @property
    def oracle(self) -> BaseNetworkCostOracle | None:
        """Return the cost oracle if enabled."""
        return self._oracle

    @property
    def topology(self) -> NetworkTopology:
        """Return the configured network topology."""
        return self._topology

    def estimate_all_reduce_cost(
        self,
        num_bytes: int,
        world_size: int,
    ) -> float:
        """Estimate all-reduce communication cost."""
        if not self._enabled or self._oracle is None:
            return 0.0
        return self._oracle.get_all_reduce_latency_us(num_bytes, world_size, self._topology)

    def estimate_send_cost(self, num_bytes: int) -> float:
        """Estimate P2P send cost."""
        if not self._enabled or self._oracle is None:
            return 0.0
        return self._oracle.get_send_latency_us(num_bytes, self._topology)

    def estimate_recv_cost(self, num_bytes: int) -> float:
        """Estimate P2P receive cost."""
        if not self._enabled or self._oracle is None:
            return 0.0
        return self._oracle.get_recv_latency_us(num_bytes, self._topology)

    def estimate_kv_transfer_cost(
        self,
        num_bytes: int,
        direction: TransferDirection = TransferDirection.GPU_TO_GPU,
    ) -> float:
        """Estimate KV cache transfer cost between GPUs."""
        if not self._enabled or self._oracle is None:
            return 0.0
        return self._oracle.get_kv_transfer_latency_us(
            num_bytes, direction, self._topology, max(1, self._active_transfers)
        )

    def apply_all_reduce_delay(self, num_bytes: int, world_size: int) -> bool:
        """Apply oracle-estimated delay for all-reduce.
        
        Returns:
            True if oracle mode is active and delay was applied.
        """
        if not self.is_enabled:
            return False

        latency_us = self.estimate_all_reduce_cost(num_bytes, world_size)
        if latency_us > 0 and self.should_block:
            time.sleep(latency_us / 1e6)
        return True

    def apply_send_delay(self, num_bytes: int) -> bool:
        """Apply oracle-estimated delay for send."""
        if not self.is_enabled:
            return False

        self._active_transfers += 1
        try:
            latency_us = self.estimate_send_cost(num_bytes)
            if latency_us > 0 and self.should_block:
                time.sleep(latency_us / 1e6)
            return True
        finally:
            self._active_transfers = max(0, self._active_transfers - 1)

    def apply_recv_delay(self, num_bytes: int) -> bool:
        """Apply oracle-estimated delay for recv."""
        if not self.is_enabled:
            return False

        self._active_transfers += 1
        try:
            latency_us = self.estimate_recv_cost(num_bytes)
            if latency_us > 0 and self.should_block:
                time.sleep(latency_us / 1e6)
            return True
        finally:
            self._active_transfers = max(0, self._active_transfers - 1)

    def apply_kv_transfer_delay(
        self,
        num_bytes: int,
        direction: TransferDirection = TransferDirection.GPU_TO_GPU,
    ) -> bool:
        """Apply oracle-estimated delay for KV transfer."""
        if not self.is_enabled:
            return False

        self._active_transfers += 1
        try:
            latency_us = self.estimate_kv_transfer_cost(num_bytes, direction)
            if latency_us > 0 and self.should_block:
                time.sleep(latency_us / 1e6)
            return True
        finally:
            self._active_transfers = max(0, self._active_transfers - 1)


# Global hook instance for module-level installation
_global_hook: NetworkHook | None = None


def get_network_hook() -> NetworkHook:
    """Get or create the global network hook instance."""
    global _global_hook
    if _global_hook is None:
        topology = os.environ.get(NETWORK_TOPOLOGY_ENV, "nvlink")
        _global_hook = NetworkHook(topology)
    return _global_hook


def install_network_hook(topology: str = "nvlink") -> NetworkHook:
    """Install the network hook as the global instance.
    
    Args:
        topology: Network topology ("nvlink", "pcie", or "ib").
        
    Returns:
        The installed hook instance.
    """
    global _global_hook
    _global_hook = NetworkHook(topology)
    return _global_hook
