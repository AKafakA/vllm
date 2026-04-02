"""Offload worker hook for emulator mode cost estimation."""

from __future__ import annotations

import os
import threading
import time
from typing import TYPE_CHECKING, Any

from vllm_emulator.oracle import (
    BaseOffloadCostOracle,
    TransferDirection,
    create_offload_oracle_from_profile_pack,
)
from vllm_emulator.profile.loader import load_profile_pack

if TYPE_CHECKING:
    from vllm.v1.kv_offload.worker.worker import OffloadingWorker, TransferSpec


# Environment variable to enable emulator offload oracle
OFFLOAD_ORACLE_ENABLED_ENV = "VLLM_EMULATOR_ENABLE_OFFLOAD_ORACLE"
OFFLOAD_PROFILE_PATH_ENV = "VLLM_EMULATOR_OFFLOAD_PROFILE_PACK"
OFFLOAD_BLOCKING_MODE_ENV = "VLLM_EMULATOR_OFFLOAD_BLOCKING_MODE"

# Blocking modes
BLOCKING_MODE_ONLINE = "online"  # Block for estimated latency (default)
BLOCKING_MODE_OFFLINE = "offline"  # No blocking (virtual time)


# Default bytes per KV block (assuming block_size=16, num_kv_heads=8, head_size=128)
DEFAULT_BYTES_PER_BLOCK = 16 * 8 * 128 * 2  # ~16KB per block (fp16)


class OffloadWorkerHook:
    """Hook that intercepts offload worker execution for emulator cost estimation.
    
    When enabled (via environment variables), this hook:
    1. Loads a profile pack from disk
    2. Creates an offload cost oracle
    3. Intercepts transfer_async() calls to estimate costs instead of running real transfers
    """

    def __init__(self, worker: "OffloadingWorker", bytes_per_block: int = DEFAULT_BYTES_PER_BLOCK):
        self._worker = worker
        self._bytes_per_block = bytes_per_block
        self._oracle: BaseOffloadCostOracle | None = None
        self._enabled = False
        self._blocking_mode = BLOCKING_MODE_ONLINE
        self._active_transfers = 0
        self._lock = threading.Lock()
        self._initialize_oracle()

    def _initialize_oracle(self) -> None:
        """Initialize oracle from profile pack if enabled."""
        if not os.environ.get(OFFLOAD_ORACLE_ENABLED_ENV, "").lower() in ("1", "true", "yes"):
            return

        profile_path = os.environ.get(OFFLOAD_PROFILE_PATH_ENV)
        if not profile_path:
            raise ValueError(
                f"{OFFLOAD_ORACLE_ENABLED_ENV} is set but {OFFLOAD_PROFILE_PATH_ENV} is not configured"
            )

        # Determine blocking mode
        blocking_mode = os.environ.get(OFFLOAD_BLOCKING_MODE_ENV, BLOCKING_MODE_ONLINE).lower()
        if blocking_mode == BLOCKING_MODE_OFFLINE:
            self._blocking_mode = BLOCKING_MODE_OFFLINE
        else:
            self._blocking_mode = BLOCKING_MODE_ONLINE

        try:
            profile_pack = load_profile_pack(profile_path)
            self._oracle = create_offload_oracle_from_profile_pack(profile_pack)
            self._enabled = True
        except Exception as e:
            raise RuntimeError(
                f"Failed to load offload profile pack for oracle: {e}"
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
        """Return whether we should block for timing simulation.
        
        True for online serving (default), False for offline (virtual time).
        """
        return self._blocking_mode == BLOCKING_MODE_ONLINE

    @property
    def oracle(self) -> BaseOffloadCostOracle | None:
        """Return the cost oracle if enabled."""
        return self._oracle

    def estimate_transfer_cost(
        self, 
        src_spec: Any, 
        dst_spec: Any,
    ) -> dict[str, float]:
        """Estimate the cost of a KV transfer operation.
        
        Analyzes transfer specs and returns estimated latencies.
        
        Returns:
            Dict with keys:
            - lookup_latency_us: Estimated lookup latency
            - transfer_latency_us: Estimated transfer latency
            - total_estimated_us: Combined estimate
        """
        if not self._enabled or self._oracle is None:
            return {"lookup_latency_us": 0, "transfer_latency_us": 0, "total_estimated_us": 0}

        # Calculate number of blocks from specs
        num_blocks = self._estimate_num_blocks(src_spec, dst_spec)
        num_bytes = num_blocks * self._bytes_per_block

        # Determine direction
        src_medium = src_spec.medium() if hasattr(src_spec, 'medium') else "UNKNOWN"
        dst_medium = dst_spec.medium() if hasattr(dst_spec, 'medium') else "UNKNOWN"
        
        if src_medium == "CPU" and dst_medium == "GPU":
            direction = TransferDirection.CPU_TO_GPU
        elif src_medium == "GPU" and dst_medium == "CPU":
            direction = TransferDirection.GPU_TO_CPU
        else:
            # Unknown direction, assume no transfer cost
            return {"lookup_latency_us": 0, "transfer_latency_us": 0, "total_estimated_us": 0}

        # Estimate lookup latency
        lookup_latency = self._oracle.get_lookup_latency_us(num_blocks)

        # Estimate transfer latency (with current concurrency)
        transfer_latency = self._oracle.get_transfer_latency_us(
            num_bytes, direction, concurrency=max(1, self._active_transfers)
        )

        return {
            "lookup_latency_us": lookup_latency,
            "transfer_latency_us": transfer_latency,
            "total_estimated_us": lookup_latency + transfer_latency,
        }

    def _estimate_num_blocks(self, src_spec: Any, dst_spec: Any) -> int:
        """Estimate number of blocks from transfer specs."""
        # Try to get block_ids from specs
        src_blocks = getattr(src_spec, 'block_ids', None)
        dst_blocks = getattr(dst_spec, 'block_ids', None)
        
        if src_blocks is not None and hasattr(src_blocks, '__len__'):
            return len(src_blocks)
        if dst_blocks is not None and hasattr(dst_blocks, '__len__'):
            return len(dst_blocks)
        
        # Default fallback
        return 1

    def should_use_oracle(self) -> bool:
        """Determine if oracle should be used."""
        return self._enabled

    def apply_oracle_delay(self, src_spec: Any, dst_spec: Any) -> bool:
        """Apply oracle-estimated delay if enabled.
        
        Returns:
            True if oracle mode is active and delay was applied.
            False if real transfer should proceed.
        """
        if not self.should_use_oracle():
            return False

        with self._lock:
            self._active_transfers += 1

        try:
            cost_estimate = self.estimate_transfer_cost(src_spec, dst_spec)
            total_latency_us = cost_estimate["total_estimated_us"]

            if total_latency_us > 0 and self.should_block:
                time.sleep(total_latency_us / 1e6)

            return True
        finally:
            with self._lock:
                self._active_transfers = max(0, self._active_transfers - 1)


def install_offload_worker_hook(
    worker: "OffloadingWorker", 
    bytes_per_block: int = DEFAULT_BYTES_PER_BLOCK
) -> OffloadWorkerHook:
    """Install the offload worker hook onto an OffloadingWorker instance.
    
    Args:
        worker: The OffloadingWorker to hook.
        bytes_per_block: Number of bytes per KV block (default: ~16KB).
        
    Returns:
        The installed hook instance.
    """
    return OffloadWorkerHook(worker, bytes_per_block)
