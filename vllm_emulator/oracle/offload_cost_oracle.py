"""Profile-driven offload cost oracle for vLLM emulator."""

from __future__ import annotations

from typing import Any

from .base import BaseOffloadCostOracle, TransferDirection


class ProfileOffloadCostOracle(BaseOffloadCostOracle):
    """Offload cost oracle that interpolates from a profile pack.
    
    Uses linear interpolation between profile pack samples to estimate
    lookup, transfer, and evict latencies for KV offload operations.
    """

    def __init__(self, profile_pack: dict[str, Any]):
        """Initialize oracle with a validated profile pack.
        
        Args:
            profile_pack: Validated profile pack dict with offload samples.
                Expected keys:
                - lookup: List[{"num_blocks": int, "latency_us": float}]
                - transfer.cpu_to_gpu: List[{"bytes": int, "latency_us": float}]
                - transfer.gpu_to_cpu: List[{"bytes": int, "latency_us": float}]
                - evict: List[{"num_blocks": int, "latency_us": float}]
        """
        self._profile = profile_pack
        self._lookup_samples = profile_pack.get("lookup", [])
        self._cpu_to_gpu_samples = profile_pack.get("transfer", {}).get("cpu_to_gpu", [])
        self._gpu_to_cpu_samples = profile_pack.get("transfer", {}).get("gpu_to_cpu", [])
        self._evict_samples = profile_pack.get("evict", [])

    def get_lookup_latency_us(self, num_blocks: int) -> float:
        """Estimate latency for looking up offloaded blocks via interpolation.

        Raises RuntimeError if the profile pack has no `lookup` samples — the
        emulator must refuse to predict rather than silently return a fabricated
        latency (see project rule: no magic numbers, no silent degradation).
        """
        samples = self._lookup_samples

        if not samples:
            raise RuntimeError(
                "ProfileOffloadCostOracle.get_lookup_latency_us called but profile "
                "pack has no 'lookup' samples. Capture offload lookup profile data "
                "before enabling this oracle, or disable it. Refusing to fabricate "
                "a fallback latency."
            )

        block_counts = [s["num_blocks"] for s in samples]
        
        if num_blocks <= block_counts[0]:
            return samples[0]["latency_us"]
        
        if num_blocks >= block_counts[-1]:
            return samples[-1]["latency_us"]
        
        # Linear interpolation
        for i in range(len(block_counts) - 1):
            if block_counts[i] <= num_blocks <= block_counts[i + 1]:
                lo, hi = samples[i], samples[i + 1]
                ratio = (num_blocks - lo["num_blocks"]) / (hi["num_blocks"] - lo["num_blocks"])
                return lo["latency_us"] + ratio * (hi["latency_us"] - lo["latency_us"])
        
        return samples[-1]["latency_us"]

    def get_transfer_latency_us(
        self, 
        num_bytes: int, 
        direction: TransferDirection,
        concurrency: int = 1
    ) -> float:
        """Estimate transfer latency via interpolation.
        
        Accounts for concurrency by scaling down latency when multiple
        transfers are active (simulating pipelining/bandwidth sharing).
        """
        if direction == TransferDirection.CPU_TO_GPU:
            samples = self._cpu_to_gpu_samples
            direction_name = "cpu_to_gpu"
        else:
            samples = self._gpu_to_cpu_samples
            direction_name = "gpu_to_cpu"

        if not samples:
            raise RuntimeError(
                f"ProfileOffloadCostOracle.get_transfer_latency_us called but "
                f"profile pack has no 'transfer.{direction_name}' samples. "
                f"Capture offload transfer profile data before enabling this "
                f"oracle, or disable it. Refusing to fabricate a fallback."
            )

        byte_counts = [s["bytes"] for s in samples]
        
        if num_bytes <= byte_counts[0]:
            base_latency = samples[0]["latency_us"]
        elif num_bytes >= byte_counts[-1]:
            base_latency = samples[-1]["latency_us"]
        else:
            # Linear interpolation
            for i in range(len(byte_counts) - 1):
                if byte_counts[i] <= num_bytes <= byte_counts[i + 1]:
                    lo, hi = samples[i], samples[i + 1]
                    ratio = (num_bytes - lo["bytes"]) / (hi["bytes"] - lo["bytes"])
                    base_latency = lo["latency_us"] + ratio * (hi["latency_us"] - lo["latency_us"])
                    break
            else:
                base_latency = samples[-1]["latency_us"]
        
        # Scale for concurrency: more concurrent transfers = less bandwidth per transfer
        # Simple model: latency scales with sqrt of concurrency (bandwidth saturation)
        if concurrency > 1:
            # Use square root scaling to model bandwidth contention
            scale_factor = (concurrency ** 0.5)
            base_latency *= scale_factor
        
        return base_latency

    def get_evict_latency_us(self, num_blocks: int) -> float:
        """Estimate eviction latency via interpolation.

        Raises RuntimeError if the profile pack has no `evict` samples.
        """
        samples = self._evict_samples

        if not samples:
            raise RuntimeError(
                "ProfileOffloadCostOracle.get_evict_latency_us called but profile "
                "pack has no 'evict' samples. Capture offload evict profile data "
                "before enabling this oracle, or disable it."
            )

        block_counts = [s["num_blocks"] for s in samples]
        
        if num_blocks <= block_counts[0]:
            return samples[0]["latency_us"]
        
        if num_blocks >= block_counts[-1]:
            return samples[-1]["latency_us"]
        
        # Linear interpolation
        for i in range(len(block_counts) - 1):
            if block_counts[i] <= num_blocks <= block_counts[i + 1]:
                lo, hi = samples[i], samples[i + 1]
                ratio = (num_blocks - lo["num_blocks"]) / (hi["num_blocks"] - lo["num_blocks"])
                return lo["latency_us"] + ratio * (hi["latency_us"] - lo["latency_us"])
        
        return samples[-1]["latency_us"]


def create_offload_oracle_from_profile_pack(profile_pack: dict[str, Any]) -> ProfileOffloadCostOracle:
    """Factory function to create an offload oracle from a profile pack."""
    return ProfileOffloadCostOracle(profile_pack)
