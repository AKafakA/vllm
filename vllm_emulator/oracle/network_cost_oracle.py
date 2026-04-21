"""Profile-driven network cost oracle for inter-GPU communication."""

from __future__ import annotations

from typing import Any

from .base import BaseNetworkCostOracle, NetworkTopology, TransferDirection


class ProfileNetworkCostOracle(BaseNetworkCostOracle):
    """Network cost oracle that interpolates from profile measurements.
    
    Supports NVLink, PCIe, and InfiniBand topologies with linear interpolation
    between profile samples. Models bandwidth saturation with concurrency.
    """

    def __init__(self, profile_pack: dict[str, Any]):
        """Initialize oracle with a validated profile pack.
        
        Args:
            profile_pack: Validated profile pack dict with network samples.
                Expected structure:
                {
                    "all_reduce": {
                        "nvlink": [{"bytes": int, "world_size": int, "latency_us": float}, ...],
                        "pcie": [...],
                        "ib": [...]
                    },
                    "send_recv": {
                        "nvlink": [{"bytes": int, "latency_us": float}, ...],
                        ...
                    },
                    "kv_transfer": {
                        "nvlink": [{"bytes": int, "latency_us": float}, ...],
                        ...
                    }
                }
        """
        self._profile = profile_pack
        
        # Extract samples by topology
        all_reduce = profile_pack.get("all_reduce", {})
        self._all_reduce_samples = {
            NetworkTopology.NVLINK: all_reduce.get("nvlink", []),
            NetworkTopology.PCIE: all_reduce.get("pcie", []),
            NetworkTopology.INFINIBAND: all_reduce.get("ib", []),
        }
        
        send_recv = profile_pack.get("send_recv", {})
        self._send_samples = {
            NetworkTopology.NVLINK: send_recv.get("nvlink", []),
            NetworkTopology.PCIE: send_recv.get("pcie", []),
            NetworkTopology.INFINIBAND: send_recv.get("ib", []),
        }
        self._recv_samples = self._send_samples  # Send/recv typically symmetric
        
        kv_transfer = profile_pack.get("kv_transfer", {})
        self._kv_transfer_samples = {
            NetworkTopology.NVLINK: kv_transfer.get("nvlink", []),
            NetworkTopology.PCIE: kv_transfer.get("pcie", []),
            NetworkTopology.INFINIBAND: kv_transfer.get("ib", []),
        }

    def _interpolate(
        self,
        samples: list[dict[str, Any]],
        x_key: str,
        x_value: float,
    ) -> float:
        """Linear interpolation between samples.
        
        Args:
            samples: List of profile samples.
            x_key: Key to extract x-axis value (e.g., "bytes", "world_size").
            x_value: Value to interpolate at.
            
        Returns:
            Interpolated latency in microseconds.

        Raises:
            RuntimeError: if `samples` is empty. Callers must provide profiled
                samples; no silent-zero fallback.
        """
        if not samples:
            raise RuntimeError(
                f"ProfileNetworkCostOracle._interpolate called with empty samples "
                f"for key '{x_key}'. Profile pack is missing required network "
                f"measurements for the requested topology/operation. Refusing to "
                f"fabricate a zero-latency fallback."
            )

        # Extract x values
        x_values = [s[x_key] for s in samples]
        
        if x_value <= x_values[0]:
            return samples[0]["latency_us"]
        if x_value >= x_values[-1]:
            return samples[-1]["latency_us"]
        
        # Find bracket and interpolate
        for i in range(len(x_values) - 1):
            if x_values[i] <= x_value <= x_values[i + 1]:
                lo, hi = samples[i], samples[i + 1]
                ratio = (x_value - lo[x_key]) / (hi[x_key] - lo[x_key])
                return lo["latency_us"] + ratio * (hi["latency_us"] - lo["latency_us"])
        
        return samples[-1]["latency_us"]

    def _interpolate_2d(
        self,
        samples: list[dict[str, Any]],
        x_key: str,
        y_key: str,
        x_value: float,
        y_value: float,
    ) -> float:
        """Bilinear interpolation for all-reduce (bytes × world_size).
        
        Args:
            samples: List of profile samples with x and y dimensions.
            x_key: Primary key (e.g., "bytes").
            y_key: Secondary key (e.g., "world_size").
            x_value: Primary value.
            y_value: Secondary value.
            
        Returns:
            Interpolated latency in microseconds.

        Raises:
            RuntimeError: if `samples` is empty.
        """
        if not samples:
            raise RuntimeError(
                f"ProfileNetworkCostOracle._interpolate_2d called with empty "
                f"samples for keys '{x_key}', '{y_key}'. Profile pack is missing "
                f"required all-reduce measurements. Refusing to fabricate a "
                f"zero-latency fallback."
            )

        # Group by x (bytes)
        by_bytes: dict[int, list[dict]] = {}
        for s in samples:
            b = s[x_key]
            if b not in by_bytes:
                by_bytes[b] = []
            by_bytes[b].append(s)
        
        bytes_list = sorted(by_bytes.keys())
        
        if x_value <= bytes_list[0]:
            # At lowest bytes, interpolate by world_size only
            return self._interpolate(by_bytes[bytes_list[0]], y_key, y_value)
        if x_value >= bytes_list[-1]:
            return self._interpolate(by_bytes[bytes_list[-1]], y_key, y_value)
        
        # Find bytes bracket
        for i in range(len(bytes_list) - 1):
            if bytes_list[i] <= x_value <= bytes_list[i + 1]:
                lo_bytes, hi_bytes = bytes_list[i], bytes_list[i + 1]
                # Interpolate at each bytes level
                lo_latency = self._interpolate(by_bytes[lo_bytes], y_key, y_value)
                hi_latency = self._interpolate(by_bytes[hi_bytes], y_key, y_value)
                # Interpolate between bytes levels
                ratio = (x_value - lo_bytes) / (hi_bytes - lo_bytes)
                return lo_latency + ratio * (hi_latency - lo_latency)
        
        return samples[-1]["latency_us"]

    def get_all_reduce_latency_us(
        self,
        num_bytes: int,
        world_size: int,
        topology: NetworkTopology,
    ) -> float:
        """Estimate all-reduce latency via interpolation.

        Raises RuntimeError if the profile pack has no samples for the
        requested topology — no magic-bandwidth fallbacks.
        """
        samples = self._all_reduce_samples.get(topology, [])

        if not samples:
            raise RuntimeError(
                f"ProfileNetworkCostOracle.get_all_reduce_latency_us called but "
                f"profile pack has no 'all_reduce.{topology.name.lower()}' "
                f"samples. Capture all-reduce profile data for this topology "
                f"before enabling multi-GPU emulation, or disable it. Refusing "
                f"to fabricate a bandwidth-model fallback."
            )

        return self._interpolate_2d(samples, "bytes", "world_size", float(num_bytes), float(world_size))

    def get_send_latency_us(
        self,
        num_bytes: int,
        topology: NetworkTopology,
    ) -> float:
        """Estimate send latency via interpolation."""
        samples = self._send_samples.get(topology, [])

        if not samples:
            raise RuntimeError(
                f"ProfileNetworkCostOracle.get_send_latency_us called but "
                f"profile pack has no 'send_recv.{topology.name.lower()}' "
                f"samples. Capture P2P send profile data for this topology "
                f"before enabling multi-GPU emulation, or disable it."
            )

        return self._interpolate(samples, "bytes", float(num_bytes))

    def get_recv_latency_us(
        self,
        num_bytes: int,
        topology: NetworkTopology,
    ) -> float:
        """Estimate receive latency via interpolation."""
        samples = self._recv_samples.get(topology, [])

        if not samples:
            raise RuntimeError(
                f"ProfileNetworkCostOracle.get_recv_latency_us called but "
                f"profile pack has no 'send_recv.{topology.name.lower()}' "
                f"samples. Capture P2P recv profile data for this topology "
                f"before enabling multi-GPU emulation, or disable it."
            )

        return self._interpolate(samples, "bytes", float(num_bytes))

    def get_kv_transfer_latency_us(
        self,
        num_bytes: int,
        direction: TransferDirection,
        topology: NetworkTopology,
        concurrency: int = 1,
    ) -> float:
        """Estimate KV transfer latency between GPUs.
        
        Accounts for concurrency by scaling latency when multiple transfers
        are active (bandwidth contention).
        """
        samples = self._kv_transfer_samples.get(topology, [])

        if not samples:
            raise RuntimeError(
                f"ProfileNetworkCostOracle.get_kv_transfer_latency_us called but "
                f"profile pack has no 'kv_transfer.{topology.name.lower()}' "
                f"samples. Capture KV-transfer profile data for this topology "
                f"before enabling multi-GPU emulation, or disable it."
            )

        base_latency = self._interpolate(samples, "bytes", float(num_bytes))

        # Scale for concurrency (bandwidth contention model)
        if concurrency > 1:
            scale_factor = (concurrency ** 0.5)
            base_latency *= scale_factor

        return base_latency


def create_network_oracle_from_profile_pack(profile_pack: dict[str, Any]) -> ProfileNetworkCostOracle:
    """Factory function to create a network oracle from a profile pack."""
    return ProfileNetworkCostOracle(profile_pack)
