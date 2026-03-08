"""Network profiler stub.

This module will measure collective-communication latency (all-reduce,
send/recv, KV transfer) across different topologies when a multi-GPU
setup is available.  For now it exposes only the public configuration
dataclass.

TODO: Implement full sweep once network oracle (P1.3) lands.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class NetworkProfilingConfig:
    """Parameters for a network profiling run (stub)."""

    bytes_grid: list[int] = field(
        default_factory=lambda: [4096, 65536, 1048576, 16777216],
    )
    world_sizes: list[int] = field(default_factory=lambda: [2, 4, 8])
    topologies: list[str] = field(
        default_factory=lambda: ["nvlink", "pcie", "ib"],
    )
    warmup_iters: int = 3
    measure_iters: int = 10
    aggregation: Literal["median", "mean"] = "median"
