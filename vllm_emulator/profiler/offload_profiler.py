"""CPU↔GPU offload profiler stub.

This module will measure KV-cache offload latency (lookup, transfer, evict)
when a real GPU + host-memory path is available.  For now it exposes only the
public configuration dataclass so that downstream code can reference it.

TODO: Implement full sweep once offload oracle (P1.2) lands.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class OffloadProfilingConfig:
    """Parameters for an offload profiling run (stub)."""

    num_blocks_grid: list[int] = field(
        default_factory=lambda: [1, 8, 64, 256],
    )
    transfer_bytes_grid: list[int] = field(
        default_factory=lambda: [4096, 65536, 1048576],
    )
    directions: list[str] = field(
        default_factory=lambda: ["cpu_to_gpu", "gpu_to_cpu"],
    )
    warmup_iters: int = 3
    measure_iters: int = 10
    aggregation: Literal["median", "mean"] = "median"
