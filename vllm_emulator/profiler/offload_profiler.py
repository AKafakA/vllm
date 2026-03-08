"""Optional profiler for CPU<->GPU transfer latency samples."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


def _measure_transfer_us(size_bytes: int, direction: str, runs: int = 5) -> float:
    elem_count = max(1, size_bytes // 4)
    latencies: list[float] = []

    for _ in range(runs):
        if direction == "cpu_to_gpu":
            src = torch.randn(elem_count, dtype=torch.float32, device="cpu", pin_memory=True)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = src.to("cuda", non_blocking=False)
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1e6)
        elif direction == "gpu_to_cpu":
            src = torch.randn(elem_count, dtype=torch.float32, device="cuda")
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = src.to("cpu", non_blocking=False)
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1e6)
        else:
            raise ValueError(f"Unknown direction: {direction}")

    return sorted(latencies)[len(latencies) // 2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Optional offload profiler")
    parser.add_argument("--sizes", default="4096,1048576,16777216")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    sizes = [int(s.strip()) for s in args.sizes.split(",") if s.strip()]
    rows = {
        "cpu_to_gpu": [{"bytes": sz, "latency_us": _measure_transfer_us(sz, "cpu_to_gpu")} for sz in sizes],
        "gpu_to_cpu": [{"bytes": sz, "latency_us": _measure_transfer_us(sz, "gpu_to_cpu")} for sz in sizes],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"transfer": rows}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
