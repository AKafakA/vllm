#!/usr/bin/env python3
"""
Simulation-only benchmark for vllm-emulator oracles.

This script tests the oracle interpolation logic in isolation,
without requiring a full vLLM build or GPU hardware.

Run: python3 benchmarks/sim_benchmark.py
"""

import json
import time
from pathlib import Path
from typing import Any

# Sample profile pack for testing
SAMPLE_PROFILE_PACK = {
    "version": "1.0",
    "gpu_model": "A100-SXM-80GB",
    "prefill": [
        {"seq_len": 128, "batch_size": 1, "latency_us": 15000},
        {"seq_len": 256, "batch_size": 1, "latency_us": 28000},
        {"seq_len": 512, "batch_size": 1, "latency_us": 55000},
        {"seq_len": 1024, "batch_size": 1, "latency_us": 108000},
        {"seq_len": 2048, "batch_size": 1, "latency_us": 215000},
    ],
    "decode": [
        {"active_seqs": 1, "latency_us_per_token": 500},
        {"active_seqs": 4, "latency_us_per_token": 1800},
        {"active_seqs": 8, "latency_us_per_token": 3500},
        {"active_seqs": 16, "latency_us_per_token": 6800},
        {"active_seqs": 32, "latency_us_per_token": 13000},
    ],
    "network": {
        "all_reduce": {
            "nvlink": [
                {"bytes": 1024, "world_size": 2, "latency_us": 15},
                {"bytes": 1024, "world_size": 4, "latency_us": 25},
                {"bytes": 1024, "world_size": 8, "latency_us": 40},
                {"bytes": 16384, "world_size": 2, "latency_us": 50},
                {"bytes": 16384, "world_size": 4, "latency_us": 80},
                {"bytes": 16384, "world_size": 8, "latency_us": 120},
            ],
            "pcie": [
                {"bytes": 1024, "world_size": 2, "latency_us": 25},
                {"bytes": 1024, "world_size": 4, "latency_us": 45},
                {"bytes": 16384, "world_size": 2, "latency_us": 80},
                {"bytes": 16384, "world_size": 4, "latency_us": 140},
            ],
        },
        "send_recv": {
            "nvlink": [
                {"bytes": 1024, "latency_us": 8},
                {"bytes": 16384, "latency_us": 25},
                {"bytes": 65536, "latency_us": 80},
            ],
            "pcie": [
                {"bytes": 1024, "latency_us": 12},
                {"bytes": 16384, "latency_us": 40},
                {"bytes": 65536, "latency_us": 150},
            ],
        },
        "kv_transfer": {
            "nvlink": [
                {"bytes": 4096, "latency_us": 20},
                {"bytes": 16384, "latency_us": 50},
                {"bytes": 65536, "latency_us": 150},
            ],
        },
    },
    "offload": {
        "lookup": [
            {"num_blocks": 1, "latency_us": 10},
            {"num_blocks": 8, "latency_us": 60},
            {"num_blocks": 32, "latency_us": 200},
            {"num_blocks": 128, "latency_us": 700},
        ],
        "transfer": {
            "cpu_to_gpu": [
                {"bytes": 4096, "latency_us": 50},
                {"bytes": 16384, "latency_us": 150},
                {"bytes": 65536, "latency_us": 500},
            ],
            "gpu_to_cpu": [
                {"bytes": 4096, "latency_us": 45},
                {"bytes": 16384, "latency_us": 140},
                {"bytes": 65536, "latency_us": 480},
            ],
        },
        "evict": [
            {"num_blocks": 1, "latency_us": 5},
            {"num_blocks": 8, "latency_us": 30},
            {"num_blocks": 32, "latency_us": 100},
            {"num_blocks": 128, "latency_us": 350},
        ],
    },
}


class GpuCostOracle:
    """Simple GPU cost oracle for simulation testing."""

    def __init__(self, profile_pack: dict[str, Any]):
        self._profile = profile_pack
        self._prefill_samples = profile_pack["prefill"]
        self._decode_samples = profile_pack["decode"]

    def estimate_prefill_latency_us(self, prompt_tokens: int, batch_size: int = 1) -> float:
        samples = self._prefill_samples
        seq_lens = [s["seq_len"] for s in samples]

        if prompt_tokens <= seq_lens[0]:
            return samples[0]["latency_us"]
        if prompt_tokens >= seq_lens[-1]:
            return samples[-1]["latency_us"]

        for i in range(len(seq_lens) - 1):
            if seq_lens[i] <= prompt_tokens <= seq_lens[i + 1]:
                lo, hi = samples[i], samples[i + 1]
                ratio = (prompt_tokens - lo["seq_len"]) / (hi["seq_len"] - lo["seq_len"])
                return lo["latency_us"] + ratio * (hi["latency_us"] - lo["latency_us"])

        return samples[-1]["latency_us"]

    def estimate_decode_latency_us(self, active_seqs: int) -> float:
        samples = self._decode_samples
        seq_counts = [s["active_seqs"] for s in samples]

        if active_seqs <= seq_counts[0]:
            return samples[0]["latency_us_per_token"]
        if active_seqs >= seq_counts[-1]:
            return samples[-1]["latency_us_per_token"]

        for i in range(len(seq_counts) - 1):
            if seq_counts[i] <= active_seqs <= seq_counts[i + 1]:
                lo, hi = samples[i], samples[i + 1]
                ratio = (active_seqs - lo["active_seqs"]) / (hi["active_seqs"] - lo["active_seqs"])
                return lo["latency_us_per_token"] + ratio * (
                    hi["latency_us_per_token"] - lo["latency_us_per_token"]
                )

        return samples[-1]["latency_us_per_token"]


class NetworkCostOracle:
    """Simple network cost oracle for simulation testing."""

    def __init__(self, profile_pack: dict[str, Any]):
        self._profile = profile_pack
        network = profile_pack.get("network", {})
        all_reduce = network.get("all_reduce", {})
        self._all_reduce_samples = {
            "nvlink": all_reduce.get("nvlink", []),
            "pcie": all_reduce.get("pcie", []),
        }
        send_recv = network.get("send_recv", {})
        self._send_samples = {
            "nvlink": send_recv.get("nvlink", []),
            "pcie": send_recv.get("pcie", []),
        }

    def _interpolate(self, samples, x_key, x_value):
        if not samples:
            return 0.0
        x_values = [s[x_key] for s in samples]
        if x_value <= x_values[0]:
            return samples[0]["latency_us"]
        if x_value >= x_values[-1]:
            return samples[-1]["latency_us"]
        for i in range(len(x_values) - 1):
            if x_values[i] <= x_value <= x_values[i + 1]:
                lo, hi = samples[i], samples[i + 1]
                ratio = (x_value - lo[x_key]) / (hi[x_key] - lo[x_key])
                return lo["latency_us"] + ratio * (hi["latency_us"] - lo["latency_us"])
        return samples[-1]["latency_us"]

    def get_all_reduce_latency_us(self, num_bytes: int, world_size: int, topology: str) -> float:
        samples = self._all_reduce_samples.get(topology, [])
        if not samples:
            # Default estimate
            if topology == "nvlink":
                return 5.0 + (num_bytes / (600 * 1e9)) * 1e6 * (world_size - 1).bit_length()
            elif topology == "pcie":
                return 15.0 + (num_bytes / (32 * 1e9)) * 1e6 * (world_size - 1).bit_length()
            return 20.0

        # 2D interpolation (bytes x world_size)
        by_bytes = {}
        for s in samples:
            b = s["bytes"]
            if b not in by_bytes:
                by_bytes[b] = []
            by_bytes[b].append(s)
        bytes_list = sorted(by_bytes.keys())

        if num_bytes <= bytes_list[0]:
            return self._interpolate(by_bytes[bytes_list[0]], "world_size", float(world_size))
        if num_bytes >= bytes_list[-1]:
            return self._interpolate(by_bytes[bytes_list[-1]], "world_size", float(world_size))

        for i in range(len(bytes_list) - 1):
            if bytes_list[i] <= num_bytes <= bytes_list[i + 1]:
                lo_latency = self._interpolate(by_bytes[bytes_list[i]], "world_size", float(world_size))
                hi_latency = self._interpolate(by_bytes[bytes_list[i + 1]], "world_size", float(world_size))
                ratio = (num_bytes - bytes_list[i]) / (bytes_list[i + 1] - bytes_list[i])
                return lo_latency + ratio * (hi_latency - lo_latency)

        return samples[-1]["latency_us"]

    def get_send_latency_us(self, num_bytes: int, topology: str) -> float:
        samples = self._send_samples.get(topology, [])
        if not samples:
            if topology == "nvlink":
                return 2.0 + (num_bytes / (900 * 1e9)) * 1e6
            elif topology == "pcie":
                return 8.0 + (num_bytes / (32 * 1e9)) * 1e6
            return 10.0
        return self._interpolate(samples, "bytes", float(num_bytes))


class OffloadCostOracle:
    """Simple offload cost oracle for simulation testing."""

    def __init__(self, profile_pack: dict[str, Any]):
        self._profile = profile_pack
        offload = profile_pack.get("offload", {})
        self._lookup_samples = offload.get("lookup", [])
        self._cpu_to_gpu = offload.get("transfer", {}).get("cpu_to_gpu", [])
        self._gpu_to_cpu = offload.get("transfer", {}).get("gpu_to_cpu", [])
        self._evict_samples = offload.get("evict", [])

    def _interpolate(self, samples, x_key, x_value):
        if not samples:
            return 0.0
        x_values = [s[x_key] for s in samples]
        if x_value <= x_values[0]:
            return samples[0]["latency_us"]
        if x_value >= x_values[-1]:
            return samples[-1]["latency_us"]
        for i in range(len(x_values) - 1):
            if x_values[i] <= x_value <= x_values[i + 1]:
                lo, hi = samples[i], samples[i + 1]
                ratio = (x_value - lo[x_key]) / (hi[x_key] - lo[x_key])
                return lo["latency_us"] + ratio * (hi["latency_us"] - lo["latency_us"])
        return samples[-1]["latency_us"]

    def get_lookup_latency_us(self, num_blocks: int) -> float:
        if not self._lookup_samples:
            return num_blocks * 10.0
        return self._interpolate(self._lookup_samples, "num_blocks", float(num_blocks))

    def get_transfer_latency_us(self, num_bytes: int, direction: str, concurrency: int = 1) -> float:
        samples = self._cpu_to_gpu if direction == "cpu_to_gpu" else self._gpu_to_cpu
        if not samples:
            base = (num_bytes / 1_048_576) * 100.0
        else:
            base = self._interpolate(samples, "bytes", float(num_bytes))
        if concurrency > 1:
            base *= (concurrency ** 0.5)
        return base

    def get_evict_latency_us(self, num_blocks: int) -> float:
        if not self._evict_samples:
            return num_blocks * 5.0
        return self._interpolate(self._evict_samples, "num_blocks", float(num_blocks))


def run_benchmark():
    """Run simulation benchmark."""
    print("=" * 60)
    print("vllm-emulator Simulation Benchmark")
    print("=" * 60)

    # Initialize oracles
    gpu_oracle = GpuCostOracle(SAMPLE_PROFILE_PACK)
    network_oracle = NetworkCostOracle(SAMPLE_PROFILE_PACK)
    offload_oracle = OffloadCostOracle(SAMPLE_PROFILE_PACK)

    results = []

    # Test 1: GPU Prefill Interpolation
    print("\n[Test 1] GPU Prefill Interpolation")
    print("-" * 40)
    test_cases = [
        (128, "exact min"),
        (256, "exact sample"),
        (384, "interpolated"),
        (512, "exact sample"),
        (768, "interpolated"),
        (1024, "exact max"),
        (1536, "extrapolated"),
    ]
    for tokens, desc in test_cases:
        latency = gpu_oracle.estimate_prefill_latency_us(tokens)
        print(f"  {tokens:5d} tokens ({desc:15s}): {latency:8.1f} us")
        results.append(("prefill", tokens, latency))

    # Test 2: GPU Decode Interpolation
    print("\n[Test 2] GPU Decode Interpolation")
    print("-" * 40)
    test_cases = [
        (1, "exact min"),
        (4, "exact sample"),
        (6, "interpolated"),
        (8, "exact sample"),
        (12, "interpolated"),
        (16, "exact max"),
        (24, "extrapolated"),
    ]
    for seqs, desc in test_cases:
        latency = gpu_oracle.estimate_decode_latency_us(seqs)
        print(f"  {seqs:5d} seqs  ({desc:15s}): {latency:8.1f} us/token")
        results.append(("decode", seqs, latency))

    # Test 3: Network All-Reduce
    print("\n[Test 3] Network All-Reduce (NVLink)")
    print("-" * 40)
    test_cases = [
        (1024, 2, "small, 2 GPUs"),
        (1024, 4, "small, 4 GPUs"),
        (16384, 2, "medium, 2 GPUs"),
        (16384, 4, "medium, 4 GPUs"),
        (16384, 8, "medium, 8 GPUs"),
        (65536, 8, "large, 8 GPUs"),
    ]
    for bytes_, ws, desc in test_cases:
        latency = network_oracle.get_all_reduce_latency_us(bytes_, ws, "nvlink")
        print(f"  {bytes_:6d} bytes, {ws} GPU ({desc:15s}): {latency:8.1f} us")
        results.append(("all_reduce", bytes_, latency))

    # Test 4: Network Send/Recv
    print("\n[Test 4] Network Send/Recv (NVLink)")
    print("-" * 40)
    test_cases = [
        (1024, "1 KB"),
        (16384, "16 KB"),
        (65536, "64 KB"),
        (262144, "256 KB"),
    ]
    for bytes_, desc in test_cases:
        latency = network_oracle.get_send_latency_us(bytes_, "nvlink")
        print(f"  {bytes_:6d} bytes ({desc:10s}): {latency:8.1f} us")
        results.append(("send", bytes_, latency))

    # Test 5: Offload Lookup
    print("\n[Test 5] Offload Lookup")
    print("-" * 40)
    test_cases = [
        (1, "1 block"),
        (8, "8 blocks"),
        (16, "interpolated"),
        (32, "32 blocks"),
        (64, "interpolated"),
        (128, "128 blocks"),
    ]
    for blocks, desc in test_cases:
        latency = offload_oracle.get_lookup_latency_us(blocks)
        print(f"  {blocks:4d} blocks ({desc:12s}): {latency:8.1f} us")
        results.append(("lookup", blocks, latency))

    # Test 6: Offload Transfer
    print("\n[Test 6] Offload Transfer (CPU -> GPU)")
    print("-" * 40)
    test_cases = [
        (4096, 1, "4 KB"),
        (16384, 1, "16 KB"),
        (65536, 1, "64 KB"),
        (16384, 2, "16 KB, 2x concurrent"),
        (16384, 4, "16 KB, 4x concurrent"),
    ]
    for bytes_, conc, desc in test_cases:
        latency = offload_oracle.get_transfer_latency_us(bytes_, "cpu_to_gpu", conc)
        print(f"  {bytes_:6d} bytes, {conc}x ({desc:20s}): {latency:8.1f} us")
        results.append(("transfer", bytes_, latency))

    # Test 7: Consistency check (interpolation is monotonic)
    print("\n[Test 7] Monotonicity Check")
    print("-" * 40)
    prev_latency = 0
    monotonic = True
    test_tokens = list(range(128, 2049, 64))
    for tokens in test_tokens:
        latency = gpu_oracle.estimate_prefill_latency_us(tokens)
        if latency < prev_latency:
            monotonic = False
            print(f"  VIOLATION: {tokens} tokens: {latency} < {prev_latency}")
        prev_latency = latency
    if monotonic:
        print("  ✓ Prefill interpolation is monotonic")
    else:
        print("  ✗ Prefill interpolation is NOT monotonic!")

    # Test 8: Performance benchmark
    print("\n[Test 8] Performance (10000 queries)")
    print("-" * 40)
    start = time.perf_counter()
    for _ in range(10000):
        gpu_oracle.estimate_prefill_latency_us(512)
        gpu_oracle.estimate_decode_latency_us(8)
        network_oracle.get_all_reduce_latency_us(16384, 4, "nvlink")
        offload_oracle.get_lookup_latency_us(32)
    elapsed = time.perf_counter() - start
    print(f"  40000 oracle calls: {elapsed*1000:.2f} ms")
    print(f"  Average per call: {elapsed/40000*1e6:.3f} us")

    print("\n" + "=" * 60)
    print("Benchmark Complete")
    print("=" * 60)

    return results


if __name__ == "__main__":
    run_benchmark()
