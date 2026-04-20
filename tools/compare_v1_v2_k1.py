#!/usr/bin/env python3
"""Compare v1 sweep (1D) vs v2 sweep (2D at k=1) for the same N values.

The Apr 19 hypothesis was that v1 k=1 median and v2 k=1 cells would differ
only by measurement noise. Observed: v5-2d-burst-tight at r=2 overshoots
by ~25ms despite k=1 almost always; predicted shift from lookup change
was ~5ms. This script quantifies the per-N disagreement and looks at
raw samples to see whether the distributions are meaningfully different
or same.
"""
import json
import statistics
from pathlib import Path

V1 = Path("/home/wd312/Code/llm/vllm-emulator/results/RTX-8000-adaptive-archive-5r/ipc_overhead_v2.json")
V2 = Path("/home/wd312/Code/llm/vllm-emulator/results/RTX-8000-adaptive-archive-5r/ipc_overhead_v2_2d.json")


def load_v1():
    """v1 is 1D (per-N k=1 only)."""
    data = json.load(open(V1))
    out = {}
    for e in data:
        n = e.get("num_reqs")
        raw = e.get("raw_ttft_samples_us") or []
        if not raw:
            continue
        out[n] = {
            "samples": raw,
            "median": statistics.median(raw),
            "mean": statistics.mean(raw),
            "stdev": statistics.stdev(raw) if len(raw) > 1 else 0,
            "n": len(raw),
        }
    return out


def load_v2_k1():
    """v2 is 2D; extract k=1 column."""
    data = json.load(open(V2))
    out = {}
    for e in data:
        if e.get("burst_k") != 1:
            continue
        n = e.get("num_reqs")
        # Raw samples are nested per-sample × per-slot. k=1 means 1 slot.
        samples = []
        for s in e.get("samples", []):
            samples.extend(s.get("ttft_us_by_slot") or [])
        if not samples:
            continue
        out[n] = {
            "samples": samples,
            "median": statistics.median(samples),
            "mean": statistics.mean(samples),
            "stdev": statistics.stdev(samples) if len(samples) > 1 else 0,
            "n": len(samples),
        }
    return out


v1 = load_v1()
v2k1 = load_v2_k1()
common_N = sorted(set(v1.keys()) & set(v2k1.keys()))

print(f"{'N':>4} | {'v1 n':>5} {'v1 med':>7} {'v1 mean':>8} {'v1 σ':>6} | "
      f"{'v2 n':>5} {'v2 med':>7} {'v2 mean':>8} {'v2 σ':>6} | "
      f"{'Δmed':>6} {'Δmean':>6}")
print("-" * 105)

for n in common_N:
    a = v1[n]; b = v2k1[n]
    dm = b["median"] - a["median"]
    dmn = b["mean"] - a["mean"]
    print(f"{n:>4} | {a['n']:>5} {a['median']/1000:>7.2f} {a['mean']/1000:>8.2f} {a['stdev']/1000:>6.2f} | "
          f"{b['n']:>5} {b['median']/1000:>7.2f} {b['mean']/1000:>8.2f} {b['stdev']/1000:>6.2f} | "
          f"{dm/1000:>+6.2f} {dmn/1000:>+6.2f}")

print()
print("(all values in ms)")
print()
print("Expectation: v1 and v2 k=1 measure the same physical quantity,")
print("so medians should agree within variance bounds. Systematic bias")
print("in the Δ columns indicates the sweeps captured different distributions.")
