#!/usr/bin/env python3
"""Variance analysis of ipc_overhead_v2.json raw samples.

Reads results/RTX-8000-adaptive-archive-5r/ipc_overhead_v2.json,
for each N computes mean, median, stdev, p10, p90 across the 15 raw TTFT
samples, and writes paper/apr_20/00_ipc_variance.md.
"""
import json
import statistics
from pathlib import Path

IPC = Path("results/RTX-8000-adaptive-archive-5r/ipc_overhead_v2.json")
OUT = Path("paper/apr_20/00_ipc_variance.md")
OUT.parent.mkdir(parents=True, exist_ok=True)


def p(samples, q):
    s = sorted(samples)
    idx = int(q * (len(s) - 1))
    return s[idx]


data = json.load(open(IPC))

L = []
L.append("# Apr 20 — IPC overhead variance analysis\n")
L.append("Input: `results/RTX-8000-adaptive-archive-5r/ipc_overhead_v2.json` "
         "(raw TTFT samples retained by profile_ipc_overhead.py v2).\n")

L.append("## Per-N summary (TTFT includes prefill step; overhead = TTFT − prefill_step)\n")
L.append("| N | n | median ms | mean ms | stdev ms | p10 ms | p90 ms | (mean − median) ms | σ / median |")
L.append("|---|---|---|---|---|---|---|---|---|")

for e in data:
    n = e.get("num_reqs")
    raw_us = e.get("raw_ttft_samples_us", [])
    if not raw_us:
        L.append(f"| {n} | 0 | (no raw samples) |")
        continue
    raw_ms = [s / 1000.0 for s in raw_us]
    med = statistics.median(raw_ms)
    mean = statistics.mean(raw_ms)
    stdev = statistics.stdev(raw_ms) if len(raw_ms) > 1 else 0.0
    p10 = p(raw_ms, 0.10)
    p90 = p(raw_ms, 0.90)
    L.append(f"| {n} | {len(raw_ms)} | {med:.2f} | {mean:.2f} | {stdev:.2f} | "
             f"{p10:.2f} | {p90:.2f} | {mean - med:+.2f} | {stdev/med:.2%} |")

L.append("")
L.append("## Interpretation\n")

# Compute aggregate stats.
all_stdev_ms = []
all_mean_minus_median = []
for e in data:
    raw_us = e.get("raw_ttft_samples_us", [])
    if not raw_us:
        continue
    raw_ms = [s / 1000.0 for s in raw_us]
    if len(raw_ms) < 2:
        continue
    all_stdev_ms.append(statistics.stdev(raw_ms))
    all_mean_minus_median.append(statistics.mean(raw_ms) - statistics.median(raw_ms))

avg_stdev = sum(all_stdev_ms) / len(all_stdev_ms) if all_stdev_ms else 0
avg_mean_med_delta = sum(all_mean_minus_median) / len(all_mean_minus_median) if all_mean_minus_median else 0

L.append(f"- Mean stdev across all N: **{avg_stdev:.2f} ms**.")
L.append(f"- Mean (mean − median) across all N: **{avg_mean_med_delta:+.2f} ms** "
         "(positive = right-skewed distribution, tail pulls mean above median).")
L.append("")

if avg_stdev < 2:
    L.append("**Variance verdict**: TIGHT (σ < 2 ms). Flat-per-N model is physically accurate; "
             "median vs mean differ by a small constant shift per N.")
elif avg_stdev < 5:
    L.append("**Variance verdict**: MODERATE (σ 2–5 ms). Single-scalar model is lossy but "
             "not catastrophic; mean vs median debate has real impact (~{} ms) on TTFT emu accuracy."
             .format(round(avg_mean_med_delta, 1)))
else:
    L.append("**Variance verdict**: WIDE (σ > 5 ms). Single-scalar model loses a lot of information. "
             "Distribution-aware admission delay (sample per request from raw_ttft_samples_us) may be "
             "meaningfully more accurate than flat median or flat mean.")

L.append("")
L.append("## Connection to v4-mean result\n")
L.append("v4-arrival-mean (TTFT at r=2 = +5.9%) overshot the target (v3 median: −9.5%). That "
         f"means emu added ~{avg_mean_med_delta:.1f} ms per arrival vs real's average — consistent "
         "with a right-skewed IPC distribution where median underestimates average. ")
L.append("")
L.append("If variance is wide (σ ≳ 5 ms), sampling per-request from the raw distribution (rather "
         "than using scalar median or mean) would give each request a realistic draw and the "
         "average would converge to real's mean naturally.")

OUT.write_text("\n".join(L) + "\n")
print(f"wrote {OUT}")
print()
print(f"Mean stdev: {avg_stdev:.2f} ms | Mean (mean−median): {avg_mean_med_delta:+.2f} ms")
