# Apr 20 — IPC overhead variance analysis

Input: `results/RTX-8000-adaptive-archive-5r/ipc_overhead_v2.json` (raw TTFT samples retained by profile_ipc_overhead.py v2).

## Per-N summary (TTFT includes prefill step; overhead = TTFT − prefill_step)

| N | n | median ms | mean ms | stdev ms | p10 ms | p90 ms | (mean − median) ms | σ / median |
|---|---|---|---|---|---|---|---|---|
| 1 | 15 | 40.16 | 40.70 | 2.17 | 39.97 | 40.83 | +0.54 | 5.40% |
| 2 | 15 | 39.28 | 41.73 | 10.27 | 38.15 | 39.89 | +2.44 | 26.14% |
| 3 | 15 | 39.36 | 41.57 | 9.31 | 38.20 | 39.86 | +2.21 | 23.65% |
| 5 | 15 | 39.85 | 43.09 | 12.63 | 39.37 | 40.45 | +3.24 | 31.69% |
| 8 | 15 | 39.69 | 41.68 | 8.41 | 39.20 | 39.84 | +1.99 | 21.18% |
| 12 | 15 | 40.01 | 44.37 | 12.06 | 39.39 | 40.53 | +4.37 | 30.14% |
| 20 | 15 | 39.80 | 44.42 | 12.25 | 39.40 | 40.78 | +4.62 | 30.79% |
| 30 | 15 | 39.87 | 43.85 | 11.97 | 39.45 | 40.52 | +3.99 | 30.02% |
| 50 | 15 | 39.91 | 44.42 | 11.99 | 39.50 | 40.79 | +4.51 | 30.05% |
| 100 | 15 | 40.08 | 45.14 | 13.37 | 39.68 | 41.29 | +5.06 | 33.35% |
| 150 | 15 | 39.93 | 44.70 | 13.02 | 39.26 | 40.63 | +4.77 | 32.61% |
| 200 | 15 | 39.22 | 43.82 | 13.21 | 38.78 | 40.84 | +4.61 | 33.70% |
| 256 | 15 | 39.84 | 45.38 | 14.65 | 39.66 | 40.39 | +5.54 | 36.77% |

## Interpretation

- Mean stdev across all N: **11.18 ms**.
- Mean (mean − median) across all N: **+3.68 ms** (positive = right-skewed distribution, tail pulls mean above median).

**Variance verdict**: WIDE (σ > 5 ms). Single-scalar model loses a lot of information. Distribution-aware admission delay (sample per request from raw_ttft_samples_us) may be meaningfully more accurate than flat median or flat mean.

## Connection to v4-mean result

v4-arrival-mean (TTFT at r=2 = +5.9%) overshot the target (v3 median: −9.5%). That means emu added ~3.7 ms per arrival vs real's average — consistent with a right-skewed IPC distribution where median underestimates average. 

If variance is wide (σ ≳ 5 ms), sampling per-request from the raw distribution (rather than using scalar median or mean) would give each request a realistic draw and the average would converge to real's mean naturally.
