# Apr 20 afternoon — Saturation-supplementary reprofile: NEGATIVE RESULT

**Hypothesis**: archive-r2's profile has sparse samples at (tt > 300, conc > 200); filling this region with additional saturation rounds should close the r=16 TTFT gap from −14% to ≤10%.

**Result**: **hypothesis refuted.** r=16 TTFT got *worse* (−14.00% → −28.53%). r=8 TPOT regressed by 5pp (−2.87% → −8.10%). Profile not promoted; kept archive-r2 as reference.

## Setup

- Supplementary trace: archive-r2's 108k trace (verbatim) + 3 additional saturation rounds on the same server session
  - rate=16, 256/128, 10,000 prompts
  - rate=32, 256/128, 10,000 prompts
  - rate=inf, 256/128, 10,000 prompts
- Warmup (rate=4 × 500) excluded via `profiling_start` / `profiling_stop` markers
- Combined trace rebuilt with `build_serving_profile_filtered.py` (same settings as archive-r2)
- Validation: archive-r2-sat-supp profile × random 256/128 × 5 rates × 2000p, v3 hook (median)

## Validation matrix (sat-supp vs archive-r2 baseline, random workload)

| rate | archive-r2 TPOT | sat-supp TPOT | archive-r2 TTFT | sat-supp TTFT | verdict |
|---|---|---|---|---|---|
| 2  | −0.09 | −0.6 | −10.56 | −10.0 | tied |
| 4  | −0.79 | −1.7 | −4.34 | −5.7 | tied |
| 8  | **−2.87** | **−8.10** | +1.59 | −5.48 | **FAIL (TPOT)** |
| 16 | −3.02 | −7.7 | **−14.00** | **−28.53** | **FAIL WORSE** |
| 32 | +1.17 | +2.4 | −0.35 | +0.3 | tied |

**Targets**: TPOT/E2E ≤6%, TTFT ≤10%. sat-supp fails 2/5 (r=8 TPOT −8.1%, r=16 TTFT −28.5%).

## Why the hypothesis failed — forensic

### Target region barely filled

Expected: fill the (tt ≥ 300, conc ≥ 200) gap with thousands of samples.
Actual:
- archive-r2: 3 cells / 3 samples in target region
- sat-supp:   5 cells / 5 samples in target region

The saturation rounds produced virtually no new data in the target region. Root cause: once running queue hits max_num_seqs (256), new prefills wait in the *waiting* queue (not scheduled). Saturation is dominated by **pure decode at conc=256 with tt=256**, NOT mixed prefill+decode at high tt.

### Where the new samples landed

| bucket | archive-r2 samples | sat-supp samples | growth |
|---|---|---|---|
| (tt=256, c=257) pure decode | 1083 | 3173 | **+193%** |
| (tt=255, c=257) | 88 | 509 | +478% |
| (tt=254, c=252) | 53 | 108 | +104% |
| conc=27-37 buckets (r=8 range) | 300-600 ea | +10-20% | contamination |

5833 new samples went to:
- ~4000 at pure-decode saturation buckets (just duplicating existing data)
- ~1800 scattered across mid-conc buckets (ramp-up phases at the start of each rate's bench) — this is the **contamination** source

### r=8 regression mechanism

At r=8, max conc ≈ 119. Oracle queries conc buckets 102-127. Those buckets are **sparse** in archive-r2 (1-20 samples each). Sat-supp doubled bucket size but the new samples come from **ramp-up** phases of the r=16/r=32/inf benches (concurrency climbing from 0 to saturation). Ramp-up samples have different step-timing characteristics than archive-r2's steady-state samples. Oracle's `random.choice` now mixes two distributions → systematically faster predictions → emu too fast → TPOT ballasted toward "−" (emu < real).

### r=16 regression mechanism

Oracle at saturation queries (tt=256, conc=257). With 3× more samples at that bucket (1083 → 3173), the distribution stayed similar in median (90.29 → 91.17 ms, +0.88ms) but the tail behaviour shifted: more short-latency samples pulled oracle selections slightly shorter on average. Over the course of a 125s bench with thousands of high-conc steps, those small shifts compound. Net emu faster → queue drains faster → TTFT shorter.

## What's ruled out now for r=16

After today:
- **Hook batch-composition effect**: REFUTED (April 20 morning, batch-comp diagnostic)
- **v1 vs v2 k=1 sweep disagreement**: REFUTED (v1/v2 agree within 3-4ms)
- **Burst-aware hook**: REFUTED (v5-2d-burst, v5-2d-burst-tight both made r=16 worse)
- **Profile density at saturation**: REFUTED (this result)

## What r=16 likely is

A structural discrepancy between emu and real at the exact rate where service ≈ arrival. r=16 sits on the edge of saturation. r=8 is below service capacity, queue stays short, TTFT dominated by prefill + IPC (predictable). r=32 is deeply saturated, TTFT dominated by queue wait (dominates modelling errors). r=16 is the sensitive transition zone.

Possible remaining causes (none tested):
- Subtle differences in scheduler chunk-prefill token budget accounting between emu and real
- CUDA graph vs eager-mode timing differences that don't affect profile samples
- vLLM v1's async scheduling batching behaviour at near-saturation differs from profile's observations

Would need per-step-by-step timing trace comparison between emu and real at r=16 to isolate. Expensive to diagnose; deferred.

## Verdict

- **Profile not promoted**. archive-r2 remains the reference.
- **r=16 TTFT accepted as known emulator limitation**: −14% at one rate, within ±20% of archive's other rate accuracies. Does not block paper's headline story (5/6 rates within target).
- **Lesson**: additive profile extension has a failure mode when supplementary rounds' ramp-up phases land in buckets with archive-r2 steady-state data. Future "additive" profiles must use per-rate profiling markers that exclude ramp-up windows.
- **Focus shift**: stop investigating r=16. Spend remaining Apr 20 budget on workload study (shareptsampled rebuild) per the original plan.

## Artifacts

- `results/RTX-8000-profile-archive-r2-sat-supp/serving-full.json` — the bad profile (kept for reference, not promoted)
- `results/sat-supp-validate-random/` — 5-rate validation matrix
- `tools/adaptive_profile_saturation_supp.sh`, `tools/validate_sat_supp.sh` — chain scripts (kept for reproducibility)
