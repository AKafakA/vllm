# Apr 19 Overnight Results — v6 5-round single-session + 5-feature ablation

**Invariant status: FAIL** — v6 (5-round single-session) does NOT match archive's emu accuracy. Round count is the bias driver, not session continuity. Server restart hypothesis REJECTED.

## Phase A — v6 profile (5-round single-session)

- Trace records: 309,200 (5× v5's 65k, 3× archive's 108k)
- Cells: 2044 step_cycle · 258 decode · 1786 prefill
- Build finished: 05:12 BST Apr 19
- File: `results/RTX-8000-adaptive-v6-5r-single/serving-full.json`
- **Shape vs archive**: p50 Δ median −0.23% / mean −0.43% (match); p90 Δ median −6.54% / mean −22.93% (AS BAD AS V4's −24%). Single-session did NOT close the shape gap. Gap scales with round count.

## Phase B — v6 emu validation vs real baseline (2000 prompts)

Archive reference in parens. Threshold: TPOT ±2pp, TTFT ±3pp.

| Rate | v6 TPOT% | v6 TTFT% | Arch TPOT% | Arch TTFT% | ΔTPOT | ΔTTFT | Verdict |
|---|---|---|---|---|---|---|---|
| 2  | −5.7  | −34.5 | −0.6  | −32.5 | −5.1pp | −2.0pp | **FAIL** |
| 4  | −12.1 | −36.3 | −1.6  | −30.7 | −10.5pp | −5.6pp | **FAIL** |
| 8  | −19.0 | −34.4 | −6.0  | −28.5 | −13.0pp | −5.9pp | **FAIL** |
| 16 | −9.7  | −33.5 | −5.6  | −25.0 | −4.1pp | −8.5pp | **FAIL** |
| 32 | +0.3  | −2.9  | +4.9  | +1.9  | −4.6pp | −4.8pp | FAIL |

**v6 numbers essentially REPLICATE v3** (v3 withsurr: −5.7/−11.7/−20.9/−5.0/−0.2). Single-session methodology did not improve on multi-session methodology. The bias source is round count (or sample density per bucket), not server-state-boundary.

## Phase C — 5-feature ablation on v6 baseline (r=2, r=8; 500 prompts)

Each A/B ran 500 prompts per rate with surrogate ON. Comparison is on-vs-off at the same rate (both use v6 baseline).

| Feature | Rate | Off TPOT | On TPOT | ΔTPOT | Off TTFT | On TTFT | ΔTTFT | Verdict |
|---|---|---|---|---|---|---|---|---|
| F1 / IQR | 2 | −6.4 | −9.8 | −3.4pp | −30.0 | −32.1 | −2.1pp | DROP |
| F1 / IQR | 8 | −37.8 | −43.0 | −5.2pp | −38.2 | −42.8 | −4.6pp | DROP |
| F1 / MAD | 2 | −5.8 | −9.8 | −4.0pp | −29.4 | −32.2 | −2.8pp | DROP |
| F1 / MAD | 8 | −35.6 | −43.2 | −7.6pp | −36.1 | −42.9 | −6.8pp | DROP |
| F1 / Winsor | 2 | −6.0 | −6.1 | −0.1pp | −28.9 | −29.1 | −0.2pp | NEUTRAL |
| F1 / Winsor | 8 | −36.2 | −36.3 | −0.1pp | −37.3 | −38.2 | −0.9pp | NEUTRAL |
| F3 sample_tokens | 2 | −5.8 | −6.0 | −0.2pp | −29.0 | −29.3 | −0.3pp | NEUTRAL |
| F3 sample_tokens | 8 | −36.4 | −36.0 | **+0.4pp** | −36.5 | −36.5 | 0.0pp | NEUTRAL |
| F5 kNN (K=3 vs K=1) | 2 | −6.2 | −6.7 | −0.5pp | −29.7 | −30.4 | −0.7pp | NEUTRAL/DROP |
| F5 kNN (K=3 vs K=1) | 8 | −35.7 | −37.5 | −1.8pp | −36.8 | −38.4 | −1.6pp | DROP |
| F2 parallel surrogate | 2 | −5.9 | −6.0 | −0.1pp | −29.5 | −29.2 | +0.3pp | NEUTRAL |
| F2 parallel surrogate | 8 | −35.1 | −36.5 | −1.4pp | −36.5 | −36.9 | −0.4pp | NEUTRAL/mild DROP |
| F4 3D prefill axis | 2 | −5.9 | −6.2 | −0.3pp | −29.3 | −30.1 | −0.8pp | NEUTRAL |
| F4 3D prefill axis | 8 | −35.3 | −37.2 | −1.9pp | −36.0 | −37.8 | −1.8pp | NEUTRAL/mild DROP |

## Verdict summary

| Feature | Verdict | One-line reasoning |
|---|---|---|
| F1 / IQR filter | **DROP** | Removes legitimate heavy-tail variance → emu under-predicts latency. Confirms Apr 18. |
| F1 / MAD filter | **DROP** | Same failure mode as IQR, slightly worse. |
| F1 / Winsor filter | **NEUTRAL** | 1/99 clipping too mild to affect sampling noticeably. |
| F3 / sample_tokens_delay | **NEUTRAL** | avg_sample_ms is small relative to step_cycle; adds noise, not signal. |
| F5 / kNN (K=3) | **DROP** | Shepard weighting blurs the bucket distribution; v6 baseline isn't sparse enough to benefit. |
| F2 / parallel surrogate | **NEUTRAL** | Persistence-forecast prediction ≈ synchronous timing at steady state. |
| F4 / 3D prefill axis | **NEUTRAL** | num_new_reqs conditioning doesn't separate the workload in this profile. |

**NONE of the 7 feature variants improved accuracy on the v6 baseline.**

## Root-cause reaffirmation (from Phase B)

Three hypotheses were eliminated tonight:
1. **Variable-shape contamination**: REJECTED (no diff between partial-v5 without variables and full-v5 with variables).
2. **CUDA warmup sweep**: already removed in v4; v4 and v6 still fail → not sole cause.
3. **Server restarts between rounds**: REJECTED (v6 single-session matches v3 multi-session ~exactly).

The remaining un-tested variable is **round count** itself:
- 1 round (~65k samples): matches archive p90 shape, emu TPOT ~3-5pp WORSE than archive (under-sampled).
- 5 rounds (~325k samples): shape deviates ~22% at p90 tail, emu TPOT 5-13pp WORSE than archive.
- 2 rounds (~108k samples = archive): best known point.

Hypothesis: sample-count-dependent bucket-sample distribution. With more rounds, steady-state samples dominate each bucket's `samples` array, and the oracle's `random.choice` pulls proportionally fewer outlier values. The archive's 108k-sample sweet spot has enough samples to populate buckets but not so many that steady-state swamps the tail.

## Agg-mode A/B (Apr 19 09:37 BST) — isolate oracle aggregation vs profile data

Added experiment-gate `VLLM_EMULATOR_ORACLE_AGG={sample|median|mean}`. Tested 6 combinations at r=2, r=8 × 500 prompts.

| Profile | Agg | r=2 TPOT | r=2 TTFT | r=8 TPOT | r=8 TTFT |
|---|---|---|---|---|---|
| v6 | sample | −6.2% | −29.8% | **−35.3%** | −36.2% |
| v6 | median | −9.8% | −32.3% | **−43.3%** | −43.1% |
| v6 | mean | −6.4% | −29.6% | **−36.5%** | −38.8% |
| archive | sample | **−1.6%** | −26.9% | **−13.2%** | −11.1% |
| archive | median | −9.9% | −32.1% | −29.6% | −19.3% |
| archive | mean | −1.6% | −28.2% | −11.5% | −11.0% |

**Findings:**
- **Sample ≥ Mean ≫ Median on both profiles**: variance from tail matters. Sample drops 8-16pp by switching to median. Mean nearly matches sample (mean still reflects tail contribution; median is robust to outliers → ignores them).
- **Archive beats v6 at every aggregation mode**: archive_sample r=8 TPOT −13.2% vs v6_sample −35.3% (22pp gap). Even archive_median (−29.6%) beats v6_sample (−35.3%). **The profile data is the bottleneck, not the oracle aggregation algorithm.**

## Dilution diagnostic (10:20 BST)

`tools/diag_bucket_counts.py` computed per-bucket tail-sample ratio: given archive's per-bucket p75 as threshold, what fraction of v6's samples exceed it?

- Archive: 247 buckets, 93k samples, mean 378/bucket, max 7009/bucket.
- v6: 258 buckets, 280k samples, mean 1085/bucket (3× denser), max 39042/bucket.
- **Aggregate tail ratio (57 common well-populated buckets)**: p10 = 0.33, p50 = 0.67, p90 = 1.00, mean 0.69.

**Dilution confirmed but non-uniform**: high-population buckets (tt=2..10, c=2..7) have tail ratios near 1.0. Lower-population buckets lose the tail disproportionately. Overall v6 has ~30% LESS tail than archive at the same threshold.

## Reservoir experiment (10:28–10:37 BST, DONE)

Added `--reservoir-size N` to builder (Vitter 1985 Algorithm R). Rebuilt v6 with cap=7009 (archive's max per-bucket count): 10 buckets capped, decode samples reduced from 279k → 154k (55% retained).

**Result matches predicted null:**

| Profile | r=2 TPOT | r=8 TPOT | gap to archive |
|---|---|---|---|
| v6_sample (full) | −6.2% | −35.3% | −4.6pp / −22.1pp |
| v6 reservoir=7009 | −6.4% | −36.4% | −4.8pp / −23.2pp |
| archive_sample | −1.6% | −13.2% | reference |

**Δ(reservoir − full) = (−0.2pp, −1.1pp)** — within noise on both rates. Reservoir gives **zero accuracy benefit**, confirming that the dilution is distributional (later-round bulk shifts DOWN from earlier-round bulk due to warmer server state), not volumetric. Reservoir sampling preserves distribution shape → preserves the bias.

**Implication**: a non-knob data-level fix isn't possible from existing v6 trace alone. Options:
- Round-weighted storage (adds a per-sample round-index field; "knob-adjacent" but defensible as structural change).
- Profile at archive-matching round count (the previously-rejected "cut rounds").
- Trace preprocessing: preferentially keep early-round samples when pooling per-bucket.

## Implication for "more data → better" invariant

The invariant doesn't hold with this oracle architecture unless the later-round bulk's speedup is somehow excluded. Options (none are truly knob-free):
1. **Round-weighted storage** — tag each sample with a round index, weight inversely to round population. Requires adding round-index field to trace and to bucket storage.
2. **Per-round oracle instances** — sample uniformly from rounds, not from pooled samples. Same data collection but changes sampling structure.
3. **Accept the methodology** — profile at archive's round count (~2 rounds). This IS the "cut rounds" fix the user previously rejected.

## Directions for next session

1. **Validate sample-count hypothesis**: build v6-cap profiles that sub-sample each bucket to 50 / 100 / 500 / 1000 / unlimited samples. If accuracy improves then degrades as cap grows, that confirms the hypothesis and gives us a principled sub-sampling rule.
2. **If confirmed, build a profile quality invariant check before any feature ablation**: profile must match archive per-rate TPOT within ±2pp before any F-A/B is reported.
3. **Paper direction**: if no methodology reaches the "more data → better" invariant, path-1 fixed-workload paper stands; features don't close the gap either. Path-3 (Δ-on-policy validation) becomes more attractive — the emu doesn't need absolute accuracy if the claim is "relative deltas preserved".

## Files

- `paper/apr_19/progress.md` — append-only timeline (every 30 min check-in)
- `paper/apr_19/INDEX.md` — pointer
- `results/RTX-8000-adaptive-v6-5r-single/` — v6 profile + trace (309k records)
- `results/RTX-8000-v6-validate/` — Phase B emu A/B (5 rates × 2000 prompts)
- `results/RTX-8000-f{1_iqr,1_mad,1_winsor,2,3,4,5}-{off,on}-apr19/` — 7 feature A/Bs × 2 passes = 14 dirs
- `tools/adaptive_profile_v6_5r_single.sh` — Phase A
- `tools/validate_v6_full.sh` — Phase B
- `tools/overnight_phase_c_v6.sh` — Phase C (intervention script after initial chain's missing-feature bug)
- `tools/overnight_summarize_v6.sh` — Phase D (auto-summary had regex bug, this file rebuilt manually)
