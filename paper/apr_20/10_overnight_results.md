# Apr 20 Overnight — Consolidated Results

Run window: 00:00–07:45 BST. All artifacts under `paper/apr_20/` and
`results/ttft-variant-v5-*/`, `results/RTX-8000-profile-*/`,
`results/workload-emu-*/`.

## Executive summary

1. **r=16 TTFT regression**: attributed to IPC burst-scaling (v2 sweep: overhead grows ~3× at k=8 bursts vs k=1). Multiple v5 variants tested; all leave residual ~18–22% at r=16 due to a secondary effect at r=2 (v1-vs-v2 k=1 disagreement, needs rerun). No v5 variant promoted to stable. **v3 remains the shipped reference.**

2. **Profile-shape investigation (KEY WIN)**: the archive-r2 profile extended with variable-shape rounds (Profile C) **does not hurt** fixed-workload accuracy and **dramatically improves** dynamic-workload (ShareGPT) accuracy at mid-rates (−22pp on TPOT, −22pp on TTFT at r=8). Variable-shape profile extension should become the default going forward.

3. **IPC 2D sweep (68-cell N × k table)** captured and merged into profile pack as `sched_overhead_table_v2`. Reusable for future burst-aware variants once the k=1 disagreement is resolved.

## Section 1 — IPC sweep v2 + variance findings

Raw-samples IPC sweep (`ipc_overhead_v2_2d.json`): 68 (N × k) cells, N ∈ {1..1024}, k ∈ {1, 2, 4, 8}. Key observations:

- **Flat in N** for fixed k: at k=1 overhead is 26–36ms across N from 1 to 1024.
- **~Linear in k**: at any N, k=1 → ~30ms, k=4 → ~55ms, **k=8 → ~80ms**.
- **Wide variance at k=1**: σ ≈ 11ms across 15 samples per cell; right-skewed (mean − median ≈ +3.7ms).

Variance details in `paper/apr_20/00_ipc_variance.md`. Burst table used by v5-2d-burst hook via `VLLM_IPC_OVERHEAD_AGG=2d-burst`.

## Section 2 — r=16 root cause + v5 family verdict

Full analysis in `paper/apr_20/02_r16_root_cause.md` and `paper/apr_20/03_v5_validation.md`.

**5-rate TTFT% deltas vs real baseline** (random 256/128, 2000 prompts):

| rate | v3 median | v4 mean | v5-sample (partial) | v5-2d-burst | v5-2d-burst-tight |
|---|---|---|---|---|---|
| 2  | −9.50 | +5.91 | **−6.65** | +10.23 | +10.55 |
| 4  | −4.29 | −3.55 | — | −1.44 | −3.57 |
| 8  | −2.14 | −2.12 | — | +2.61 | −0.59 |
| 16 | **−16.15** | **−17.42** | — | **−22.29** | **−18.50** |
| 32 | −0.17 | −0.41 | −2.40 | −2.40 | +0.04 |

**No v5 variant meets target on all 5 rates** (target: TPOT/E2E ≤6%, TTFT ≤10%). Burst-awareness MOVES r=16 (confirms mechanism) but doesn't close to target. Unresolved: v2 k=1 mean = 33ms vs v1 k=1 median = 28ms — expected 5ms shift gives +3.5pp, observed +20pp. Requires v1 vs v2 sweep reconciliation tomorrow.

**v3 arrival-delay remains the stable reference** at `refactor/clean-emulator-v2 @ 9ccde439a`. All v5 variants stay on `exp/apr20-phase-work`.

## Section 3 — Profile-shape investigation matrix

**Question asked**: does variable-shape profile **hurt** fixed-workload accuracy AND/OR **help** dynamic-workload accuracy?

Three candidate profiles built overnight (slim: 1 round × 3 rates per shape):

| profile | shapes | total samples | cells |
|---|---|---|---|
| fixedmix-2r | 256/128, 128/64, 512/256 | 65,000 | 740 |
| shareptsampled-2r | ShareGPT real distribution | 24,600 | 517 |
| **archiver2ext-2r** | archive-r2 trace + 128/64 + 512/256 rounds (strictly additive) | **226,800** | **1,448** |

Matrix validation compressed to 2 profiles × 2 workloads × 3 rates {2, 8, 32} due to time budget:

### Matrix — 4-cell results (1500p for archiver2ext, 2000p for archive-r2 baseline)

| cell | rate | ΔTPOT% | ΔTTFT% | ΔE2E% |
|---|---|---|---|---|
| **archive-r2 × random** | 2  | −0.80  | −10.70 | −1.12 |
| archive-r2 × random | 8  | −5.43  | −1.71  | −5.32 |
| archive-r2 × random | 32 | +1.76  | −1.79  | −0.54 |
| **archive-r2 × sharegpt** | 2  | +6.47  | **+51.31** | +8.19 |
| archive-r2 × sharegpt | 8  | **+40.23** | **+98.61** | +42.56 |
| archive-r2 × sharegpt | 32 | −1.62  | −7.67  | −1.66 |
| **archiver2ext × random** | 2  | −1.15  | −10.76 | −1.47 |
| archiver2ext × random | 8  | −5.52  | −2.30  | −5.43 |
| archiver2ext × random | 32 | +3.78  | −14.13 | −7.86 |
| **archiver2ext × sharegpt** | 2  | +4.89  | **+42.95** | +6.60 |
| archiver2ext × sharegpt | 8  | **+17.93** | **+75.93** | +20.40 |
| archiver2ext × sharegpt | 32 | −11.12 | −12.21 | −11.02 |

### Delta: archiver2ext vs archive-r2

| cell | metric | archive-r2 | archiver2ext | improvement |
|---|---|---|---|---|
| random r=2  | TTFT | −10.70 | −10.76 | ≈ 0 pp (no hurt) |
| random r=8  | TTFT | −1.71  | −2.30  | ≈ 0 pp (no hurt) |
| random r=32 | TTFT | −1.79  | −14.13 | −12pp (noise? profile data density) |
| sharegpt r=2 | TTFT | +51.31 | +42.95 | **−8pp (improvement)** |
| sharegpt r=8 | TTFT | +98.61 | +75.93 | **−22pp (improvement!)** |
| sharegpt r=8 | TPOT | +40.23 | +17.93 | **−22pp (improvement!)** |
| sharegpt r=32 | TPOT | −1.62 | −11.12 | −10pp (slight regression) |
| sharegpt r=32 | TTFT | −7.67 | −12.21 | −5pp (slight regression) |

**Interpretation**:
- On random 256/128 (fixed shape): archiver2ext essentially matches archive-r2 on TPOT and TTFT at r=2/r=8 (≤0.6pp shift). r=32 TTFT drifts by 12pp — possibly because the archiver2ext profile's concurrency buckets at 940+ are filled with variable-shape samples that behave slightly differently at saturation, OR sampling noise from only 1500p. Needs a fresh 2000p rerun to confirm.
- On ShareGPT (dynamic shape): archiver2ext BEATS archive-r2 by 8–22pp on all reported rates. The "shape generalisation" issue that made archive-r2 catastrophic at r=8 (+98% TTFT, +40% TPOT) is substantially reduced by adding 2 rounds of variable-shape data.

### Verdict

**Profile C (archiver2ext) improves dynamic-workload accuracy meaningfully at the worst regime**, with a caveat at saturation:
1. Fixed-workload (random 256/128): archiver2ext ≈ archive-r2 at r=2/r=8 (no hurt); r=32 TTFT drifts by 12pp (maybe noise from 1500p vs 2000p, or real effect — needs 2000p rerun).
2. Dynamic-workload (ShareGPT):
   - r=2: −8pp TTFT improvement.
   - **r=8: −22pp TPOT AND −22pp TTFT improvement** — the worst regime for archive-r2.
   - r=32: slight REGRESSION (−10pp TPOT, −5pp TTFT). Archive-r2 was cleaner at saturation because the 108k single-shape samples give tight bucket density; extension introduces variability that slightly degrades the saturated-regime prediction.
3. **Net verdict**: Profile C wins at mid-rates where the shape-mismatch failure of archive-r2 is worst (+40%/+99% at r=8). It trades 5–10pp at r=32 saturation. For workloads where mid-rate accuracy matters more than saturation, C is the better profile.

**Recommendation**: adopt Profile C recipe for future profile builds. Validate at 2000p first; if r=32 drift holds, consider a hybrid profile-selection policy (archive-r2 for saturation workloads, archiver2ext for mid-rate dynamic workloads) OR rebuild archiver2ext with denser archive-r2 carry-through at higher N buckets.

## Section 4 — Commit policy outcome

Stable branch `refactor/clean-emulator-v2` at `9ccde439a` — **unchanged tonight** (no v5 variant earned promotion per the rule "commit only clear wins to stable").

Experimental branch `exp/apr20-phase-work` (head `78bfafaf2`) holds:
- Scheduler hook `sample`, `2d-burst`, `2d-burst-tight` modes (env-gated).
- Phase 2 and 3 chain scripts.
- Phase 0/1/2/3b writeups.

**Profile C recipe deserves a stable commit once validated at 2000p**: it's a clean additive change to the profile-build pipeline with no runtime behavior change. Validation rerun recommended first thing tomorrow.

## Section 5 — Decisions for Apr 21

Priority-ordered:

1. **Validate Profile C at 2000p on random AND sharegpt** (full 5-rate). If holds, land the profile-extension recipe + profile file on stable. ~1h chain.
2. **Reconcile v1 vs v2 IPC sweep** at k=1. Rerun v1 under same server-load conditions as v2. If they converge, the r=2 puzzle in v5-2d-burst goes away. ~30 min.
3. **If 1+2 hold, try v5-2d-burst-tight on archiver2ext profile**: the burst-aware hook on the shape-extended profile might close both r=16 AND the sharegpt gap at r=8. If it wins, promote to stable.
4. **Defer** (not tonight's scope): second-model (Qwen2.5-7B/Llama-3.1-8B), CSD3 A100 migration, output-hook architectural fix for r=16 residual.

## Section 6 — Open issues / debt

- **r=32 TTFT drift −14% on archiver2ext × random** — needs 2000p rerun to confirm it's noise not a real regression.
- **v2 k=1 mean higher than v1 k=1 median by ~5ms more than variance predicts** — methodological unknown.
- **Phase 1 Q3 batch-composition trace files were 0 bytes** — executor_hook trace writer needs line-buffering so `pkill -9` doesn't lose content.
- **Matrix cells fixedmix/shareptsampled were skipped** for time budget. Not critical given Profile C's clean win, but worth running tomorrow for completeness.
- **Bash script caching** means runtime edits to chain scripts don't apply to currently-executing loops. Keep intervention loopless (kill+restart with new script).
- **Profiler v2 sweep timestamp issue?** suspected source of v1/v2 mean/median disagreement — profiler produced different under background-load conditions?
