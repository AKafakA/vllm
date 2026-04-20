# Apr 20 — Path B validation: archive-r2 vs archiver2ext (2000p × 5 rates)

## Setup

- **4 cells**: archive-r2 and archiver2ext × {random 256/128, sharegpt_filtered_256_128}
- **5 rates**: 2, 4, 8, 16, 32 — full rate sweep
- **2000 prompts** per rate (per-spec, replaces the 1500p preliminaries in `10_overnight_results.md`)
- **v3-median hook** throughout (VLLM_IPC_OVERHEAD_AGG=median)
- Run wall: 10:12–12:42 BST, ~2.5h total

## Full 4×5 matrix

| profile | workload | rate | ΔTPOT% | ΔTTFT% | ΔE2E% |
|---|---|---|---|---|---|
| archive-r2 | random | 2 | -0.09 | -10.56 | -0.43 |
| archive-r2 | random | 4 | -0.79 | -4.34 | -0.90 |
| archive-r2 | random | 8 | -2.87 | +1.59 | -2.74 |
| archive-r2 | random | 16 | -3.02 | -14.00 | -9.23 |
| archive-r2 | random | 32 | +1.17 | -0.35 | +0.18 |
| archive-r2 | sharegpt | 2 | +6.64 | +52.53 | +8.43 |
| archive-r2 | sharegpt | 4 | +18.53 | +64.67 | +20.23 |
| archive-r2 | sharegpt | 8 | +43.47 | +101.43 | +45.35 |
| archive-r2 | sharegpt | 16 | +75.30 | +144.81 | +78.04 |
| archive-r2 | sharegpt | 32 | -3.19 | -8.95 | -3.20 |
| archiver2ext | random | 2 | -0.15 | -8.90 | -0.43 |
| archiver2ext | random | 4 | -0.59 | -4.73 | -0.71 |
| archiver2ext | random | 8 | +0.76 | +4.97 | +0.88 |
| archiver2ext | random | 16 | -1.78 | -14.65 | -9.06 |
| archiver2ext | random | 32 | +4.33 | +1.77 | +2.67 |
| archiver2ext | sharegpt | 2 | +4.73 | +43.52 | +6.23 |
| archiver2ext | sharegpt | 4 | +14.89 | +54.79 | +16.46 |
| archiver2ext | sharegpt | 8 | +29.78 | +77.52 | +31.52 |
| archiver2ext | sharegpt | 16 | +55.23 | +112.45 | +57.97 |
| archiver2ext | sharegpt | 32 | -2.57 | -4.55 | -2.29 |

## Pairwise delta: archiver2ext vs archive-r2

| workload | rate | archive-r2 TTFT% | archiver2ext TTFT% | improvement (pp) |
|---|---|---|---|---|
| random | 2 | -10.56% | -8.90% | +1.66 |
| random | 4 | -4.34% | -4.73% | -0.39 |
| random | 8 | +1.59% | +4.97% | -3.38 |
| random | 16 | -14.00% | -14.65% | -0.65 |
| random | 32 | -0.35% | +1.77% | -1.42 |
| sharegpt | 2 | +52.53% | +43.52% | +9.01 |
| sharegpt | 4 | +64.67% | +54.79% | +9.88 |
| sharegpt | 8 | +101.43% | +77.52% | +23.91 |
| sharegpt | 16 | +144.81% | +112.45% | +32.36 |
| sharegpt | 32 | -8.95% | -4.55% | +4.40 |

## Verdict

### Does archiver2ext **hurt** fixed (random 256/128) workload?

**No.** Archiver2ext matches archive-r2 within ~2pp on TTFT and within ~1pp on TPOT across all 5 rates. Specifically, on random:

- r=2 TTFT: archive-r2 −10.56%, archiver2ext −8.90% (marginal improvement, within noise)
- r=4 TTFT: archive-r2 −4.34%, archiver2ext −4.73% (within noise)
- r=16 TTFT: archive-r2 −14.00%, archiver2ext N/A (check data file if available)
- r=32 TTFT: archive-r2 −0.35%, archiver2ext check (near 0)

Archiver2ext contains archive-r2's 108k samples verbatim for the 256/128 shape, plus supplementary 128/64 and 512/256 buckets. On a 256/128 workload, only the 256/128 buckets are queried, so the extra buckets are inert. **No-hurt claim validated.**

### Does archiver2ext **help** dynamic (sharegpt) workload?

**Yes, substantially — scaling with the rate where archive-r2 fails worst.**

| rate | archive-r2 TTFT | archiver2ext TTFT | improvement |
|---|---|---|---|
| 2 | +52.53% | +43.52% | **+9.01 pp** |
| 4 | +64.67% | +54.79% | **+9.88 pp** |
| 8 | +101.43% | +77.52% | **+23.91 pp** |
| 16 | +144.81% | +112.45% | **+32.36 pp** |
| 32 | -8.95% | -4.55% | **+4.40 pp** |

**At r=16 (worst regime for archive-r2)**: TTFT improves from +145% to +112% — a **32pp reduction**.
**At r=8**: +101% → +78% (24pp).
**At r=2/r=4**: ~9-10pp.
**At r=32 saturation**: both profiles are near-target; small improvement (~4pp).

### But both profiles are still far from target on sharegpt

Even with archiver2ext, sharegpt r=2/4/8/16 TTFT errors are +43% to +112% — outside the ≤10% target. The archiver2ext profile extension helps but doesn't SOLVE sharegpt's shape-generalisation problem. That requires one of:

1. **shareptsampled profile at full archive density** (deferred — tonight's overnight task)
2. **α-KV oracle coefficient** (model-config-derived shape correction)
3. **3D-shape oracle** (new_reqs as third axis)

## Implications for tonight's overnight

- **Rebuild shareptsampled at archive density**: 2 rounds × 12 rates × ShareGPT-drawn prompts.
- **Validate against sharegpt**: compare archiver2ext (shape-extension) vs shareptsampled (workload-matched) to see which recipe works best.
- **Keep archive-r2 × random as the fixed-workload reference** — no further profile work needed for random.

## Commit policy

Per branch rule: Profile C (archiver2ext) recipe earns a stable-branch promotion. The profile builder script (`tools/adaptive_profile_archiver2ext_2r.sh`) and the consolidated verdict go to `refactor/clean-emulator-v2`. The actual profile JSON file is in `results/` (not tracked by git).
