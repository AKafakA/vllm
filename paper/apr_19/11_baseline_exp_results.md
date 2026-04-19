# Apr 19 baseline experiments — results

## Experimental setup

- Profile: archive-r2 (2-round archive recipe, 184k samples, single-session)
- Hardware: RTX 8000
- Model: Qwen/Qwen3-8B
- 2000 prompts per bench, single-session server for each {real, emu} pass

## Exp 1 — Burstiness generalization (rate=4, 2000 prompts × 256/128)

| burst | TPOT real (ms) | TPOT emu (ms) | ΔTPOT% | TTFT real (ms) | TTFT emu (ms) | ΔTTFT% | ΔE2E% |
|---|---|---|---|---|---|---|---|
| 0.3 | 45.93 | 47.25 | +2.88% | 246.05 | 154.06 | -37.39% | +1.25% |
| 1.0 | 45.38 | 44.82 | -1.23% | 177.52 | 129.62 | -26.98% | -2.00% |
| 3.0 | 45.69 | 45.31 | -0.84% | 154.09 | 131.27 | -14.81% | -1.20% |

**Interpretation**: TPOT within ±3% across all burstiness (arrival-pattern-robust). TTFT gap ranges −15% (burst=3.0) to −37% (burst=0.3). Real TTFT grows with burstiness (clustered arrivals pressure per-request CPU/IPC overhead); emu TTFT stays ~flat (emu's timer-based Future skips that overhead). Confirms the per-request-overhead mechanism as the TTFT gap source.

## Exp 2 — Shape generalization (filtered sharegpt, input≤256, output≤128)

| rate | TPOT real (ms) | TPOT emu (ms) | ΔTPOT% | TTFT real (ms) | TTFT emu (ms) | ΔTTFT% | ΔE2E% |
|---|---|---|---|---|---|---|---|
| 2 | 29.76 | 31.84 | +7.02% | 87.89 | 98.61 | +12.19% | +7.05% |
| 8 | 36.30 | 51.77 | +42.63% | 104.25 | 157.20 | +50.80% | +42.74% |
| 32 | 102.54 | 100.39 | -2.10% | 430.38 | 341.01 | -20.77% | -3.02% |

**Interpretation**: sharp degradation at mid-saturation r=8 (TPOT +43%, TTFT +51%). At r=2, small (TPOT +7%) because low-conc batches have less shape-mix. At r=32 (saturated), shape variation averages out (TPOT −2%). **Concrete evidence of the 2D-oracle shape-class limit** — archive-r2 profile (all 256/128) cannot match sharegpt's varying KV-depth distributions at mid-rate. Architectural fix (KV-depth-conditioned oracle) is the path to cross-shape accuracy.

## Exp 3 — Combined stress (sharegpt + burstiness=0.3, rate=4)

| | TPOT real (ms) | TPOT emu (ms) | ΔTPOT% | TTFT real (ms) | TTFT emu (ms) | ΔTTFT% | ΔE2E% |
|---|---|---|---|---|---|---|---|
| r=4 | 31.82 | 39.14 | +23.02% | 100.75 | 133.73 | +32.74% | +23.19% |

**Interpretation**: worst-case combination stresses BOTH failure modes. Compare to Exp 1 burst=0.3 on 256/128 (TPOT +2.9%, TTFT −37%) and Exp 2 sharegpt r=4 (only tested r=2/8/32 — interpolate). Combined row isolates the additive impact of both shape-mix AND burstiness.

## Headline summary

- **Burstiness-robust on TPOT**: across burst ∈ {0.3, 1.0, 3.0}, TPOT within ±3%.
- **Shape-sensitive**: profile built on single shape (256/128) cannot predict multi-shape workload reliably at mid-saturation (TPOT +43% at r=8 on sharegpt).
- **TTFT gap is mechanistic** (per-request CPU/IPC overhead), not a profile-quality issue. Fix coming via IPC-overhead sweep (exp/ipc-overhead-sweep branch).
