# Apr 19 — Exp 2 full 5-rate sharegpt matrix

## Setup

- Profile: archive-r2 (2-round archive, 184k samples)
- Hardware: RTX 8000, Model: Qwen/Qwen3-8B
- 2000 prompts per bench, filtered sharegpt (input≤256, output≤128)
- r=2/8/32 from `exp2-sharegpt-*` (baseline Exp 2), r=4/16 from `supp-sharegpt-*` (gap-fill)

## 5-rate sharegpt results

| rate | TPOT real (ms) | TPOT emu (ms) | ΔTPOT% | TTFT real (ms) | TTFT emu (ms) | ΔTTFT% | ΔE2E% |
|---|---|---|---|---|---|---|---|
| 2 | 29.76 | 31.84 | +7.02% | 87.89 | 98.61 | +12.19% | +7.05% |
| 4 | 30.83 | 42.59 | +38.15% | 93.27 | 158.22 | +69.64% | +39.47% |
| 8 | 36.30 | 51.77 | +42.63% | 104.25 | 157.20 | +50.80% | +42.74% |
| 16 | 41.85 | 87.69 | +109.51% | 123.51 | 274.97 | +122.63% | +110.04% |
| 32 | 102.54 | 100.39 | -2.10% | 430.38 | 341.01 | -20.77% | -3.02% |

## Interpretation

- **r=2 light load**: +7% TPOT, +12% TTFT. Small shape mixing at low conc; errors begin.
- **r=4 (+38%), r=8 (+43%)**: monotonic growth — KV-depth mismatch amplifies as batching intensifies.
- **r=16 (+110% TPOT, +123% TTFT)**: worst regime. Real saturation not yet hit on long sharegpt sequences, but emu profile lacks the matching decode-bucket distribution → huge over-estimation.
- **r=32 saturation bailout**: real hits throughput ceiling (102ms TPOT, 430ms TTFT queue buildup); emu's bounded prediction converges (TPOT −2%, TTFT −21%). Saturation ceiling masks profile gap.
- **Architectural verdict**: 2D oracle is KV-depth-blind — shape generalisation cap is structural, not a sample-density issue. Fix path: KV-depth-conditioned oracle (α-KV from model_config extension or 3D axis on sum_kv).

## Comparison to random 256/128 (archive-r2 baseline)

| rate | ΔTPOT sharegpt | ΔTPOT random | ΔTTFT sharegpt | ΔTTFT random |
|---|---|---|---|---|
| 2 | +7.02% | -0.02% | +12.19% | -31.65% |
| 4 | +38.15% | -0.56% | +69.64% | -29.16% |
| 8 | +42.63% | -2.52% | +50.80% | -23.47% |
| 16 | +109.51% | +0.04% | +122.63% | -6.35% |
| 32 | -2.10% | +1.11% | -20.77% | -1.14% |

**Shape-sensitivity signal**: sharegpt TPOT deltas are larger-magnitude at mid-rates, confirming 2D-oracle shape-class limit. TTFT deltas follow similar pattern because per-request IPC overhead was not yet applied at time of measurement.
