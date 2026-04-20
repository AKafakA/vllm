# Apr 20 — Batch composition diagnostic, r=16 × 2000 prompts

Compares per-step batch composition between emu with v3 arrival-delay hook (SCHEDULER_HOOK=1) and emu with hook disabled (SCHEDULER_HOOK=0), same random 256/128 workload, same profile (archive-r2).

## Summary

| metric | hook_on | hook_off | delta |
|---|---|---|---|
| steps | 1675 | 1671 | +4.0 |
| prefill_steps | 861 | 854 | +7.0 |
| pure_decode_steps | 814 | 817 | -3.0 |
| mixed_steps | 859 | 852 | +7.0 |
| pure_prefill_steps | 2 | 2 | +0.0 |
| n_reqs_mean | 156.7 | 157.1 | -0.4 |
| n_reqs_p50 | 249 | 250 | -1.0 |
| n_reqs_max | 256 | 256 | +0.0 |
| n_decode_mean | 155.5 | 155.8 | -0.4 |
| n_new_mean | 1.2 | 1.2 | -0.0 |
| n_new_max | 8 | 8 | +0.0 |
| tt_mean | 459.9 | 461.0 | -1.1 |
| tt_p90 | 1224 | 1272 | -48.0 |
| oracle_us_mean | 121237.3 | 121192.0 | +45.3 |

## n_new distribution per step

| n_new | hook_on count | hook_on % | hook_off count | hook_off % |
|---|---|---|---|---|
| 0 | 814 | 48.6% | 817 | 48.9% |
| 1 | 341 | 20.4% | 344 | 20.6% |
| 2 | 219 | 13.1% | 203 | 12.1% |
| 3 | 125 | 7.5% | 119 | 7.1% |
| 4 | 71 | 4.2% | 83 | 5.0% |
| 5 | 61 | 3.6% | 52 | 3.1% |
| 6 | 5 | 0.3% | 22 | 1.3% |
| 7 | 35 | 2.1% | 29 | 1.7% |
| 8 | 4 | 0.2% | 2 | 0.1% |

## Interpretation

- hook_on: pure-decode steps = 48.6%, mixed = 51.3%
- hook_off: pure-decode steps = 48.9%, mixed = 51.0%
- Δ pure-decode: -0.3pp
- Δ mixed: +0.3pp

**Verdict**: batch composition is **essentially identical** between hook_on and hook_off (<2pp delta on pure-decode and mixed step fractions). The r=16 TTFT drift is NOT from batch-composition shifting; it's likely an oracle miscalibration or queue-drain artefact amplified by the hook's per-step scheduling delay jitter.
