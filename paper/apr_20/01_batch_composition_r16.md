# Apr 20 — Batch composition diagnostic, r=16 × 2000 prompts

Compares per-step batch composition between emu with v3 arrival-delay hook (SCHEDULER_HOOK=1) and emu with hook disabled (SCHEDULER_HOOK=0), same random 256/128 workload, same profile (archive-r2).

## Summary

| metric | hook_on | hook_off | delta |
|---|---|---|---|
| steps | 0 | 0 | +0.0 |
| prefill_steps | — | — | — |
| pure_decode_steps | — | — | — |
| mixed_steps | — | — | — |
| pure_prefill_steps | — | — | — |
| n_reqs_mean | — | — | — |
| n_reqs_p50 | — | — | — |
| n_reqs_max | — | — | — |
| n_decode_mean | — | — | — |
| n_new_mean | — | — | — |
| n_new_max | — | — | — |
| tt_mean | — | — | — |
| tt_p90 | — | — | — |
| oracle_us_mean | — | — | — |

## n_new distribution per step

| n_new | hook_on count | hook_on % | hook_off count | hook_off % |
|---|---|---|---|---|

## Interpretation

