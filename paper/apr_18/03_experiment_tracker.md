# Apr 18 Experiment Tracker

Live table of every A/B run today. Row is added when an experiment is launched. Status progresses `PLAN → RUN → PASS/FAIL`.

Schema: `<branch_short>-<variant>-r<rate>` for run id.

## Runs

| Run ID | Branch | Feature | Gate setting | Profile | Rate | TPOT% | TTFT% | E2E% | tok/s% | Status | progress.md link | Commit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v3-nosurr-r02 | (S0 baseline) | — | (no surrogate) | v3 | 2 | tbd | tbd | tbd | tbd | RUN | [TBD](#) | `a102eed27` |
| v3-nosurr-r08 | (S0 baseline) | — | (no surrogate) | v3 | 8 | tbd | tbd | tbd | tbd | RUN | [TBD](#) | `a102eed27` |
| v3-nosurr-r16 | (S0 baseline) | — | (no surrogate) | v3 | 16 | tbd | tbd | tbd | tbd | RUN | [TBD](#) | `a102eed27` |
| v3-withsurr-r02 | (S0 baseline) | — | `VLLM_EMULATOR_PREP_SURROGATE=1` | v3 | 2 | tbd | tbd | tbd | tbd | RUN | [TBD](#) | `a102eed27` |
| v3-withsurr-r08 | (S0 baseline) | — | `VLLM_EMULATOR_PREP_SURROGATE=1` | v3 | 8 | tbd | tbd | tbd | tbd | RUN | [TBD](#) | `a102eed27` |
| v3-withsurr-r16 | (S0 baseline) | — | `VLLM_EMULATOR_PREP_SURROGATE=1` | v3 | 16 | tbd | tbd | tbd | tbd | RUN | [TBD](#) | `a102eed27` |

Feature blocks will be appended below as each feature's slot begins.

---

## F1 — outlier filter (DROP)

Baseline profile: archive `results/_archive/serving-dense.json`. Variant: post-hoc IQR via `tools/apply_outlier_filter.py` (dropped 9.72% of samples). A/B at r=2/8/16, 1000 prompts.

| Run ID | Branch | Gate | Rate | TPOT% | TTFT% | E2E% | tok/s% | Status | progress | Commit |
|---|---|---|---|---|---|---|---|---|---|---|
| f1-off-r02 | exp/f1-outlier-filter | --outlier-filter none | 2 | −0.7 | −30.3 | −1.6 | −0.4 | PASS | baseline | 944238c32 |
| f1-off-r08 | exp/f1-outlier-filter | --outlier-filter none | 8 | −11.4 | −31.9 | −12.0 | −2.8 | PASS | baseline | 944238c32 |
| f1-off-r16 | exp/f1-outlier-filter | --outlier-filter none | 16 | −10.7 | −71.7 | −45.2 | +2.6 | PASS | baseline | 944238c32 |
| f1-on-r02  | exp/f1-outlier-filter | --outlier-filter iqr  | 2 | −9.4 | −36.1 | −10.2 | −0.4 | FAIL | variant | 944238c32 |
| f1-on-r08  | exp/f1-outlier-filter | --outlier-filter iqr  | 8 | −16.5 | −35.1 | −17.0 | −2.3 | FAIL | variant | 944238c32 |
| f1-on-r16  | exp/f1-outlier-filter | --outlier-filter iqr  | 16 | −10.7 | −74.9 | −47.0 | +2.0 | FAIL | variant | 944238c32 |

Verdict: **DROP**. IQR removed legitimate heavy-tail variance; emu ran too fast → TPOT regressed by 5-9pp at low rates.


## Conflict matrix

Filled at S13 (combined validation).

| (X, Y) pair | additivity worst-rate | regression? | sign-flip? | variance? | schema? | Verdict |
|---|---|---|---|---|---|---|
| _(populated during S13)_ | | | | | | |

Verdict codes: `independent` (stacks freely) / `weak-interaction` (needs a tuning step) / `conflict` (one feature must be reformulated) / `hard-conflict` (structural).
