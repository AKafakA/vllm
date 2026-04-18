# v3 Baseline A/B (slot S0) — Results + Fallback Decision

## A/B results (v3 profile = `results/RTX-8000-adaptive-v3/serving-full.json`, 318k records, 1989 combined buckets)

Pass A — **nosurr** (`results/RTX-8000-v3-nosurr/`):

| Rate | TPOT% | TTFT% | E2E% | tok/s% | max_conc r/e | verdict |
|---|---|---|---|---|---|---|
|  2 | −7.5  | −36.4 | −8.5  | +0.0 | 26/22   | FAIL |
|  4 | −15.3 | −39.5 | −16.1 | −0.0 | 50/43   | FAIL |
|  8 | −22.9 | −36.7 | −23.3 | +0.0 | 119/89  | FAIL |
| 16 | −10.2 | −35.1 | −24.3 | +10.0| 940/767 | FAIL |
| 32 | −5.3  | −8.7  | −7.5  | +6.1 | 1023/1021 | FAIL |

Pass B — **withsurr** (`results/RTX-8000-v3-withsurr/`):

| Rate | TPOT% | TTFT% | E2E% | tok/s% | max_conc r/e | verdict |
|---|---|---|---|---|---|---|
|  2 | −5.7  | −34.5 | −6.6  | −0.0 | 26/22   | FAIL |
|  4 | −11.7 | −35.7 | −12.4 | −0.0 | 50/46   | FAIL |
|  8 | −20.9 | −36.5 | −21.3 | −0.1 | 119/93  | FAIL |
| 16 | −5.0  | −16.7 | −11.6 | +4.3 | 940/842 | FAIL |
| 32 | −0.2  | −2.2  | −1.5  | +0.6 | 1023/1024 | **PASS** |

## Archive reference (locked earlier today, `results/_archive/serving-dense.json`, 108k records)

| Rate | TPOT% | TTFT% |
|---|---|---|
|  2 | −0.6  | −32.5 |
|  4 | −1.6  | −30.7 |
|  8 | −6.0  | −28.5 |
| 16 | −5.6  | −25.0 |
| 32 | +4.9  | +1.9  |

## Fallback-rule evaluation (v3 withsurr vs archive withsurr at r=2/4/8/16)

| Rate | Δ TPOT (v3 − arch) | Δ TTFT (v3 − arch) | within ±2pp both? |
|---|---|---|---|
|  2 | −5.1 pp worse      | −2.0 pp worse      | **NO** (TPOT fails) |
|  4 | −10.1 pp worse     | −5.0 pp worse      | **NO** |
|  8 | −14.9 pp worse     | −8.0 pp worse      | **NO** |
| 16 | +0.6 pp better     | +8.3 pp better     | **YES** |

**Selection:** v3 regresses at r=2/4/8 on both TPOT and TTFT, each by more than 2pp. Triggers the fallback clause.

### Baseline profile for feature ablations

Use `results/_archive/serving-dense.json` (archive, 108k records).

## Interesting anomaly at r=16

v3 is BETTER than archive at r=16 TTFT (−16.7 vs −25.0). Suggests the 3× profile data is helping at saturation where samples naturally accumulate, but hurting at low rates where the finer bucket grid yields fewer-sample-per-bucket noise. Document for tomorrow's v3-debug.

## Tomorrow's top priority (Apr 19)

**Debug the v3 profile regression.** Three concrete hypotheses to investigate:

1. **Fine-bucket variance amplification**: v3 uses `tt-width=1 conc-width=5` producing ~4× more buckets than the archive's bucketing. Each bucket has fewer samples → `random.choice` has higher variance → concurrency feedback loop amplifies. **Experiment**: rebuild v3 with `tt-width=5 conc-width=5` (archive-like grid) and re-run withsurr A/B. If r=8 TPOT recovers toward −6% (archive), variance amplification is the cause.

2. **Cross-round thermal drift**: 5 rounds span ~4h of GPU runtime. Late-round samples may have different latency distribution than early-round due to thermal state. **Experiment**: filter v3 trace to a single round, rebuild profile, compare.

3. **Variable-shape contamination**: rate-sweep queries (256/128 prompts) land in (tt, conc) buckets that also hold variable-shape samples (64/32, 512/256, 128/64 at fixed rates). **Experiment**: rebuild v3 excluding variable-shape phases (would need to tag them with a separate marker).

Any of these, alone or in combination, could explain the regression. Tomorrow's morning session should pick the cheapest first (hypothesis 1: just rebuild with coarser bucketing).

## Other decisions made here

- All F1–F5 ablation A/Bs will run against `results/_archive/serving-dense.json`.
- The per-feature success criteria in `01_ablation_study_plan.md` stay unchanged.
- v3 profile is NOT discarded — it is kept under `results/RTX-8000-adaptive-v3/` for tomorrow's investigation.
