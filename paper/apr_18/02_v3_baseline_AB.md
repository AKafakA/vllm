# v3 Baseline A/B (slot S0) — Results, Fallback Decision, and Root-Cause Diagnostic

## UPDATED root-cause (established 22:55 UTC, supersedes 18:25 finding below)

The v3/v4 bias has **two stacked causes**, not one. The 18:25 finding (CUDA warmup sweep) is correct but only explains ~60% of the gap. The remaining cause is **server restarts between rounds.**

Evidence — per-bucket p90 Δ% vs archive, 3-profile comparison on `256/128` random workload, no variable shapes included:

| Profile | Structure | CUDA sweep | Server restarts | p50 Δ% | **p90 Δ%** |
|---|---|---|---|---|---|
| v3 (5-round, warmup sweep on) | multi-session | yes | yes (4) | matches | −45% |
| v4 (5-round, warmup sweep removed) | multi-session | no | yes (4) | −0.55% | **−24%** |
| v5 partial (6 rates, warmup sweep removed, single session) | single-session | no | **none** | −0.23% | **−3.14%** |

Eliminated as causes by this comparison:
- Variable-shape contamination (none of these profiles has variable shapes)
- CUDA warmup sweep (v4 and v5 both remove it)
- Profiling-window marker bugs (fixed in `a102eed27`, present in all)

**Remaining variable: server restarts between rounds.** Archive is single-session; v5 partial is single-session. V4 restarts between rounds. Each restart re-initialises CUDA context, IPC state, memory fragmentation — subsequent rounds may start with a freshly-warm but differently-fragmented state that truncates the heavy tail the oracle's `random.choice` needs.

Conclusion: the **"more data → better results" invariant holds IFF profiling is single-session.** Multi-round-with-restarts introduces structured bias that widens the profile-vs-real gap even when the individual samples are correct.

---

## Original root-cause (established 18:25 UTC — still relevant for v3)

The v3 bias is a **per-bucket DISTRIBUTION SHAPE issue, not an average-metric issue.**

Warm-vs-cold real baselines recaptured at 17:40 UTC (`tools/recapture_real_warm.sh` → `results/RTX-8000-v31-2000p-warm/`): all metrics within 2% of the original cold baselines. Confirms CUDA graph capture cost is amortized across thousand-second runs → <1% effect on average TPOT/TTFT.

Per-bucket percentile analysis (`tools/diag_v3_deep.py`):

```
(tt=20, c=22):
  v3      p50=37275  p90=38738  p95=82630  p99=88072    (n=2961)
  archive p50=37358  p90=83070  p95=85445  p99=126279   (n=1382)
                     ^^^^ 2.1× gap at p90
```

**Medians match. Heavy tails are missing from v3.** Archive's tails are the CUDA graph capture events (~50–100 ms spikes) on first encounter of each batch-size graph during rate sweep. V3's explicit warmup sweep pre-captures all graphs outside the profile window → rate-sweep samples miss capture events entirely → profile is artificially tight.

When the oracle's `random.choice` draws from v3, it never pulls the 80+ ms "cold-capture" samples that archive's oracle occasionally draws. Emu's per-step latency variance is lower than real's → concurrency-feedback loop amplifies → TPOT regresses by 10%+ even though per-step mean error is only 10%.

So the earlier "remove CUDA warmup sweep from adaptive_profile_full.sh" fix is the right move, but the reason is subtler than initially framed: not "match real's average behaviour" but "preserve the per-bucket distribution SHAPE that includes capture-cost outliers."

## Invariant violated (enforceable going forward)

**More profile data must produce better (or at least equal) emulator accuracy — never worse.**

v3 has 3× archive's data but regressed emulator accuracy by 5–15 pp TPOT and 2–8 pp TTFT at low-mid rates. That is an invariant violation. When this invariant is violated, the methodology is wrong, not the algorithm. Feature ablations built on top of a methodology that violates this invariant produce relative-between-broken-conditions comparisons, not absolute-accuracy measurements — so they cannot be published or shipped without the methodology first being validated.

**Today's feature ablations are paused at F1 (DROP verdict, archive baseline) pending invariant restoration.**


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

## Fallback-rule evaluation

V3 regressed by >2pp TPOT/TTFT at r=2/4/8 → triggers archive fallback.

**Baseline profile for feature ablations: `results/_archive/serving-dense.json`.**

## Diagnostic executed today (Apr 18, while F3 A/B runs)

Three hypotheses tested directly on the v3 trace:

### Hypothesis 1: Cross-round thermal drift — REJECTED

`tools/diag_v3_rounds.py` split v3 trace by markers into 5 rounds and computed per-round stats:

| Round | n | mean | median | std | p10 | p90 |
|---|---|---|---|---|---|---|
| 1 | 36,800 | 33,150 | 29,791 | 9,707 | 28,290 | 40,431 |
| 2 | 57,800 | 33,363 | 29,782 | 12,995 | 27,474 | 40,142 |
| 3 | 57,800 | 33,351 | 29,777 | 12,825 | 27,475 | 40,129 |
| 4 | 78,400 | 39,629 | 29,679 | 36,338 | 27,364 | 53,030 |
| 5 | 78,400 | 39,564 | 29,671 | 35,963 | 27,353 | 52,999 |

Medians nearly identical across rounds (~29,680 us). No thermal drift. R4/R5 higher mean only because base=800 prompts (vs 300/500) generates more saturation samples. **Rejected.**

### Hypothesis 2: Variable-shape contamination — REJECTED

`tools/test_f4_rescue.py` replicated F4's 3D logic on the v3 trace — grouped samples by (tt, conc, num_new_reqs) and compared `new_reqs=0` (pure decode) against archive.

| Comparison | median Δmean% | mean Δmean% |
|---|---|---|
| v3 2D (all new_reqs) vs archive | −10.28% | −11.34% |
| v3 3D (new_reqs=0) vs archive | −10.28% | −11.34% |

Identical bias whether or not samples are split by num_new_reqs. **Variable-shape contamination is not the cause. F4 does not rescue v3.**

### Hypothesis 3: CUDA-graph warmup asymmetry — IDENTIFIED AS PRIMARY CAUSE

Per-round within-window samples start at ~28,650 us with no capture-cost spike. For a fresh server with cold CUDA graphs, we'd expect the first samples to include graph-capture cost (50–100 ms per new capture, per memory `project_ttft_trace_analysis`). V3 shows no such spike → **graphs were already warm by the time rate-sweep started**.

Reason: `adaptive_profile_full.sh` lines 138–145 run a **CUDA graph warmup sweep** — 15 benchmarks at `--random-input-len 1 --random-output-len 1 --request-rate inf` across batch sizes 1..256. This pre-captures every CUDA graph. The subsequent standard warmup + rate sweep run with fully-warm graphs.

Archive's script (`adaptive_profiling.sh`) has no CUDA warmup sweep. Graphs are captured in-situ during the rate sweep — early samples include capture cost. **Real baseline (`RTX-8000-v31-2000p`) was also captured without pre-warming** — includes capture cost too.

**Result:** v3 samples come from warm-graph steady-state; archive and real baseline include capture latency. V3's oracle predicts ~10% faster than real behaviour → observed regression at every low-mid rate.

## Broader asymmetry (confirmed via script inspection)

The v3 regression is one instance of a general **three-run methodology mismatch** between profile, real baseline, and emulator validation. Inventory:

| Run | Pre-warmup? | Measurement includes cold-graph capture? |
|---|---|---|
| **Profile: archive** (`adaptive_profiling.sh` L49–50) | 500 prompts @ rate=4, 256/128 — partial | YES (rate sweep encounters new batch sizes in-situ) |
| **Profile: v3** (`adaptive_profile_full.sh` L138–158) | 3 warmups: CUDA sweep (all 19 batch sizes at input_len=1) + high-conc burst + rate=4 warmup | NO (every graph pre-captured) |
| **Real baseline** (`RTX-8000-v31-2000p/rN_real.json`, duration 1003.9s for 1000s target — 4s of startup cost visible) | NO explicit pre-warmup | YES (first prompts capture graphs in-situ) |
| **Emu validation** (`validate_feature_ab.sh` L100–103) | 200 prompts @ rate=4, 256/128 (explicit `.warmup_done` marker) | N/A — oracle samples IID from profile; no time-dependence |

**Consequences:**

- Archive matches real baseline's capture-cost structure → emu using archive is valid to compare vs real (modulo the oracle-IID issue below).
- V3 has no capture-cost samples at all → emu predicts artificially fast → ~10% TPOT/TTFT regression even when the profile data is "more" and "cleaner".
- Emu validation's explicit 200-prompt pre-warmup is OK because the oracle has no time-dependence anyway — the warmup just primes the engine/scheduler.
- But: **oracle-IID vs real-sequential is a permanent structural asymmetry.** Real pays cold-graph once at the start; oracle samples are stateless so every call is drawn without regard to "first" vs "later" step. This is the same asymmetry that caused the earlier TTFT −14% residual flagged in `project_ttft_trace_analysis` (CUDA graph compilation 50 ms + IPC 39 ms).

## Fix directions for tomorrow (Apr 19)

1. **Cheapest: profile-methodology parity.** Remove lines 138–145 (CUDA sweep) and 146–152 (high-conc burst) from `adaptive_profile_full.sh`. Keep only the 200-prompt rate=4 warmup. Matches archive's methodology and real baseline's capture-cost structure. Expected: v3 per-bucket means climb ~10% to match archive. No magic numbers, no gap-fitting.

2. **Medium: emu-validation symmetry.** Optionally remove the 200-prompt warmup from `validate_feature_ab.sh` / `validate_wiring_fix.sh` so the measurement window matches real baseline's "no pre-warmup" behaviour. Won't affect per-step oracle predictions but makes the workload envelope comparable. Tiny script edit.

3. **Expensive (structural): time-dependent oracle.** Add a first-occurrence-of-batch-size counter to the oracle and inject a profile-derived capture-cost sample the first time each batch size is queried. Address the oracle-IID asymmetry directly. Large feature — tracked as future work, not for tomorrow.

Tomorrow's session should do #1 + #2 together (both are a few lines of script edit) and re-run the v3 A/B. If v3 withsurr then matches or beats archive, adopt v3 as the permanent baseline and proceed to #3 as a longer-horizon improvement.

## Selection decision

**Profile chosen for feature ablations:** `results/_archive/serving-dense.json`.

**Rationale:** V3's explicit CUDA graph warmup pre-captures graphs, skipping the capture-cost samples that real validation workloads incur. Archive's methodology matches real (graphs captured in-situ). Until the adaptive-profile script is fixed, archive is the only profile whose distribution aligns with the validation workload.

**Tomorrow's top priority:** fix `adaptive_profile_full.sh` (remove CUDA warmup sweep + high-conc burst), reprofile (~4h), rerun the v3 A/B. If v3 withsurr then matches or beats archive, adopt v3 as the permanent baseline.
