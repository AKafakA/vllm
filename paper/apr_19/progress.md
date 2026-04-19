# Apr 19 Overnight Progress Tracker

Append-only timeline. Every cron check adds a timestamped entry.

## Session metadata

- Chain launched: 01:33 BST Apr 19, PID 2226284 on personal_gpu_vm
- Budget: 8h 45min (01:15 → 10:00 BST sync)
- Phases: A reprofile (4h) → B validate (45min) → C ablate (3h) → D summarize (30min)
- Plan: `.claude/plans/it-looks-good-let-unified-steele.md`

## Timeline

### 01:33 BST — Phase A start
- Killed prior overnight chain (F3 on archive, F5 in-progress). Port 8100 freed.
- Deployed 4 new scripts to remote; launched `overnight_chain_v6.sh` via nohup.
- Phase A: 5-round single-session reprofile (new methodology: server stays alive across all rounds).

### 02:49 BST — Phase A round 2/5, r=0.5 of 14

- Current: Phase A, round 2/5 near end of rate sweep (r=0.5, n=166).
- Records: 88,004 (from 23,602 at 01:49 check).
- Results so far: none yet — trace still accumulating.
- Issues: none; chain process tree healthy (overnight_chain_v6 PID 2226284 + adaptive_profile_v6 child PID 2226289).
- Pace: round 1 took ~37 min; projecting 5 rounds ≈ 3h 5min → Phase A done ~04:38 BST; Phase B ~05:23; Phase C ~08:23; Phase D ~08:53. On track for 09:30 deliverable.

### 03:19 BST — Phase A round 3/5, r=10 of 14

- Current: Phase A, round 3/5 (round 3 started 02:38:49, now at r=10/14 after 40 min in round 3).
- Records: 136,606 (+48,602 in last 30 min).
- Results so far: none yet.
- Issues: none; procs healthy.
- Pace refined: round 1+2 took 62 min combined (~31 min avg per round). Projecting 5 × 31 min = 155 min → Phase A done ~04:11 BST (faster than prior estimate). Phase B ~04:56; Phase C ~07:56; Phase D ~08:26. **60 min buffer before 09:30 deliverable.**

### 03:49 BST — Phase A round 4/5, r=2 of 14

- Current: Phase A, round 4/5 (round 4 started 03:16:16 at r=1 n=800; now r=2 at 03:29:53).
- Records: 184,208 (+47,602 in last 30 min). Round 3 end total was 153,607.
- Results so far: none yet.
- Issues: none; procs healthy.
- Pace revised upward for rounds 4–5: `get_num_prompts` uses base=800 for round>3 (vs 500 for rounds 2–3, 300 for round 1), so each round 4–5 should take ~54 min (r=1 alone: 13m37s actual, matching 800/rate prediction). Round 4 end ~04:10 BST; round 5 end ~05:04; Phase A build ~05:05. Phase B ~05:50, Phase C ~08:50, Phase D ~09:20. **~10 min buffer before 09:30 deliverable** — acceptable but tighter than earlier estimate.

### 04:19 BST — Phase A round 4/5, r=0.5 of 14

- Current: Phase A, round 4/5 at r=0.5 n=266 (13/14 rates); r=0.5 started 04:01:08.
- Records: 215,608 (+31,400 in last 30 min; slower growth at higher rates).
- Results so far: none yet.
- Issues: none; procs healthy.
- Expected timing: round 4 ends ~04:16 BST (r=0.5 ~9 min + r=inf + 3 variable shapes). Round 5 (n=800 again) ends ~05:10 BST. Phase A build ~05:15. Phase B ~06:00. Phase C ~09:00. Phase D ~09:30. **On the deadline**, zero buffer — watch for slip next check.

### 04:49 BST — Phase A round 5/5 (final), r=2 of 14

- Current: Phase A, round 5/5 at r=2 n=800; round 5 started 04:14:26.
- Records: 265,810 (+50,202 in last 30 min). Round 4 closed at 232,009.
- Results so far: none yet.
- Issues: none; procs healthy. Round 5 r=1 took 13:37 (matching round 4 r=1 — pace consistent, no slippage).
- Expected timing: round 5 ends ~05:08 BST → Phase A build ~05:12 → Phase B ~05:57 → Phase C ~08:57 → Phase D ~09:27. **3 min buffer before 09:30 deadline.**

### 05:19 BST — Phase A round 5/5, r=0.5 of 14 (~20 min SLIP)

- Current: Phase A, round 5/5 at r=0.5 n=266; r=0.5 started 04:59:21, still running at 05:19 (20 min in).
- Records: 297,010 (+31,200 last 30 min).
- Results so far: none yet.
- Issues: **r=0.5 benchmark running ~2× expected** (predicted ~9 min for n=266 @ rate=0.5, actual 20+ min). Possibly GPU thermal slowdown after 3h 45min sustained load, or scheduler accumulating state after round 5's high-rate sweep.
- Revised timing: round 5 end ~05:30–05:35 → Phase A build ~05:36 → Phase B ~06:21 → Phase C ~09:21 → Phase D ~09:51. **Past 09:30 target by ~20 min but before 10:00 user sync.** Acceptable. No intervention yet; watch next check.

### 09:37 BST — Agg-mode A/B launched (oracle median/mean vs sample)

- Purpose: isolate whether ORACLE VARIANCE (from `random.choice(samples)`) or SAMPLE DENSITY drives accuracy. If median-only oracle approaches archive-sample accuracy on v6, the variance was hurting more than helping — deterministic oracle suffices.
- New env var (experiment gate only, not a production feature): `VLLM_EMULATOR_ORACLE_AGG={sample|median|mean}`. Default `sample` = byte-identical to current `random.choice`. Oracle code changes: 1 helper `_aggregate(samples)`, 4 `self._rng.choice(...)` call sites replaced.
- 6 passes: v6 profile × {sample, median, mean} + archive profile × {sample, median, mean}. Each pass: r=2, r=8 × 500 prompts. ~6 min/pass × 6 = ~36 min total. ETA done ~10:13 BST.
- Ordering (v6 first): v6_sample → v6_median → v6_mean → archive_sample → archive_median → archive_mean. Monitoring cron `61aaf900` polls at :01/:16/:31/:46.

### 08:19 BST — Phase C v6: F2 DONE, F4 on-pass running (LAST FEATURE)

- Current: F4 on-variant (started 07:58, r=2). Off-variant done at ~07:58 (shown earlier, TPOT −5.9% r=2, −35.3% r=8 — matches v6 baseline profile).
- Pace: 6 features done in 1h 48min. F4 should finish ~08:05 (~17 min per feature). Phase D ~08:20 BST. Under 10:00 deadline by ~1h 40min.
- **F2 parallel surrogate verdict: DROP / NEUTRAL**

| Rate | off TPOT | on TPOT | ΔTPOT | off TTFT | on TTFT | ΔTTFT |
|---|---|---|---|---|---|---|
| 2 | −5.9% | −6.0% | −0.1pp | −29.5% | −29.2% | **+0.3pp** |
| 8 | −35.1% | −36.5% | −1.4pp | −36.5% | −36.9% | −0.4pp |

Essentially neutral. Persistence forecast's predicted surrogate time slightly overestimates first-step warmup (small r=8 TPOT regression).

- **Cumulative verdict table**:

| Feature | Verdict | ΔTPOT r=2 | ΔTPOT r=8 | Notes |
|---|---|---|---|---|
| F1/IQR | DROP | −3.4pp | −5.2pp | Removes heavy tail |
| F1/MAD | DROP | −4.0pp | −7.6pp | Same as IQR |
| F1/Winsor | NEUTRAL | −0.1pp | −0.1pp | Too mild to change |
| F3 | NEUTRAL | −0.2pp | +0.4pp | avg_sample_ms small relative to step |
| F5 | DROP | −0.5pp | −1.8pp | kNN smoothing hurts |
| F2 | NEUTRAL | −0.1pp | −1.4pp | Persistence prediction overestimates first step |
| F4 | ? | TBD | TBD | Running |

- Issues: none; chain healthy.

### 07:49 BST — Phase C v6: F3+F5 DONE, F2 running, F4 queued

- Current: F2 parallel surrogate A/B (started 07:32:11). F4 queued.
- Pace: 5 features done in 1h 22min (06:10 → 07:32). Remaining: F2 + F4 × ~16 min = 32 min → all done ~08:05 BST. Phase D ~08:20 BST.
- **F3 sample_tokens_delay verdict: DROP / NEUTRAL**

| Rate | off TPOT | on TPOT | ΔTPOT | off TTFT | on TTFT | ΔTTFT |
|---|---|---|---|---|---|---|
| 2 | −5.8% | −6.0% | −0.2pp | −29.0% | −29.3% | −0.3pp |
| 8 | −36.4% | −36.0% | **+0.4pp** | −36.5% | −36.5% | 0pp |
| 16 (bonus) | −9.9% | −12.3% | −2.4pp | −71.5% | −75.5% | −4.0pp |

Effectively neutral at r=2,8 (within noise). r=16 regresses but that was a bonus rate.

- **F5 kNN K=3 verdict: DROP**

| Rate | off TPOT | on TPOT | ΔTPOT | off TTFT | on TTFT | ΔTTFT |
|---|---|---|---|---|---|---|
| 2 | −6.2% | −6.7% | −0.5pp | −29.7% | −30.4% | −0.7pp |
| 8 | −35.7% | −37.5% | −1.8pp | −36.8% | −38.4% | −1.6pp |

Small but consistent regression. K=3 inverse-distance weighting smears the distribution; v6 baseline isn't sparse enough to benefit from kNN smoothing.

- **Running summary so far: F1/F3/F5 all DROP. F2 and F4 pending.** No positive feature verdict yet.
- Issues: F5 on-variant at r=16 missing (possible timeout/crash at high concurrency under K=3 compute overhead). Not blocking; r=2,8 deltas are the verdict rates.

### 07:19 BST — Phase C v6: F1 sweep DONE, F3 running

- Current: Phase C v6, F3 A/B (started 06:59:25, r=2 running since 07:01).
- Pace: each feature ~16.5 min. Remaining: F3, F5, F2, F4 (4 × 16 min = 64 min) → all done ~08:03 BST. Phase D ~08:18.
- **F1 sweep complete — all 3 variants DROP on v6 baseline:**

| F1 variant | off r=2 | on r=2 | ΔTPOT r=2 | off r=8 | on r=8 | ΔTPOT r=8 | verdict |
|---|---|---|---|---|---|---|---|
| IQR | −6.4% | −9.8% | **−3.4pp** | −37.8% | −43.0% | **−5.2pp** | DROP (regress) |
| MAD | −5.8% | −9.8% | **−4.0pp** | −35.6% | −43.2% | **−7.6pp** | DROP (regress) |
| Winsor | −6.0% | −6.1% | −0.1pp | −36.2% | −36.3% | −0.1pp | DROP (neutral) |

- Interpretation: IQR and MAD remove legitimate heavy-tail variance → emu under-predicts latency → worse accuracy. Winsor's 1/99 clipping is too mild to affect sampling. **Feature F1 as a whole: DROP**. This confirms Apr 18's IQR-only DROP verdict AND extends it to MAD/Winsor — no outlier filter helps here.
- Issues: none; procs healthy.

### 06:49 BST — Phase C v6: F1 IQR DONE, F1 MAD running

- Current: Phase C v6, F1 MAD A/B (started 06:26:36). Bench serve r=2 running since 06:28.
- Per-feature pace: F1 IQR took 16 min (06:10 → 06:26). 6 features remaining at ~16 min each = ~96 min → all done ~08:02 BST.
- Results so far — **F1 IQR** (rsynced locally):
  - off (unfiltered v6): r=2 TPOT −6.4% · r=8 TPOT −37.8%
  - on (IQR filter):     r=2 TPOT −9.8% · r=8 TPOT −43.0%
  - **Verdict: DROP** (IQR regresses TPOT by 3.4pp at r=2 and 5.2pp at r=8 — confirms Apr 18's DROP).
- **⚠ BASELINE INSTABILITY FLAG**: v6 off-variant at r=8 shows TPOT −37.8% in Phase C (500 prompts), but Phase B's 2000-prompt v6 measurement showed TPOT −19.0%. Gap of ~19pp is NOT feature-related — this is baseline drift between 500-prompt and 2000-prompt runs. Possible causes: GPU thermal state, RNG variance, or prompt-count sensitivity. **Per-feature deltas (on vs off at same prompt count) remain internally consistent**; absolute comparisons to archive reference are less reliable at 500 prompts.
- Issues: baseline instability at 500 prompts flagged but not blocking.

### 06:19 BST — Phase B DONE (FAIL), Phase C intervention

- Phase B v6 emu validation finished 05:51 BST. Results (archive reference in parens):
  - r=2 TPOT −5.7% (−0.6%) · TTFT −34.5% (−32.5%) · FAIL
  - r=4 TPOT −12.1% (−1.6%) · TTFT −36.3% (−30.7%) · FAIL
  - r=8 TPOT −19.0% (−6.0%) · TTFT −34.4% (−28.5%) · FAIL
  - r=16 TPOT −9.7% (−5.6%) · TTFT −33.5% (−25.0%) · FAIL
  - r=32 TPOT +0.3% (+4.9%) · TTFT −2.9% (+1.9%) · PASS
- **Invariant: FAIL**. v6 replicates v3's accuracy exactly — server restart was NOT the cause; round count is. Single-session didn't help.
- **Chain Phase C had 3 bugs**:
  1. F1 (`--outlier-filter`) CLI args not on deployed branch — builds failed → skipped.
  2. F4 (`--profile-axes`) CLI args not on deployed branch — builds failed → skipped.
  3. F5 env var `VLLM_EMULATOR_ORACLE_K` and F2 env var `VLLM_EMULATOR_PARALLEL_SURROGATE` not on deployed branch — would silent no-op. (Remote was on exp/f3-sample-tokens-delay; other features on separate branches.)
- **Intervention (06:19–06:40 BST)**:
  - Killed chain + all children.
  - Created local branch `exp/combined-apr19` by merging f1/f4/f5/f2 into f3. Resolved 4 conflicts (builder CLI + oracle methods + hook init/env/chain-timer).
  - Syntax-checked all 3 modified files, confirmed CLI args present locally + remote.
  - Rsynced merged vllm_emulator/ to remote.
  - Rebuilt F1 iqr/mad/winsor + F4 3D profiles from v6 trace on remote (all succeeded; 280k/281k/309k/309k samples).
  - Wrote tools/overnight_phase_c_v6.sh: r=2,8 at 500 prompts for all 7 A/Bs. Deployed.
- **Phase C v6 relaunched at 06:10 BST** as PID 2353513. ETA ~08:10 BST — under 10:00 sync deadline.

- Current: **Phase B: v6 emu validation** (r=2 done 05:32, r=4 running since 05:32:04). r=8/16/32 remaining.
- Phase A final: 309,200 samples · 2044 step_cycle cells · 258 decode cells · 1786 prefill cells. Finished 05:12:47 BST.
- Results: `/tmp/vllm_v6_profile.done` marker exists; `serving-full.json` rsynced down (10.4MB).
- **v6 vs archive shape comparison** (aggregate across 57 common decode buckets ≥100 samples each):
  - **p50 Δ% median -0.23%, mean -0.43%** — match.
  - **p90 Δ% median -6.54%, mean -22.93%** — **AS BAD AS V4 (-24%)**.
- **UNEXPECTED FINDING**: single-session methodology did NOT close v4's p90 gap. v6 and v4 both have ~-23% p90 deficit; v5 1-round had -3% p90 deficit. So the variable is **round count**, not server restart.
  - Revised root-cause hypothesis: 5-round profiling (regardless of session continuity) produces enough samples in thin buckets that the bucket average becomes steady-state-dominated, truncating the heavy tail archive has from smaller sample counts.
  - If true, the "more data → better" invariant may not hold on archive-as-reference — the reference itself benefits from low sample count.
- Issues: shape gap replicates v4. Phase B invariant gate will tell us if emu accuracy follows shape (likely FAIL) or compensates via sample variance (possible PASS).
- ETA: Phase B ends ~06:22 · Phase C ~09:22 · Phase D ~09:52.
