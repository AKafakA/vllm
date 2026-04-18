# Apr 18 Progress Log

Append-only. Newest entries at bottom. Every entry starts with `### HH:MM UTC — <topic>` and ends with `Tracker updated: <run ids>` or `Tracker updated: (none)`.

This file exists per-branch during the day; branches may have their own slice. Consolidated onto `exp/combined-apr18` at slot S14.

---

### 13:45 UTC — Session setup (on refactor/clean-emulator-v2)

Plan file `/home/wd312/.claude/plans/it-looks-good-let-unified-steele.md` approved. Entered auto mode.

Paper directory `paper/apr_18/` initialized. v3 profiling still running on `personal_gpu_vm` (Round 4).

Tracker updated: (none)

### 13:55 — 14:10 UTC — F1 slot (branch exp/f1-outlier-filter)

Design → review → implement → smoke test cycle:
- Design doc committed (`7b993a104`).
- Review agent `ab6d9d3057d8bd6da`: APPROVED WITH CONDITIONS.
- Verdict committed (`135b79588`).
- Implementation committed (`f7b1b8983`).
- Byte-identity smoke test ALL PASS (parent vs F1-default/explicit-none IDENTICAL; F1-iqr DIFFER as expected).
- Progress entries committed on F1 branch (`c77b7726f`).

F1 branch is implementation-ready; A/B awaits GPU free + baseline profile lock.

Tracker updated: (none)

### 14:15 UTC — F3 slot begin (branch exp/f3-sample-tokens-delay)

- Branched from `a102eed27`. Note: F1's progress.md does not exist on this branch (it committed on the F1 branch separately).
- Design doc `05_f3_sample_tokens_delay.md` committed (`ad09cacac`).
- Review agent `aca7c84a213c09970`: **APPROVED WITH CONDITIONS** (once-guard on warning, verify builder writes avg_sample_ms). Both addressable in implementation. Verdict committed (`799c8b16f`).
- Builder pre-run check: `build_serving_profile_filtered.py` lines ~200-221 already write `avg_sample_ms` when `--step-timing-csv` is provided. No builder fix needed.

Tracker updated: (none)

### 14:25 UTC — F3 implementation

Applied to `executor_hook.py`:
- New env const `SAMPLE_TOKENS_DELAY_ENV`.
- Module-level `_WARNED_NO_AVG_SAMPLE_MS` once-guard (condition 1).
- `__init__`: `self._sample_tokens_delay_s = 0.0`.
- `_initialize`: reads env + profile pack; sets delay in seconds or warns once if missing; init log line now reports on/off + ms.
- `_handle_async` chain accumulation: `end_time = start_time + latency_s + self._last_surrogate_time_s + self._sample_tokens_delay_s`.

Off-is-noop by inspection: env unset ⇒ `_sample_tokens_delay_s = 0.0` (never reassigned) ⇒ adding 0.0 to end_time is bit-level identical. Live verification deferred to A/B Pass A vs baseline (Pass A with env unset acts as the byte-identity smoke test).

Ready for commit + A/B.

Tracker updated: (none)

### 14:35 — 15:00 UTC — F5 + F4 + F2 slots (all implemented, none ablated)

F5 (exp/f5-knn-conditioning) — Shepard p=2 inverse-distance kNN, env `VLLM_EMULATOR_ORACLE_K`. Review `aced7ab88f684a214` APPROVED WITH CONDITIONS (3 cond). Commits `0aa43da1f`, `23468b162`, `8d53683f9`. Unit tests pass.

F4 (exp/f4-prefill-axis) — Schema v2.0→v2.1, third axis `num_new_reqs_bucket`. Review `ae2d425501d39b712` APPROVED WITH CONDITIONS (3 cond). Commits `431dc8ebd`, `c2a083f33`, `d3115ded6`. Byte-identity + 3D smoke tests pass.

F2 (exp/f2-parallel-surrogate) — First draft REJECTED for magic-window-size + fabricated citation; rewrite with persistence-forecast APPROVED (`a1dd898d9e4042072`). Commits `6a351a449`, `aec22242f`, `3ee754a2c`, `6b86cb9a1`.

Generic A/B harness (`tools/validate_feature_ab.sh`) committed on F2. v3 baseline scaffold (`02_v3_baseline_AB.md`) committed.

Tracker updated: (none — harness-only phase)

### ~14:30–16:00 UTC — v3 A/B + archive-fallback + F1 A/B

V3 profiling completed (318,215 records, 1,989 combined buckets). Auto A/B against v3 ran (nosurr + withsurr vs real baseline). Result: v3 REGRESSED by 5-15 pp TPOT at r=2/4/8 vs archive-wiring-fix-withsurr reference. Fallback rule triggered → **archive** = baseline for ablations.

F1 A/B (off/iqr) against archive: verdict **DROP**. r=2 TPOT regressed 8.7 pp, r=8 regressed 5.1 pp when iqr applied. Committed `944238c32`, `dfb0569b1`, `e9d76ec49`.

F3 A/B launched. Pass A (off) completed (byte-identical to baseline). Pass B (on, delay=0.7536ms/step) started.

### 16:15–16:40 UTC — Root-cause diagnostic on v3

Three hypotheses tested without re-running anything on GPU (pure trace analysis):

1. `compare_archive_v3.py`: v3 means systematically −10.28 % vs archive across 97 common buckets; std ratio 0.79 (v3 TIGHTER). Signature of systematic bias, not variance expansion.
2. `diag_v3_rounds.py`: per-round medians all ~29,680 μs — no thermal drift. Within-round first-N vs last-N divergence is natural rate-sweep progression.
3. `test_f4_rescue.py`: F4's 3D axis filtered to new_reqs=0 gives identical −10.28 % bias. Variable-shape contamination is NOT the cause; F4 does not rescue v3.

Root cause: **CUDA graph warmup asymmetry** in `adaptive_profile_full.sh` (pre-captures all 19 padded batch sizes before rate sweep → v3 profile is 100 % warm-graph). Archive's script has no such pre-warmup → samples include in-situ captures. Real baseline also has no pre-warmup → matches archive. **Archive matches real; v3 diverges.**

User expanded insight: the three runs (profile, real baseline, emu validation) must be methodologically symmetric. Documented in `02_v3_baseline_AB.md`.

Tracker updated: (none)

### 16:45 UTC — Pivot: profiler is TOP blocker, ablations paused

User directive: invariant "more data → better results" must hold. v3 violates it; therefore methodology is wrong, therefore feature ablations on any asymmetric baseline are relative-between-broken-conditions comparisons, not trustworthy accuracy metrics.

Decision:
- Kill F3 A/B mid-Pass-B. F5/F4/F2 ablations deferred.
- Launch warm-real-baseline recapture (`tools/recapture_real_warm.sh`) with v3-matching pre-warmup (CUDA sweep + 500@inf burst + 200@rate=4) so real baseline becomes symmetric with v3 methodology.
- Then auto-launch v3 A/B against warm baselines. If v3+warm matches expectations → methodology fix validated. If v3 still regresses → reprofile v3 tomorrow without CUDA sweep.

Commits: `824c11e80` (diagnostics + expanded 3-run asymmetry doc), `ed5e839b4` (update 02_v3 with asymmetry inventory), `acd110922` (invariant section + recapture script).

Cron `d252b5fc` at :07/:27/:47 monitors recapture; auto-progresses to v3-warm A/B when `.done` sentinel arrives.

### Pending (blocks everything else)

- Warm-baseline recapture completes (~40 min from 16:45).
- Auto-launched v3 A/B against warm baselines (~60 min).
- Analysis + decision: invariant holds → v3 is paper baseline; else → reprofile tomorrow.
- End-of-day handoff once decision is made.

Tracker updated: (none)
