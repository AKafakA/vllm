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
