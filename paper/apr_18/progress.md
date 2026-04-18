# Apr 18 Progress Log

Append-only. Consolidated onto `exp/combined-apr18` at slot S14.

---

### 13:45 UTC — Session setup

Plan approved. `paper/apr_18/` initialized. v3 profiling still running.

Tracker updated: (none)

### F1 slot (exp/f1-outlier-filter)

Design → review (`ab6d9d3057d8bd6da` APPROVED WITH CONDITIONS) → impl → smoke test. Commits: `7b993a104`, `135b79588`, `f7b1b8983`, `c77b7726f`. Byte-identity smoke: parent vs F1-default IDENTICAL; F1-iqr DIFFER as expected.

### F3 slot (exp/f3-sample-tokens-delay)

Design → review (`aca7c84a213c09970` APPROVED WITH CONDITIONS) → impl. Commits: `ad09cacac`, `799c8b16f`, `2427e1bb0`. Off-is-noop by inspection.

### F5 slot (exp/f5-knn-conditioning)

Design → review (`aced7ab88f684a214` APPROVED WITH CONDITIONS) → impl. Commits: `0aa43da1f`, `23468b162`, `8d53683f9`. Unit tests: K=1, K=3, exact-match short-circuit, invalid K handling all pass.

### F4 slot (exp/f4-prefill-axis)

Design → review (`ae2d425501d39b712` APPROVED WITH CONDITIONS) → impl. Commits: `431dc8ebd`, `c2a083f33`, `d3115ded6`. Byte-identity smoke: parent vs F4-default-2d IDENTICAL; 3D schema `2.1` with axis fields. Oracle unit tests pass: 2D default, 3D new_reqs=0 and =2, 3D miss → 2D fallback, invalid env raises.

### F2 slot (exp/f2-parallel-surrogate)

- First design doc committed (`6a351a449`).
- Review agent `ac4de8a6815d2772c`: **REJECTED**. Three issues:
  - Window size `32` with fabricated Tibshirani citation — magic number.
  - `__init__` unconditionally set `self._surrogate_time_window` on off-path (not bit-identical).
  - Initial prediction = 0 framed as seed instead of additive identity.
- Rewrite committed (`aec22242f`): persistence forecasting (prediction = previous step's measured time, cited Hyndman & Athanasopoulos ch.5), env-gated state, initial = additive identity. No running statistic, no window, no new constants.
- Review agent `a1dd898d9e4042072`: **APPROVED** (not "with conditions"). Verdict committed (`3ee754a2c`).
- Implementation applied to `executor_hook.py`:
  - Module-level `_PARALLEL_SURROGATE_POOL` + lazy `_get_parallel_pool()`.
  - `PARALLEL_SURROGATE_ENV` constant.
  - `__init__`: `self._parallel_surrogate_enabled = False` default.
  - `_initialize`: env-gated set `self._parallel_surrogate_enabled = True` + lazy pool init.
  - `create_delayed_future`: when parallel mode (and non_block), submit surrogate to pool, keep `self._last_surrogate_time_s` as persistence prediction. Synchronous path unchanged otherwise.
  - `_handle_async`: timer callback joins surrogate Future and updates `self._last_surrogate_time_s` before resolving sample Future. Ordering invariant preserved on all paths (realtime, accelerated, delay-below-timer-granularity).

Off-is-noop by inspection: env unset → `_parallel_surrogate_enabled = False` → `use_parallel = False` → executes the prior synchronous code unchanged.

Ready to commit.

Tracker updated: (none)
