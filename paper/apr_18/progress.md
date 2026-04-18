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

- Design doc `08_f4_prefill_axis.md` committed (`431dc8ebd`).
- Review agent `ae2d425501d39b712`: **APPROVED WITH CONDITIONS**. Three conditions:
  1. 3D neighbor rule must be exact-match only (no cross-axis weighting).
  2. Once-only warning on 3d gate without axis tables.
  3. A/B harness must byte-diff `--profile-axes 2d` output vs parent builder.
- Verdict committed (`c2a083f33`).
- Implementation applied across 3 files:
  - `build_serving_profile_filtered.py`: `--profile-axes {2d,3d}` (default 2d), `--new-reqs-bucket-width` (default 1), `build_3d_distribution` helper, version bumps to `2.1` only when 3d.
  - `gpu_cost_oracle.py`: load optional axis tables; read `VLLM_EMULATOR_PROFILE_AXES` env (default 2d, raises on invalid); new `_sample_3d_distribution` uses exact-match new_reqs filter (condition 1); once-only warning on 3d gate without axis tables (condition 2); `estimate_step_latency_us` signature gains `num_new_reqs` kwarg.
  - `executor_hook.py`: passes `num_new_reqs=num_new` to oracle. 2D mode ignores it.
- Smoke tests (`tools/f4_smoke_test.py`):
  - parent vs F4-default-2d: **IDENTICAL** ✓ (condition 3)
  - parent vs F4-explicit-2d: **IDENTICAL** ✓
  - F4-3d: version `2.1`, axis fields present, 2D sub-fields equal to plain 2d build ✓
- Oracle unit tests (standalone, no torch): 2D default returns 2D sample (30000); 3D mode returns correct new_reqs bucket samples (11111 for new_reqs=0, 22222 for new_reqs=2); 3D miss at new_reqs=99 falls back to 2D (30000); invalid env raises ValueError.

Ready to commit F4.

Tracker updated: (none)
