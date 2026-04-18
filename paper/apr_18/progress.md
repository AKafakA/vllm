# Apr 18 Progress Log

Append-only. Newest entries at bottom. Consolidated onto `exp/combined-apr18` at slot S14.

---

### 13:45 UTC — Session setup (refactor/clean-emulator-v2)

Plan approved. `paper/apr_18/` initialized.

Tracker updated: (none)

### 13:55 — 14:10 UTC — F1 slot (exp/f1-outlier-filter)

Design (`7b993a104`) → review agent `ab6d9d3057d8bd6da` APPROVED WITH CONDITIONS → verdict (`135b79588`) → impl (`f7b1b8983`) → progress (`c77b7726f`). Byte-identity smoke tests: parent-vs-F1-default and parent-vs-F1-explicit-none both IDENTICAL; iqr DIFFER as expected.

Tracker updated: (none)

### 14:15 — 14:30 UTC — F3 slot (exp/f3-sample-tokens-delay)

Design (`ad09cacac`) → review agent `aca7c84a213c09970` APPROVED WITH CONDITIONS (once-guard, verify builder) → verdict (`799c8b16f`) → impl+progress (`2427e1bb0`). Builder pre-run check passed. Off-is-noop by inspection; live byte-identity verification deferred to A/B Pass A.

Tracker updated: (none)

### 14:35 — 14:55 UTC — F5 slot (exp/f5-knn-conditioning)

- Design `06_f5_knn_conditioning.md` committed (`0aa43da1f`).
- Review agent `aced7ab88f684a214`: **APPROVED WITH CONDITIONS**. Three conditions: (1) `_sample_knn_2d` uses `self._rng` only, (2) env parser raises on invalid, (3) K=1 path literally unchanged code. Verdict committed (`23468b162`).
- Implementation applied to `gpu_cost_oracle.py`:
  - `__init__` reads `VLLM_EMULATOR_ORACLE_K` env, stores `self._oracle_k`, raises on invalid (condition 2).
  - `_sample_2d_distribution`: K=1 branch is the literal parent-commit code, unchanged (condition 3). K>1 branch calls `_sample_knn_2d`.
  - `_sample_knn_2d` uses only `self._rng` (condition 1). Shepard 1968 IDW with p=2 over range-normalised Euclidean distance, exact-match short-circuit.
- Unit tests (oracle standalone, no torch required): K=1 samples from profile ✓, K=3 samples from profile ✓, invalid K (`0`, `-1`, `abc`) raises ValueError ✓, exact-match short-circuit (K=3 query at bucket `(1,2)` yields 20/20 samples from that bucket) ✓.

Ready to commit. K=1 byte-identity holds by literal-unchanged-code guarantee per condition 3.

Tracker updated: (none)
