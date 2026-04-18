# Apr 18 Progress Log

Append-only. Newest entries at bottom. Every entry starts with `### HH:MM UTC — <topic>` and ends with `Tracker updated: <run ids>` or `Tracker updated: (none)`.

---

### 13:45 UTC — Session setup

Plan file `/home/wd312/.claude/plans/it-looks-good-let-unified-steele.md` approved. Entering auto mode for execution.

Paper directory `paper/apr_18/` created with:
- `00_session_context.md` (frozen context, commits, baselines)
- `01_ablation_study_plan.md` (per-feature spec overview)
- `03_experiment_tracker.md` (live run tracker, populated at v3 A/B completion + each feature A/B)
- `progress.md` (this file)

Currently running on `personal_gpu_vm`: adaptive profiling v3 (Round 4, ~13:30 UTC checkpoint showed r=2/14 with 188K trace records). ETA ~14:25 UTC. Auto-A/B via cron `bebd21f5` follows.

Next: wait for S0 to finish, apply fallback rule, then start F1 (simple → hard order).

Tracker updated: (none)

### 13:55 UTC — F1 design + review

- Created `exp/f1-outlier-filter` from `a102eed27`.
- Wrote `04_f1_outlier_filter.md` with all 8 required sections. Committed as `7b993a104 docs(F1): design for outlier filter in profile builder`.
- Invoked general-purpose review agent (id `ab6d9d3057d8bd6da`). Verdict: **APPROVED WITH CONDITIONS**. All 6 binding requirements COMPLY. Three conditions to apply at implementation:
  1. Inline citations for each constant.
  2. Runtime print showing per-bucket reduction.
  3. Off-is-noop byte-identical smoke test recorded here before A/B.
- Updated design doc with verdict and committed as `135b79588 docs(F1): design review — APPROVED WITH CONDITIONS`.

Tracker updated: (none)

### 14:05 UTC — F1 implementation + smoke test

- Implemented `_filter_outliers(samples, method)` in `build_serving_profile_filtered.py` with inline textbook citations per review condition 1 (Tukey 1977, Iglewicz-Hoaglin 1993, MAD consistency Φ⁻¹(0.75)).
- Added `--outlier-filter {none,iqr,mad,winsor}` CLI arg, default `none`.
- Added per-filter runtime print `"outlier_filter=iqr: dropped N of M samples (X.XX%) across K buckets"` per review condition 2.
- Unit tests for each filter method pass (passthrough, IQR drops 1000, MAD drops 1000, winsor clips to [1, 99]).
- Byte-identity smoke test (`tools/f1_byte_identity_test.py`) per review condition 3:
  - parent-commit builder vs F1-default-none: **IDENTICAL** ✓
  - parent-commit builder vs F1-explicit-none: **IDENTICAL** ✓
  - parent-commit builder vs F1-iqr: **DIFFER** (iqr path ran as expected) ✓
- Committed as `f7b1b8983 feat(F1): implement --outlier-filter {none,iqr,mad,winsor}`.

Off-is-noop contract verified. Ready for A/B after v3 profile is available.

Tracker updated: (none)
