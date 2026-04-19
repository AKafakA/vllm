# F4 — Prefill-Composition Third Axis in Profile

**Branch:** `exp/f4-prefill-axis`
**Parent commit:** `a102eed27`
**Gates:** builder CLI `--profile-axes {2d,3d}`; oracle env `VLLM_EMULATOR_PROFILE_AXES=3d`
**Default:** `2d` for both (current behaviour, byte-identical)

## 1. Motivation

The current profile keys buckets by `(total_tokens, concurrency)` with a binary split on `has_prefill`. Two batches with identical `(tt, conc, has_prefill=True)` can still differ materially: one is a single 256-token prefill, the other is 4 × 64-token prefills mixed with decodes. The GPU behaves differently. Current decode/prefill 2D tables average both — oracle prediction widens.

A third axis `num_new_reqs_bucket` (coarse binning of how many of the batch's requests are new / still in prefill) resolves this without adding tunable constants. Trace records already contain `num_new_reqs` (used only for the binary `has_prefill` split today), so **no re-profiling is required** — we rebuild from the existing trace with new bucketing.

## 2. Mechanism

### Profile schema version bump 2.0 → 2.1

Profile packs gain three optional fields:

- `prefill_axis_distribution`: list of `{tt, conc, new_reqs, num_samples, samples}` buckets.
- `decode_axis_distribution`: same shape but for records with `num_new_reqs == 0`.
- `step_cycle_axis_distribution`: combined.

Existing fields `prefill_2d_distribution` / `decode_2d_distribution` / `step_cycle_2d_distribution` are ALWAYS written, unchanged. The 3D fields are written **in addition** when `--profile-axes 3d` is passed. This keeps every new profile backward-compatible.

### Builder change

Add CLI flag `--profile-axes {2d,3d}` (default `2d`). When `3d`:

1. Bucketing adds a third coordinate:
   ```
   new_reqs_bucket = (new_reqs // new_reqs_bucket_width) * new_reqs_bucket_width + new_reqs_bucket_width // 2
   ```
   `new_reqs_bucket_width` defaults to 1 (fine-grained, same as `tt-bucket-width`). Exposed as `--new-reqs-bucket-width` CLI flag for user-chosen resolution (same category as the existing `--tt-bucket-width`, not a tuned constant).
2. Populate the three new `*_axis_distribution` lists alongside the existing 2D lists.
3. Schema version written as `"2.1"`.

### Loader back-compat

`loader.py` + `validator.py` accept both `2.0` (no axis fields) and `2.1` (with axis fields). Missing-field behaviour falls back to the 2D tables — no exception.

### Oracle change

`ProfileGpuCostOracle.__init__` reads `VLLM_EMULATOR_PROFILE_AXES` env (default `2d`). When `3d` AND the loaded profile has axis tables, oracle uses them; otherwise falls back to 2D (logged once).

New method `_sample_3d_distribution(tt, conc, new_reqs, has_prefill)`:

1. Pick 3D table (prefill / decode / combined) same way as 2D pick.
2. Nearest-neighbor in 3D `(tt, conc, new_reqs)`. If range-normalised distance pulls neighbors from outside the query's own new_reqs bucket, fall back to 2D behaviour for that query (no invention of mixed new_reqs samples).

`_sample_2d_distribution` logic unchanged. Oracle chooses which method via `self._profile_axes`.

## 3. Gate + default

- Builder: `--profile-axes 2d` (default) — byte-identical profile output (2D-only fields, schema version `2.0`).
- Builder: `--profile-axes 3d` — adds 2.1 axis fields + version bump to `2.1`. 2D fields identical to 2D-only build (same records go in).
- Oracle: env `VLLM_EMULATOR_PROFILE_AXES` unset OR `2d` — uses `_sample_2d_distribution` path unchanged.
- Oracle: env `VLLM_EMULATOR_PROFILE_AXES=3d` — uses 3D path when profile has axis tables, else falls back to 2D with a once-only warning.

## 4. Files touched

- `/home/wd312/Code/llm/vllm-emulator/vllm_emulator/profile/build_serving_profile_filtered.py`
  - Add `--profile-axes` CLI flag (default `2d`).
  - Add `--new-reqs-bucket-width` CLI flag (default `1`).
  - New helper `build_3d_distribution(data, label)`.
  - `main()`: when `args.profile_axes == "3d"`, also populate axis dicts + call the new helper; write axis fields + version `2.1`.
- `/home/wd312/Code/llm/vllm-emulator/vllm_emulator/oracle/gpu_cost_oracle.py`
  - Read `VLLM_EMULATOR_PROFILE_AXES` env in `__init__`.
  - Load axis tables (list-of-dicts → 3-tuple keyed dict), only if present.
  - New method `_sample_3d_distribution`.
  - `estimate_step_latency_us` passes `num_new_reqs` through if 3D is active; otherwise same as before.
- `/home/wd312/Code/llm/vllm-emulator/vllm_emulator/profile/validator.py`
  - Optional axis fields accepted; version check loosened to accept `2.0` or `2.1`.
- `/home/wd312/Code/llm/vllm-emulator/vllm_emulator/hooks/executor_hook.py`
  - `create_delayed_future` already computes `num_new = len(scheduler_output.scheduled_new_reqs)`; pass it to `estimate_step_latency_us` as a new kwarg `num_new_reqs`. Oracle ignores it in 2D mode.

No vllm-core touches.

## 5. Input/output invariants

- `--profile-axes 2d` → output JSON byte-identical to parent-commit builder output at the same other flags.
- `--profile-axes 3d` + oracle env unset → still byte-identical timer behaviour (oracle uses 2D path).
- `--profile-axes 3d` + oracle env `2d` → identical to above.
- `--profile-axes 3d` + oracle env `3d`, but profile lacks axis fields → 2D fallback + once-only warning.
- Schema version `2.0` profiles always load unchanged.

## 6. A/B protocol

- Baseline profile: chosen per `00_session_context.md` fallback rule. Rebuild it with `--profile-axes 3d` from the existing trace (no re-profiling; ~1 min).
- Harness: `tools/validate_f4_ab.sh`:
  - Pass A (2d): `VLLM_EMULATOR_PROFILE_AXES` unset, `PROFILE=<baseline_3d.json>`, output `results/RTX-8000-f4-2d-apr18/`.
  - Pass B (3d): `VLLM_EMULATOR_PROFILE_AXES=3d`, same profile (back-compat exercise), output `results/RTX-8000-f4-3d-apr18/`.
  - Both: `VLLM_EMULATOR_PREP_SURROGATE=1`, `VLLM_EMULATOR_SAMPLE_TRIM="2,98"`.
  - Rates r=2/8/16; 1000 prompts per rate.

## 7. Success metric

- At **r=4 and r=8**: `ΔTPOT ≤ −2pp` (variant vs baseline).
- At every rate: no metric regresses by > 2pp (F4 is a bigger change so the guard is loosened slightly).

If both conditions hold → **KEEP**.

## 8. Rollback

- Unset `VLLM_EMULATOR_PROFILE_AXES` (or set to `2d`). 3D fields in the profile are ignored — safe to leave in place.
- Zero-code rollback.

## Review agent verdict

**APPROVED WITH CONDITIONS** (agent id `ae2d425501d39b712`).

All 6 binding requirements COMPLY. Three conditions:

1. Implementation must use a **concrete deterministic 3D neighbor rule**: exact `new_reqs_bucket` match required; if absent, fall back to the existing 2D path. No cross-bucket distance weighting on the new axis (the reviewer explicitly prohibited implicit tunable weighting between axes).
2. Once-only warning when `VLLM_EMULATOR_PROFILE_AXES=3d` is set but the loaded profile lacks axis fields — fail-loud, not silent 2D degradation.
3. A/B harness must include a byte-identical check: the generated profile from the baseline trace with `--profile-axes 2d` must `diff`-equal the pre-F4 builder output for the same trace, confirmed BEFORE the rate sweep runs.

## Results

_(populated after A/B run)_

| Rate | 2d TPOT | 3d TPOT | 2d TTFT | 3d TTFT | Verdict |
|---|---|---|---|---|---|
| 2 | tbd | tbd | tbd | tbd | — |
| 8 | tbd | tbd | tbd | tbd | — |
| 16 | tbd | tbd | tbd | tbd | — |

**Overall verdict:** _(KEEP / DROP)_
