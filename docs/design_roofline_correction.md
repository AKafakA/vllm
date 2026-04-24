# Roofline correction in the oracle — design

## Problem

At saturation on workloads with long prompts, TPOT grows with batch KV
size (memory-bandwidth bound). The current (tt, conc) profile surface
captures this only within its sampled domain. Validation queries that
exceed the profile's captured `sum_kv` range land in under-predicted
territory (~25-38% TPOT gap on A10 + full ShareGPT at r≥8).

Regressing `step_us ~ sum_kv` on the existing sat-band profile gives
R² ≈ 0.15 globally and near 0 per-conc — the profile's captured
`sum_kv` range (up to ~27k at conc=64) is too narrow for a confident
slope fit, even though the physics says the slope exists.

## Approach

Add a **roofline correction term** to the oracle, derived from a
dedicated bandwidth calibration captured once per (hardware, model)
pair. Not a fallback path — a correction applied on top of the
empirical bucket lookup.

Final formula:

```
step_us(tt, conc, sum_kv) =
    empirical_lookup(tt, conc)                         # existing oracle
  + bw_slope_us_per_token * (sum_kv - bw_reference)    # roofline correction
```

Where:

- `empirical_lookup` is the current 2D/3D sample-aggregation path.
- `bw_slope_us_per_token` is fit from a dedicated BW calibration run
  (single long-sequence decode sweep on the target hardware).
- `bw_reference` is the mean `sum_kv` observed in the empirical profile
  samples, so within-domain queries get a ≈zero correction and the
  correction only bites for out-of-domain `sum_kv` extrapolation.
- Applied only to **decode steps** (not prefill). Prefill compute
  scales with tt differently (quadratic attention), not modeled here.

## Calibration workflow

1. New tool `tools/profile_bw_calibration.py`:
   - Starts vLLM server with target model and args.
   - Submits one long request (prompt length ≈ max_model_len - 500,
     decode length 500).
   - Captures step_cycle_trace during the decode phase.
   - Filters to decode-only steps (`num_decode_seqs == 1`,
     `num_new_reqs == 0`).
   - Regresses `step_cycle_us ~ sum_kv` across the decode trajectory
     (sum_kv grows by 1 per step as the sequence extends).
   - Writes `bw_calibration.json`:
     ```
     {
       "gpu_name": "NVIDIA A10",
       "model_name": "Qwen/Qwen3-8B",
       "kv_per_token_bytes": 147456,
       "bw_slope_us_per_token": 0.30,
       "bw_intercept_us": 44200.0,
       "bw_r_squared": 0.97,
       "n_samples": 480,
       "sum_kv_range": [3800, 4280]
     }
     ```
2. Profile-pack build step (`build_serving_profile_filtered.py`):
   - If a `bw_calibration.json` sibling exists, merge its
     `bw_slope_us_per_token` into the output profile pack.
   - Also compute `bw_reference_sum_kv` from the profile's own decode
     samples (mean `sum_kv` across all decode buckets).
3. Oracle (`ProfileGpuCostOracle.__init__`):
   - Read `bw_slope_us_per_token` and `bw_reference_sum_kv` from the
     profile pack. Default 0.0 keeps behaviour byte-identical.
4. `estimate_step_latency_us`:
   - Existing code path unchanged.
   - After empirical lookup, if `bw_slope > 0` and step is not prefill,
     add `bw_slope × (sum_kv - bw_reference)`.

## Why this is "correction on top of empirical" not "analytical fallback"

- In-domain queries (sum_kv near bw_reference): correction ≈ 0.
  Empirical lookup dominates. No change from current behaviour.
- Out-of-domain queries (sum_kv >> bw_reference, e.g. deep saturation
  on long-prompt workloads): correction grows linearly. Physics-bounded.

The slope is NOT a guess — it is measured on the target GPU via a
calibration that takes ~1 minute of server time. BW varies by
hardware (A10 ~400 GB/s sustained, RTX 8000 ~600 GB/s sustained), so
the calibration is mandatory per hardware target.

## Knobs introduced

None new. `bw_slope_us_per_token` and `bw_reference_sum_kv` are read
from the profile pack, not from env vars or CLI flags. When absent
(e.g., existing profiles built before this branch), oracle uses 0.0
and behaves exactly as before.

## Test plan (A10, tonight)

1. Capture BW calibration on A10 with Qwen3-8B.
2. Merge slope into the existing sat-band profile pack.
3. Re-run validate (Test A: sample aggregation + roofline correction)
   against `a10-qwen38b-full-5rate-real` baseline.
4. Compare against Test A from earlier this session (sample agg, no
   roofline correction):
   - Expected: TPOT gap at r=8/16/32 closes from -25..-38% toward <10%.
   - Unexpected: TPOT stays flat → hypothesis refuted, design doesn't ship.
5. If pass: document deltas in `paper/apr_22/progress.md`, consider
   merging to stable.
