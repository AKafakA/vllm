# F5 — kNN Conditioning in Oracle

**Branch:** `exp/f5-knn-conditioning`
**Parent commit:** `a102eed27`
**Gate:** env `VLLM_EMULATOR_ORACLE_K=N` (integer, N ≥ 1)
**Default:** `1` (current nearest-neighbor behaviour, byte-identical)

## 1. Motivation

`_sample_2d_distribution` currently does a two-step nearest-neighbor snap: pick the nearest `tt`, then the nearest `conc` within that `tt`. When the query lands between populated buckets, the oracle draws from a single bucket — which amplifies the random-choice variance problem that triggers the concurrency-feedback loop (see earlier today's `tt=22,conc=22` 4× spread and the resulting −25% TPOT at r=8 on v2).

kNN with inverse-distance weighting is the standard textbook fix: instead of snapping to one bucket, borrow statistical mass from the K nearest buckets weighted by proximity. Sparse buckets get smoothing from surrounding buckets; dense buckets stay dominated by their own samples. No tuned coefficients, only standard-kNN constants (Shepard p=2, Euclidean distance on range-normalised axes).

## 2. Mechanism

Two execution paths, selected by the gate:

**K=1 path (default):** identical to the current code, invoked unchanged. No rng ordering change. Byte-identical behaviour.

**K>1 path:**

1. Compute Euclidean distance from query `(tt, conc)` to every populated bucket in the selected table (decode / prefill / combined), using range-normalised axes:
   ```
   tt_range  = max(tts)  - min(tts)   (or 1 if all equal)
   conc_range = max(concs) - min(concs) (or 1 if all equal)
   d(k) = sqrt( ((k.tt - q.tt)/tt_range)^2 + ((k.conc - q.conc)/conc_range)^2 )
   ```
   Range normalisation is the classical fix when axes have different scales (tt ~ 1-2048 vs conc ~ 1-257). It is the standard kNN-input-normalisation default, not tuned.

2. Sort buckets by distance, take the K nearest.

3. **Exact-match short-circuit** (standard Shepard's method clause): if any of the K has `d = 0`, return a uniform sample from that bucket alone. This preserves exactness when the query hits a populated bucket.

4. Otherwise weight buckets by Shepard's p=2 inverse-distance weighting:
   ```
   w_i = 1 / d_i^2
   ```
   Shepard (1968) is the textbook inverse-distance-weighting default; p=2 is the classical choice for spatial interpolation.

5. Pick a bucket via weighted random choice (weights normalised to sum 1), then pick one sample uniformly from that bucket's samples list.

Two rng calls in the K>1 path (bucket pick + sample pick) vs one in K=1. That's why K=1 must reuse the existing code path unchanged — to preserve identical RNG advancement.

## 3. Gate + default

- Env var: `VLLM_EMULATOR_ORACLE_K=N`, integer ≥ 1. Parsed with `int(os.environ.get(...))`; invalid → raise at init.
- Default (env unset or `=1`): K=1, current nearest-neighbor path, byte-identical output.
- Off-is-noop contract: env unset ⇒ sample sequence identical to parent commit for a given RNG seed.

## 4. Files touched

- `/home/wd312/Code/llm/vllm-emulator/vllm_emulator/oracle/gpu_cost_oracle.py`
  - `__init__`: read env var, store `self._oracle_k`.
  - `_sample_2d_distribution`: K=1 keeps the existing nearest-neighbor code path; K>1 calls a new helper `_sample_knn_2d`.
  - New helper `_sample_knn_2d(table, tt, conc)` implementing steps 1-5 above.

No other source file. No vllm-core touches.

## 5. Input/output invariants

- K=1 (env unset or `VLLM_EMULATOR_ORACLE_K=1`): `_sample_2d_distribution` executes the exact same code as the parent commit. Same RNG state, same sample sequence.
- K>1 and query hits an exact-match bucket: returns a uniform sample from that bucket (same distribution as current, one RNG call extra from the weighted-pick bookkeeping is acceptable).
- K>1 and sparse query: weighted-sample across K nearest buckets. Still pure samples from profile data; no synthesis.
- Oracle interface unchanged (same method signature, same return semantics).

## 6. A/B protocol

- Baseline profile: the chosen baseline per `00_session_context.md` fallback rule.
- Harness: `tools/validate_f5_ab.sh` (derivative of `validate_wiring_fix.sh`):
  - Pass A (K=1): env `VLLM_EMULATOR_ORACLE_K=1`, output `results/RTX-8000-f5-k1-apr18/`.
  - Pass B (K=3): env `VLLM_EMULATOR_ORACLE_K=3`, output `results/RTX-8000-f5-k3-apr18/`.
  - Both passes: `VLLM_EMULATOR_PREP_SURROGATE=1`, `VLLM_EMULATOR_SAMPLE_TRIM="2,98"`.
  - Rates r=2/8/16; 1000 prompts per rate.
- Optional extra passes: K=5 and K=7 if K=3 shows signal and time permits.

## 7. Success metric

- At under-sampled rates (typically **r=16** in current data): `ΔTPOT ≤ −2pp` between K=1 and K=3.
- At every rate (r=2, 8, 16): **no metric regresses by > 1pp**.

If both conditions hold → **KEEP**. If K=3 regresses TPOT at low rates but helps at high rates, try K=5/K=7 — the K parameter itself is the gate, so "best K" is choosable from the A/B without re-code.

## 8. Rollback

- Set `VLLM_EMULATOR_ORACLE_K=1` (or unset).
- Zero-code rollback.

## Review agent verdict

_(populated after Step 2)_

## Results

_(populated after A/B run)_

| Rate | K=1 TPOT | K=3 TPOT | K=1 TTFT | K=3 TTFT | Verdict |
|---|---|---|---|---|---|
| 2 | tbd | tbd | tbd | tbd | — |
| 8 | tbd | tbd | tbd | tbd | — |
| 16 | tbd | tbd | tbd | tbd | — |

**Overall verdict:** _(KEEP / DROP — after A/B)_
