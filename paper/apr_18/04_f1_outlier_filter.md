# F1 — Outlier Filter in Profile Builder

**Branch:** `exp/f1-outlier-filter`
**Parent commit:** `a102eed27`
**Gate:** CLI `--outlier-filter {none,iqr,mad,winsor}` on `build_serving_profile_filtered.py`
**Default:** `none` (byte-identical to current output)

## 1. Motivation

The earlier v2 profile (now the polluted-profile cautionary tale) showed that the oracle's `random.choice(samples)` amplifies per-bucket sample variance into TPOT errors via the concurrency feedback loop — see `paper/emulator-docs/executor-hook-implementation.md` §5.1 and today's AM `RTX-8000-v2-nosurr-apr17` result where r=8 TPOT reached −25.2% despite oracle mean-error of only ~3%.

Even in the clean-marker v3 profile, heavy-tailed samples in a bucket can push `random.choice` toward atypical values and trigger the loop. Applying a well-known statistical outlier filter at the **profile-build** stage (not at oracle-sample stage) reduces tail pollution without touching the oracle.

Specifically the `tt=22, conc=22` bucket in the v2 comparison had `min=31811 max=138091` µs — a 4× spread. Trimming the tails tightens the distribution.

## 2. Mechanism

Implemented inside `build_serving_profile_filtered.py` `build_2d_distribution` immediately before the samples list is stored into each bucket.

Filter options (all standard textbook statistical defaults — no parameter tuning):

- **`none`**: pass-through (current behaviour). No filter applied.
- **`iqr`** (Tukey 1.5×): compute `Q1 = percentile(samples, 25)`, `Q3 = percentile(samples, 75)`, `IQR = Q3 − Q1`. Drop any sample `x` with `x < Q1 − 1.5·IQR` or `x > Q3 + 1.5·IQR`. Requires ≥ 4 samples per bucket to apply (otherwise `none`).
- **`mad`** (modified Z-score with MAD, threshold 3.5): compute `median`, `MAD = median(|x − median|)`. Drop sample if `|0.6745·(x − median)/MAD| > 3.5`. Requires ≥ 4 samples per bucket.
- **`winsor`** (Winsorize at 1/99 percentiles): clip `x` to `[percentile(samples, 1), percentile(samples, 99)]`. No drops; tails pulled in. Requires ≥ 100 samples per bucket (otherwise `none`).

All constants (`1.5`, `0.6745`, `3.5`, `1`, `99`, `4`, `100`) are **named textbook defaults**:
- `1.5` is Tukey's original fence constant (*Exploratory Data Analysis*, 1977).
- `0.6745 = Φ⁻¹(0.75)` standard normal MAD consistency constant; threshold `3.5` is the widely-cited Iglewicz-Hoaglin (1993) default for moderate outlier detection.
- `1`/`99` are the classical Winsorize percentiles when nothing else is specified.
- `4` / `100` are minimum-sample guards for the filter to be meaningful.

Pseudocode:

```python
def filter_outliers(samples, method):
    if method == "none" or len(samples) < 4:
        return samples
    if method == "iqr":
        q1, q3 = np.percentile(samples, [25, 75])
        iqr = q3 - q1
        return [x for x in samples if q1 - 1.5*iqr <= x <= q3 + 1.5*iqr]
    if method == "mad":
        median = np.median(samples)
        mad = np.median(np.abs(samples - median))
        if mad == 0:
            return samples
        return [x for x in samples if abs(0.6745*(x - median)/mad) <= 3.5]
    if method == "winsor":
        if len(samples) < 100:
            return samples
        lo, hi = np.percentile(samples, [1, 99])
        return [min(max(x, lo), hi) for x in samples]
```

Observable inputs: only `samples` (profile data). No model_config, no scheduler state, no gap measurements.

## 3. Gate + default

- CLI flag: `--outlier-filter {none,iqr,mad,winsor}` on `build_serving_profile_filtered.py`.
- Argparse default: `none`.
- Off-is-noop contract: `--outlier-filter none` (or flag omitted) → byte-identical profile output to current code. Smoke-tested by `diff` against the v3 profile.

## 4. Files touched

- `/home/wd312/Code/llm/vllm-emulator/vllm_emulator/profile/build_serving_profile_filtered.py` — add CLI arg, filter function, call in `build_2d_distribution`.

No other files. No vllm-core touches.

## 5. Input/output invariants

- `--outlier-filter none` produces **byte-identical output JSON** to pre-F1 code, verified by `diff`.
- Profile schema unchanged (same top-level keys, same bucket field shape).
- `validator.py` accepts all filter outputs without change.
- Every filter monotonically **reduces or equals** the sample count per bucket (winsor = equals with clipped values). Never introduces fake samples.

## 6. A/B protocol

- Baseline profile: the chosen baseline per `00_session_context.md` fallback rule (v3 or archive).
- Variant build: rebuild that profile with `--outlier-filter iqr`. Save as `results/RTX-8000-f1-iqr-profile.json`.
- Harness: `tools/validate_f1_ab.sh` (derivative of `tools/validate_wiring_fix.sh`):
  - Pass A (none): `PROFILE=<baseline>`, output `results/RTX-8000-f1-off-apr18/`.
  - Pass B (iqr): `PROFILE=results/RTX-8000-f1-iqr-profile.json`, output `results/RTX-8000-f1-iqr-apr18/`.
  - Both passes use `VLLM_EMULATOR_PREP_SURROGATE=1` (best-known config from wiring-fix validation).
  - Rates r=2/8/16; 1000 prompts per rate; `VLLM_EMULATOR_SAMPLE_TRIM="2,98"` held constant.
- If `iqr` is promising (meets success criterion), time permits additional `mad` and `winsor` passes.

## 7. Success metric

- At **r=8**: `ΔTPOT ≤ −2pp` (i.e. variant TPOT is ≥ 2pp closer to zero than baseline TPOT).
- At **every rate** (r=2, 8, 16): **no metric regresses by > 1pp**.

If both conditions hold → **KEEP** (eligible for combined branch).
Otherwise → **DROP** (revert CLI flag on main, feature dead for today).

## 8. Rollback

- Rebuild profile with `--outlier-filter none` (or simply omit the flag).
- The CLI flag being default-`none` means rollback is zero-code — merging the branch is safe even if we later decide not to use `iqr` at runtime.

## Review agent verdict

**APPROVED WITH CONDITIONS** (agent id `ab6d9d3057d8bd6da`).

Conditions to apply in implementation:

1. Inline-cite each constant in the implementation comments (`# Tukey 1977`, `# Iglewicz-Hoaglin 1993`, `# MAD consistency constant Φ⁻¹(0.75)`) so future reviewers can audit without re-reading the design doc.
2. Add a runtime print per filter run showing per-bucket sample-count reduction (e.g. `iqr: dropped N of M samples across K buckets`) so the A/B operator can verify the filter actually fired.
3. The off-is-noop byte-identical `diff` smoke test mentioned in §3 and §5 must be executed and recorded in `progress.md` before the A/B run begins.

All six binding requirements checked COMPLIES. No design changes needed.

## Results

_(populated after A/B run)_

| Rate | Baseline TPOT | Variant TPOT | Baseline TTFT | Variant TTFT | Verdict |
|---|---|---|---|---|---|
| 2 | tbd | tbd | tbd | tbd | — |
| 8 | tbd | tbd | tbd | tbd | — |
| 16 | tbd | tbd | tbd | tbd | — |

**Overall verdict:** _(KEEP / DROP — written after A/B completes)_
