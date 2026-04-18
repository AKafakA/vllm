# F3 — Sample-Tokens Delay in Executor Hook

**Branch:** `exp/f3-sample-tokens-delay`
**Parent commit:** `a102eed27`
**Gate:** env `VLLM_EMULATOR_SAMPLE_TOKENS_DELAY=1`
**Default:** off (hook returns instantly from sample path, current behaviour)

## 1. Motivation

Per `paper/emulator-docs/executor-hook-implementation.md` §5.1 step-timing table:

| Component | Real (ms) | Emu (ms) | Gap |
|---|---|---|---|
| sched_ms | 0.16 | 0.15 | ~0 |
| exec_ms | **2.85** | 1.05 | −1.80 |
| **sample_ms** | **0.73** | **0.00** | **−0.73** |
| wait_ms | 29.69 | 31.84 | +2.15 |
| update_ms | 0.14 | 0.13 | ~0 |

The emulator's `sample_tokens()` returns instantly (`get_sample_future()` returns a pre-resolved Future) while real hardware spends ~0.73 ms per step sampling tokens. This is a **measured** per-step cost that the emulator currently ignores.

The 0.73 ms comes from profiled real-GPU traces captured in `step_timing.csv` (written via `VLLM_DIAG_STEP_TIMING_LOG`). Doc §6.2 lists this as an open improvement path: "Add delay at `get_sample_future()` — this is AFTER the chain timer is created, so the timer delay is unaffected."

## 2. Mechanism

`_handle_async` currently computes `end_time = start_time + latency_s + self._last_surrogate_time_s`. F3 adds the profiled sample time on top:

```python
end_time = start_time + latency_s + self._last_surrogate_time_s \
         + self._sample_tokens_delay_s
```

Where `self._sample_tokens_delay_s` is set once at hook init:

```python
# In _initialize, after oracle loads:
if os.environ.get("VLLM_EMULATOR_SAMPLE_TOKENS_DELAY", "").lower() in ("1", "true", "yes"):
    avg_sample_ms = profile_pack.get("avg_sample_ms")
    if avg_sample_ms is not None:
        self._sample_tokens_delay_s = avg_sample_ms / 1000.0
```

When gate is off OR profile has no `avg_sample_ms` → `self._sample_tokens_delay_s = 0.0` → no change to `end_time`. Byte-identical.

Observable inputs: `avg_sample_ms` from profile pack (written by the builder's existing `--step-timing-csv` flag, which reads `sample_ms` column from a real-GPU-captured CSV). No model_config. No scheduler state. No emu-vs-real comparison.

`avg_sample_ms` is the **mean of real per-step sample times** measured during profiling — it is profile data, not a tuned constant.

## 3. Gate + default

- Env var: `VLLM_EMULATOR_SAMPLE_TOKENS_DELAY=1` (accepts `1|true|yes`, case-insensitive).
- Default: env unset ⇒ `_sample_tokens_delay_s = 0.0` ⇒ no change to end_time.
- Safety: if gate is on but profile has no `avg_sample_ms` field, `_sample_tokens_delay_s` stays 0.0 and a warning is logged once.
- Off-is-noop contract: env unset ⇒ byte-identical hook trace + timer sequencing vs parent commit.

## 4. Files touched

- `/home/wd312/Code/llm/vllm-emulator/vllm_emulator/hooks/executor_hook.py`
  - `__init__`: new field `self._sample_tokens_delay_s = 0.0`.
  - `_initialize`: set the field from profile + env.
  - `_handle_async`: add the field into `end_time`.
  - `ORACLE_*`-style module const: `SAMPLE_TOKENS_DELAY_ENV = "VLLM_EMULATOR_SAMPLE_TOKENS_DELAY"`.

No other source file. No vllm-core touches.

Profile packs will need the `avg_sample_ms` field. The profile builder already supports this via `--step-timing-csv` (pre-existing CLI flag). For the A/B, we rebuild the baseline profile from the existing trace file using `--step-timing-csv=results/RTX-8000-step-overhead/step_timing.csv` (15 700-sample reference real-GPU step timings already on remote per memory `project_session_handoff_apr16_night`).

## 5. Input/output invariants

- Env unset ⇒ `end_time` identical to parent-commit behaviour (bit-level).
- Env set but profile lacks `avg_sample_ms` ⇒ still bit-level identical (guard set to 0).
- Env set + profile has `avg_sample_ms` ⇒ every step's `end_time` grows by `avg_sample_ms / 1000` seconds; nothing else changes.
- No new output fields in emu response or ModelRunnerOutput.
- Profile schema unchanged (F3 only reads an existing-and-optional field).

## 6. A/B protocol

- Baseline profile: the chosen baseline per `00_session_context.md` fallback rule. Pre-A/B step: rebuild it with `--step-timing-csv=results/RTX-8000-step-overhead/step_timing.csv` to include `avg_sample_ms`. Save as `<baseline>_with_timing.json`.
- Harness: `tools/validate_f3_ab.sh` (derivative of `validate_wiring_fix.sh`):
  - Pass A (off): env `VLLM_EMULATOR_SAMPLE_TOKENS_DELAY` unset, `PROFILE=<baseline>_with_timing.json`, output `results/RTX-8000-f3-off-apr18/`.
  - Pass B (on): env `VLLM_EMULATOR_SAMPLE_TOKENS_DELAY=1`, same profile, output `results/RTX-8000-f3-on-apr18/`.
  - Both passes: `VLLM_EMULATOR_PREP_SURROGATE=1`, `VLLM_EMULATOR_SAMPLE_TRIM="2,98"`.
  - Rates r=2/8/16; 1000 prompts per rate.

## 7. Success metric

- At **r=8 and r=16**: `ΔTTFT ≤ −1pp` (variant TTFT closer to zero than baseline).
- At **every rate** (r=2, 8, 16): `ΔTPOT ≥ −1pp` (does not worsen TPOT by more than 1 pp).

If both conditions hold → **KEEP**. Otherwise → **DROP**.

The success criterion is narrower than F1 because F3 explicitly trades TPOT for TTFT; a tight "no worse TPOT" guard is the defence against the TPOT-TTFT conflict discussed in the feature-brainstorm notes.

## 8. Rollback

- Unset `VLLM_EMULATOR_SAMPLE_TOKENS_DELAY`.
- Zero-code rollback. Profile with `avg_sample_ms` is safe to keep — the field is passively ignored.

## Review agent verdict

**APPROVED WITH CONDITIONS** (agent id `aca7c84a213c09970`).

All 6 binding requirements COMPLY. Two conditions:

1. Implementer must add a once-only guard on the "profile lacks `avg_sample_ms`" warning to avoid log spam (design said "once" but did not specify the mechanism).
2. Before running the A/B, verify the profile builder's `--step-timing-csv` path actually writes `avg_sample_ms` into the profile pack JSON.

Pre-run check for condition 2: `vllm_emulator/profile/build_serving_profile_filtered.py` lines ~200-221 contain the CSV read + `profile["avg_sample_ms"] = round(...)` write. Verified — contract holds. No builder fix needed.

## Results

_(populated after A/B run)_

| Rate | Baseline TPOT | Variant TPOT | Baseline TTFT | Variant TTFT | Verdict |
|---|---|---|---|---|---|
| 2 | tbd | tbd | tbd | tbd | — |
| 8 | tbd | tbd | tbd | tbd | — |
| 16 | tbd | tbd | tbd | tbd | — |

**Overall verdict:** _(KEEP / DROP — written after A/B completes)_
