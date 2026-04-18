# F2 — Parallel Surrogate

**Branch:** `exp/f2-parallel-surrogate`
**Parent commit:** `a102eed27`
**Gate:** env `VLLM_EMULATOR_PARALLEL_SURROGATE=1`
**Default:** off (synchronous surrogate; current F-plus-chain behaviour)

## 1. Motivation

The current chain-timer surrogate wiring (commit `409fd8dc3`) runs the surrogate **synchronously** on the engine thread before the timer callback fires. This serialises the CPU prep with the virtual GPU time — the behaviour the doc §2.2 intends. But real hardware **pipelines** CPU prep with the previous step's GPU work via the batch queue double-buffer, and today's brainstorm identified this as the root of the TPOT-TTFT coupling: every ms of TTFT help comes out of TPOT.

F2 breaks the coupling by running the surrogate in a separate thread that **overlaps** with the timer sleep, and feeds the chain accumulation a *persistence-forecast* of surrogate time so the chain state is decided synchronously.

## 2. Mechanism

A module-level `ThreadPoolExecutor(max_workers=1)` is created lazily on the first `__init__` where the env gate is set. `create_delayed_future` with gate on:

1. Oracle samples `latency_us` (unchanged).
2. Submit `run_prep_surrogate(scheduler_output)` to the executor → returns `surr_fut` immediately.
3. Use `self._last_surrogate_time_s` as the **persistence-forecast prediction** of THIS step's surrogate wall-clock. `self._last_surrogate_time_s` already exists (set in commit `409fd8dc3` for the chain-timer path) — we reuse it rather than introducing a new statistic or window.
4. Chain accumulate: `end_time = start_time + latency_s + self._last_surrogate_time_s`. This is **identical to the synchronous chain formula** from commit `409fd8dc3`; the only difference is WHEN `self._last_surrogate_time_s` is set (see step 6).
5. Timer scheduled for `delay = end_time - now`.
6. Timer callback (on a different thread):
   a. Calls `surr_fut.result()` to ensure the surrogate has actually completed.
   b. Measures `actual_surrogate_time` (wall-clock measured during the executor's run).
   c. Updates `self._last_surrogate_time_s = actual_surrogate_time` so the NEXT step uses THIS step's measurement as its prediction.
   d. Resolves `sample_future.set_result(fake_output)`.

### Why persistence forecasting

Persistence forecasting (`prediction_t = observation_{t-1}`) is the textbook baseline for one-step-ahead prediction when no richer model is justified. It is the default naïve forecast in every time-series textbook (e.g. Hyndman & Athanasopoulos, *Forecasting: Principles and Practice*, chapter 5). It uses no tuning parameter, no window size, no statistic — just last-value.

### Initial-step handling

On the very first `create_delayed_future` call, `self._last_surrogate_time_s == 0.0` (set in `__init__`). The chain formula `end_time = start_time + latency_s + 0.0` is **the additive identity** — identical to the surrogate-off baseline, not a tuned seed. After the first measurement flows back through the timer callback, subsequent steps use last-measurement as prediction.

### Ordering invariant

The sample Future is never resolved before the surrogate Future: `timer_callback: surr_fut.result(); sample_future.set_result(...)`. Even if the predicted time was too optimistic and the timer fired before the surrogate finished, `surr_fut.result()` blocks until it completes.

Off-is-noop: env unset ⇒ `self._parallel_surrogate_enabled = False` ⇒ current synchronous path runs **unchanged**, including `self._last_surrogate_time_s` being set synchronously to the measured time (as it is today). No new per-step attribute is touched on the off-path.

## 3. Gate + default

- Env var: `VLLM_EMULATOR_PARALLEL_SURROGATE=1` (accepts `1|true|yes`, case-insensitive).
- Default: off — synchronous surrogate path from commit `409fd8dc3` is executed unchanged.
- Off-is-noop: env unset ⇒ bit-level identical timer sequencing vs parent commit. Guaranteed by (a) env check gates ALL new state creation in `__init__`; (b) the chain-formula expression using `self._last_surrogate_time_s` is literally identical to today's code.

## 4. Files touched

- `/home/wd312/Code/llm/vllm-emulator/vllm_emulator/hooks/executor_hook.py`
  - New constant `PARALLEL_SURROGATE_ENV`.
  - Module-level `_PARALLEL_SURROGATE_POOL = None` initialised lazily.
  - `__init__`: env-gated section creates the pool and sets `self._parallel_surrogate_enabled = True`. On the off-path, no new attribute is touched (condition 2).
  - `create_delayed_future`: when enabled, submit surrogate to pool + use `self._last_surrogate_time_s` as prediction (unchanged formula); pass the Future through to `_handle_async`'s timer callback for joining + measurement update.
  - `_handle_async`: timer-callback variant (only when parallel mode active) joins the surrogate Future, updates `self._last_surrogate_time_s`, then resolves the sample Future.

No vllm-core touches.

## 5. Input/output invariants

- Env unset ⇒ bit-level identical to parent-commit behaviour. Verified by:
  - `_initialize` does NOT touch any F2 state on off-path (condition 2).
  - `create_delayed_future` has an `if self._parallel_surrogate_enabled:` guard; off-path runs the synchronous code unchanged.
  - Chain formula is literally `end_time = start_time + latency_s + self._last_surrogate_time_s`; same expression on both paths.
- Env set + first call ⇒ chain formula adds `self._last_surrogate_time_s == 0.0` (additive identity, condition 3).
- Pool lifetime: created lazily, lives until process exit. No restart logic.
- Sample Future ordering invariant preserved by `surr_fut.result()` inside timer callback.

## 6. A/B protocol

- Baseline profile: chosen per `00_session_context.md` fallback rule.
- Harness: `tools/validate_f2_ab.sh`:
  - Pass A (off): env unset ⇒ synchronous surrogate. Output `results/RTX-8000-f2-off-apr18/`.
  - Pass B (on): env `VLLM_EMULATOR_PARALLEL_SURROGATE=1`. Output `results/RTX-8000-f2-on-apr18/`.
  - Both: `VLLM_EMULATOR_PREP_SURROGATE=1`, `VLLM_EMULATOR_SAMPLE_TRIM="2,98"`.
  - Rates r=2/8/16; 1000 prompts per rate.

## 7. Success metric

- At **r=8 and r=16**: `ΔTTFT ≤ −3pp` (variant TTFT closer to zero).
- At **every rate**: `ΔTPOT ≥ −1pp` (does not worsen TPOT by more than 1pp).

If both conditions hold → **KEEP**.

## 8. Rollback

- Unset `VLLM_EMULATOR_PARALLEL_SURROGATE`. Zero-code rollback. Pool idles until process exit.

## Review agent verdict

_(populated after Step 2 — resubmission pending after rewrite.)_

## Results

_(populated after A/B run)_

| Rate | Off TPOT | On TPOT | Off TTFT | On TTFT | Verdict |
|---|---|---|---|---|---|
| 2 | tbd | tbd | tbd | tbd | — |
| 8 | tbd | tbd | tbd | tbd | — |
| 16 | tbd | tbd | tbd | tbd | — |

**Overall verdict:** _(KEEP / DROP)_
