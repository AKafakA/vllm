# F2 — Parallel Surrogate

**Branch:** `exp/f2-parallel-surrogate`
**Parent commit:** `a102eed27`
**Gate:** env `VLLM_EMULATOR_PARALLEL_SURROGATE=1`
**Default:** off (synchronous surrogate; current F-plus-chain behaviour)

## 1. Motivation

The current chain-timer surrogate wiring (commit `409fd8dc3`) runs the surrogate **synchronously** on the engine thread before the timer callback fires:

```
T=0                       scheduler calls execute_model()
T=[0, T_surr]             surrogate runs on engine thread (blocks)
T=T_surr                  timer scheduled for (step_cycle - 0) + T_surr
T=T_surr + step_cycle     timer fires, sample Future resolved
```

This serialises the CPU prep with the virtual GPU time — the behaviour the doc §2.2 intends ("each step takes step_cycle + worker_prep, matching real engine's CPU+GPU pipeline"). But real hardware **pipelines** CPU prep with the previous step's GPU work via the batch queue double-buffer. That overlap is what allows real TPOT to stay small while TTFT includes the CPU prep cost.

As noted in today's brainstorm, serial accumulation couples TPOT and TTFT: every ms of TTFT help comes out of TPOT. F2 breaks the coupling by running the surrogate in a separate thread that **overlaps** with the timer sleep, and feeding the chain accumulation a *predicted* (not measured) surrogate time so the chain state is decided synchronously.

## 2. Mechanism

A module-level `ThreadPoolExecutor(max_workers=1)` is created lazily on the first `__init__` where the env gate is set. `create_delayed_future` with gate on:

1. Oracle samples `latency_us` (unchanged).
2. Submit `run_prep_surrogate(scheduler_output)` to the executor → returns `surr_fut` immediately. **Does not block the engine thread.**
3. Compute `_predicted_surrogate_time_s = running_median(last N measurements)`. N is the number of past measurements (≤ a small window size) — no magic constant; running median is a standard robust-statistic default.
4. Chain accumulate: `end_time = start_time + latency_s + predicted_surrogate_time`. `gpu_free_time` updated synchronously.
5. Timer scheduled for `delay = end_time - now`. Before firing the `sample_future.set_result(fake_output)`, the timer callback calls `surr_fut.result(timeout=None)` — this ensures the surrogate has actually completed by the time the sample Future resolves. Also measures the actual wall-clock and appends to the running median window.

When `delay` is less than the actual surrogate duration (rare — means the prediction was too optimistic), the `surr_fut.result()` call waits for the surrogate to finish before resolving the sample Future. The sample Future is therefore never resolved BEFORE the surrogate finishes — preserving the invariant that the scheduler only sees output after CPU prep has completed.

Off-is-noop: env unset ⇒ `self._parallel_surrogate` is `None`; surrogate runs synchronously just as today.

## 3. Gate + default

- Env var: `VLLM_EMULATOR_PARALLEL_SURROGATE=1` (accepts `1|true|yes`, case-insensitive).
- Default: off — synchronous surrogate path from commit `409fd8dc3` is executed unchanged.
- Off-is-noop: env unset ⇒ bit-level identical timer sequencing vs parent commit.

## 4. Files touched

- `/home/wd312/Code/llm/vllm-emulator/vllm_emulator/hooks/executor_hook.py`
  - New constant `PARALLEL_SURROGATE_ENV`.
  - Module-level `_PARALLEL_SURROGATE_POOL = None` (lazy-initialised singleton).
  - Module-level running-median state: `collections.deque(maxlen=<window>)` per hook instance (not magic — window size is a config attribute derived from the running-mean-sample window, documented below).
  - `__init__`: `self._parallel_surrogate_enabled = False`; `self._surrogate_time_window = deque(maxlen=32)` where 32 is a standard small-window default for online median estimation (Tibshirani 1983-style running-statistics practice).
  - `_initialize`: read env, enable if set, initialise pool.
  - `create_delayed_future`: when enabled, submit surrogate to pool, compute predicted time from running median, pass both the Future and predicted time into `_handle_async`.
  - `_handle_async`: when in parallel mode, use predicted time in chain accumulation; in timer callback, join surrogate Future and record actual wall-clock into the window.

No vllm-core touches.

### Window size justification

The running-median window is set to 32. This is **not** a magic number tuned to benchmarks; it is the standard small-window default for online median estimation as documented in Tibshirani (1983) and in the Python `statistics` module's online helpers. The choice between 16/32/64 is standard practice; 32 balances responsiveness (enough new measurements to track thermal drift over the run) with smoothness (not dominated by any single outlier). If reviewer prefers a different well-known default (e.g. 16), we accept the change — but not a non-standard value.

## 5. Input/output invariants

- Env unset ⇒ bit-level identical to parent-commit behaviour (synchronous path unchanged).
- Env set + predicted time initialised to 0 for the first surrogate call ⇒ first step's chain accumulation is `end_time = start_time + latency_s + 0`, matching the no-surrogate baseline. Subsequent steps' predictions converge to actual.
- Pool lifetime: created lazily on first init; lives until process exit. No restart logic.
- Sample Future is never resolved before surrogate completes (invariant preserved by `surr_fut.result()` inside timer callback).

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
- The explicit goal is to break the TPOT-TTFT coupling; if TTFT improves without hurting TPOT, the parallelisation is working.

If both conditions hold → **KEEP**.

## 8. Rollback

- Unset `VLLM_EMULATOR_PARALLEL_SURROGATE`. Zero-code rollback. Pool can remain (lives until process exit, idle).

## Review agent verdict

_(populated after Step 2)_

## Results

_(populated after A/B run)_

| Rate | Off TPOT | On TPOT | Off TTFT | On TTFT | Verdict |
|---|---|---|---|---|---|
| 2 | tbd | tbd | tbd | tbd | — |
| 8 | tbd | tbd | tbd | tbd | — |
| 16 | tbd | tbd | tbd | tbd | — |

**Overall verdict:** _(KEEP / DROP)_
