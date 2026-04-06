# TTFT Compensation Round 2: Decoupled Timer Approach

**Date**: April 6, 2026  
**Previous**: See `ttft-compensation-attempts.md` for Round 1 (6 attempts, all failed).

## What Changed: Decoupled Compensation

Based on Codex review recommending `_gpu_free_time > now` as trigger, we implemented:

**Key idea**: Add compensation to the timer DELAY only, not to `_gpu_free_time`. This way:
- The prefill timer takes longer (TTFT increases)
- But `_gpu_free_time` isn't extended (decode timers unaffected → TPOT preserved)

```python
# vllm_emulator/hooks/executor_hook.py
pipeline_comp_s = 0.0
if (has_prefill
        and self._pipeline_compensation_us > 0
        and self._gpu_free_time > now):
    pipeline_comp_s = self._pipeline_compensation_us / 1e6

start_time = max(now, self._gpu_free_time)
end_time = start_time + latency_s
self._gpu_free_time = end_time          # Chain uses ORIGINAL latency only
delay = end_time - now + pipeline_comp_s # Timer uses original + compensation
```

**Compensation amount**: `pipeline_compensation_us = 19.6ms` (avg decode step cycle at tt=1-4, from profile).

## Results (200 prompts, RTX 3060, Qwen 1.5B)

| Rate | Mean TTFT | Median TTFT | Mean TPOT | Median TPOT | E2E | Throughput |
|------|-----------|-------------|-----------|-------------|-----|------------|
| 1 | -22.4% | -39.8% | -4.4% ✓ | -5.2% | -5.2% | +0.1% ✓ |
| 2 | +0.3% ✓ | +16.1% | -6.0% | -7.3% | -5.7% | +0.5% ✓ |
| 4 | +19.2% | +21.6% | +9.2% | +11.2% | +9.6% | -0.0% ✓ |

## Analysis

### Rate=1: Compensation helps mean but not median
- **Mean improved**: -39% → -22.4% (compensation fires on ~30% of requests)
- **Median unchanged**: -39.8% (most requests arrive when `_gpu_free_time <= now`)
- At rate=1, engine alternates between busy (timer running) and idle (timer finished)
- ~70% of requests arrive during idle moments → trigger doesn't fire → no compensation
- The 30% that DO get compensated improve the mean but not the median

### Rate=2: Mean TTFT passes but median overestimates
- Mean TTFT +0.3% ✓ — compensation is about right on average
- Median +16.1% — some requests get overcompensated (those with long pending timers)
- TPOT -6% to -7.3% — compensation somehow still affects decode slightly

### Rate=4: Everything overestimates
- TTFT +19-22% — compensation fires too often at high rate
- TPOT +9-11% — even though `_gpu_free_time` isn't extended, the batch queue interaction causes cascade
- Throughput still perfect (-0.0%)

### Why TPOT is affected despite "decoupled" approach
The timer `delay` includes the compensation. Even though `_gpu_free_time` uses the original latency, the TIMER for the prefill step takes 19.6ms longer to fire. During this extra time:
- The batch queue's oldest entry (the prefill) hasn't resolved yet
- The engine can't pop and process it → decode steps that follow are delayed
- The batch queue is "stuck" for 19.6ms extra → TPOT increases indirectly

This is a subtler cascade: not through `_gpu_free_time`, but through the **batch queue occupancy**. The extra timer delay keeps the batch queue entry pending longer, blocking subsequent step processing.

## Why the Trigger Doesn't Work at Rate=1

At rate=1 with ~3 concurrent requests:
```
Timeline:
t=0ms:    Timer N starts (20ms), gpu_free_time = t+20ms
t=0.1ms:  Engine returns early, checks IPC → picks up new request
t=20ms:   Timer N fires, gpu_free_time expires
t=20.1ms: Engine processes output, gpu_free_time = now (expired)
t=500ms:  New request arrives ← gpu_free_time < now (IDLE) → NO COMPENSATION
t=500ms:  Prefill scheduled, Timer starts
t=520ms:  Timer fires
...
t=1000ms: Another new request arrives ← gpu_free_time < now → NO COMPENSATION
```

At rate=1, the inter-arrival time (~1000ms) is much larger than the step cycle (~20ms). The engine finishes all timers long before the next request arrives. `_gpu_free_time` is always expired when new requests come in.

## Why Removing the Trigger Would Overestimate at Rate=4

At rate=4:
- Baseline TTFT already overestimates (+9% without compensation)
- Adding 19.6ms to every prefill pushes it to +22%
- The baseline overestimate comes from profile overestimation at high total_tokens

## The Fundamental Tension

```
Rate=1: Engine idle between requests → no trigger fires → underestimates TTFT
Rate=4: Engine always busy → trigger always fires → overestimates TTFT
```

The trigger `_gpu_free_time > now` fires at the WRONG rates:
- Fires too rarely at rate=1 (where compensation is needed most)
- Fires too often at rate=4 (where compensation is not needed)

This is because the pipelining advantage is INVERSE to engine busyness:
- Idle engine: pipelining advantage is LARGE (engine picks up requests instantly)
- Busy engine: pipelining advantage is SMALL (engine is already processing)

But the trigger detects busyness (`gpu_free_time > now` = busy), which is the opposite of what we need.

## Remaining Options

### Option A: Invert the trigger
Compensate when `_gpu_free_time <= now` (engine idle) instead of `> now` (engine busy). But this fires on the FIRST request to an idle engine, which shouldn't be compensated (no prior batch to wait for).

### Option B: Track "was-busy" state
Add a flag that's set when pipelining occurs and cleared when the batch queue drains. Compensate when the flag was recently set (engine was busy when the request was in the IPC queue).

### Option C: Compensation based on batch queue history
Track the fraction of recent steps that triggered pipelining. Use this as a scaling factor. High pipelining rate → more compensation. Low → less.

### Option D: Two-tier compensation
- When `_gpu_free_time > now`: full compensation (engine busy, pipelining active)
- When `_gpu_free_time <= now` but `_gpu_free_time > now - step_cycle`: partial compensation (engine JUST became idle, request likely waited during previous busy period)

### Option E: Compensation in the timer chain itself
Instead of adding to `delay`, add to the timer's `start_time`:
```python
# Normal: start_time = max(now, gpu_free_time)
# Compensated: start_time = max(now + comp, gpu_free_time)
```
This delays the timer START without extending gpu_free_time. But need to think through if this has different cascade behavior than delaying the timer END.

### Option F: Accept and scope claims
Best results so far (clean baseline, no compensation):
- TPOT <5% at all rates ✓
- Throughput <1% at all rates ✓  
- E2E <7% at all rates ✓
- TTFT: limitation at low rates

## Key Data Points Across All Attempts

| Approach | Rate=1 Mean TTFT | Rate=2 TPOT | Rate=4 TPOT | Notes |
|----------|-----------------|-------------|-------------|-------|
| Baseline (no comp) | -39% | -3.9% ✓ | -2.3% ✓ | Clean, everything else passes |
| Timer duration (Attempt 1) | -14.1% | -7.4% | +22.7% | Best rate=1, broke rate=4 |
| Decoupled timer | -22.4% | -6.0% | +9.2% | Better than Attempt 1 at rate=4, worse at rate=1 |
| ADD delay (engine core) | -24.7% | -11.3% | +1.7% ✓ | Best rate=4, broke rate=2 E2E |
| After step_fn (all) | -30.6% | -10.6% | +8.8% | TPOT degraded everywhere |

No single approach achieves <10% TTFT at rate=1 without breaking rate=2+ TPOT or E2E.

## Infrastructure Note

The Vast RTX 3060 host experienced GPU memory exhaustion and bench serve failures after extended testing. A `pkill -9 -f python3` + server restart resolves this. The bench serve `random` dataset with `--num-prompts N --request-rate R` requires `N > R × 3` to get enough completed requests for meaningful metrics (requests arriving late in the window time out).
