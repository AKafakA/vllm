# TTFT Pipeline Compensation: All Attempts and Results

**Date**: April 6, 2026  
**Context**: Implementing Proposal 2b (timer compensation) to fix TTFT underestimation at low rates.

## Background

The async scheduler requires pending Futures (pipelining) in the batch queue. The pipelining gives the emulator an IPC scheduling advantage: the engine picks up new requests during the timer wait, while real GPU blocks the engine thread. See `ttft-investigation.md` for root cause.

**Goal**: Add compensation to model the real GPU's blocking behavior without breaking the async scheduler's `num_output_placeholders` tracking.

## The Engine Loop (Key to Understanding All Attempts)

```
run_busy_loop:
    while True:
        _process_input_queue()    # ← Picks up new requests from IPC
        _process_engine_step()    # ← Calls step_fn()

step_with_batch_queue (step_fn):
    1. scheduler.schedule()       # ← Creates batch, increments placeholders
    2. execute_model()            # ← Hook intercepts: starts timer, returns pending Future
    3. sample_tokens()            # ← Returns pending Future from hook
    4. batch_queue.append(future)
    5. IF queue not full AND future.done()=False:
         return None, True        # ← EARLY RETURN (pipelining) ★
    6. batch_queue.pop()          # ← Waits for oldest timer to fire
    7. update_from_output()       # ← Decrements placeholders, advances request
    8. return outputs

★ = The pipelining point. After early return, engine goes back to
    _process_input_queue() which picks up new requests DURING the timer wait.
```

## The IPC Advantage Timeline (Rate=1)

```
t=0ms:   Decode step N scheduled, timer(20ms) started → pending Future
         Step returns early (line 5) → back to busy loop
t=0.1ms: _process_input_queue() → picks up NEW REQUEST from IPC ★
t=0.2ms: Prefill step N+1 scheduled (includes new request), timer(20ms)
         Step returns early → back to busy loop
t=0.3ms: _process_input_queue() → nothing new
t=0.4ms: step_fn() → queue full → pop step N → waits for timer
t=20ms:  Timer N fires → processes step N output
t=20ms:  Pop step N+1 → waits for timer
t=40ms:  Timer N+1 fires → processes step N+1 output → FIRST TOKEN

TTFT = 40ms (from request arrival at t≈0ms to first token at t≈40ms)

★ On real GPU: engine is BLOCKED in execute_model at t=0ms.
  New request waits until t=20ms (step N finishes) before being picked up.
  TTFT = 20ms (wait) + 20ms (prefill) = 40ms + overhead ≈ 60ms+
```

**The advantage happens at t=0.1ms** — the engine picks up the request ~20ms earlier than real GPU would.

## Compensation Attempts

### Attempt 1: Add to Timer Duration (in executor hook)

**Code** (`executor_hook.py:create_delayed_future`):
```python
if has_prefill and self._gpu_free_time > now:
    latency_us += self._pipeline_compensation_us  # Add 19.6ms
    latency_s = latency_us / 1e6
# Timer uses extended latency_s
start_time = max(now, self._gpu_free_time)
end_time = start_time + latency_s  # ← Extended
self._gpu_free_time = end_time     # ← CASCADES to all subsequent timers
```

**Results**:
| Rate | TTFT | TPOT | Issue |
|------|------|------|-------|
| 1 | -14.1% (improved!) | -0.6% ✓ | |
| 2 | +17.3% | -7.4% | Overcorrected |
| 4 | +32.6% | +22.7% | Badly overcorrected |

**Why it fails at high rates**: The `_gpu_free_time` extension cascades through the timer chain. Adding 19.6ms to one prefill timer delays ALL subsequent decode timers by 19.6ms, destroying TPOT.

### Attempt 2: Sleep Before Timer (in executor hook)

**Code** (`executor_hook.py:create_delayed_future`):
```python
if has_prefill and self._gpu_free_time > now:
    queue_depth = getattr(scheduler_output, '_batch_queue_depth', 0)
    if queue_depth < 2:
        time.sleep(self._pipeline_compensation_us / 1e6)  # Sleep 19.6ms
        now = time.perf_counter()  # Update now
# Timer uses original latency (no cascade)
start_time = max(now, self._gpu_free_time)
```

**Results**:
| Rate | TTFT | TPOT | Issue |
|------|------|------|-------|
| 1 | -39.3% | -3.2% ✓ | No improvement |
| 2 | -6.5% | -10.4% | TPOT degraded |
| 4 | +14.4% | +6.0% | Overcorrected |

**Why rate=1 didn't improve**: The sleep happens inside `execute_model`, which runs AFTER the engine already scheduled the batch. The IPC advantage happened BEFORE scheduling — when the engine returned early from the PREVIOUS step and picked up the request. By the time `execute_model` runs, the advantage is already taken.

### Attempt 3: Sleep After step_fn (all pipelined steps, in engine core)

**Code** (`core.py:_process_engine_step`):
```python
outputs, model_executed = self.step_fn()
# Compensate on ALL pipelined returns
if (outputs is None and model_executed
        and getattr(self, '_emulator_pipeline_comp_s', 0) > 0):
    time.sleep(self._emulator_pipeline_comp_s)  # Sleep 19.6ms
```

**Results**:
| Rate | TTFT | TPOT | Issue |
|------|------|------|-------|
| 1 | -30.6% (slight improvement) | -2.5% ✓ | |
| 2 | median -0.7% ✓ | -10.6% | TPOT degraded |
| 4 | +15.0% | +8.8% | Both degraded |

**Why TPOT degraded**: The compensation fires on EVERY pipelined step (decode AND prefill). At rate=2, ~2 decode steps per second get 19.6ms compensation → 39ms/s overhead → TPOT +10%.

### Attempt 4: Sleep After step_fn (prefill only, in engine core)

**Code** (`core.py:_process_engine_step`):
```python
outputs, model_executed = self.step_fn()
if (outputs is None and model_executed
        and getattr(self, '_emulator_pipeline_comp_s', 0) > 0):
    so = getattr(self, '_last_scheduler_output', None)
    if so is not None and len(so.scheduled_new_reqs) > 0:  # PREFILL ONLY
        time.sleep(self._emulator_pipeline_comp_s)
```

**Results**:
| Rate | TTFT | TPOT | Issue |
|------|------|------|-------|
| 1 | **-41.0%** | -3.0% ✓ | NO improvement over baseline |
| 2 | median +0.2% ✓ | -10.7% | TPOT still degraded |
| 4 | +14.8% | +9.4% | Both degraded |

**Why rate=1 TTFT didn't improve**: The compensation fires on the PREFILL step (step N+1), but the IPC advantage happened BEFORE that — when the DECODE step (step N) returned early and the engine picked up the new request. By the time the prefill step runs, the request was already picked up ~20ms early.

**Why rate=2/4 TPOT still degraded**: Even though the sleep only fires on prefill steps, at rate=2 there are ~2 prefills/second × 19.6ms = 39ms/s of blocking. This delays the batch queue processing for decode steps that follow.

### Attempt 5: Variable Remaining Time (in executor hook)

**Code** (`executor_hook.py:create_delayed_future`):
```python
if has_prefill and self._gpu_free_time > now:
    remaining_us = (self._gpu_free_time - now) * 1e6  # Variable amount
    latency_us += remaining_us
```

**Results**: Catastrophic — `remaining_us` can be the ENTIRE timer chain backlog (2-3 step cycles). At rate=1, compensation was 40ms+. At rate=4 with 10+ timers chained, it was 200ms+.

### Attempt 6: Scaled by num_reqs (in executor hook)

**Code**:
```python
if has_prefill and self._gpu_free_time > now:
    num_reqs = len(scheduler_output.num_scheduled_tokens)
    scale = min(1.0, 3.0 / max(num_reqs, 1))
    latency_us += self._pipeline_compensation_us * scale
```

**Not tested** — user correctly identified `3.0` as a magic number. Replaced by queue depth approach.

## Why Every Placement Fails

The core problem is **timing**: the IPC advantage happens at `_process_input_queue()`, which runs BETWEEN steps. All our compensation attempts fire either:
- **During execute_model** (too late — request already picked up)
- **After step_fn returns** (either too late for prefill, or hurts decode TPOT)
- **In the timer duration** (cascades through timer chain)

The only correct placement is **inside `_process_input_queue` itself** — delaying the IPC drain when a timer is in flight. This would:
```python
def _process_input_queue(self):
    # IF emulator timer is in flight, delay picking up new requests
    # to match real GPU's blocked behavior
    if self._emulator_pipeline_comp_s > 0:
        if self.batch_queue and not self.batch_queue[-1][0].done():
            time.sleep(self._emulator_pipeline_comp_s)
    # ... existing IPC processing
```

**Not yet attempted** — this deeply changes the engine's core input processing loop.

## Compensation Amount

All attempts used `pipeline_compensation_us = 19.6ms` computed from the profile:
```python
# Average decode step cycle at low total_tokens (tt=1-4)
decode_fwd = profile_pack.get("decode_forward_pass", [])
low_tt = [e["latency_us"] for e in decode_fwd if e["total_tokens"] <= 4]
pipeline_compensation_us = sum(low_tt) / len(low_tt)  # ≈ 19.6ms
```

This represents one decode step cycle — the time the engine would be blocked on real GPU before picking up a new request.

## Baseline Comparison (No Compensation)

For reference, the timer approach WITHOUT any compensation:

| Rate | Mean TTFT | TPOT | E2E | Throughput |
|------|-----------|------|-----|------------|
| 1 | -39% | <5% ✓ | <7% | <1% ✓ |
| 2 | -15% | <5% ✓ | <5% ✓ | <1% ✓ |
| 4 | +9% | <5% ✓ | <5% ✓ | <1% ✓ |

## Remaining Options

### Option A: Compensation in _process_input_queue
Sleep before IPC drain when a timer-based batch is pending. This is the only placement that blocks the engine BEFORE it picks up new requests.

**Concern**: Changes the engine's core loop. May affect request handling latency for ALL requests, not just new prefills.

### Option B: Accept TTFT Limitation
Commit clean timer approach. Report TPOT <5%, throughput <1%, E2E <7% as the primary contribution. Document TTFT as a known limitation of timer-based emulation with async scheduling.

### Option C: Virtual Time (REVATI-style)
Implement virtual time semantics where the engine sees GPU completion in zero wall-clock time. The async scheduler sees the same event sequence as real GPU. Major architectural change.

## Key Insight from Codex Review

The Codex review (external) recommended using `_gpu_free_time > now` as the trigger for compensation. This is correct — it directly measures "is prior emulated GPU work in flight?" The trigger works perfectly. The problem is the PLACEMENT and AMOUNT of compensation, not the trigger.

The review also suggested: "if TTFT drops into a defensible band, ship it. If not, scope claims around TPOT/throughput/E2E."
