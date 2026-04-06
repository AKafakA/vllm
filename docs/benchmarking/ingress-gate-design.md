# Ingress-Gated Realtime Mode: Design Document

## Goal

Improve rate=1 TTFT to <10-15% error while preserving TPOT/E2E/throughput <5% at all rates.

## Architecture

A new emulator mode in the engine core that controls ADD request admission during pending-batch windows. NOT a timer compensation — a request-ordering change.

## Key Insight

The TTFT error comes from a specific moment in the engine loop:

```
step_with_batch_queue():
    schedule batch → execute_model → pending Future
    → early return (line 596)

_process_input_queue():          ← HERE: engine picks up ADD 
    input_queue.get() → ADD      ← during pending-batch window
    _handle_client_request(ADD)  ← request ingested early
```

On real GPU, `execute_model` blocks → no early return → `_process_input_queue` only runs AFTER batch completes. New requests wait.

## Signal: `_last_step_pipelined`

Instead of checking batch queue state or `_gpu_free_time` (both failed):

Track whether the **previous step returned early** (pipelined):

```python
# In _process_engine_step, after step_fn returns:
outputs, model_executed = self.step_fn()
self._last_step_pipelined = (outputs is None and model_executed)
```

This is the DIRECT signal that the engine just took the early return path. It means:
- A pending batch exists in the queue
- The engine is about to drain IPC during the timer window
- Any ADD picked up now has an unfair timing advantage

## Gate Design

In `_process_input_queue`, before handling each request:

```python
def _gate_add_request(self, request_type):
    if not self._emu_ingress_gate:
        return
    if request_type != EngineCoreRequestType.ADD:
        return  # Never gate control traffic
    if not self._last_step_pipelined:
        return  # Previous step was not pipelined → no advantage
    
    # One-shot: clear the flag so subsequent ADDs in this drain pass through
    self._last_step_pipelined = False
    
    # Capped delay: don't fully block, just remove the early-pickup advantage
    # Amount: profiled from avg decode step cycle, capped
    time.sleep(self._emu_ingress_cap_s)
```

## Why This Is Different From Previous Attempts

| Previous | Problem | This Design |
|----------|---------|-------------|
| `_gpu_free_time > now` trigger | Fires wrong rates (idle=no trigger, busy=always trigger) | `_last_step_pipelined` fires when pipelining actually occurred |
| Timer compensation (delay/duration) | Cascades through timer chain or batch queue | Separate sleep, no timer interaction |
| `future.result()` (full block) | Stalls engine, -11% rate=2 TPOT | Capped short sleep, one-shot |
| After step_fn | Too late (request already ingested) | Before request handling (correct timing) |

## Why `_last_step_pipelined` Works at Rate=1

At rate=1 with 3 concurrent requests:
```
t=0:    Decode step → pipeline → _last_step_pipelined = True
t=0.1:  _process_input_queue runs
t=0.1:  No ADD in queue (next request arrives at t=1000ms)
t=0.1:  _process_engine_step runs → another decode → pipeline again
...
t=940:  Last decode step → pipeline → _last_step_pipelined = True
t=940:  _process_input_queue → still no ADD
t=960:  Step completes, requests done → _last_step_pipelined = False
...
t=1000: New request arrives
t=1000: _process_input_queue → ADD found
t=1000: _last_step_pipelined = False ← flag was cleared by non-pipelined step
t=1000: No gate fires ← SAME PROBLEM AS BEFORE
```

**Problem**: at rate=1, the flag is set during decode pipelining but cleared before the ADD arrives (940ms later).

## Fix: Persist the Flag Until Consumed

Instead of clearing `_last_step_pipelined` on every step, only clear it when an ADD is gated:

```python
# In _process_engine_step:
if outputs is None and model_executed:
    self._last_step_pipelined = True
# Note: do NOT clear on non-pipelined steps

# In _gate_add_request:
if self._last_step_pipelined:
    self._last_step_pipelined = False  # Consumed
    time.sleep(cap)
```

But this makes the flag sticky — it stays True from the last pipelined step until the next ADD. At rate=1, this means every ADD gets gated (the flag persists from 940ms ago).

At rate=4, the flag is set on every step and consumed on every ADD. With 4 ADDs/second and 50 pipelined steps/second, every ADD gets gated too.

**Same problem**: sticky flag fires at all rates equally.

## Better: Track Time Since Last Pipelined Step

```python
# In _process_engine_step:
if outputs is None and model_executed:
    self._last_pipeline_time = time.perf_counter()

# In _gate_add_request:
if request_type != ADD:
    return
elapsed = time.perf_counter() - self._last_pipeline_time
if elapsed > self._emu_ingress_window_s:  # e.g., 50ms window
    return  # Pipeline was too long ago, no advantage
# Within window → gate this ADD
time.sleep(min(self._emu_ingress_cap_s, elapsed))
```

The window parameter controls how long after a pipelined step the gate remains active:
- At rate=1: decode steps finish at ~t=60ms. Flag set. Request arrives at ~t=1000ms. elapsed=940ms > window(50ms) → **no gate**. Still doesn't fire!

**The fundamental issue remains**: at rate=1, by the time the ADD arrives, the pipelining advantage is long over. The engine processed the batch, went idle, and the ADD arrives fresh to an idle engine.

## Alternative: Gate Based on Concurrent Request State

The pipelining advantage affects TTFT when there ARE concurrent requests being decoded while a new request arrives. The advantage = the engine can schedule the new request's prefill during an ongoing decode timer.

What if we gate based on whether the engine is CURRENTLY processing ongoing requests?

```python
# In _gate_add_request:
if request_type != ADD:
    return
if self.scheduler.get_num_unfinished_requests() == 0:
    return  # Idle engine, no concurrent work → no advantage
# Engine has ongoing work → gate this ADD
time.sleep(cap)
```

At rate=1 with 3 concurrent: `num_unfinished > 0` → gates the ADD. ✓
At rate=1 when idle: `num_unfinished == 0` → no gate. ✓
At rate=4 with 12 concurrent: `num_unfinished > 0` → gates every ADD. ✗

Still gates at rate=4. Need scaling: `cap * min(1, K/num_unfinished)` but that's back to heuristics.

## Conclusion

Every ingress-gate variant faces the same fundamental tension:
- At rate=1: the advantage exists during concurrent decode, but ADDs arrive during idle → gate doesn't fire
- At rate=4: everything is concurrent → gate always fires → over-penalizes

The only approach that correctly targets the advantage is blocking during the ACTUAL pipelined step (attempt #7: `future.result()`), but that's too aggressive for rate=2.

## Recommendation

The narrowest viable approach combines:
1. `num_unfinished > 0` as the gate trigger (engine has ongoing work)
2. `min(cap, step_cycle / num_unfinished)` as the delay (scales inversely)
3. One-shot per batch cycle

This is essentially the `3.0/num_reqs` scaling approach but with `num_unfinished` from the scheduler instead of `num_reqs` from the batch. It's a heuristic but grounded in engine state.
