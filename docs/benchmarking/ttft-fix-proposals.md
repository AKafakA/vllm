# TTFT Fix Proposals: Compensating for Async Scheduler Pipelining

**Date**: April 6, 2026  
**Prerequisite**: Read `ttft-investigation.md` for the root cause analysis.

## Problem Summary

The async scheduler (`batch_queue_size=2`) requires pending Futures (pipelining) to keep `num_output_placeholders > 0`. This pipelining gives the emulator an IPC scheduling advantage, causing TTFT underestimation (~20-40ms at rate=1).

The advantage comes from the engine returning early at this check:
```python
# vllm/v1/engine/core.py, step_with_batch_queue()
if (model_executed
    and len(batch_queue) < self.batch_queue_size
    and not batch_queue[-1][0].done()):  # pending Future → True
    return None, True  # EARLY RETURN → engine processes IPC
```

## Proposal 1: Placeholder Injection

**Idea**: Before each `schedule()` call, set `num_output_placeholders = 1` for all running requests. This fakes the "in-flight batch" that the async scheduler expects, allowing blocking/no-pipeline without deadlock.

**Implementation** (in `vllm/v1/core/sched/scheduler.py`):
```python
def schedule(self):
    # Inject placeholder if emulator is in no-pipeline mode
    if getattr(self, '_emulator_inject_placeholders', False):
        for req in self.running:
            if req.num_output_placeholders == 0:
                req.num_output_placeholders = 1
                req.num_computed_tokens += 1  # Keep consistent
    # ... rest of schedule()
```

**Pros**:
- Minimal code change
- Allows blocking/no-pipeline approach
- Async scheduler stays happy

**Cons**:
- Modifies scheduler internal state artificially
- `num_computed_tokens` manipulation could cause cascading inconsistencies
- The `num_computed_tokens += 1` might break KV cache block allocation (the scheduler uses this to determine which blocks need to be allocated)
- Would need extensive testing across all scheduler paths (preemption, chunked prefill, spec decode)

**Risk**: High — scheduler state manipulation is fragile.

---

## Proposal 2: Timer Duration Compensation (Prefill-Only)

**Idea**: Increase the prefill step's timer duration to account for the time a previous pipeline step would have blocked on real GPU. On real GPU, a new request must wait for the current GPU step to finish (~T). With the timer, the new request is picked up immediately during the timer. Adding T to the prefill timer compensates.

**How it works**:
```
Real GPU:   [prev_step BLOCKS 20ms] → [schedule prefill] → [prefill BLOCKS 20ms] → output
            Total TTFT ≈ 10ms (avg remaining) + 20ms = 30ms

Emulator:   [prev_timer 20ms] ← [prefill scheduled DURING timer] → [prefill timer 20ms]
            Both overlap → Total TTFT ≈ 20ms

Fixed:      [prev_timer 20ms] ← [prefill scheduled DURING timer] → [prefill timer 40ms]
            Even with overlap, TTFT ≈ 40ms ← closer to real
```

**Implementation** (in `vllm_emulator/hooks/executor_hook.py`):
```python
def create_delayed_future(self, scheduler_output, non_block=False):
    ...
    latency_us = self._oracle.estimate_step_latency_us(total_tokens, has_prefill=has_prefill)

    # Compensate for pipeline overlap: on real GPU, a new request
    # arriving during an in-flight batch must wait for it to finish.
    # The timer approach overlaps the prefill with the previous batch.
    # Add the avg decode step duration to prefill timers when there
    # are concurrent requests (pipeline is active).
    if has_prefill and self._pipeline_compensation_us > 0:
        num_reqs = len(scheduler_output.num_scheduled_tokens)
        if num_reqs > 1:  # Concurrent work exists → pipeline is active
            latency_us += self._pipeline_compensation_us
    
    latency_s = latency_us / 1e6
    ...
```

Where `_pipeline_compensation_us` is computed at init:
```python
def _initialize(self):
    ...
    # Pipeline compensation: avg decode step cycle from profile.
    # This is the time a previous batch blocks on real GPU that
    # the timer approach overlaps away.
    decode_fwd = profile_pack.get("decode_forward_pass", [])
    if decode_fwd:
        low_tt = [e["latency_us"] for e in decode_fwd if e["total_tokens"] <= 4]
        self._pipeline_compensation_us = sum(low_tt) / len(low_tt) if low_tt else 0
    else:
        self._pipeline_compensation_us = 0
```

**Pros**:
- No scheduler modification
- Async scheduler stays intact (pending Futures, placeholders work)
- Only affects TTFT (prefill steps), not TPOT (decode steps)
- Profile-driven (avg decode step from profile)
- Conditional on concurrent requests (no compensation for isolated requests)

**Cons**:
- The compensation is approximate (avg decode step ≈ one pipeline slot)
- At rate=1 with 3 concurrent: `num_reqs > 1` is True → compensates.
  But the actual pipelining advantage depends on batch queue depth, not just num_reqs.
- Single request scenario: `num_reqs = 1` → no compensation ✓
- But a prefill step with `num_reqs = 1` could still have a decode batch in the queue
  from a just-finished request. The `num_reqs` check isn't perfect.

**Risk**: Low — only changes timer duration, doesn't touch scheduler state.

---

## Proposal 3: Virtual Time (REVATI-style)

**Idea**: Replace real-time timers with virtual time advancement. Instead of `threading.Timer(T)`, advance a virtual clock by T. The engine thread doesn't wait in real time — it just sees the virtual clock jump forward.

**How REVATI does it** (from paper arxiv:2601.00397):
- Intercepts CUDA API calls at the driver level
- Virtualizes device management (memory, streams, events)
- Fast-forwards virtual time by predicted kernel durations
- Uses barrier-based protocol for multi-process synchronization
- Engine code runs at full CPU speed, GPU "executes" in zero wall-clock time

**Implementation sketch**:
```python
# Instead of:
timer = threading.Timer(delay, _resolve)
timer.start()

# Virtual time approach:
virtual_clock.advance(delay)  # No real waiting
sample_fut.set_result(fake_output)  # Immediately resolved
```

But this requires the ENTIRE vLLM stack to use the virtual clock for timing:
- `time.time()` / `time.perf_counter()` must return virtual time
- ZMQ IPC timestamps must use virtual time
- The API server's streaming must use virtual time
- The bench serve client must use virtual time for TTFT measurement

**Pros**:
- Correct by construction — same event ordering as real GPU
- No scheduler issues (placeholders work because virtual time sequence is identical)
- Can run faster than real-time (accelerated mode)
- Proven approach (REVATI)

**Cons**:
- Major architectural change (essentially reimplementing REVATI)
- Requires patching `time.time()` globally or per-process
- Multi-process vLLM (API server ↔ EngineCore via ZMQ) complicates virtual time sync
- The bench serve client runs in a SEPARATE process — it can't share the virtual clock
- Would need a custom TTFT measurement approach

**Risk**: Very high implementation effort. Essentially a different project.

---

## Proposal 4: Batch Queue Depth-Aware Timer Compensation

**Idea**: Same as Proposal 2 but uses actual batch queue depth instead of `num_reqs > 1`.
The executor hook receives the batch queue depth from the engine core and adjusts the timer.

**Implementation**: Pass `batch_queue_depth` to the executor hook:

In `vllm/v1/engine/core.py`:
```python
def step_with_batch_queue(self):
    ...
    if self.scheduler.has_requests():
        scheduler_output = self.scheduler.schedule()
        # Pass queue depth to executor for timer compensation
        scheduler_output._emulator_queue_depth = len(batch_queue)
        exec_future = self.model_executor.execute_model(scheduler_output, non_block=True)
```

In `vllm_emulator/hooks/executor_hook.py`:
```python
def create_delayed_future(self, scheduler_output, non_block=False):
    ...
    if has_prefill and self._pipeline_compensation_us > 0:
        queue_depth = getattr(scheduler_output, '_emulator_queue_depth', 0)
        if queue_depth > 0:  # In-flight batch exists
            latency_us += self._pipeline_compensation_us
```

**Pros**:
- More precise than Proposal 2 (checks actual queue state)
- Correct for isolated requests (queue_depth=0 → no compensation)
- Correct for concurrent requests (queue_depth=1 → compensates)

**Cons**:
- Requires modifying the engine core (adding attribute to scheduler_output)
- Slightly invasive (touches the engine ↔ executor interface)

**Risk**: Low-medium — small engine modification.

---

## Proposal 5: Post-hoc TTFT Correction

**Idea**: Don't fix TTFT in the emulator. Instead, apply a correction factor when reporting results. The correction = measured pipelining advantage from the two-pass N-sweep.

**Implementation**: Pure post-processing:
```python
# After benchmarking:
emu_ttft = measured_emu_ttft
correction = ipc_overhead_table[num_concurrent_reqs]  # From N-sweep
corrected_ttft = emu_ttft + correction
```

**Pros**:
- Zero emulator code changes
- Can be applied to any existing results
- The N-sweep data already exists

**Cons**:
- Post-hoc correction feels like cheating
- The N-sweep measured overhead doesn't match online overhead (as we discovered)
- Doesn't help clients who use the emulator for real TTFT predictions

**Risk**: Low effort but weak contribution.

---

## Recommendation

**Proposal 2 (Timer Duration Compensation)** is the best balance of correctness, simplicity, and risk:
- Only adds latency to prefill timers when concurrent requests exist
- Profile-driven (avg decode step cycle)
- No scheduler state modification
- Async scheduler stays intact
- TPOT unaffected (decode timers unchanged)

The compensation value `pipeline_compensation_us` = avg decode step cycle at low tt (1-4), which is ~18-20ms on RTX 3060 with Qwen 1.5B. This should close most of the TTFT gap at rate=1 while not affecting higher rates (where the pipelining advantage is smaller because the engine is busy).

**Key consideration**: The `num_reqs > 1` check is imperfect. A request arriving at an idle engine (`num_reqs = 1` in its prefill batch) should NOT be compensated, and this check handles it correctly. But a request whose prefill batch has `num_reqs = 1` (the only request is the new one) but the batch queue has a pending entry from a previous decode batch — this case needs the queue depth check (Proposal 4).

**Simplest correct approach**: Combine Proposals 2 and 4 — use `num_reqs > 1 OR queue_depth > 0` as the compensation trigger.

## Comparison with REVATI

REVATI avoids this entire problem by using virtual time — the engine never actually waits during "GPU execution," so the pipelining advantage doesn't exist. Our timer-based approach is fundamentally different: the engine DOES wait in real time, creating the pipelining opportunity.

REVATI's approach is correct by construction but requires deep CUDA API interception and virtual time synchronization across processes. Our approach is simpler (hook at executor level) but has this TTFT artifact that needs compensation.

For the paper, this is a valid trade-off to discuss: simplicity of implementation vs. TTFT accuracy.
