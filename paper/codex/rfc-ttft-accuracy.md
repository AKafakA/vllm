# RFC: Closing the TTFT Gap — Timer Architecture Options

**Date**: 2026-04-08
**Status**: Design review requested
**Context**: GhostServe (vLLM emulator) online serving accuracy

---

## Problem Statement

The emulator's TTFT (Time To First Token) consistently underestimates real GPU by 15-27%, while TPOT and E2E are within 5%. The gap exists because the emulator's async Future mechanism doesn't accurately model how real GPU blocks the vLLM engine thread during `execute_model()`.

### Current Best Results (2D oracle + hybrid overhead)

| Config | R=1 TPOT | R=1 TTFT | R=4 TPOT | R=4 TTFT | R=8 TPOT | R=8 TTFT |
|--------|----------|----------|----------|----------|----------|----------|
| Pool   | -0.4%    | -21.3%   | +1.8%    | -27.9%   | +0.8%    | -25.6%   |
| Chain  | -0.7%    | -15.3%   | +3.5%    | -21.0%   | +6.3%    | -16.9%   |

**Targets**: TPOT <5%, E2E <5%, TTFT <15%

### Root Cause Analysis

On real GPU, `execute_model()` in UniProcExecutor **blocks the engine thread** for the full step-cycle duration. During this blocking:
- No IPC processing occurs
- New requests queue in the IPC pipe
- Requests wait for the current step to finish before being scheduled

The emulator returns a pending Future without blocking the engine thread. The engine processes IPC immediately, picking up new requests too early → TTFT is too low.

---

## Architecture: How vLLM's Async Engine Works

```
Engine Core Main Loop:
  1. Check IPC for new requests          ← blocked during real GPU execute_model
  2. Run scheduler → produce batch
  3. Call execute_model(batch)            ← returns Future
  4. Call sample_tokens()                 ← returns sample Future
  5. If sample_fut pending → go to 1     ← pipelining (batch_queue_size=2)
  6. If sample_fut done → process output, send to client
```

**Key constraint**: `sample_fut` must be pending (not resolved) when returned, otherwise `num_output_placeholders` drops to 0 → scheduler deadlocks (generates 0 tokens).

---

## Option A: Uncapped Chain (current best for TTFT)

### Mechanism
```python
start_time = max(now, gpu_free_time)
end_time = start_time + step_latency
gpu_free_time = end_time
delay = end_time - now
timer = threading.Timer(delay, resolve_future)
```

Each step's Future resolves only after all prior steps complete (via `gpu_free_time` chaining). This delays Future resolution, causing the engine to see work as pending longer → IPC processing is delayed → TTFT increases toward real.

### Pros
- No hardcoded parameters — uses profiled step_cycle directly from 2D table
- Naturally models GPU serialization
- Best TTFT at R=1 (-15.3%) and R=8 (-16.9%)

### Cons
- **TPOT drift at high rates**: At R=8, TPOT=+6.3% (over 5% target). The chain accumulates because:
  - Engine is NOT blocked between steps (unlike real GPU)
  - Engine schedules ahead faster than timers fire
  - `gpu_free_time` advances ahead of wall-clock
  - Each step's delay grows: `(gpu_free_time - now)` increases over time
- The drift is NOT physical — on real GPU, engine blocking prevents this accumulation

### Open Questions
1. Can we cap drift without introducing hardcoded parameters?
2. Could a self-correcting chain (periodically reset gpu_free_time) work?
3. Is +6.3% TPOT at R=8 acceptable for the paper?

---

## Option B: ThreadPool (current best for TPOT)

### Mechanism
```python
def _gpu_step():
    time.sleep(step_latency)
    return fake_output
sample_fut = thread_pool_executor.submit(_gpu_step)  # max_workers=1
```

Single-worker pool serializes steps naturally. Engine thread is never blocked.

### Pros
- Excellent TPOT at all rates (R=1: -0.4%, R=4: +1.8%, R=8: +0.8%)
- No drift — pool backpressure naturally limits how far engine gets ahead
- Thread-switch overhead (~0.5ms) accidentally matches real scheduling gap

### Cons
- **TTFT always too fast** (-21 to -28%) — engine processes IPC immediately after submit
- Adding compensation to the pool sleep doesn't help (tested: only 1.7% improvement)
  because the engine already picked up new requests before the pool worker starts

### Why Compensation Inside Pool Fails
```python
latency_us += sched_compensation   # added to pool sleep duration
pool.submit(sleep(latency_s))       # engine returns immediately ← IPC already processed
```
The compensation makes the Future resolve later, but the engine already picked up new requests from IPC. The damage happens at submit time, not at resolution time.

---

## Option C: Pool + Blocking Sleep Before Submit

### Mechanism
```python
time.sleep(blocking_delay)           # Block engine thread first
sample_fut = pool.submit(sleep(remaining))  # Then submit remainder to pool
```

### Pros
- Models IPC blocking directly
- Pool handles Future lifecycle (no drift)

### Cons
- **Requires splitting step_latency into blocking + pool portions**
- Any fixed split (e.g., 95%/5%, 50%/50%) is a hardcoded hyperparameter
- The split is not a hardware property — cannot be profiled universally
- Tested previously at R=4: only improved TTFT from -22.7% to -21.0%
  (but that test may have used the wrong splitting approach)

### Critical Issue
The split between "how much to block" vs "how much to leave in pool" is an implementation artifact of vLLM's async scheduler constraint (need pending Future). It's not derivable from hardware profiling. **Any fixed ratio will not generalize across hardware/models.**

---

## Option D: Chain with Profiled Drift Correction

### Mechanism
Use uncapped chain but correct the drift using profiled data.

```python
start_time = max(now, gpu_free_time)
# Drift correction: don't let gpu_free_time get more than
# max_profiled_step_cycle ahead of now
max_drift = lookup_max_step_cycle(current_concurrency)  # from 2D table
gpu_free_time = min(gpu_free_time, now + max_drift)
end_time = gpu_free_time + step_latency
```

### Pros
- Chain's TTFT benefit preserved
- Drift capped by profiled data (max step_cycle at current concurrency)
- No arbitrary fractions — the cap comes from the 2D profile

### Cons
- `max_step_cycle(concurrency)` needs to be derived from the 2D table
- May still have some residual drift within the cap window
- Needs validation: does capping preserve TTFT improvement?

---

## Option E: Hybrid — Pool for Decode, Chain for Prefill

### Mechanism
- **Prefill steps** (new request arriving): Use chain to model IPC blocking → correct TTFT
- **Decode-only steps** (ongoing requests): Use pool for correct TPOT → no drift

### Rationale
TTFT is determined by prefill steps (first step with new request). TPOT is determined by decode steps (subsequent steps). Using different mechanisms for each targets the right metric.

```python
if has_prefill:
    # Chain: delays Future, models IPC wait for new request
    delay = max(0, gpu_free_time - now) + step_latency
    timer = Timer(delay, resolve_future)
    gpu_free_time = now + delay
else:
    # Pool: correct per-step timing, no drift
    pool.submit(sleep(step_latency))
```

### Pros
- Targets TTFT and TPOT independently
- No arbitrary split parameters
- Both mechanisms use profiled step_cycle directly from 2D table

### Cons
- Switching between pool and chain mid-sequence could create timing discontinuities
- `gpu_free_time` only updated on prefill steps — may get stale
- Needs careful handling of the pool/chain transition

---

## Option F: Full Engine-Thread Blocking (Architectural Change)

### Mechanism
Instead of working around the async constraint, modify vLLM's engine loop to support synchronous execute_model that returns a pending Future.

### Approach
Block inside `execute_model()` for the full step_cycle, but arrange for `sample_tokens()` to return a Future that resolves with minimal delay afterward:

```python
# In executor hook:
def create_delayed_future(self, scheduler_output, non_block=True):
    time.sleep(step_latency)    # Block engine thread (matches real GPU)
    # Return immediately-resolving Future for sample_tokens
    # But modify engine loop to not deadlock on resolved Future
```

This requires patching vLLM's `EngineCore` to handle `num_output_placeholders == 0` without deadlocking.

### Pros
- Physically exact — matches real GPU behavior
- No timer/chain/pool complexity
- All timing from profiled step_cycle, no parameters

### Cons
- Requires modifying vLLM core engine loop (invasive)
- May break other scheduler assumptions
- Higher maintenance burden across vLLM versions

---

## Recommendation

**Short term (paper deadline)**: Option D (chain + profiled drift correction) or Option E (hybrid pool/chain). Both avoid hardcoded parameters and use the 2D table directly.

**For discussion**: Is +6.3% TPOT at R=8 acceptable? If so, uncapped chain (Option A) is already sufficient — simplest approach, all timing from profile.

---

## Current Profiling Infrastructure

The 2D table provides `step_cycle(tt_bucket, concurrency_bucket) → latency_us` from 60,000 step-cycle records across 13 arrival rates. Key data points:

| tt | conc=2 | conc=7 | conc=15 | conc=35 | conc=75 | conc=150 | conc=250 |
|----|--------|--------|---------|---------|---------|----------|----------|
| 2  | 11.6ms | 12.4ms | 12.4ms  | —       | —       | —        | —        |
| 262| —      | —      | —       | —       | —       | —        | 53.4ms   |
| 267| 17.3ms | 17.3ms | 17.6ms  | 18.0ms  | —       | 84.6ms   | 113.4ms  |

The profile builder (`build_serving_profile_2d.py`) generates this table with:
- TT bucket width: 5 tokens
- Concurrency boundaries: [1, 3, 5, 10, 20, 50, 100, 200, 300]
- Outlier filtering: samples >3× or <⅓× median removed
- Minimum 3 samples per cell

All oracle modes (step_cycle, hybrid, 2d, corrected) and timer modes (pool, chain) can be combined via environment variables. No code changes needed to test any combination.

---

## Files

| File | Role |
|------|------|
| `vllm_emulator/hooks/executor_hook.py` | Timer dispatch: pool vs chain, hybrid overhead |
| `vllm_emulator/oracle/gpu_cost_oracle.py` | 2D table lookup + interpolation |
| `paper/.../build_serving_profile_2d.py` | Profile builder: step_cycle → 2D table |
| `tools/e2e/test_2d_chain.sh` | Chain evaluation script |
| `tools/e2e/test_2d_hybrid.sh` | Pool evaluation script |
