# TTFT Investigation: Timer Pipelining and Async Scheduler Constraint

**Date**: April 6, 2026  
**Status**: Root cause confirmed, architectural constraint identified

## Problem Statement

The emulator systematically underestimates TTFT (Time To First Token) at low request rates (rate=1: -28% to -39% error) while TPOT, throughput, and E2E latency are accurate (<5% error at all rates).

## Root Cause

### 1. The Timer Pipelining Effect

The emulator uses `threading.Timer` to model GPU compute time, returning a **pending Future** (`done()=False`). The vLLM engine core's `step_with_batch_queue` checks this:

```python
# vllm/v1/engine/core.py, line ~596
if (model_executed
    and len(batch_queue) < self.batch_queue_size
    and not batch_queue[-1][0].done()):  # ← pending Future
    return None, True  # ← EARLY RETURN (pipelining)
```

- **Real GPU**: `execute_model` blocks → returns resolved Future → `done()=True` → falls through → processes output immediately
- **Emulator**: timer starts → returns pending Future → `done()=False` → returns early → engine processes IPC, picks up new requests DURING timer wait

This gives the emulator an IPC scheduling advantage: new requests are picked up ~20ms earlier than on real GPU.

### 2. Why It's Worse at Low Rates

At **rate=1** (~3 concurrent requests):
- Engine has room in batch queue (1 of 2 slots used)
- New request arrives → emulator picks it up immediately during timer wait
- Real GPU: must finish current blocked step first → ~20ms delay
- **Full pipelining advantage ≈ 20-40ms**

At **rate=4** (~12 concurrent requests):
- Engine batch queue is typically full (2 of 2 slots)
- Both real and emulator must pop before scheduling → less advantage
- Profile overestimation at high tt partially offsets the advantage
- **Net advantage ≈ 0ms or negative**

### 3. TTFT Trace Evidence

EngineCore-side TTFT tracing (VLLM_EMULATOR_TRACE_TTFT=1) confirmed:
- **Scheduling delay = 0ms** in both real and emulator (not the bottleneck)
- Real GPU exec at num_reqs=2-5: **110-118ms** per request
- Emulator exec at num_reqs=2-5: **37-40ms** per request
- Gap ≈ **78ms** at the EngineCore level

### 4. Why the Async Scheduler Requires Pipelining

The `AsyncScheduler` (used when `batch_queue_size > 1`) tracks `num_output_placeholders`:

```python
# During schedule(): 
request.num_output_placeholders += 1  # Expects in-flight batch

# During update_from_output():
request.num_output_placeholders -= len(new_token_ids)  # Batch completed
```

The scheduling formula:
```python
num_new_tokens = num_tokens_with_spec + num_output_placeholders - num_computed_tokens
```

**With pipelining** (timer, pending Future):
- Schedule N: `placeholders=1`
- Schedule N+1: `placeholders=2` (N not yet processed)
- Process N: `placeholders=1`
- When scheduling N+2: `num_new_tokens = spec + 1 - computed = 1` ✓

**Without pipelining** (any blocking approach):
- Schedule N: `placeholders=1`
- Process N: `placeholders=0` (processed before next schedule)
- When scheduling N+1: `num_new_tokens = spec + 0 - computed = 0` ✗ **DEADLOCK**

## Approaches Attempted

| Approach | TTFT | TPOT | Deadlock? | Why |
|----------|------|------|-----------|-----|
| Timer (pipelining) | -28% to -39% | <5% ✓ | No | Pipelining advantage |
| Blocking in execute_model | Would fix TTFT | N/A | **Yes** at 30+ reqs | Placeholders=0 → sched deadlock |
| BlockingFuture (done()=True) | Would fix TTFT | N/A | **Yes** | Same: placeholders=0 |
| Timer + no-pipeline flag | Would fix TTFT | N/A | **Yes** | Same: prevents early return → placeholders=0 |
| Disable async scheduling | Would fix TTFT | Would fix | No, but **different behavior** | Non-default vLLM config |
| sched_overhead (sleep before timer) | Partial improvement | Varies | No | Added ~10ms delay, doesn't fix root cause |
| N-sweep IPC overhead | Rate=2 ✓ | <5% ✓ | No | Measured overhead, but rate=1 gap too large |

## Conclusion

**The TTFT gap is an inherent architectural constraint**: the vLLM async scheduler requires pipelining (`num_output_placeholders > 0` from in-flight batches), and pipelining gives the emulator an IPC scheduling advantage. Any attempt to remove pipelining breaks the async scheduler.

This is **not a bug** — it's a fundamental incompatibility between:
1. The emulator's timer-based approach (pending Futures needed for placeholder tracking)
2. The desire for no pipelining (to match real GPU's sequential behavior)

## What DOES Work Accurately

| Metric | Rate=1 | Rate=2 | Rate=4 | Offline |
|--------|--------|--------|--------|---------|
| **TPOT** | <5% ✓ | <5% ✓ | <5% ✓ | Needs profile coverage |
| **Throughput** | <1% ✓ | <1% ✓ | <1% ✓ | <1% ✓ |
| **E2E Latency** | <7% | <5% ✓ | <5% ✓ | Needs profile coverage |
| **TTFT** | -28% | -15% | +9% | N/A |

## Possible Future Directions

1. **Accept TTFT limitation**: Focus paper on TPOT/throughput/E2E accuracy. Document TTFT gap as known limitation.

2. **Custom scheduler mode**: Implement a non-async scheduler variant that works with blocking but maintains the same scheduling decisions as async. Would require significant vLLM core changes.

3. **Placeholder injection**: Before each `schedule()` call, artificially set `num_output_placeholders=1` for all running requests. This fakes the "in-flight batch" that the async scheduler expects. Hacky but might work.

4. **Post-hoc TTFT correction**: Apply a measured correction factor to TTFT based on the profiled pipelining advantage at each concurrency level (the two-pass N-sweep approach).

5. **Upstream vLLM change**: Propose a scheduler mode that doesn't depend on placeholders for token counting, making it compatible with both blocking and non-blocking execution.

## CUDA Graph Investigation Summary

During the investigation, we confirmed:
- CUDA graphs are captured **once at startup** (not recompiled during serving)
- The 110ms real GPU exec time is from **batch queue pipeline depth** (2 sequential steps), not CUDA graph compilation
- No existing simulator (Vidur, REVATI, Splitwise) models CUDA graph startup overhead
- Our CUDA graph warmup model (first-encounter shape overhead) is valid for startup effects

## References

- vLLM async scheduler: `vllm/v1/core/sched/async_scheduler.py`
- Batch queue pipelining: `vllm/v1/engine/core.py:step_with_batch_queue()`
- CUDA graph utils: `vllm/v1/worker/gpu/cudagraph_utils.py`
- TTFT tracer: `vllm/v1/engine/core.py:_TTFTTracer`
