# Profiling Methodology: Offline Sweep + Online Step-Cycle

## Current Approach (Single Source)

The emulator timer uses `step_cycle` data from online serving traces:
```
step_cycle = gpu_forward_pass + scheduling_overhead + output_processing
           ≈ 12ms            + 3ms                 + 5ms = 20ms
```

The timer sleeps for 20ms (the full step_cycle). The engine also runs scheduling/output code, but since the emulator uses fake outputs, this overhead is lighter (~3ms instead of ~8ms). Net effect: the timer absorbs the overhead difference, giving correct TPOT.

**Problem**: The 20ms timer enables 20ms of pipelining advantage → TTFT underestimated by ~40ms at rate=1.

## Proposed Approach (Two Sources)

### Source 1: Offline Batch Sweep
- Pure GPU forward pass time at each `total_tokens`
- Controlled conditions: single batch, no serving overhead
- Already exists: `sweep-1.5b-tp1-v14.json` → `forward_pass` section
- Gives: `f(total_tokens) → gpu_compute_time_us` (~12ms at tt=1)

### Source 2: Online Step-Cycle Trace
- Full step-to-step time during real serving at various rates
- Includes GPU + scheduling + output processing
- Already exists: `step_cycle_*.jsonl` → `decode_forward_pass` section
- Gives: `f(total_tokens) → step_cycle_time_us` (~20ms at tt=1)

### The Overhead Delta
```
overhead = step_cycle - sweep_gpu_time
         = 20ms      - 12ms = 8ms (real GPU overhead per step)

emulator_overhead ≈ 3ms (lighter: no CUDA sync, fake tensors)

delta = real_overhead - emulator_overhead = 8ms - 3ms = 5ms
```

## How They're Used Together

### For the Timer (GPU compute model)
```python
timer_duration = step_cycle(total_tokens)  # NOT sweep
```
Why step_cycle, not sweep? Because:
- The emulator's lighter overhead means using sweep alone → TPOT too fast (-25%)
- The step_cycle compensates by including overhead the emulator doesn't naturally add
- This gives TPOT <5% accuracy (verified across all rates)

### For Offline Evaluation Coverage
The sweep profile covers `total_tokens` up to 512-8192 (from controlled batch sweep). The step-cycle trace only covers tt≈1-25 (from online serving at rates 0.5-8). For offline evaluation (rate=inf, tt=50-100+), we need the sweep data for interpolation.

### For Profile Quality
Both sources provide redundant data in the overlapping range (tt=1-25):
- If sweep and step-cycle agree: high confidence in the profile
- If they diverge: indicates overhead variation (thermal, contention)

## Profiling Protocol

### Step 1: Warmup (200 prompts at rate=4)
- Reaches GPU thermal equilibrium (~60s sustained load)
- Exercises all CUDA graph shapes
- Ensures subsequent profiling captures steady-state performance

### Step 2: Online Step-Cycle Trace
- Run server with `VLLM_EMULATOR_TRACE_STEP_CYCLE=1`
- Send traffic at rates: 0.5, 1, 2, 3, 4, 6, 8, 12 (200 prompts each)
- Also rate=inf (offline, 200 prompts) for high-concurrency coverage
- Captures: `step_cycle_us` per step with `total_tokens`, `num_new_reqs`, `num_decode_seqs`
- Target: 30,000+ records covering tt=1-100+

### Step 3: Profile Build
- `build_serving_profile_2d.py` processes the trace into:
  - `prefill_forward_pass`: median step-cycle for prefill steps per tt bucket
  - `decode_forward_pass`: median step-cycle for decode steps per tt bucket
  - `forward_pass`: combined, merged with sweep data for high-tt coverage
  - `cuda_graph_warmup_us`: first-encounter shape overhead
  - `sched_overhead_table`: (optional) measured IPC overhead per concurrency

### Step 4: Evaluation
- Same GPU, same session (thermal state matched)
- 200-prompt warmup before each rate (same as profiling warmup)
- Independent server start per rate (no sequential thermal drift)
- A2A comparison: real GPU vs emulator with the profile

## Why This Protocol Matters

### Previous Issue: Profile Staleness
The original profile was captured hours before evaluation. GPU thermal state drifted:
- Profiling: GPU at steady state → 20ms step-cycle
- Evaluation (hours later): GPU degraded → 21ms real TPOT
- Result: emulator 20ms vs real 21ms → -5% TPOT (acceptable)

But when we re-profiled on a "fresh" GPU:
- Profiling: cold GPU → 15ms step-cycle  
- Evaluation: GPU warms up during test → 21ms real TPOT
- Result: emulator 15ms vs real 21ms → -28% TPOT (terrible)

**Fix**: Heavy warmup before profiling ensures the GPU is at the same thermal state as during evaluation.

### Previous Issue: Coverage Gaps
Original profile: 13,000 records from rates 0.5-8 → tt coverage 1-25
- Rate=4 evaluation (tt=10-15): covered ✓
- Offline evaluation (tt=50-100): NOT covered → extrapolation error (+50% TPOT)

**Fix**: Add rate=12 and rate=inf to profiling → tt coverage 1-100+

### Previous Issue: Record Count
Fresh profile: 7,000 records → sparse buckets → noisy medians
Original profile: 13,000 records → denser buckets → stable medians

**Fix**: 200 prompts × 9 rates = ~230,000 step records → very dense coverage

## Summary

| Component | Source | Purpose |
|-----------|--------|---------|
| Timer duration | Online step-cycle | Models GPU time + overhead delta (TPOT accuracy) |
| High-tt coverage | Offline sweep + rate=inf trace | Extends profile to tt=100+ (offline evaluation) |
| Thermal calibration | 200-prompt warmup | Matches profiling and evaluation thermal state |
| Profile density | 200 prompts × 9+ rates | Reduces bucket noise, improves interpolation |
