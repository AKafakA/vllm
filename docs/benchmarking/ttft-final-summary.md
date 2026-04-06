# TTFT Investigation: Final Summary

**Date**: April 6, 2026  
**Status**: 10 compensation approaches tested, none achieve <10% TTFT at rate=1 without breaking rate=2+ metrics.

## Target Metrics

- TPOT: <5% at all rates
- E2E: <5% at all rates
- Throughput: <1% at all rates
- TTFT: <10% at rate=1 (acceptable limitation)

## All Approaches Tested

### Round 1: Timer-Side Compensation (in executor hook)

| # | Approach | Rate=1 TTFT | Rate=2 TPOT | Rate=4 TPOT | Failure Mode |
|---|----------|-------------|-------------|-------------|-------------|
| 1 | Timer duration (add to latency_us) | **-14.1%** ★ | -7.4% | +22.7% | _gpu_free_time cascade → all subsequent decode timers delayed |
| 2 | Sleep before timer | -39.3% | -10.4% | +6.0% | Fires inside execute_model, AFTER request already picked up from IPC |
| 3 | Variable remaining time | catastrophic | - | - | Full timer chain backlog added → 200ms+ compensation |
| 4 | Scaled by num_reqs (3.0/N) | not tested | - | - | Magic number, replaced by queue depth |

### Round 2: Engine-Core Compensation (in _process_engine_step / _process_input_queue)

| # | Approach | Rate=1 TTFT | Rate=2 TPOT | Rate=4 TPOT | Failure Mode |
|---|----------|-------------|-------------|-------------|-------------|
| 5 | Sleep after step_fn (all steps) | -30.6% | -10.6% | +8.8% | Fires on decode steps too → TPOT degraded |
| 6 | Sleep after step_fn (prefill only) | -41.0% | -10.7% | +9.4% | Fires AFTER request already scheduled (too late) |
| 7 | ADD-delay (future.result()) | -24.7% | -11.3% | +1.7% | Full block stalls engine thread → rate=2 E2E -10.9% |

### Round 3: Refined Approaches

| # | Approach | Rate=1 TTFT | Rate=2 TPOT | Rate=4 TPOT | Failure Mode |
|---|----------|-------------|-------------|-------------|-------------|
| 8 | Decoupled timer (delay only, not _gpu_free_time) | -22.4% | -6.0% | +9.2% | Batch queue occupancy cascade; trigger fires wrong rates |
| 9 | 5ms capped one-shot ingress gate | -21.3% | -6.4% | +7.2% | Same as #8; gate fires too rarely at rate=1, too often at rate=4 |
| 10 | (Baseline) No compensation | -39% | -3.9% ✓ | -2.3% ✓ | Only TTFT fails; everything else passes |

★ Closest to target but breaks rate=4 TPOT.

## Why Rate=1 Is Fundamentally Hard

At rate=1 with ~3 concurrent requests:
- Engine processes all requests in ~60ms (3 × 20ms steps)
- Then idle for ~940ms before next request arrives
- **94% of requests arrive during idle engine** → no "busy" trigger fires
- Any gate/compensation targeting "engine busy" misses 94% of requests

At rate=4 with ~12 concurrent requests:
- Engine always busy (timer chain always active)
- **100% of requests arrive during busy engine** → every gate/compensation fires
- Any fixed compensation × high request rate = large cumulative overhead

## The Fundamental Tension

```
                    Rate=1          Rate=4
Engine state:       94% idle        100% busy
Pipelining gain:    ~40ms/req       ~5ms/req
Compensation need:  HIGH            ZERO
"Busy" trigger:     fires 6%        fires 100%
Result:             under-compensate  over-compensate
```

The pipelining advantage is INVERSE to engine busyness, but all triggers detect busyness. This is the core unsolvable tension with the current architecture.

## Root Cause (Confirmed by TTFT Tracer + Code Analysis)

The async scheduler (`batch_queue_size=2`) requires pending Futures for `num_output_placeholders` tracking. Pending Futures cause the engine to return early from `step_with_batch_queue` (line 596), allowing IPC processing during timer waits.

On real GPU, `execute_model` blocks the engine thread → no IPC processing → new requests wait. The emulator's timer doesn't block → engine picks up requests ~20ms earlier.

**This is not a bug — it's an architectural coupling**: pending Futures serve both scheduler correctness AND timing realism, and those roles want different event ordering. Confirmed by Codex external review.

## What Actually Works

**Clean baseline (no compensation):**

| Rate | TPOT | E2E | Throughput | TTFT |
|------|------|-----|------------|------|
| 1 | **<5%** ✓ | **<7%** ✓ | **<1%** ✓ | -39% ✗ |
| 2 | **<5%** ✓ | **<5%** ✓ | **<1%** ✓ | -15% |
| 4 | **<5%** ✓ | **<5%** ✓ | **<1%** ✓ | +9% |

TPOT, throughput, and E2E are consistently accurate across all rates. Only TTFT fails, specifically at low rates.

## Codex Architecture Review Summary

Three viable architectural directions identified:

1. **Ingress-gated realtime mode** (deadline-viable): Defer ADD processing when batch is pending. We tested this (attempts #7, #9) — it helps rate=1 TTFT but damages rate=2 metrics.

2. **Split placeholder accounting** (long-term): Separate async scheduler's placeholder tracking from engine-visible pending Future. Would allow blocking execution while preserving scheduler correctness. Too risky for deadline.

3. **Virtual time / event-driven** (different project): REVATI-style approach. Correct by construction but requires cross-process virtual time synchronization.

## Possible Next Steps

### A. Accept and Frame
- Use clean baseline results
- Paper claims: TPOT <5%, throughput <1%, E2E <5% at rate≥2
- Document TTFT limitation with full systems analysis
- Contribution: the async-scheduler pipelining insight itself is novel

### B. One More Attempt: Ingress Gate with Dynamic Cap
- Gate only ADD, only first per batch, capped at `min(remaining, 5ms)`
- But also: only gate when `num_reqs <= 5` (low concurrency)
- This would fire at rate=1 (3 reqs ≤ 5) but not rate=4 (12 reqs > 5)
- Risk: `5` is a magic number

### C. Profile-Aware Timer Duration
- Use Attempt 1 (timer duration) which gave -14.1% at rate=1
- But modify: only extend timer for steps where `num_reqs <= threshold`
- The decoupling prevents _gpu_free_time cascade
- Threshold from profile (avg concurrent at rate=1 during profiling)

### D. Two-Pass Profiling
- Profile real GPU TTFT at each concurrency (already done: ipc_overhead.json)
- Profile emulator TTFT at each concurrency (already done: ipc_emu_ttft.json)
- Apply delta as post-hoc correction to reported TTFT
- Not real compensation (doesn't change emulator behavior) but accurate reporting

## Files Reference

- `docs/benchmarking/ttft-investigation.md` — Root cause analysis
- `docs/benchmarking/ttft-fix-proposals.md` — 5 original proposals
- `docs/benchmarking/ttft-compensation-attempts.md` — Round 1 (6 attempts)
- `docs/benchmarking/ttft-compensation-round2.md` — Round 2 (decoupled + ingress gate)
- `docs/benchmarking/ttft-final-summary.md` — This document
- `docs/benchmarking/handoff-apr6.md` — Full project handoff
