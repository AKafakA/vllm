# Apr 19 — TTFT overnight variant chain

## Setup

- Profile: archive-r2 with `sched_overhead_table` merged in (IPC sweep 19:17).
- Variant v2-additive-divide: oracle adds `IPC_overhead(N)/num_new_reqs` to prefill-step
  latency. Async chain-timer architecture preserved (no synchronous engine-thread sleep).
- Hardware: RTX 8000, Model: Qwen/Qwen3-8B, 2000 prompts × 5 rates (random 256/128).

## Variant 5-rate matrix

| rate | archive-r2 TPOT% | archive-r2 TTFT% | IPC-current TPOT% | IPC-current TTFT% | v2 TPOT% | v2 TTFT% |
|---|---|---|---|---|---|---|
| 2 | -0.02% | -31.65% | +6.92% | -8.67% | +6.93% | -9.49% |
| 4 | -0.56% | -29.16% | +16.76% | -4.20% | +14.68% | -7.77% |
| 8 | -2.52% | -23.47% | +15.95% | -5.87% | +14.28% | -11.09% |
| 16 | +0.04% | -6.35% | +5.82% | +8.15% | +4.93% | +1.06% |
| 32 | +1.11% | -1.14% | +6.57% | +3.01% | +6.32% | +4.21% |

## Verdict

**Winner: archive-r2 (no IPC injection) remains the best overall setting.**

- **v2 vs IPC-current**: v2 mildly improves TPOT at mid-rates (r=4 +14.7 vs +16.8; r=8 +14.3 vs +16.0) and mildly improves TTFT (r=4 −7.8 vs −4.2; r=8 −11.1 vs −5.9). Division by num_new_reqs does *not* rescue the TPOT regression introduced by additive overhead in the oracle latency path.
- **v2 vs archive-r2**: archive-r2 wins TPOT on every rate (|Δ|≤2.5% vs v2's +5 to +15%). v2 wins TTFT at r=2/4/8 (smaller magnitude gap) but overshoots TTFT at r=16 (+1%) and r=32 (+4%) where archive-r2 had it close to zero.
- **Architectural implication**: additive oracle-latency injection contaminates scheduler feedback (downstream chain-timer cascades). The correct place for per-request IPC overhead is either (a) a synchronous block in the engine thread at prefill-step boundaries (blocks `num_output_placeholders` the way real IPC does), OR (b) left absent — accepting the TTFT gap as a known residual and investing that effort elsewhere.
- **Tonight's decision**: revert oracle to pre-IPC state; keep `sched_overhead_table` captured in profile pack for future architectural-fix variants (Apr 20 Direction A).

## Comparison to archive-r2 baseline shape-sensitivity (Exp 2)

The TTFT gap of −32 to −23% at r=2/4/8 in archive-r2 is the *mechanistic TTFT gap*. Exp 2 on filtered sharegpt (paper/apr_19/13_exp2_fullrate.md) shows TPOT deltas up to +110% at r=16 from *shape generalisation*. The two failure modes are orthogonal — fixing one does not improve the other. Apr 20 focus should be shape-generalisation (α-KV model extension), since it affects more metrics and is a more fundamental oracle model error.
