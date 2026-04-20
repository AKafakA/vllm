# Apr 19 — TTFT arrival-delay hook (v3)

## Mechanism

Patched vLLM `Scheduler.__init__` loads `vllm_emulator.hooks.scheduler_hook` via
env var `VLLM_EMULATOR_SCHEDULER_HOOK=1`. The hook wraps three methods on the
scheduler instance:

- `add_request` — stashes new request in `_emu_pending_arrivals` with
  `admission_time = now + IPC_overhead(current_concurrency)`. Overhead comes
  from `sched_overhead_table` in the profile pack (measured on real hardware
  by `tools/profile_ipc_overhead.py`; flat ~28ms across N=1..256 on RTX 8000).
- `schedule` — drains pending arrivals whose admission time has passed into
  the real waiting queue, then delegates to original `schedule`.
- `get_num_unfinished_requests` — includes pending arrivals so the engine
  loop keeps iterating (otherwise `has_requests()` returns False and the
  engine skips `schedule()`, stranding the pending list forever).

The oracle's `estimate_step_latency_us` is unchanged (IPC injection reverted).
Step latency stays clean → no chain-timer contamination → TPOT untouched by
the mechanism in the first-order model.

## 5-rate results (random 256/128, 2000 prompts, archive-r2 profile + sched_overhead_table)

| rate | TPOT real (ms) | TPOT emu (ms) | ΔTPOT% | TTFT real (ms) | TTFT emu (ms) | ΔTTFT% |
|---|---|---|---|---|---|---|
|  2 |  33.91 |  33.83 | **−0.24%** |    144.96 |    131.20 | **−9.50%** |
|  4 |  46.42 |  46.06 | **−0.76%** |    184.83 |    176.90 | **−4.29%** |
|  8 |  72.51 |  68.39 | **−5.68%** |    271.60 |    265.78 | **−2.14%** |
| 16 | 181.58 | 176.20 | **−2.96%** |  29989.93 |  25147.02 | **−16.15%** |
| 32 | 176.19 | 179.30 | **+1.77%** |  41503.28 |  41433.58 | **−0.17%** |

## Comparison to prior approaches

| rate | archive-r2 TPOT | v2-additive-divide TPOT | **v3 TPOT** | archive-r2 TTFT | v2 TTFT | **v3 TTFT** |
|---|---|---|---|---|---|---|
|  2 | −0.02% | +6.93%  | **−0.24%** | −31.65% | −9.49% | **−9.50%** |
|  4 | −0.56% | +14.68% | **−0.76%** | −29.16% | −7.77% | **−4.29%** |
|  8 | −2.52% | +14.28% | **−5.68%** | −23.47% | −11.09% | **−2.14%** |
| 16 | +0.04% | +4.93%  | **−2.96%** |  −6.35% | +1.06% | **−16.15%** |
| 32 | +1.11% | +6.32%  | **+1.77%** |  −1.14% | +4.21% | **−0.17%** |

**Accuracy vs project targets (TPOT ≤5%, TTFT ≤15%):**

- **TPOT**: v3 passes 4/5 (r=8 borderline at −5.7%); archive-r2 passes 5/5; v2 fails 5/5.
- **TTFT**: v3 passes 4/5 (r=16 borderline at −16.2%); archive-r2 passes 2/5; v2 passes 5/5 but trades TPOT.

**v3 is the new best overall** — dramatic TTFT improvement (~20–25pp closer to zero at low rates) with near-clean TPOT, except at the saturation edge cases noted below.

## Second-order interactions (honest reporting)

The hook's first-order model is "admission delay only affects TTFT." Empirically,
we see drift on TPOT at r=8 and TTFT at r=16.

**r=8 TPOT −5.68% (vs baseline −2.52%, +3pp drift)**: The admission delay
smooths arrival bursts. In real vLLM, new requests enter the waiting queue
immediately, and the next step runs a MIXED prefill+decode batch that
contends for GPU and slows decode tokens. In emu v3, new requests are
stashed 28ms; during that window the scheduler runs decode-only steps
(no prefill contention). Net effect: slightly less prefill–decode mixing
than real → emu's average TPOT ticks down. Becomes visible at r=8+ where
arrival bursts are dense.

**r=16 TTFT −16.15% (vs baseline −6.35%, −10pp drift)**: At r=16, real
TTFT is 30s — saturated queueing regime. Queue wait time dominates the
28ms IPC overhead. Arrival-delay dilutes emu's sustained concurrency
slightly, which reduces queue growth rate → smaller queue wait → emu TTFT
drifts lower. Same mechanism as the r=8 TPOT drift.

**r=32 lands cleanest** (TPOT +1.77%, TTFT −0.17%) because at full
saturation, the queue wait is so large (~41s) that every second-order
effect gets absorbed.

## Architectural note: the "right" fix for r=16

First-order model wants: new request enters batch for scheduling purposes
at t=0 (same concurrency pressure as real), but its first-token response
emission is delayed by 28ms. Current hook puts the request at
`admission_time = t+28ms` for BOTH purposes. A cleaner model would admit
immediately but hold first-token buffering downstream — requires hooking
into the output_processor path, not the scheduler. Deferred.

## Verdict

**v3 arrival-delay is the winning variant for this iteration.** Oracle reverted
to clean state. Scheduler-layer IPC overhead is architecturally the right
place — no chain-timer contamination, step latency stays profile-driven,
only new-request admission pays the measured IPC cost. Second-order drifts
at r=8 TPOT and r=16 TTFT are understood mechanistically and bounded.

## Design pattern note (for future refactor)

The current touch-point in `vllm/v1/core/sched/scheduler.py` is an env-var
gated import of the emulator hook. For a cleaner upstream submission, the
pattern to adopt is vLLM's existing `scheduler_cls` / `worker_cls` style:

```python
# vLLM side — one generic line at end of Scheduler.__init__:
_load_scheduler_plugins(self)

# Utility (scheduler/plugin.py, ~15 lines):
def _load_scheduler_plugins(scheduler):
    names = os.environ.get("VLLM_SCHEDULER_PLUGINS", "").strip()
    for dotted in names.split(","):
        if dotted:
            importlib.import_module(dotted).apply(scheduler)

# Emulator side — stable plugin ABI:
def apply(scheduler):  # vllm_emulator.hooks.scheduler_hook.apply
    install_arrival_delay(scheduler)
```

Activates via `VLLM_SCHEDULER_PLUGINS=vllm_emulator.hooks.scheduler_hook`.
No emulator-specific symbol in vLLM. Compose with multiple plugins. Deferred
to follow-up; current implementation is fine for paper experimentation.

## v4 mean-aggregation result (IPC-mean chain, 00:08 BST)

**Setup**: same v3 arrival-delay hook, but the IPC overhead lookup uses `mean` instead of `median` of per-N TTFT samples (VLLM_IPC_OVERHEAD_AGG=mean).

| rate | v3 TPOT% | v3 TTFT% | **v4 TPOT%** | **v4 TTFT%** |
|---|---|---|---|---|
| 2 | -0.24% | -9.50% | **+0.04%** | **+5.91%** |
| 4 | -0.76% | -4.29% | **-1.15%** | **-3.55%** |
| 8 | -5.68% | -2.14% | **-5.26%** | **-2.12%** |
| 16 | -2.96% | -16.15% | **-2.75%** | **-17.42%** |
| 32 | +1.77% | -0.17% | **+1.08%** | **-0.41%** |

**Observation**: r=2 TTFT flipped sign (−9.50% → +5.9%). Mean added ~15pp vs median — a much larger shift than expected from the IPC sweep's reported 28–29ms flat value. Variance analysis (`paper/apr_20/00_ipc_variance.md`) reveals why: σ ≈ 11 ms across N>1 with right-skewed distribution (mean − median ≈ +3.7 ms per N). The "flat 28ms" characterisation was lossy. Real IPC draws are from a wide distribution, not a constant.

**Implication**: neither flat median (undershoots) nor flat mean (overshoots) matches real. The correct model is **per-arrival sampling from the raw distribution** — each admitted request draws its own IPC overhead from `raw_ttft_samples_us[N]`. This will be implemented as v5-arrival-sample in Phase 2.

**r=16**: v4-mean −17.4% is slightly worse than v3 −16.15%. Confirms the r=16 gap is NOT overhead-magnitude driven; it's a batch-composition / structural effect independent of whether we use median, mean, or raw-sample draw.


## v5 family results (Apr 20 overnight)

Full validation and analysis in `paper/apr_20/03_v5_validation.md`. Short version:

| rate | v3 (stable) | v5-sample | v5-2d-burst | v5-2d-burst-tight |
|---|---|---|---|---|
| 2  TTFT% | −9.50  | **−6.65** | +10.23 | +10.55 |
| 4  TTFT% | −4.29  | — | −1.44  | −3.57 |
| 8  TTFT% | −2.14  | — | +2.61  | −0.59 |
| 16 TTFT% | **−16.15** | — | −22.29 | **−18.50** |
| 32 TTFT% | −0.17  | — | −2.40  | +0.04 |

Findings:
- Per-arrival sampling (v5-sample) improves r=2 by 3pp (r=2 only measured; rest damaged by operator kill).
- Burst-aware 2D lookup (v5-2d-burst-tight) moves r=16 by 4pp vs pipelined and is the direction for r=16 fix, but still 2pp above target.
- v2 k=1 mean disagrees with v1 k=1 median by ~5ms more than variance predicts — suspicious; investigate tomorrow.
- **v3 remains the shipped reference on stable.** No v5 variant promoted.
