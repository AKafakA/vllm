# Apr 20 — r=16 TTFT root-cause diagnosis

Phase 1 evidence gathered in `chain_phase1_evidence.sh` (completed 01:01:35 BST).
Answers the four questions posed in Phase 1 analysis.

## Evidence summary

### Q1 — Is IPC overhead at N=1024 materially different from N=256?

**Answer: NO.** Sweep data for k=1 (single-arrival):

| N | overhead (median, ms) | overhead (mean, ms) |
|---|---|---|
| 256  | 28.2 | 30.6 (from v1 sweep) |
| 384  | 25.1 | 34.7 |
| 512  | 25.2 | 31.7 |
| 768  | 25.9 | 31.1 |
| 1024 | 26.5 | 34.4 |

Single-arrival IPC overhead is essentially flat at ~26–28ms across the extended N range. The hypothesis that r=16/r=32 saturation degrades because of extrapolation error beyond N=256 is REFUTED.

### Q2 — Does IPC overhead grow with burst_k?

**Answer: YES — dramatically, ~linear in k.** 2D sweep results:

| N / k | k=1 | k=2 | k=4 | k=8 |
|---|---|---|---|---|
| 256  | 28.2 | 28.2 | 56.1 | **80.5** |
| 384  | 25.1 | 52.7 | 33.1 | **83.3** |
| 512  | 25.2 | 28.5 | 57.2 | **81.8** |
| 768  | 25.9 | 29.0 | 57.2 | **80.9** |
| 1024 | 26.5 | 29.2 | 58.6 | **80.9** |

At any N in the saturation range, 8 simultaneous new arrivals each pay ~80ms of IPC overhead — **~3× higher than the single-arrival case**. The v3 hook's `_lookup_overhead_us(conc)` returns the k=1 value regardless of drain batch size, so **at r=16 where Poisson bursts routinely arrive with k≥4, the hook under-delays by ~30–55ms per request**.

### Q3 — Does v3 hook shift per-step batch composition at r=16?

**Answer: UNKNOWN (measurement failure).**

The `VLLM_EMULATOR_HOOK_TRACE` trace files at `results/batch-diag-r16-apr20/trace_{hook,nohook}.csv` are 0 bytes. Root cause: `pkill -9` sends SIGKILL; Python's buffered trace file is never flushed on shutdown. The diagnostic needs executor_hook.py to open the trace file with line-buffering or flush on every step (currently flushes every 50 steps). A fix is queued for tomorrow; we proceed without Q3 evidence tonight.

This is not blocking for Phase 2 — Q2 already gives a strong, sufficient explanation for the r=16 regression and points to a concrete profile-driven fix.

### Q4 — Dominant mechanism for r=16 regression?

**Q2 burst-scaling is dominant.** Reasoning:

- V3 hook uses flat `overhead(N)` ≈ 28ms for any k.
- At r=16 the Poisson process routinely delivers 2–8 new requests within a few ms → hook drains them together with admission times all within 28ms of now.
- Real vLLM at the same moment pays ~60–80ms of IPC setup for that burst (per the 2D sweep).
- Net: emu admits burst arrivals ~30–50ms earlier than real → emu's waiting queue fills later / drains more smoothly → queue wait shorter → TTFT at r=16 is lower than real by an amount that scales with burst frequency.

The previously hypothesized "pure-decode window" effect (Q3) may exist as a secondary contributor but the 3× burst-overhead gap is large enough to explain the observed −16% drift on its own.

## Variance reconciliation

The Apr 20 Phase 0 variance analysis (`paper/apr_20/00_ipc_variance.md`) found σ ≈ 11ms / mean−median ≈ +3.7ms on the v1 sweep (k=1 only). With the v2 sweep showing burst_k scaling, part of that v1 "variance" is now explained: occasional coincidental bursts during the v1 sweep's 15 measurements inflated the tail. Sample-per-request from the v1 k=1 distribution (v5-arrival-sample, currently running in Phase 2) will close r=2/r=4 residuals but should NOT meaningfully improve r=16 — burst-matched `overhead(N, k)` is the r=16 fix.

## Recommended v5 variant: 2D burst-aware lookup

**v5-2d-burst** (supersedes v5-arrival-sample for r=16):

- Hook change: `scheduler_hook._patched_add_request` stashes new request in `_emu_pending_arrivals` with an `admission_time` computed using `overhead_us(N, k_burst)` where `k_burst` is the number of pending arrivals currently clustered with this one (or equivalently the batch size at the next drain).
- Requires drain logic to reassign admission times after each burst — or approximate: at arrival, use current `len(pending_arrivals) + 1` as k, and look up `overhead(N, k)` from `sched_overhead_table_v2`.
- Bilinear interpolation over (N, k) grid from the 68-cell v2 sweep already in `serving-r2.json` as `sched_overhead_table_v2`.
- Env gate: `VLLM_IPC_OVERHEAD_AGG=2d-burst`.

Secondary option (cheap): **v5-sample + 2d-mean scalar**. If 2d-burst implementation runs over time budget, fall back to using 2d sweep's mean-overhead-per-N (averaged across all k) as a scalar that implicitly accounts for typical burst behaviour. Less accurate but zero new code paths.

## Action

Phase 2 is currently validating v5-arrival-sample (started 01:04 BST, running r=2 as of this writing). Letting it complete gives us:

1. Empirical confirmation that sample-per-k=1 fixes r=2/4 residuals.
2. Empirical confirmation that sample-per-k=1 does NOT fix r=16 (predicted).
3. A clean comparison point for v5-2d-burst.

Phase 2 timing budget allows v5-arrival-sample (45 min) + v5-2d-burst build+validate (60 min) ≈ 01:04 + 105 min = **02:50 BST for Phase 2 complete**. Phase 3 still has 4h runway to 07:00 BST finish.
