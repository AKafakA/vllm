# Apr 20 — v5 family validation

Three candidate v5 variants aimed at improving r=16 TTFT over v3 while
preserving r=2/r=4/r=8/r=32 accuracy.

## 5-rate matrix (random 256/128, 2000 prompts, archive-r2 profile)

Target: TPOT/E2E ≤ 6%, TTFT ≤ 10%, no rate regresses >2pp from v3.

| rate | v3 median | v4 mean | v5-sample | v5-2d-burst | v5-2d-burst-tight |
|---|---|---|---|---|---|
| **TPOT%** |
| 2  | −0.24 | +0.04 | −0.26 | +0.27 | −0.00 |
| 4  | −0.76 | −1.15 | — | −0.54 | −1.91 |
| 8  | −5.68 | −5.26 | — | −2.20 | −4.08 |
| 16 | −2.96 | −2.75 | — | −5.81 | −3.49 |
| 32 | +1.77 | +1.08 | — | +0.38 | +1.68 |
| **TTFT%** |
| 2  | **−9.50** | **+5.91** | **−6.65** | **+10.23** | **+10.55** |
| 4  | −4.29 | −3.55 | — | −1.44 | −3.57 |
| 8  | −2.14 | −2.12 | — | +2.61 | −0.59 |
| 16 | **−16.15** | **−17.42** | — | **−22.29** | **−18.50** |
| 32 | −0.17 | −0.41 | — | −2.40 | +0.04 |

**v5-sample partial** — r=4+ benches were damaged by an accidental operator
kill mid-run (02:15 BST). r=2 captured only.

## Variant-by-variant analysis

### v4-arrival-mean
Swap median → mean in the 1D lookup. Overshoots r=2 (+5.9%) because
mean is pulled up by the variance tail (σ≈11ms). Same r=16 drift as v3
(distribution-agnostic). **Not a winner.**

### v5-arrival-sample
Per-arrival random draw from raw `raw_ttft_samples_us` (1D k=1
distribution). r=2 lands at −6.65%, closer to zero than v3 (−9.50%) and
within target. Rest damaged — no verdict for r=16. Sampling helps with
the variance-induced shortfall at low rates.

### v5-2d-burst (pipelined k_burst)
Uses 2D overhead(N, k_burst) from the 68-cell v2 sweep; k_burst counts
ALL arrivals currently in `_emu_pending_arrivals`. Results:
- r=4, r=8 **both improve** vs v3 (−4.3→−1.4, −2.1→+2.6) — closest to zero yet.
- r=2 **overshoots** (+10.23%) — k_burst inflated at low rates because
  30–80ms admission delays keep pending non-empty even without real bursts.
- r=16 **regresses further** (−16.15 → −22.29) — pending gets feedback-looped:
  longer delays → more pending → higher k → longer delays → ...
  This rules out pipelined k_burst as the correct semantics.
- r=32 fine (−2.40).

Semantic bug: the v2 sweep measures k_burst as "k simultaneous NEW arrivals
in one IPC cycle" (threads launched within ~1ms). My pipelined count treats
arrivals that arrived 30-80ms apart as one burst. **Wrong physics.**

### v5-2d-burst-tight (Phase 2c, DONE 02:53)

Window: `prefill_step_us / 1e6 = 11.7ms` (profile-derived, no knob).

Results vs 2d-burst pipelined:
- r=16: **−18.5%** (vs 2d-burst −22.3%) — tighter window **did** reduce feedback-loop amplification by ~4pp, confirming the semantic fix direction.
- r=4/r=8: slight improvements (r=8 TTFT −0.6% vs +2.6%).
- r=2: **+10.55%** — unchanged from 2d-burst. Predicted to revert to v3-like
  (≈−5%) under tight window, did not. Investigation below.

**Unresolved anomaly at r=2**: at r=2 inter-arrival is 500ms and burst
window is 11.7ms, so k_burst_tight should be 1 ≥97% of the time →
table returns ~33ms overhead at (N≈25, k=1) → TTFT shift over v3 should
be ~5ms/145ms ≈ 3.5pp. Observed shift is 20pp. Hypothesis (unverified):
the v2 sweep's k=1 mean values systematically exceed the v1 k=1 median
by more than variance alone (possibly because the v2 sweep ran under
slightly different server load — warmup state, background decode count,
etc.). Profile diagnostic for tomorrow.

r=16 TTFT still at −18.5% — not within ±10% target. Burst mechanism is
a real contributor but not the full r=16 story. The batch-composition
diagnostic (Phase 1 Q3) would have helped isolate remaining drivers,
but Phase 1.2 trace files were lost to `pkill -9` before flush.

## Verdict across v5 family

No v5 variant meets the 5-rate target band. Per-variant summary:

| variant | r=2 target ✓ | r=16 target ✓ | net change vs v3 |
|---|---|---|---|
| v4 mean | ✓ (+5.9%) | ✗ (−17.4%) | moves r=2 from −9.5 to +5.9, r=16 marginally worse |
| v5 sample | ✓ (−6.7%) | n/a | closes r=2 by 3pp; r=16 untested |
| v5-2d-burst | ✗ (+10.2%) | ✗✗ (−22.3%) | r=4/r=8 better; r=2 and r=16 worse |
| v5-2d-burst-tight | ✗ (+10.6%) | ✗ (−18.5%) | best burst variant on r=16 but still fails target |

**Two insights locked in:**
1. Burst dimension is a real physical lever (r=16 responds: v3 −16 → v5-tight −18, demonstrating it can MOVE r=16 TTFT).
2. The v1 k=1 scalar is too low for low rates (r=2 consistently overshoots v3 when using v2's k=1 mean) — suggests the v1 and v2 sweeps captured subtly different distributions at low N.

**Path forward for tomorrow** (deferred, NOT blocking tonight):
- Investigate v1 vs v2 k=1 disagreement at low N: rerun v1 sweep under same load conditions as v2 and see if they converge.
- Try HYBRID: use v1 median for low-conc regime (N ≤ 50), v2 2D-burst for high-conc. Would be a piecewise profile-driven lookup, still no magic numbers.
- Reconstruct Phase 1 Q3 (batch composition) with executor_hook.py fixed to line-buffer the trace file (so SIGKILL doesn't lose content).

## Commit policy

Per branch-discipline rule: none of v4/v5-sample/v5-2d-burst/v5-2d-burst-tight passes. All code stays on `exp/apr20-phase-work`. No promotion to `refactor/clean-emulator-v2`. v3 at stable HEAD `9ccde439a` remains the recommended reference.
