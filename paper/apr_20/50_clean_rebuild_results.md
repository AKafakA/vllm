# Apr 20 PM — clean profile/bench rebuild

Plan: `.claude/plans/it-looks-good-let-unified-steele.md`.

## Progress Log
- 19:11:36 BST — Phase 1 clean baseline DONE: 15/15 benches (3 configs × 5 rates, fresh server per bench, seed=0). All JSONs present, no FAIL lines.
- 19:40 BST — clean IPC sweep launched (chain_ipc_sweep_clean.sh --mode=clean --max-n=256 --burst-k="1 2 4 8" --samples=5); ETA ~35 min.
- 00:43:15 BST — adaptive profile (per-rate fresh server, 2 rounds, matched warmup) DONE: 1451 cells, 211,800 samples, tt range 1-2048, 52 conc buckets. Trace 227,478 lines (~2× archive-r2 density). Output: results/RTX-8000-adaptive-apr20-clean/serving-full.json.
- 00:47 BST — merge step: injected 13 IPC-overhead cells from existing ipc_overhead_v2.json into new profile pack. Backup at serving-full.json.bak.
- 00:48 BST — validation chain staged on remote (chain_validate_5rate_clean.sh); not launched — awaiting user go.
- 01:44:06 BST — validation DONE (5/5 emu hookOn benches, fresh server each, new profile). Results vs Phase 1 real:
  - **TPOT: 5/5 PASS** (max |Δ| = 5.24% at r=16); E2E 4/5 (r=16 −11.51%); ITL 4/5 (r=32 +39.92%); TTFT 3/5 (r=2 −11.57%, r=16 −16.28%).
  - New profile improves r=4/r=8 dramatically (r=8 TTFT +15% → +7.6%) but regresses r=16 (TTFT −2.7% → −16.3%, E2E −2.3% → −11.5%) and r=32 ITL (+0.1% → +40%).
  - Hypothesis: IPC overhead table was captured against OLD profile's prefill baseline; the new profile's saturation-regime predictions are decoupled from the stale IPC table. Next step: rerun patched v3 IPC sweep against new profile baseline.
- 06:40:05 BST (Apr 21) — apr21-dense adaptive profile DONE. 1711 cells (+18% vs apr20-clean), 224,800 samples, trace 242,890 lines. Density at target conc 112-237 boosted +100% to +700% in decode buckets and +100% to +360% in prefill buckets (where rates 14, 18 dwell).
- 06:40 BST — Phase 2 merge done: 13 IPC cells injected from archive-r2 IPC table into new profile. Backup saved.
- 06:40 BST — Phase 3a exp1 (NEW profile + default oracle K=1) validation started. ~45min.
- 07:26 BST — **exp1 (NEW profile + default oracle) DONE**: r=16 TTFT closed from -16.28% → -8.59% (PASS). r=8 slight regression +7.64% → +10.55% (barely FAIL 0.55pp). r=2 unchanged -11.91% (FAIL, low-conc not addressed). r=32 ITL unchanged +39.58% (bimodal persists). Pass counts: TPOT 5/5, ITL 4/5, E2E 4/5, TTFT 3/5.
- 08:13 BST — **exp3 (NEW profile + KNN K=4) DONE**: mixed result. r=8 TTFT closes (+10.55% → +7.78% ✓) but r=16/r=32 BREAK dramatically (r=16 TTFT -8.59% → +17.00%, r=32 TPOT -1.27% → +7.61%). K=4 pools too aggressively at saturation — includes neighbor-bucket outliers from rate-ramp transients. Validates need for ADAPTIVE K (sample-count floor), not uniform K=4. Pass counts: TTFT 2/5, TPOT 3/5, E2E 3/5 — strictly worse than exp1's 3/5 TTFT / 5/5 TPOT / 4/5 E2E.
- 09:47 BST — **exp4 (apr21-dense + adaptive-K oracle, K=auto M=30) DONE — NEW LEADER 13/15**: r=16 all three pass (TTFT -3.40%, TPOT -1.54%, E2E -5.11%); r=8 all pass; only r=2 TTFT -11.62% (1.62pp over) and r=32 E2E +6.94% (0.94pp over) fail. Worst-case magnitude 1.62pp. Adaptive-K BREAKS the r=8/r=16 tradeoff observed in all prior uniform-K configs.
- 10:33 BST — **exp5 (response-side IPC) FAIL**: TTFT systematically under-predicted at low rates (-34%, -26%, -18% at r=2/4/8). Removing arrival-delay hook without full compensation in oracle broke TTFT magnitude. Response-side IPC as implemented does not work. Revert to arrival-delay for production configs.
- 12:00 BST — **exp7 (apr21-dense-iqr + KNN K=4) FAIL 6/15**: IQR filter dropped 10% tail samples at dense buckets (r=2/r=4 have 47-55k decode samples). Removing "outliers" removed legitimate tail latencies → bucket means shifted 15-20% downward → emu under-predicts at r=2/r=4/r=8. r=16 passes (as expected — IQR helped at sparse buckets) but r=2/r=4 broken. Worst case -20%. **IQR is the WRONG lever.** Outliers at dense buckets are not outliers — they're real workload variance.
- Paper verdict: **exp4 (apr21-dense + adaptive-K, K=auto M=30) remains the winner: 13/15 pass, worst-case 1.62pp** at r=2 TTFT.
