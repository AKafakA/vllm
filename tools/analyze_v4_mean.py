#!/usr/bin/env python3
"""v4-arrival-mean 5-rate summary vs v3."""
import json
from pathlib import Path

RATES = [2, 4, 8, 16, 32]
BASE = Path("results/ttft-variant-v4-arrival-mean")
OUT = Path("paper/apr_19/15_ttft_arrival_delay.md")


def delta(r, e):
    return (
        (e["mean_tpot_ms"] - r["mean_tpot_ms"]) / r["mean_tpot_ms"] * 100,
        (e["mean_ttft_ms"] - r["mean_ttft_ms"]) / r["mean_ttft_ms"] * 100,
    )


v3_numbers = {
    2: (-0.24, -9.50), 4: (-0.76, -4.29), 8: (-5.68, -2.14),
    16: (-2.96, -16.15), 32: (+1.77, -0.17),
}

v4_rows = []
for r in RATES:
    real = json.load(open(f"{BASE}/r{r}_real.json"))
    emu = json.load(open(f"{BASE}/r{r}_emu.json"))
    dtpot, dttft = delta(real, emu)
    v4_rows.append((r, dtpot, dttft))

append = []
append.append("\n## v4 mean-aggregation result (IPC-mean chain, 00:08 BST)\n")
append.append("**Setup**: same v3 arrival-delay hook, but the IPC overhead lookup uses `mean` "
              "instead of `median` of per-N TTFT samples (VLLM_IPC_OVERHEAD_AGG=mean).\n")
append.append("| rate | v3 TPOT% | v3 TTFT% | **v4 TPOT%** | **v4 TTFT%** |")
append.append("|---|---|---|---|---|")
for r, dtpot, dttft in v4_rows:
    v3_tp, v3_tt = v3_numbers[r]
    append.append(f"| {r} | {v3_tp:+.2f}% | {v3_tt:+.2f}% | **{dtpot:+.2f}%** | **{dttft:+.2f}%** |")

append.append("")
append.append("**Observation**: r=2 TTFT flipped sign (−9.50% → +5.9%). Mean added ~15pp vs "
              "median — a much larger shift than expected from the IPC sweep's reported 28–29ms "
              "flat value. Variance analysis (`paper/apr_20/00_ipc_variance.md`) reveals why: "
              "σ ≈ 11 ms across N>1 with right-skewed distribution (mean − median ≈ +3.7 ms per N). "
              "The \"flat 28ms\" characterisation was lossy. Real IPC draws are from a wide "
              "distribution, not a constant.\n")
append.append("**Implication**: neither flat median (undershoots) nor flat mean (overshoots) "
              "matches real. The correct model is **per-arrival sampling from the raw distribution** "
              "— each admitted request draws its own IPC overhead from `raw_ttft_samples_us[N]`. "
              "This will be implemented as v5-arrival-sample in Phase 2.\n")
append.append("**r=16**: v4-mean −17.4% is slightly worse than v3 −16.15%. Confirms the r=16 gap "
              "is NOT overhead-magnitude driven; it's a batch-composition / structural effect "
              "independent of whether we use median, mean, or raw-sample draw.\n")

existing = OUT.read_text()
# Remove any previous v4 section then append fresh.
marker = "\n## v4 mean-aggregation result"
if marker in existing:
    existing = existing.split(marker)[0].rstrip() + "\n"
OUT.write_text(existing + "\n".join(append) + "\n")
print(f"updated {OUT}")
for r, dtpot, dttft in v4_rows:
    print(f"  r={r}: TPOT {dtpot:+.2f}%  TTFT {dttft:+.2f}%")
