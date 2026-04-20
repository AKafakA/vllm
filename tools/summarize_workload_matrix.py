#!/usr/bin/env python3
"""Build 4-cell workload matrix summary from archive-r2 and archiver2ext runs."""
import json
from pathlib import Path

CELLS = {
    "archive-r2 × random":   "workload-emu-archive-r2-random",
    "archive-r2 × sharegpt": "workload-emu-archive-r2-sharegpt",
    "archiver2ext × random":   "workload-emu-archiver2ext-random",
    "archiver2ext × sharegpt": "workload-emu-archiver2ext-sharegpt",
}

print(f"{'cell':<26} | {'rate':>4} | {'TPOT%':>8} | {'TTFT%':>8} | {'E2E%':>8}")
print("-" * 80)
for label, d in CELLS.items():
    base = Path(f"/home/wd312/Code/llm/vllm-emulator/results/{d}")
    for R in (2, 8, 32):
        rp = base / f"r{R}_real.json"
        ep = base / f"r{R}_emu.json"
        if rp.exists() and ep.exists():
            r = json.load(open(rp))
            e = json.load(open(ep))
            dtpot = (e["mean_tpot_ms"] - r["mean_tpot_ms"]) / r["mean_tpot_ms"] * 100
            dttft = (e["mean_ttft_ms"] - r["mean_ttft_ms"]) / r["mean_ttft_ms"] * 100
            de2e = (e["mean_e2el_ms"] - r["mean_e2el_ms"]) / r["mean_e2el_ms"] * 100
            print(f"{label:<26} | {R:>4} | {dtpot:>+7.2f}% | {dttft:>+7.2f}% | {de2e:>+7.2f}%")
        else:
            print(f"{label:<26} | {R:>4} | {'MISSING':>8} | {'':>8} | {'':>8}")
    print("-" * 80)
