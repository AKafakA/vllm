#!/usr/bin/env python3
"""Compare back-to-back real vs emu results for a given label."""
import json
import os
import sys

RESULT_DIR = "/workspace/eval_results/RTX-3060-12GB"
label = sys.argv[1] if len(sys.argv) > 1 else "1.5b-tp1"

print(f"{'Config':<25} {'TTFT':>8} {'P99TTFT':>9} {'TPOT':>8} {'P99TPOT':>9}")
print("-" * 62)
for rate in [1, 2, 4]:
    for pfx in ["real", "emu"]:
        f = f"{RESULT_DIR}/online/b2b_{pfx}_{label}_rate{rate}.json"
        if os.path.exists(f):
            d = json.load(open(f))
            print(f"{pfx+' rate='+str(rate):<25} "
                  f"{d['mean_ttft_ms']:>8.1f} {d['p99_ttft_ms']:>9.1f} "
                  f"{d['mean_tpot_ms']:>8.1f} {d['p99_tpot_ms']:>9.1f}")

print()
all_pass = True
for rate in [1, 2, 4]:
    rf = f"{RESULT_DIR}/online/b2b_real_{label}_rate{rate}.json"
    ef = f"{RESULT_DIR}/online/b2b_emu_{label}_rate{rate}.json"
    if os.path.exists(rf) and os.path.exists(ef):
        r, e = json.load(open(rf)), json.load(open(ef))
        te = (e["mean_ttft_ms"] - r["mean_ttft_ms"]) / r["mean_ttft_ms"] * 100
        pe = (e["mean_tpot_ms"] - r["mean_tpot_ms"]) / r["mean_tpot_ms"] * 100
        tok = "✓" if abs(te) < 5 else "✗"
        pok = "✓" if abs(pe) < 5 else "✗"
        if abs(te) >= 5 or abs(pe) >= 5:
            all_pass = False
        print(f"  rate={rate}: TTFT {te:>+6.1f}% {tok}  TPOT {pe:>+6.1f}% {pok}")

if all_pass:
    print("\nALL METRICS UNDER 5% — PASS")
else:
    print("\nSOME METRICS ABOVE 5% �� NEEDS INVESTIGATION")
