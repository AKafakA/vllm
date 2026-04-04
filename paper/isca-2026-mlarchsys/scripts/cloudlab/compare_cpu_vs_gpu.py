#!/usr/bin/env python3
"""Compare CPU emulator results against GPU real baseline."""
import json
import os
import sys

# CPU emulator results (from CloudLab)
cpu_dir = os.path.expanduser("~/vllm-emulator/results")
# GPU real baseline results (copied from Vast)
gpu_dir = os.path.expanduser("~/vllm-emulator/gpu_baseline")

print(f"{'Config':<25} {'TTFT':>8} {'TPOT':>8}")
print("-" * 44)

for rate in [1, 2, 4]:
    # GPU real
    gf = f"{gpu_dir}/cluster_real_rate{rate}.json"
    if os.path.exists(gf):
        g = json.load(open(gf))
        print(f"{'GPU real rate='+str(rate):<25} {g['mean_ttft_ms']:>8.1f} {g['mean_tpot_ms']:>8.1f}")

    # CPU emu
    cf = f"{cpu_dir}/cpu_emu_rate{rate}.json"
    if os.path.exists(cf):
        c = json.load(open(cf))
        print(f"{'CPU emu rate='+str(rate):<25} {c['mean_ttft_ms']:>8.1f} {c['mean_tpot_ms']:>8.1f}")

    # GPU emu (from Vast b2b results if available)
    ef = f"{gpu_dir}/b2b_emu_0.5b-tp1_rate{rate}.json"
    if os.path.exists(ef):
        e = json.load(open(ef))
        print(f"{'GPU emu rate='+str(rate):<25} {e['mean_ttft_ms']:>8.1f} {e['mean_tpot_ms']:>8.1f}")
    print()

print("Error analysis:")
for rate in [1, 2, 4]:
    gf = f"{gpu_dir}/cluster_real_rate{rate}.json"
    cf = f"{cpu_dir}/cpu_emu_rate{rate}.json"
    if os.path.exists(gf) and os.path.exists(cf):
        g, c = json.load(open(gf)), json.load(open(cf))
        te = (c["mean_ttft_ms"] - g["mean_ttft_ms"]) / g["mean_ttft_ms"] * 100
        pe = (c["mean_tpot_ms"] - g["mean_tpot_ms"]) / g["mean_tpot_ms"] * 100
        tok = "✓" if abs(te) < 5 else "✗"
        pok = "✓" if abs(pe) < 5 else "✗"
        print(f"  rate={rate}: TTFT {te:+.1f}% {tok}  TPOT {pe:+.1f}% {pok}")
