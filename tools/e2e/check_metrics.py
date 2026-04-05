"""Print all numeric metrics from bench serve result files."""
import json
import sys
import os

def show_metrics(path):
    if not os.path.exists(path):
        print(f"  NOT FOUND: {path}")
        return
    d = json.load(open(path))
    print(f"  File: {os.path.basename(path)}")
    for k in sorted(d.keys()):
        v = d[k]
        if isinstance(v, float):
            print(f"    {k}: {v:.2f}")
        elif isinstance(v, int):
            print(f"    {k}: {v}")

rd = "/workspace/eval_results/RTX-3060-12GB/online"
print("=== Real rate=1 ===")
show_metrics(f"{rd}/a2a_real_rate1.json")
print("\n=== Emu rate=1 ===")
show_metrics(f"{rd}/a2a_emu_rate1.json")
