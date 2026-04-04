#!/usr/bin/env python3
"""Build serving profile for 3B TP=2 from step cycle trace."""
import json
import statistics
from collections import defaultdict

RESULT_DIR = "/workspace/eval_results/RTX-3060-12GB"
STEP_CYCLE_FILE = f"{RESULT_DIR}/step_cycle_3b_tp2.jsonl"
SWEEP_PROFILE = f"{RESULT_DIR}/profiles/sweep-3b-tp2.json"
OUT = f"{RESULT_DIR}/profiles/serving-3b-tp2-step-cycle.json"

records = []
for line in open(STEP_CYCLE_FILE):
    r = json.loads(line)
    if "total_tokens" in r:
        records.append(r)

print(f"Records with batch info: {len(records)}")

by_tt = defaultdict(list)
for r in records:
    by_tt[r["total_tokens"]].append(r["step_cycle_us"])

forward_pass = []
for tt in sorted(by_tt):
    lats = by_tt[tt]
    if len(lats) >= 2:
        forward_pass.append({
            "total_tokens": tt,
            "latency_us": round(statistics.median(lats), 1),
            "num_samples": len(lats),
        })

# Merge with sweep profile for large tt
max_tt = max(e["total_tokens"] for e in forward_pass) if forward_pass else 0
try:
    sweep = json.load(open(SWEEP_PROFILE))
    for e in sweep["forward_pass"]:
        if e["total_tokens"] > max_tt:
            forward_pass.append(e)
except FileNotFoundError:
    print("Warning: no sweep profile to merge")

profile = {
    "gpu_model": "RTX-3060-12GB-TP2",
    "model_name": "Qwen/Qwen2.5-3B-Instruct",
    "profile_type": "serving_step_cycle",
    "forward_pass": sorted(forward_pass, key=lambda e: e["total_tokens"]),
}
json.dump(profile, open(OUT, "w"), indent=2)
print(f"Serving profile: {len(forward_pass)} buckets")
for e in profile["forward_pass"][:15]:
    print(f"  tt={e['total_tokens']:>4}: {e['latency_us']/1000:.1f}ms (n={e.get('num_samples',0)})")

print(f"\nSaved to {OUT}")
