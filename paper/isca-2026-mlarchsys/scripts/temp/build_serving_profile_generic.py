#!/usr/bin/env python3
"""Build serving profile from step cycle trace (generic, any GPU/model).

Usage: python build_serving_profile_generic.py <step_cycle_file> <sweep_profile> <output> <model_name> <gpu_model>
"""
import json
import statistics
import sys
from collections import defaultdict

step_cycle_file = sys.argv[1]
sweep_profile_path = sys.argv[2]
output_path = sys.argv[3]
model_name = sys.argv[4] if len(sys.argv) > 4 else "unknown"
gpu_model = sys.argv[5] if len(sys.argv) > 5 else "unknown"

records = []
for line in open(step_cycle_file):
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
    if len(lats) < 2:
        continue
    med = statistics.median(lats)
    # Filter outliers: keep values >5ms and <3x median
    filtered = [v for v in lats if v > 5000 and v < med * 3]
    if len(filtered) >= 2:
        forward_pass.append({
            "total_tokens": tt,
            "latency_us": round(statistics.median(filtered), 1),
            "num_samples": len(filtered),
        })

# Merge with sweep profile for large tt
max_tt = max(e["total_tokens"] for e in forward_pass) if forward_pass else 0
try:
    sweep = json.load(open(sweep_profile_path))
    for e in sweep["forward_pass"]:
        if e["total_tokens"] > max_tt:
            forward_pass.append(e)
except FileNotFoundError:
    print("Warning: no sweep profile to merge")

profile = {
    "gpu_model": gpu_model,
    "model_name": model_name,
    "profile_type": "serving_step_cycle",
    "forward_pass": sorted(forward_pass, key=lambda e: e["total_tokens"]),
}
json.dump(profile, open(output_path, "w"), indent=2)
print(f"Serving profile: {len(forward_pass)} buckets")
for e in profile["forward_pass"][:10]:
    print(f"  tt={e['total_tokens']:>4}: {e['latency_us']/1000:.1f}ms (n={e.get('num_samples',0)})")
print(f"Saved to {output_path}")
