#!/usr/bin/env python3
"""Analyze why sweep profile overestimates at tt=50-100.
Compare context length distributions between sweep trace and offline trace."""
import json
import statistics
from collections import defaultdict

RESULT_DIR = "/workspace/eval_results/RTX-3060-12GB"

# Load sweep trace (used to build the profile)
sweep_records = []
for line in open(f"{RESULT_DIR}/sweep_trace_1.5b_dense.jsonl"):
    sweep_records.append(json.loads(line))

# Load offline trace (real offline workload)
offline_records = []
for line in open(f"{RESULT_DIR}/offline_real_trace.jsonl"):
    offline_records.append(json.loads(line))

print("=== Context Length Distribution at Key Total Tokens ===")
print(f"{'tt':>6} {'source':>8} {'latency':>10} {'avg_ctx':>10} {'n_decode':>10} {'n':>6}")

for tt_target in [30, 50, 64, 80, 100]:
    # Find records within ±5 of target
    for name, records in [("sweep", sweep_records), ("offline", offline_records)]:
        matching = [r for r in records if abs(r["total_tokens"] - tt_target) <= 3]
        if matching:
            lat = statistics.median([r["latency_us"] for r in matching])
            ctx = statistics.median([r.get("avg_decode_context", r.get("avg_context_len", 0)) for r in matching])
            nd = statistics.median([r.get("num_decode_seqs", r.get("num_decode_tokens", 0)) for r in matching])
            print(f"{tt_target:>6} {name:>8} {lat/1000:>10.1f} {ctx:>10.0f} {nd:>10.0f} {len(matching):>6}")
    print()

print("=== Key Insight ===")
print("If sweep and offline have different avg_decode_context at the same tt,")
print("the 1D profile (total_tokens only) can't distinguish them.")
print("Solution: ensure sweep profiler covers realistic context distributions,")
print("or use step-cycle serving profile which captures real operating conditions.")
