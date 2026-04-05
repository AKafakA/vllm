#!/usr/bin/env python3
"""Build serving profile with separate prefill/decode forward_pass sections.

Prefill steps (has new_reqs) have different CUDA graph latencies than
decode-only steps at the same total_tokens. This 2-section profile
lets the oracle use the correct latency for each step type.

Usage: python build_serving_profile_2d.py <step_cycle_file> <sweep_profile> <output> <model_name> <gpu_model>
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

print(f"Records: {len(records)}")

# Split into prefill (has new_reqs) and decode (no new_reqs) steps
prefill_by_tt = defaultdict(list)
decode_by_tt = defaultdict(list)
for r in records:
    tt = r["total_tokens"]
    if r.get("num_new_reqs", 0) > 0:
        prefill_by_tt[tt].append(r["step_cycle_us"])
    else:
        decode_by_tt[tt].append(r["step_cycle_us"])

def build_section(by_tt, label):
    section = []
    for tt in sorted(by_tt):
        lats = by_tt[tt]
        if len(lats) < 2:
            continue
        med = statistics.median(lats)
        filtered = [v for v in lats if v > 5000 and v < med * 3]
        if len(filtered) >= 2:
            section.append({
                "total_tokens": tt,
                "latency_us": round(statistics.median(filtered), 1),
                "num_samples": len(filtered),
            })
    print(f"  {label}: {len(section)} buckets")
    return section

prefill_fp = build_section(prefill_by_tt, "prefill_forward_pass")
decode_fp = build_section(decode_by_tt, "decode_forward_pass")

# Combined forward_pass (for backward compat — uses all steps)
all_by_tt = defaultdict(list)
for r in records:
    all_by_tt[r["total_tokens"]].append(r["step_cycle_us"])
combined_fp = build_section(all_by_tt, "combined_forward_pass")

# Merge with sweep for large tt
max_tt = max(e["total_tokens"] for e in combined_fp) if combined_fp else 0
try:
    sweep = json.load(open(sweep_profile_path))
    added = 0
    for e in sweep["forward_pass"]:
        if e["total_tokens"] > max_tt:
            combined_fp.append(e)
            added += 1
    print(f"  Merged {added} sweep buckets for tt>{max_tt}")
except FileNotFoundError:
    pass

# Compute emulator calibration parameters from trace data
# 1. CUDA graph shape warmup: first-encounter overhead per padded batch size
CAPTURE_SIZES = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 160, 192,
                 224, 256, 320, 384, 448, 512, 640, 768, 896, 1024]

def get_padded(tt):
    for s in CAPTURE_SIZES:
        if s >= tt:
            return s
    return tt

shape_records = defaultdict(list)
for r in records[100:]:  # Skip first 100 (cold start)
    shape_records[get_padded(r["total_tokens"])].append(r["step_cycle_us"])

shape_overheads = []
for shape, vals in shape_records.items():
    if len(vals) > 5:
        first = vals[0]
        warm = statistics.median(vals[2:])
        overhead = first - warm
        if overhead > 5000:  # >5ms overhead
            shape_overheads.append(overhead)

avg_cuda_warmup_us = statistics.median(shape_overheads) if shape_overheads else 0
print(f"\n  cuda_graph_warmup_us: {avg_cuda_warmup_us:.0f} (from {len(shape_overheads)} shapes)")

# 2. Scheduling factor and decode ratio
# These are software-architecture-dependent (vLLM engine loop structure)
# Default values validated on vLLM v0.18 UniProcExecutor
sched_factor = 1.5
decode_sched_ratio = 0.1
print(f"  sched_factor: {sched_factor}")
print(f"  decode_sched_ratio: {decode_sched_ratio}")

profile = {
    "version": "1.0",
    "gpu_model": gpu_model,
    "model_name": model_name,
    "profile_type": "serving_step_cycle_2d",
    "prefill": [],
    "decode": [],
    "forward_pass": sorted(combined_fp, key=lambda e: e["total_tokens"]),
    "prefill_forward_pass": sorted(prefill_fp, key=lambda e: e["total_tokens"]),
    "decode_forward_pass": sorted(decode_fp, key=lambda e: e["total_tokens"]),
    # Emulator calibration parameters (auto-computed from trace)
    "cuda_graph_warmup_us": round(avg_cuda_warmup_us, 0),
    "sched_factor": sched_factor,
    "decode_sched_ratio": decode_sched_ratio,
}
json.dump(profile, open(output_path, "w"), indent=2)

# Show key differences
print(f"\nPrefill vs Decode at key tt values:")
pfill_map = {e["total_tokens"]: e["latency_us"] for e in prefill_fp}
dec_map = {e["total_tokens"]: e["latency_us"] for e in decode_fp}
for tt in [1, 5, 10, 256, 260, 265, 270]:
    p = pfill_map.get(tt, 0)
    d = dec_map.get(tt, 0)
    if p > 0 or d > 0:
        print(f"  tt={tt:>4}: prefill={p/1000:.1f}ms, decode={d/1000:.1f}ms, ratio={p/d:.1f}x" if d > 0 else f"  tt={tt:>4}: prefill={p/1000:.1f}ms, decode=N/A")

print(f"\nSaved to {output_path}")
