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
    # First pass: compute raw medians
    raw_medians = {}
    for tt in sorted(by_tt):
        lats = by_tt[tt]
        if len(lats) < 2:
            continue
        med = statistics.median(lats)
        filtered = [v for v in lats if v > 5000 and v < med * 3]
        if len(filtered) >= 2:
            raw_medians[tt] = statistics.median(filtered)

    # Second pass: cross-reference with neighbors to detect outlier buckets.
    # A bucket's median should be within 3x of its nearest neighbors.
    # This catches cold-start contamination in sparse buckets (e.g., tt=256
    # having 78ms when tt=258 has 17ms — the 256 bucket is an outlier).
    section = []
    sorted_tts = sorted(raw_medians.keys())
    for i, tt in enumerate(sorted_tts):
        med = raw_medians[tt]
        # Find nearest neighbors within ±10 tt
        neighbors = [raw_medians[t] for t in sorted_tts
                     if abs(t - tt) <= 10 and t != tt]
        if neighbors:
            neighbor_med = statistics.median(neighbors)
            if med > neighbor_med * 3:
                # This bucket is >3x its neighbors — likely cold-start outlier
                print(f"    WARNING: {label} tt={tt} median={med/1000:.1f}ms "
                      f"is {med/neighbor_med:.1f}x neighbors ({neighbor_med/1000:.1f}ms), "
                      f"replacing with neighbor median")
                med = neighbor_med

        lats = by_tt[tt]
        section.append({
            "total_tokens": tt,
            "latency_us": round(med, 1),
            "num_samples": len(lats),
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
sweep_fp = []
try:
    sweep = json.load(open(sweep_profile_path))
    sweep_fp = sweep.get("forward_pass", [])
    # Sweep (enforce_eager, no CUDA graphs) is NOT merged into the online
    # forward_pass. It overestimates by 3-4x at tt>271 vs graph-enabled GPU.
    # Kept as separate "sweep_forward_pass" for reference/offline/non-graph use.
    print(f"  Sweep loaded ({len(sweep_fp)} buckets) — stored separately, NOT merged into online")
except FileNotFoundError:
    print(f"  No sweep profile found")

# Compute emulator calibration parameters from trace + bench results
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

# 2. IPC scheduling overhead: directly measured per concurrent request count.
# Loaded from ipc_overhead.json produced by profile_ipc_overhead.py.
# Each entry: {num_reqs: N, overhead_us: X} measured with N-1 background
# requests in flight + 1 measurement request.
import os
ipc_overhead_path = os.path.join(os.path.dirname(output_path), "ipc_overhead.json")
sched_overhead_table = []
if os.path.exists(ipc_overhead_path):
    sched_overhead_table = json.load(open(ipc_overhead_path))
    print(f"\n  Loaded IPC overhead table: {len(sched_overhead_table)} entries")
    for e in sched_overhead_table:
        if e["num_reqs"] in [1, 2, 3, 5, 10, 20, 30, 50]:
            print(f"    N={e['num_reqs']:3d}: overhead={e['overhead_us']/1000:.1f}ms")
else:
    print(f"\n  WARNING: {ipc_overhead_path} not found. Run profile_ipc_sweep.sh first.")

# Build offline_forward_pass from offline trace (if available).
# The offline trace captures step-cycle via LLM() path (bench throughput)
# with CUDA graphs at production batch sizes. Decode-only steps
# (num_new_reqs=0) give the correct latency for offline inference.
offline_trace_path = os.path.join(os.path.dirname(step_cycle_file), "offline_step_cycle.jsonl")
offline_fp = []
if os.path.exists(offline_trace_path):
    offline_records = []
    for line in open(offline_trace_path):
        r = json.loads(line)
        if "total_tokens" in r:
            offline_records.append(r)
    offline_by_tt = defaultdict(list)
    for r in offline_records:
        offline_by_tt[r["total_tokens"]].append(r["step_cycle_us"])
    offline_fp = build_section(offline_by_tt, "offline_forward_pass")
    print(f"\n  Loaded offline trace: {len(offline_records)} decode-only steps")
else:
    print(f"\n  No offline trace found at {offline_trace_path}")

# Compute overhead_per_request_us for 2D oracle.
# Uses linear regression: step_cycle = a + b*total_tokens + c*num_requests
# The per-request overhead c captures host-side costs that scale with concurrency.
overhead_per_request_us = 0.0
try:
    import numpy as np
    # Use decode-only steps (no prefill noise)
    decode_records = [r for r in records[200:] if r.get("num_new_reqs", 0) == 0
                      and r.get("num_decode_seqs", 0) > 0]
    if len(decode_records) >= 20:
        X = np.array([[r["total_tokens"], r["num_decode_seqs"]] for r in decode_records])
        y = np.array([r["step_cycle_us"] for r in decode_records])
        # Add intercept: y = a + b*tt + c*n_reqs
        X_aug = np.column_stack([np.ones(len(X)), X])
        # Least squares fit
        coeffs, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
        intercept, coeff_tt, coeff_nreqs = coeffs
        overhead_per_request_us = max(0, coeff_nreqs)  # clamp non-negative
        print(f"\n  2D regression (decode-only, {len(decode_records)} records):")
        print(f"    latency = {intercept/1000:.1f}ms + {coeff_tt/1000:.2f}ms*tt + {coeff_nreqs/1000:.2f}ms*n_reqs")
        print(f"    overhead_per_request_us = {overhead_per_request_us:.0f} ({overhead_per_request_us/1000:.2f}ms)")
    else:
        print(f"\n  Not enough decode records for 2D regression ({len(decode_records)})")
except Exception as e:
    print(f"\n  2D regression failed: {e}")

# Build concurrency correction table.
# For each concurrency bucket, compute:
#   correction = actual_step_cycle - oracle_1d_prediction(total_tokens)
# This captures the gap between 1D profile and reality at each concurrency.
# Positive = profile underestimates (need to add), negative = overestimates.
correction_table = []
try:
    # Build 1D oracle lookup from combined forward_pass for prediction
    fp_map = {e["total_tokens"]: e["latency_us"] for e in combined_fp}
    fp_tts = sorted(fp_map.keys())

    def oracle_1d(tt):
        """Simple 1D interpolation matching what the oracle does."""
        if tt <= 0:
            return 0
        if tt in fp_map:
            return fp_map[tt]
        # Find bracketing entries
        lo = max((t for t in fp_tts if t <= tt), default=fp_tts[0])
        hi = min((t for t in fp_tts if t >= tt), default=fp_tts[-1])
        if lo == hi:
            return fp_map[lo]
        frac = (tt - lo) / (hi - lo)
        return fp_map[lo] + frac * (fp_map[hi] - fp_map[lo])

    # Use decode-only steps (no prefill noise)
    decode_recs = [r for r in records[200:] if r.get("num_new_reqs", 0) == 0
                   and r.get("num_decode_seqs", 0) > 0]

    # Group by concurrency bucket (width=5)
    from collections import defaultdict as dd
    by_conc = dd(list)
    for r in decode_recs:
        n = r["num_decode_seqs"]
        by_conc[n].append(r)

    # Compute correction per bucket
    buckets = sorted(set((n // 5) * 5 for n in by_conc.keys()))
    print(f"\n  Concurrency correction table ({len(decode_recs)} decode records):")
    for bucket in buckets:
        recs_in_bucket = []
        for n in range(bucket, bucket + 5):
            recs_in_bucket.extend(by_conc.get(n, []))
        if len(recs_in_bucket) < 5:
            continue
        actual_lats = [r["step_cycle_us"] for r in recs_in_bucket]
        predicted_lats = [oracle_1d(r["total_tokens"]) for r in recs_in_bucket]
        actual_med = statistics.median(actual_lats)
        predicted_med = statistics.median(predicted_lats)
        correction = actual_med - predicted_med
        avg_n = statistics.median([r["num_decode_seqs"] for r in recs_in_bucket])
        correction_table.append({
            "num_requests": round(avg_n),
            "correction_us": round(correction, 1),
            "num_samples": len(recs_in_bucket),
        })
        print(f"    N={avg_n:>3}: actual={actual_med/1000:.1f}ms, predicted={predicted_med/1000:.1f}ms, "
              f"correction={correction/1000:+.1f}ms (n={len(recs_in_bucket)})")
except Exception as e:
    print(f"\n  Correction table failed: {e}")

profile = {
    "version": "1.0",
    "gpu_model": gpu_model,
    "model_name": model_name,
    "profile_type": "serving_step_cycle_2d",
    "overhead_per_request_us": round(overhead_per_request_us, 1),
    "correction_table": correction_table,
    "prefill": [],
    "decode": [],
    "forward_pass": sorted(combined_fp, key=lambda e: e["total_tokens"]),
    "prefill_forward_pass": sorted(prefill_fp, key=lambda e: e["total_tokens"]),
    "decode_forward_pass": sorted(decode_fp, key=lambda e: e["total_tokens"]),
    "offline_forward_pass": sorted(offline_fp, key=lambda e: e["total_tokens"]),
    "sweep_forward_pass": sorted(sweep_fp, key=lambda e: e["total_tokens"]),
    # Emulator calibration parameters (auto-computed from trace)
    "cuda_graph_warmup_us": round(avg_cuda_warmup_us, 0),
    "sched_overhead_table": sched_overhead_table,
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

if offline_fp:
    off_map = {e["total_tokens"]: e["latency_us"] for e in offline_fp}
    print(f"\nOffline forward_pass at key tt values:")
    for tt in sorted(off_map.keys()):
        print(f"  tt={tt:>4}: {off_map[tt]/1000:.1f}ms")

print(f"\nSaved to {output_path}")
