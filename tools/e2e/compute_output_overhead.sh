#!/bin/bash
# Compute output delivery overhead: client_TPOT - step_cycle at same concurrency
# Uses existing data: real baselines (client TPOT) + profiling trace (step_cycle)
source /workspace/vllm-v18-env/bin/activate

python3 << 'PYEOF'
import json, statistics, os
from collections import defaultdict

RESULT_DIR = "/workspace/eval_results/RTX-3060-12GB"
ONLINE_DIR = f"{RESULT_DIR}/online"
TRACE = f"{RESULT_DIR}/step_cycle_adaptive.jsonl"

# Load profiling trace — decode steps only, skip warmup
records = []
for line in open(TRACE):
    r = json.loads(line)
    if "total_tokens" in r:
        records.append(r)

decode_by_conc = defaultdict(list)
for r in records[5000:]:
    if r.get("num_new_reqs", 0) == 0 and r.get("num_decode_seqs", 0) > 0:
        conc = r["num_decode_seqs"]
        decode_by_conc[conc].append(r["step_cycle_us"])

print("=== Step Cycle by Concurrency (from profiling trace) ===")
for conc in sorted(decode_by_conc):
    if conc in [1, 2, 5, 10, 15, 20, 25, 30, 40, 50]:
        lats = decode_by_conc[conc]
        med = statistics.median(lats) / 1000
        print(f"  conc={conc:>3}: n={len(lats):>6}, median_step_cycle={med:.1f}ms")

# Load real baselines — get client-visible TPOT and max_concurrent
print(f"\n=== Real Baselines (client-visible TPOT) ===")
baselines = {}
for tag in ["split2d", "fc_rate"]:
    for rate in [1, 4, 8]:
        if tag == "fc_rate":
            f = f"{ONLINE_DIR}/fc_rate{rate}_real.json"
        else:
            f = f"{ONLINE_DIR}/{tag}_r{rate}_real.json"
        if os.path.exists(f):
            d = json.load(open(f))
            tpot = d["mean_tpot_ms"]
            max_conc = d.get("max_concurrent_requests", "?")
            baselines[(tag, rate)] = d
            print(f"  {tag} R={rate}: TPOT={tpot:.1f}ms, max_conc={max_conc}")

# Compute output overhead per rate
# Estimate average concurrency from the rate and model characteristics
# At rate=R, avg_concurrent ≈ R * mean_e2e_seconds
print(f"\n=== Output Overhead Estimate ===")
print(f"  overhead = client_TPOT - step_cycle_at_avg_concurrency")

for tag in ["split2d"]:
    for rate in [1, 4, 8]:
        key = (tag, rate)
        if key not in baselines:
            continue
        d = baselines[key]
        client_tpot = d["mean_tpot_ms"]
        mean_e2e_s = d.get("mean_e2el_ms", 0) / 1000

        # Estimate average concurrency
        avg_conc = rate * mean_e2e_s
        # Find step_cycle at that concurrency
        # Use median of nearby concurrency values
        nearby = []
        for c in range(max(1, int(avg_conc) - 3), int(avg_conc) + 4):
            if c in decode_by_conc:
                nearby.extend(decode_by_conc[c])
        if nearby:
            step_cycle_ms = statistics.median(nearby) / 1000
            overhead = client_tpot - step_cycle_ms
            print(f"  R={rate}: client_TPOT={client_tpot:.1f}ms, "
                  f"avg_conc≈{avg_conc:.0f}, "
                  f"step_cycle={step_cycle_ms:.1f}ms, "
                  f"overhead={overhead:.1f}ms ({overhead/client_tpot*100:.0f}%)")
        else:
            print(f"  R={rate}: no step_cycle data at conc≈{avg_conc:.0f}")

# Build output overhead table: overhead per concurrency bucket
print(f"\n=== Output Overhead Table (for profile) ===")
# For each rate, we have one data point: (avg_conc, overhead)
# We can also estimate from the step_cycle trace itself by looking at
# the gap between consecutive decode steps' timestamps
# But that requires wall-clock timestamps which we don't have in the trace

# Simple approach: compute overhead at a few concurrency points
# from the available baselines
overheads = []
for tag in ["split2d"]:
    for rate in [1, 4, 8]:
        key = (tag, rate)
        if key not in baselines:
            continue
        d = baselines[key]
        client_tpot = d["mean_tpot_ms"]
        mean_e2e_s = d.get("mean_e2el_ms", 0) / 1000
        avg_conc = rate * mean_e2e_s

        nearby = []
        for c in range(max(1, int(avg_conc) - 3), int(avg_conc) + 4):
            if c in decode_by_conc:
                nearby.extend(decode_by_conc[c])
        if nearby:
            step_cycle_ms = statistics.median(nearby) / 1000
            overhead = client_tpot - step_cycle_ms
            overheads.append({
                "num_requests": round(avg_conc),
                "overhead_ms": round(overhead, 2),
                "rate": rate,
            })
            print(f"  conc≈{avg_conc:.0f}: overhead={overhead:.1f}ms (from R={rate})")

# This can be stored in the profile as output_overhead_table
print(f"\n=== Suggested Profile Addition ===")
print(f"output_overhead_table: {json.dumps(overheads, indent=2)}")
PYEOF
