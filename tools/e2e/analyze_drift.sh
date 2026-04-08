#!/bin/bash
source /workspace/vllm-v18-env/bin/activate
python3 << 'PYEOF'
import csv

rows = []
with open("/workspace/hook_trace_r8.csv") as f:
    for r in csv.DictReader(f):
        rows.append({k: float(v) for k, v in r.items()})

# Skip warmup
bench = rows[3000:]
print(f"Benchmark steps: {len(bench)}")

# Show drift: gpu_free_time - wall_s = how far ahead the virtual GPU is
print(f"\n=== gpu_free_time drift (seconds ahead of wall-clock) ===")
for i in range(0, min(len(bench), 200), 10):
    r = bench[i]
    drift = r["gpu_free_time"] - r["wall_s"]
    print(f"  step {int(r['step']):>5}: drift={drift:.3f}s ({drift*1000:.0f}ms), "
          f"reqs={int(r['n_reqs'])}, oracle={r['oracle_us']/1000:.1f}ms, "
          f"timer_delay={r['timer_delay_us']/1000:.0f}ms, "
          f"total_lat={r['total_latency_us']/1000:.0f}ms")

# Show drift over time
drifts = [r["gpu_free_time"] - r["wall_s"] for r in bench]
import statistics
print(f"\nDrift stats (seconds):")
print(f"  mean={statistics.mean(drifts):.3f}, median={statistics.median(drifts):.3f}")
print(f"  min={min(drifts):.3f}, max={max(drifts):.3f}")

# Does drift grow over time?
first_100 = drifts[:100]
last_100 = drifts[-100:]
print(f"\n  First 100 steps: avg drift={statistics.mean(first_100)*1000:.0f}ms")
print(f"  Last 100 steps:  avg drift={statistics.mean(last_100)*1000:.0f}ms")

# How many steps have drift > 1 step (>15ms)?
print(f"\n  Drift > 15ms: {sum(1 for d in drifts if d > 0.015)} steps ({sum(1 for d in drifts if d > 0.015)/len(drifts)*100:.1f}%)")
print(f"  Drift > 50ms: {sum(1 for d in drifts if d > 0.050)} steps ({sum(1 for d in drifts if d > 0.050)/len(drifts)*100:.1f}%)")
print(f"  Drift > 100ms: {sum(1 for d in drifts if d > 0.100)} steps ({sum(1 for d in drifts if d > 0.100)/len(drifts)*100:.1f}%)")
PYEOF
