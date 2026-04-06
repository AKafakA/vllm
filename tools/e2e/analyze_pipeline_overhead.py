"""Analyze the processing overhead that gets pipelined away by the timer approach.

For each step in the trace:
  pipeline_overhead = step_cycle - forward_pass_latency

This is the time spent on scheduling, output processing, and IPC drain
that the timer approach "hides" inside the timer wait. On real GPU, this
overhead is sequential with the GPU compute. On the emulator, it runs
concurrently during the timer.

For TTFT, this overhead compounds because the engine also processes IPC
during the timer, picking up new requests faster.
"""
import json
import sys
from collections import defaultdict

trace_file = sys.argv[1] if len(sys.argv) > 1 else \
    "/workspace/eval_results/RTX-3060-12GB/step_cycle_1.5b_full.jsonl"
profile_file = sys.argv[2] if len(sys.argv) > 2 else \
    "/workspace/eval_results/RTX-3060-12GB/profiles/serving-1.5b-tp1-calibrated.json"

# Load profile
profile = json.load(open(profile_file))
pfp = {e["total_tokens"]: e["latency_us"] for e in profile.get("prefill_forward_pass", [])}
dfp = {e["total_tokens"]: e["latency_us"] for e in profile.get("decode_forward_pass", [])}
fwd = {e["total_tokens"]: e["latency_us"] for e in profile.get("forward_pass", [])}

# Load trace
records = []
with open(trace_file) as f:
    for line in f:
        records.append(json.loads(line))
records = records[100:]  # Skip cold start

print(f"Records: {len(records)}")

# For each record, compute pipeline_overhead = step_cycle - profiled_latency
overheads = []
by_concurrency = defaultdict(list)

for r in records:
    tt = r["total_tokens"]
    has_prefill = r.get("num_new_reqs", 0) > 0
    cycle = r["step_cycle_us"]

    # Look up profiled GPU time
    if has_prefill and tt in pfp:
        gpu_us = pfp[tt]
    elif not has_prefill and tt in dfp:
        gpu_us = dfp[tt]
    elif tt in fwd:
        gpu_us = fwd[tt]
    else:
        continue

    overhead = cycle - gpu_us
    if overhead > 0:
        overheads.append(overhead)
        num_reqs = r.get("num_decode_seqs", 0) + r.get("num_new_reqs", 0)
        by_concurrency[num_reqs].append(overhead)

overheads.sort()
n = len(overheads)
print(f"\nPipeline overhead (step_cycle - profiled_gpu_time):")
print(f"  mean={sum(overheads)/n/1000:.1f}ms  median={overheads[n//2]/1000:.1f}ms  "
      f"p75={overheads[int(n*0.75)]/1000:.1f}ms  p90={overheads[int(n*0.9)]/1000:.1f}ms")

print(f"\nBy concurrency:")
buckets = [(1, 2), (3, 5), (6, 10), (11, 20), (21, 50)]
for lo, hi in buckets:
    vals = []
    for nreq in range(lo, hi + 1):
        if nreq in by_concurrency:
            vals.extend(by_concurrency[nreq])
    if vals:
        vals.sort()
        m = len(vals)
        print(f"  reqs={lo}-{hi}: median={vals[m//2]/1000:.1f}ms  "
              f"mean={sum(vals)/m/1000:.1f}ms  n={m}")
