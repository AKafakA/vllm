"""Analyze per-batch-size scheduling overhead from step-cycle trace.

The step-cycle captures full engine loop time (scheduling + GPU + output processing).
The profile captures GPU forward pass time only.
The difference is the scheduling/processing overhead per batch size.

This overhead should be added to emulator timers to model the real engine's
thread-blocking behavior that prevents IPC pipelining.
"""
import json
import sys
from collections import defaultdict

def main():
    trace_file = sys.argv[1] if len(sys.argv) > 1 else \
        "/workspace/eval_results/RTX-3060-12GB/step_cycle_1.5b_full.jsonl"
    profile_file = sys.argv[2] if len(sys.argv) > 2 else \
        "/workspace/eval_results/RTX-3060-12GB/profiles/serving-1.5b-tp1-full.json"

    # Load profile (GPU forward pass times)
    profile = json.load(open(profile_file))

    # Build lookup: total_tokens -> profiled latency_us
    pfp = {e["total_tokens"]: e["latency_us"]
           for e in profile.get("prefill_forward_pass", [])}
    dfp = {e["total_tokens"]: e["latency_us"]
           for e in profile.get("decode_forward_pass", [])}
    fwd = {e["total_tokens"]: e["latency_us"]
           for e in profile.get("forward_pass", [])}

    # Load trace
    records = []
    with open(trace_file) as f:
        for line in f:
            records.append(json.loads(line))

    print("Total trace records:", len(records))

    # Skip first 100 records (cold start / warmup)
    records = records[100:]
    print("After skipping first 100:", len(records))

    # For each record, compute: overhead = step_cycle - profiled_gpu_time
    overhead_by_tt = defaultdict(list)

    for r in records:
        tt = r["total_tokens"]
        has_prefill = r["num_new_reqs"] > 0
        cycle_us = r["step_cycle_us"]

        # Look up profiled GPU time
        if has_prefill and tt in pfp:
            gpu_us = pfp[tt]
        elif not has_prefill and tt in dfp:
            gpu_us = dfp[tt]
        elif tt in fwd:
            gpu_us = fwd[tt]
        else:
            continue  # Can't compute overhead without profile match

        overhead_us = cycle_us - gpu_us
        if overhead_us > 0:  # Ignore negative (profile > cycle, measurement noise)
            overhead_by_tt[tt].append(overhead_us)

    # Compute statistics per total_tokens bucket
    print("\nScheduling overhead by total_tokens:")
    print("%6s %6s %10s %10s %10s %10s" % (
        "tt", "count", "mean_ms", "median_ms", "p75_ms", "p90_ms"))

    all_overheads = []
    for tt in sorted(overhead_by_tt.keys()):
        vals = sorted(overhead_by_tt[tt])
        n = len(vals)
        if n < 3:
            continue
        mean = sum(vals) / n / 1000
        median = vals[n // 2] / 1000
        p75 = vals[int(n * 0.75)] / 1000
        p90 = vals[int(n * 0.90)] / 1000
        all_overheads.extend(vals)
        if n >= 10:  # Only show buckets with enough samples
            print("%6d %6d %10.1f %10.1f %10.1f %10.1f" % (
                tt, n, mean, median, p75, p90))

    if all_overheads:
        all_overheads.sort()
        n = len(all_overheads)
        print("\nOverall scheduling overhead (all batch sizes):")
        print("  mean=%.1fms  median=%.1fms  p75=%.1fms  p90=%.1fms" % (
            sum(all_overheads)/n/1000, all_overheads[n//2]/1000,
            all_overheads[int(n*0.75)]/1000, all_overheads[int(n*0.9)]/1000))

    # Group by batch size ranges for profile integration
    print("\nSuggested overhead values for profile:")
    ranges = [(1, 1), (2, 4), (5, 10), (11, 32), (33, 128), (129, 512)]
    for lo, hi in ranges:
        vals = []
        for tt in range(lo, hi + 1):
            if tt in overhead_by_tt:
                vals.extend(overhead_by_tt[tt])
        if vals:
            vals.sort()
            n = len(vals)
            median = vals[n // 2] / 1000
            print("  tt=%d-%d: median=%.1fms (n=%d)" % (lo, hi, median, n))


if __name__ == "__main__":
    main()
