"""Analyze CUDA graph shape warmup overhead from step-cycle trace.

Groups step-cycle records by padded batch size (CUDA graph shapes).
For each shape, compares first-encounter latency vs warm (subsequent) latency.
"""
import json
import sys
from collections import defaultdict

# vLLM default CUDA graph capture sizes
CAPTURE_SIZES = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128,
                 160, 192, 224, 256, 320, 384, 448, 512, 640, 768, 896, 1024]


def get_padded_size(total_tokens: int) -> int:
    for size in CAPTURE_SIZES:
        if size >= total_tokens:
            return size
    return total_tokens


def main():
    trace_file = sys.argv[1] if len(sys.argv) > 1 else "/workspace/eval_results/RTX-3060-12GB/step_cycle_1.5b_full.jsonl"

    records = []
    with open(trace_file) as f:
        for line in f:
            records.append(json.loads(line))

    print(f"Total records: {len(records)}")

    # Group by padded batch size, track first vs warm latency
    shape_records: dict[int, list[float]] = defaultdict(list)
    for r in records:
        padded = get_padded_size(r["total_tokens"])
        shape_records[padded].append(r["step_cycle_us"])

    print(f"\nShape analysis (first vs warm):")
    print(f"{'Shape':>6} {'Count':>6} {'First_ms':>10} {'Warm_ms':>10} {'Overhead_ms':>12}")

    total_overhead = 0.0
    num_shapes = 0
    for shape in sorted(shape_records.keys()):
        records_list = shape_records[shape]
        first = records_list[0] / 1000
        if len(records_list) > 2:
            warm = sum(records_list[2:]) / len(records_list[2:]) / 1000
        elif len(records_list) > 1:
            warm = records_list[-1] / 1000
        else:
            warm = first
        overhead = first - warm
        if overhead > 5:  # Only show significant overhead
            print(f"{shape:>6} {len(records_list):>6} {first:>10.1f} {warm:>10.1f} {overhead:>+11.1f}")
            total_overhead += overhead
            num_shapes += 1

    if num_shapes > 0:
        print(f"\nShapes with significant overhead: {num_shapes}")
        print(f"Average first-encounter overhead: {total_overhead/num_shapes:.1f}ms")
        print(f"Total first-encounter overhead across all shapes: {total_overhead:.1f}ms")

    # Also compute the overall distribution
    all_latencies = [r["step_cycle_us"] for r in records]
    all_latencies.sort()
    n = len(all_latencies)
    print(f"\nOverall step cycle distribution:")
    print(f"  p50={all_latencies[n//2]/1000:.1f}ms  p90={all_latencies[int(n*0.9)]/1000:.1f}ms  "
          f"p99={all_latencies[int(n*0.99)]/1000:.1f}ms  max={all_latencies[-1]/1000:.1f}ms")


if __name__ == "__main__":
    main()
