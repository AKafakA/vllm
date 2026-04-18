#!/usr/bin/env python3
"""Split v3 trace by profiling_start/stop markers into per-round chunks
and compute per-round latency stats. Tests thermal-drift + warmup
hypothesis."""
import json
import sys
import statistics
from collections import defaultdict

TRACE = sys.argv[1] if len(sys.argv) > 1 else "results/RTX-8000-adaptive-v3/step_cycle_trace.jsonl"


def main():
    rounds = []  # list of (records, label)
    current = None
    current_label = None
    with open(TRACE) as f:
        for line in f:
            r = json.loads(line)
            if r.get("__marker__") == "profiling_start":
                current = []
                current_label = f"round_{len(rounds) + 1}"
                continue
            if r.get("__marker__") == "profiling_stop":
                if current is not None:
                    rounds.append((current, current_label))
                    current = None
                continue
            if r.get("_header"):
                continue
            if current is not None and "step_cycle_us" in r:
                current.append(r)

    print(f"Parsed {len(rounds)} rounds.\n")
    print(f"{'round':<10} {'n':>8} {'mean_us':>10} {'median_us':>10} "
          f"{'std_us':>10} {'p10':>10} {'p90':>10}")

    # Overall stats per round
    for recs, label in rounds:
        lats = [r["step_cycle_us"] for r in recs]
        if not lats:
            continue
        sorted_l = sorted(lats)
        n = len(lats)
        m = sum(lats) / n
        md = sorted_l[n // 2]
        s = statistics.pstdev(lats) if n > 1 else 0
        p10 = sorted_l[n // 10]
        p90 = sorted_l[n * 9 // 10]
        print(f"{label:<10} {n:>8} {m:>10.0f} {md:>10.0f} {s:>10.0f} "
              f"{p10:>10.0f} {p90:>10.0f}")

    # First-1000 vs last-1000 per round (graph-warming signature?)
    print("\nFirst-N vs last-N per round (N=1000) means (us):")
    print(f"{'round':<10} {'first_1k_mean':>14} {'last_1k_mean':>14} "
          f"{'drift_pct':>10}")
    for recs, label in rounds:
        lats = [r["step_cycle_us"] for r in recs]
        if len(lats) < 2000:
            continue
        first_mean = sum(lats[:1000]) / 1000
        last_mean = sum(lats[-1000:]) / 1000
        drift = 100 * (last_mean - first_mean) / first_mean
        print(f"{label:<10} {first_mean:>14.0f} {last_mean:>14.0f} {drift:>9.2f}%")

    # Per-round distribution for a common bucket (tt=1, conc<=3) — decode dominant
    print("\nPer-round mean for tt=1-3 decode-heavy samples (variable-shape contamination signature):")
    print(f"{'round':<10} {'n':>8} {'mean_us':>10}")
    for recs, label in rounds:
        sub = [r["step_cycle_us"] for r in recs
               if r.get("total_tokens", 99) in (1, 2, 3)
               and r.get("num_new_reqs", 0) == 0
               and (r.get("num_decode_seqs", 0) + r.get("num_new_reqs", 0)) <= 3]
        if sub:
            print(f"{label:<10} {len(sub):>8} {sum(sub)/len(sub):>10.0f}")

    # Does the trace contain samples from input_len=1 (CUDA graph warmup) that leaked past markers?
    print("\nSanity: any records with tt=1 new_reqs=1 decode=0 (CUDA-warmup shape) inside profile window?")
    for recs, label in rounds:
        warmup_shape = [r for r in recs
                        if r.get("total_tokens") == 1
                        and r.get("num_new_reqs", 0) == 1
                        and r.get("num_decode_seqs", 0) == 0]
        print(f"  {label}: {len(warmup_shape)} / {len(recs)} ({100 * len(warmup_shape) / len(recs):.1f}%)")


if __name__ == "__main__":
    main()
