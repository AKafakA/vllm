#!/usr/bin/env python3
"""Compare per-bucket sample counts between two profiles.

Writes a summary of: how many buckets, samples per bucket (p50/p90/max),
tail-sample count above a shared p75 threshold.
"""
import json
import sys
import statistics

A, B = sys.argv[1], sys.argv[2]


def load(path, key):
    p = json.load(open(path))
    return {(b["tt"], b["conc"]): b for b in p.get(key, [])}


def stats(name, table):
    counts = [len(b.get("samples", [])) for b in table.values()]
    counts.sort()
    n = len(counts)
    total = sum(counts)
    print(f"\n=== {name}: {n} buckets, {total} samples ===")
    if n:
        print(f"  per-bucket counts  p50={counts[n//2]} "
              f"p90={counts[min(n-1, n*9//10)]} "
              f"p99={counts[min(n-1, n*99//100)]} "
              f"max={counts[-1]}")
        print(f"  mean samples/bucket = {total/n:.0f}")


def tail_compare(a_table, b_table):
    """For well-populated common buckets, count absolute tail samples."""
    common = set(a_table.keys()) & set(b_table.keys())
    well_pop = [k for k in common
                if len(a_table[k].get("samples", [])) > 100
                and len(b_table[k].get("samples", [])) > 100]
    print(f"\n=== Tail-sample comparison, {len(well_pop)} common well-populated buckets ===")
    print("Tail = samples > shared p75 of archive's samples for that bucket.")
    print()
    print(f"{'bucket':>10} | {'arch_n':>6} {'arch_p75':>8} {'arch_tail':>9} | "
          f"{'cand_n':>6} {'cand_tail':>9} | {'tail_ratio':>10}")
    print("-" * 95)
    tail_ratios = []
    for k in sorted(well_pop, key=lambda k: -len(a_table[k]["samples"]))[:15]:
        a = a_table[k]["samples"]
        b = b_table[k]["samples"]
        a_p75 = sorted(a)[len(a) * 3 // 4]
        a_tail = sum(1 for x in a if x > a_p75)
        b_tail = sum(1 for x in b if x > a_p75)
        # expected if tail scales linearly with total samples
        expected_b_tail = a_tail * len(b) / len(a)
        ratio = b_tail / expected_b_tail if expected_b_tail else 0
        tail_ratios.append(ratio)
        print(f"  tt={k[0]:>3} c={k[1]:>3} | {len(a):>6} {a_p75:>8} {a_tail:>9} | "
              f"{len(b):>6} {b_tail:>9} | {ratio:>10.2f}x")

    # aggregate for all common wellpop buckets
    all_ratios = []
    for k in well_pop:
        a = a_table[k]["samples"]
        b = b_table[k]["samples"]
        a_p75 = sorted(a)[len(a) * 3 // 4]
        a_tail = sum(1 for x in a if x > a_p75)
        b_tail = sum(1 for x in b if x > a_p75)
        expected = a_tail * len(b) / len(a)
        if expected > 0:
            all_ratios.append(b_tail / expected)
    all_ratios.sort()
    n = len(all_ratios)
    print()
    print(f"Aggregate over {n} buckets:")
    print(f"  tail ratio p10 = {all_ratios[n//10]:.2f}")
    print(f"  tail ratio p50 = {all_ratios[n//2]:.2f}")
    print(f"  tail ratio p90 = {all_ratios[n*9//10]:.2f}")
    print(f"  tail ratio mean = {sum(all_ratios)/n:.2f}")
    print()
    print("Interpretation:")
    print("  ratio = 1.0 → candidate tail scales exactly proportionally with samples")
    print("             (no dilution; uniform random.choice gives same tail probability).")
    print("  ratio < 1.0 → candidate has LESS tail than proportional (dilution: steady-state")
    print("             samples grow faster than capture-cost spikes).")
    print("  ratio > 1.0 → candidate has MORE tail than proportional (unexpected).")


if __name__ == "__main__":
    a = load(A, "decode_2d_distribution")
    b = load(B, "decode_2d_distribution")
    stats("Profile A (" + A.split("/")[-1] + ")", a)
    stats("Profile B (" + B.split("/")[-1] + ")", b)
    tail_compare(a, b)
