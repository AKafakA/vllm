#!/usr/bin/env python3
"""Analyze real vs emulator step-cycle traces to diagnose latency gap.

Compares real GPU step_cycle_us with emulator's step_cycle_us at each
concurrency level. Identifies which component (oracle, wait, scheduling)
contributes to the gap.

Usage: python analyze_real_vs_emu.py <real_trace.jsonl> <emu_trace.jsonl> [profile.json]
"""
import json
import statistics
import sys
from collections import defaultdict


def load_trace(path):
    """Load JSONL trace, skip header."""
    records = []
    for line in open(path):
        r = json.loads(line)
        if r.get("_header") or "__marker__" in r:
            continue
        if "step_cycle_us" in r:
            records.append(r)
    return records


def group_by_concurrency(records):
    """Group records by concurrent request count."""
    by_conc = defaultdict(list)
    for r in records:
        n = r.get("num_new_reqs", 0) + r.get("num_decode_seqs", 0)
        by_conc[n].append(r)
    return by_conc


def main():
    real_path = sys.argv[1]
    emu_path = sys.argv[2]
    profile_path = sys.argv[3] if len(sys.argv) > 3 else None

    real = load_trace(real_path)
    emu = load_trace(emu_path)

    print(f"Real: {len(real)} steps, Emu: {len(emu)} steps")

    # Skip warmup (first 200 steps)
    real = real[200:]
    emu = emu[200:]
    print(f"After warmup skip: Real={len(real)}, Emu={len(emu)}")

    real_by_conc = group_by_concurrency(real)
    emu_by_conc = group_by_concurrency(emu)

    # Load profile forward_pass for comparison
    profile_fp = {}
    if profile_path:
        profile = json.load(open(profile_path))
        for e in profile.get("forward_pass", []):
            profile_fp[e["total_tokens"]] = e["latency_us"]

    print("\n=== Step cycle comparison by concurrency ===")
    print(f"{'N':>4} {'Real_med':>10} {'Emu_med':>10} {'Gap':>10} {'Gap%':>8} {'Emu_wait':>10} {'Emu_exec':>10} {'Profile':>10}")
    print("-" * 82)

    all_concs = sorted(set(list(real_by_conc.keys()) + list(emu_by_conc.keys())))
    for n in all_concs:
        r_recs = real_by_conc.get(n, [])
        e_recs = emu_by_conc.get(n, [])
        if len(r_recs) < 3 or len(e_recs) < 3:
            continue

        r_med = statistics.median([r["step_cycle_us"] for r in r_recs])
        e_med = statistics.median([r["step_cycle_us"] for r in e_recs])
        gap = e_med - r_med
        gap_pct = gap / r_med * 100 if r_med else 0

        # Emu breakdown (if fields available)
        e_wait = statistics.median([r.get("wait_ms", 0) for r in e_recs]) if e_recs else 0
        e_exec = statistics.median([r.get("exec_ms", 0) for r in e_recs]) if e_recs else 0

        # Profile prediction at median tt
        r_tts = [r.get("total_tokens", 0) for r in r_recs]
        med_tt = int(statistics.median(r_tts))
        prof_val = profile_fp.get(med_tt, 0)

        print(f"{n:>4} {r_med/1000:>10.1f} {e_med/1000:>10.1f} {gap/1000:>10.1f} {gap_pct:>7.1f}% {e_wait:>10.1f} {e_exec:>10.1f} {prof_val/1000:>10.1f}")

    # Overall comparison
    r_all = [r["step_cycle_us"] for r in real]
    e_all = [r["step_cycle_us"] for r in emu]
    print(f"\n{'ALL':>4} {statistics.median(r_all)/1000:>10.1f} {statistics.median(e_all)/1000:>10.1f} "
          f"{(statistics.median(e_all)-statistics.median(r_all))/1000:>10.1f} "
          f"{(statistics.median(e_all)-statistics.median(r_all))/statistics.median(r_all)*100:>7.1f}%")

    # Analyze the gap pattern
    print("\n=== Gap analysis ===")
    gaps = []
    for n in sorted(real_by_conc.keys()):
        r_recs = real_by_conc.get(n, [])
        e_recs = emu_by_conc.get(n, [])
        if len(r_recs) < 3 and len(e_recs) < 3:
            continue
        if len(r_recs) >= 3:
            r_med = statistics.median([r["step_cycle_us"] for r in r_recs])
        else:
            continue
        if len(e_recs) >= 3:
            e_med = statistics.median([r["step_cycle_us"] for r in e_recs])
        else:
            continue
        gaps.append((n, r_med, e_med, e_med - r_med))

    if gaps:
        # Linear regression: gap vs concurrency
        import numpy as np
        ns = np.array([g[0] for g in gaps])
        gap_vals = np.array([g[3] for g in gaps])
        if len(ns) >= 3:
            coeffs = np.polyfit(ns, gap_vals, 1)
            print(f"Gap = {coeffs[1]/1000:.2f}ms + {coeffs[0]/1000:.3f}ms * N_concurrent")
            print(f"  Base gap (N=1): {(coeffs[1] + coeffs[0])/1000:.2f}ms")
            print(f"  Per-request gap: {coeffs[0]/1000:.3f}ms")

    # Emu timing breakdown
    if emu and "wait_ms" in emu[0]:
        print("\n=== Emulator timing breakdown ===")
        for field in ["wait_ms", "update_ms", "sched_ms", "exec_ms", "sample_ms", "queue_ms"]:
            vals = [r.get(field, 0) for r in emu if field in r]
            if vals:
                print(f"  {field:>12}: median={statistics.median(vals):.2f}ms, "
                      f"mean={statistics.mean(vals):.2f}ms, "
                      f"p99={sorted(vals)[int(len(vals)*0.99)]:.2f}ms")

    # Real timing: step_cycle breakdown if available
    print("\n=== Real step_cycle distribution ===")
    r_decode = [r for r in real if r.get("num_new_reqs", 0) == 0]
    r_prefill = [r for r in real if r.get("num_new_reqs", 0) > 0]
    if r_decode:
        r_d_lat = [r["step_cycle_us"] for r in r_decode]
        print(f"  Decode-only: median={statistics.median(r_d_lat)/1000:.1f}ms, "
              f"mean={statistics.mean(r_d_lat)/1000:.1f}ms (n={len(r_decode)})")
    if r_prefill:
        r_p_lat = [r["step_cycle_us"] for r in r_prefill]
        print(f"  Has-prefill: median={statistics.median(r_p_lat)/1000:.1f}ms, "
              f"mean={statistics.mean(r_p_lat)/1000:.1f}ms (n={len(r_prefill)})")


if __name__ == "__main__":
    main()
