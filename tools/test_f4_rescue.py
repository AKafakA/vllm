#!/usr/bin/env python3
"""Test whether F4's 3D axis (per-num_new_reqs split) rescues v3.

If the bias comes from variable-shape contamination (shapes with
different KV-cache-depth signatures landing in same (tt, conc)
buckets), splitting by num_new_reqs should produce sub-buckets that
more closely match archive's single-shape distribution.

Reads v3 trace, groups by (tt, conc, num_new_reqs), compares to
archive's (tt, conc) means at num_new_reqs=0 (pure decode — the
regime the 256/128 validation workload is dominantly in).
"""
import json
import statistics
import sys

ARCHIVE = "results/_archive/serving-dense.json"
V3_TRACE = "results/RTX-8000-adaptive-v3/step_cycle_trace.jsonl"


def load_archive_decode():
    p = json.load(open(ARCHIVE))
    return {(b["tt"], b["conc"]): b for b in p.get("decode_2d_distribution", [])}


def main():
    archive = load_archive_decode()
    from collections import defaultdict

    # Bucket v3 records by (tt, conc, new_reqs) inside profile windows.
    groups_by_newreqs = defaultdict(lambda: defaultdict(list))  # (new_reqs) -> (tt, conc) -> list
    in_profile = False
    with open(V3_TRACE) as f:
        for line in f:
            r = json.loads(line)
            if r.get("__marker__") == "profiling_start":
                in_profile = True
                continue
            if r.get("__marker__") == "profiling_stop":
                in_profile = False
                continue
            if not in_profile or r.get("_header"):
                continue
            if "step_cycle_us" not in r:
                continue
            tt = r["total_tokens"]
            nnr = r.get("num_new_reqs", 0)
            conc = nnr + r.get("num_decode_seqs", 0)
            if conc < 1:
                conc = 1
            # Apply archive bucketing: tt-width=1, conc-width=5.
            ttb = tt
            cb = (conc // 5) * 5 + 2
            groups_by_newreqs[nnr][(ttb, cb)].append(r["step_cycle_us"])

    # Compare (tt, conc, new_reqs=0) sub-buckets to archive (tt, conc).
    # These are pure decode steps — what validation workload hits most.
    common_keys = [k for k in groups_by_newreqs[0].keys() if k in archive]
    print(f"v3 (new_reqs=0) buckets matching archive decode: {len(common_keys)}")
    if not common_keys:
        print("no overlap — nothing to compare")
        return

    # Mean diff: v3[decode, new_reqs=0] vs archive[decode]
    diffs_3d = []
    diffs_2d = []
    for k in common_keys:
        a_samples = archive[k].get("samples") or []
        v3_3d_samples = groups_by_newreqs[0].get(k, [])
        # 2D combined (all new_reqs for this (tt, conc))
        v3_2d_samples = sum((groups_by_newreqs[nnr].get(k, [])
                             for nnr in groups_by_newreqs), [])
        if len(a_samples) < 20 or len(v3_3d_samples) < 20 or len(v3_2d_samples) < 20:
            continue
        a_mean = sum(a_samples) / len(a_samples)
        v3_3d_mean = sum(v3_3d_samples) / len(v3_3d_samples)
        v3_2d_mean = sum(v3_2d_samples) / len(v3_2d_samples)
        diffs_3d.append(100 * (v3_3d_mean - a_mean) / a_mean)
        diffs_2d.append(100 * (v3_2d_mean - a_mean) / a_mean)

    if not diffs_3d:
        print("no buckets with enough samples")
        return

    diffs_3d.sort()
    diffs_2d.sort()
    n = len(diffs_3d)
    print(f"\nBuckets with ≥20 samples in both: {n}")
    print(f"\n=== v3 2D (all new_reqs merged) vs archive ===")
    print(f"  median Δmean%: {diffs_2d[n//2]:+.2f}%")
    print(f"  mean   Δmean%: {sum(diffs_2d)/n:+.2f}%")
    print(f"  p10 Δmean%:    {diffs_2d[n//10]:+.2f}%")
    print(f"  p90 Δmean%:    {diffs_2d[n*9//10]:+.2f}%")
    print(f"\n=== v3 3D (new_reqs=0 only) vs archive ===")
    print(f"  median Δmean%: {diffs_3d[n//2]:+.2f}%")
    print(f"  mean   Δmean%: {sum(diffs_3d)/n:+.2f}%")
    print(f"  p10 Δmean%:    {diffs_3d[n//10]:+.2f}%")
    print(f"  p90 Δmean%:    {diffs_3d[n*9//10]:+.2f}%")

    # If 3D bias is much smaller than 2D bias, F4 should rescue v3.
    if abs(sum(diffs_3d) / n) < abs(sum(diffs_2d) / n) - 2:
        print("\n=> F4 (3D axis) WOULD likely reduce v3 bias materially.")
    elif abs(sum(diffs_3d) / n) < abs(sum(diffs_2d) / n):
        print("\n=> F4 would modestly help but does not eliminate bias.")
    else:
        print("\n=> F4 does NOT rescue v3; bias is not about new_reqs.")


if __name__ == "__main__":
    main()
