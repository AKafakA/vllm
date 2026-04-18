#!/usr/bin/env python3
"""Apply the F1 outlier filter post-hoc to an already-built profile pack.

The profile stores raw samples per bucket, so applying the filter on those
samples gives the same output as rebuilding from the trace with
--outlier-filter=<method>. Used for F1 A/B when the original trace is
unavailable (e.g. the archived profile).

Shares _filter_outliers implementation with the builder so method
definitions stay in one place.

Usage:
    python3 tools/apply_outlier_filter.py INPUT OUTPUT --method iqr
"""
import argparse
import json
import sys


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", help="Source profile JSON")
    p.add_argument("output", help="Filtered profile JSON")
    p.add_argument("--method",
                   choices=("none", "iqr", "mad", "winsor"),
                   default="iqr")
    return p.parse_args()


def main():
    args = parse_args()
    # Lazy import so this tool can run without torch/vllm side-deps.
    sys.path.insert(0, ".")
    from vllm_emulator.profile.build_serving_profile_filtered import (
        _filter_outliers,
    )

    with open(args.input) as f:
        prof = json.load(f)

    total_in = 0
    total_out = 0
    buckets_filtered = 0
    for table_key in ("step_cycle_2d_distribution",
                      "prefill_2d_distribution",
                      "decode_2d_distribution",
                      "prefill_axis_distribution",
                      "decode_axis_distribution",
                      "step_cycle_axis_distribution"):
        table = prof.get(table_key) or []
        for bucket in table:
            samples = bucket.get("samples") or []
            total_in += len(samples)
            if args.method == "none":
                continue
            before = len(samples)
            filtered = _filter_outliers(samples, args.method)
            bucket["samples"] = filtered
            bucket["num_samples"] = len(filtered)
            if len(filtered) != before:
                buckets_filtered += 1
            total_out += len(filtered)

    if args.method != "none" and total_in > 0:
        dropped = total_in - total_out
        pct = 100.0 * dropped / total_in
        print(f"outlier_filter={args.method}: dropped {dropped} of {total_in} "
              f"samples ({pct:.2f}%) across {buckets_filtered} buckets")

    with open(args.output, "w") as f:
        json.dump(prof, f, indent=2)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
