#!/usr/bin/env python3
"""Inject avg_sample_ms / avg_exec_ms into a pre-built profile from a
step_timing.csv. Used for F3 when the profile's source trace is
unavailable (e.g. the archived profile).

Equivalent to rebuilding the profile with --step-timing-csv. Both avg
fields come from measured real-hardware per-step timings; they are
profile data, not emu-vs-real delta.

Usage:
    python3 tools/inject_step_timing.py INPUT.json OUTPUT.json TIMING.csv
"""
import csv
import json
import sys


def main():
    if len(sys.argv) != 4:
        print("Usage: inject_step_timing.py INPUT OUTPUT TIMING.csv")
        sys.exit(1)
    input_path, output_path, csv_path = sys.argv[1:]

    with open(input_path) as f:
        prof = json.load(f)

    sample_vals = []
    exec_vals = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("sample_ms"):
                sample_vals.append(float(row["sample_ms"]))
            if row.get("exec_ms"):
                exec_vals.append(float(row["exec_ms"]))

    if sample_vals:
        prof["avg_sample_ms"] = round(sum(sample_vals) / len(sample_vals), 4)
        print(f"avg_sample_ms = {prof['avg_sample_ms']} (n={len(sample_vals)})")
    if exec_vals:
        prof["avg_exec_ms"] = round(sum(exec_vals) / len(exec_vals), 4)
        print(f"avg_exec_ms   = {prof['avg_exec_ms']} (n={len(exec_vals)})")

    with open(output_path, "w") as f:
        json.dump(prof, f, indent=2)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
