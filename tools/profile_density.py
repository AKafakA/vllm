"""Report cell/sample density of a profile pack."""
import json, sys
from collections import Counter

p = sys.argv[1]
pack = json.load(open(p))

def dist_stats(section_key):
    section = pack.get(section_key, [])
    if not section:
        return None
    per_cell_samples = []
    tt_seen = set()
    conc_seen = set()
    for e in section:
        n = len(e.get("samples", []))
        if n > 0:
            per_cell_samples.append(n)
            tt_seen.add(e.get("tt", -1))
            conc_seen.add(e.get("conc", -1))
    if not per_cell_samples:
        return None
    per_cell_samples.sort()
    total = sum(per_cell_samples)
    n = len(per_cell_samples)
    return {
        "cells": n,
        "samples_total": total,
        "samples_mean_per_cell": total / n,
        "samples_median_per_cell": per_cell_samples[n // 2],
        "samples_p10": per_cell_samples[max(0, n // 10)],
        "samples_p90": per_cell_samples[min(n - 1, 9 * n // 10)],
        "samples_min": per_cell_samples[0],
        "samples_max": per_cell_samples[-1],
        "unique_tt": len(tt_seen),
        "unique_conc": len(conc_seen),
    }

print(f"Profile: {p}")
for section in ("prefill_2d_distribution", "decode_2d_distribution",
                "step_cycle_2d_distribution"):
    s = dist_stats(section)
    if s:
        print(f"\n--- {section} ---")
        for k, v in s.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.1f}")
            else:
                print(f"  {k}: {v}")
