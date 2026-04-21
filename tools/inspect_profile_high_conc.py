"""How dense is the new profile at high concurrency (where r=16/r=32 live)?"""
import json, sys
from collections import defaultdict

p = json.load(open(sys.argv[1]))

for section_name in ("prefill_2d_distribution", "decode_2d_distribution",
                      "step_cycle_2d_distribution"):
    section = p.get(section_name, [])
    if not section:
        continue
    print(f"\n=== {section_name} ===")
    # Group by conc bucket.
    by_conc = defaultdict(lambda: {"cells": 0, "samples": 0})
    for e in section:
        c = e.get("conc", -1)
        n = len(e.get("samples", []))
        by_conc[c]["cells"] += 1
        by_conc[c]["samples"] += n
    # Sort by conc, print.
    print(f"  {'conc':>5}  {'cells':>6}  {'samples':>8}  {'samp/cell':>10}")
    for c in sorted(by_conc.keys()):
        cells = by_conc[c]["cells"]
        samples = by_conc[c]["samples"]
        sc = samples / cells if cells > 0 else 0
        print(f"  {c:>5}  {cells:>6}  {samples:>8}  {sc:>10.1f}")

# Also print IPC table summary
ipc = p.get("sched_overhead_table", [])
if ipc:
    print(f"\n=== sched_overhead_table ===")
    print(f"  {'num_reqs':>8}  {'overhead_us':>12}  {'overhead_ms':>11}")
    for e in ipc:
        n = e.get("num_reqs", 0)
        o = e.get("overhead_us", 0)
        print(f"  {n:>8}  {o:>12.0f}  {o/1000:>11.2f}")
