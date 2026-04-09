#!/bin/bash
source /workspace/vllm-v18-env/bin/activate
python3 << 'PYEOF'
import json
from collections import defaultdict

p = json.load(open("/workspace/eval_results/RTX-3060-12GB/profiles/serving-1.5b-tp1-filtered.json"))
table = p.get("step_cycle_2d_table", [])

print(f"2D table: {len(table)} cells")

# Group by conc
by_conc = defaultdict(list)
for e in table:
    by_conc[e["conc"]].append(e)

# Check for outliers: entries where latency is >3x neighbors at same conc
print(f"\nPer-conc summary:")
for conc in sorted(by_conc):
    entries = sorted(by_conc[conc], key=lambda e: e["tt"])
    lats = [e["latency_us"]/1000 for e in entries]
    tts = [e["tt"] for e in entries]
    samples = [e["num_samples"] for e in entries]
    print(f"  conc={conc:>4}: {len(entries)} entries, "
          f"tt={tts[0]}-{tts[-1]}, "
          f"lat={min(lats):.1f}-{max(lats):.1f}ms, "
          f"samples={min(samples)}-{max(samples)}")

    # Check for jumps >3x neighbor
    for i in range(1, len(entries)-1):
        prev_lat = entries[i-1]["latency_us"]
        curr_lat = entries[i]["latency_us"]
        next_lat = entries[i+1]["latency_us"]
        neighbor_avg = (prev_lat + next_lat) / 2
        if neighbor_avg > 0 and curr_lat > neighbor_avg * 3:
            print(f"    OUTLIER: tt={entries[i]['tt']} lat={curr_lat/1000:.1f}ms "
                  f"vs neighbors {prev_lat/1000:.1f}/{next_lat/1000:.1f}ms "
                  f"(n={entries[i]['num_samples']})")

# Also check 1D sections for outliers
for section in ["prefill_forward_pass", "decode_forward_pass", "forward_pass"]:
    entries = p.get(section, [])
    if not entries:
        continue
    print(f"\n{section}: {len(entries)} entries")
    for i in range(1, len(entries)-1):
        prev = entries[i-1]["latency_us"]
        curr = entries[i]["latency_us"]
        nxt = entries[i+1]["latency_us"]
        avg = (prev + nxt) / 2
        if avg > 0 and curr > avg * 2.5:
            print(f"  OUTLIER: tt={entries[i]['total_tokens']} lat={curr/1000:.1f}ms "
                  f"vs neighbors {prev/1000:.1f}/{nxt/1000:.1f}ms "
                  f"(n={entries[i].get('num_samples','?')})")
PYEOF
