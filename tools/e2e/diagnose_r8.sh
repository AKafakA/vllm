#!/bin/bash
# Diagnose R=8 underprediction: compare decode 2D table vs real step_cycle
source /workspace/vllm-v18-env/bin/activate

python3 << 'PYEOF'
import json, statistics
from collections import defaultdict

# Load profile
profile = json.load(open("/workspace/eval_results/RTX-3060-12GB/profiles/serving-1.5b-tp1-split2d.json"))

# Decode 2D table
decode_2d = profile.get("decode_2d_table", [])
print(f"=== Decode 2D Table ===")
print(f"Entries: {len(decode_2d)}")
by_conc = defaultdict(list)
for e in decode_2d:
    by_conc[e["conc"]].append(e)
for conc in sorted(by_conc):
    entries = by_conc[conc]
    lats = [e["latency_us"]/1000 for e in entries]
    print(f"  conc={conc:>4}: {len(entries)} entries, lat={min(lats):.1f}-{max(lats):.1f}ms")
    # Show entries at low tt (decode-only, tt=1-10)
    low_tt = [e for e in entries if e["tt"] <= 12]
    for e in low_tt:
        print(f"    tt={e['tt']:>4}: {e['latency_us']/1000:.1f}ms (n={e['num_samples']})")

# Now load real R=8 trace to see what concurrency actually looks like
print(f"\n=== Real R=8 Baseline Step Cycle ===")
# Check if we have a traced R=8 benchmark
import os, glob
traces = glob.glob("/workspace/eval_results/RTX-3060-12GB/step_cycle_adaptive.jsonl")
if traces:
    records = []
    for line in open(traces[0]):
        r = json.loads(line)
        if "total_tokens" in r:
            records.append(r)

    # Look at decode steps at R=8-like concurrency (10-30 concurrent)
    decode_steps = [r for r in records[5000:] if r.get("num_new_reqs", 0) == 0
                    and r.get("num_decode_seqs", 0) >= 10]

    print(f"Decode steps with conc>=10: {len(decode_steps)}")

    # Group by concurrency
    by_conc_real = defaultdict(list)
    for r in decode_steps:
        conc = r["num_decode_seqs"]
        by_conc_real[conc].append(r["step_cycle_us"])

    print(f"\nReal decode step_cycle by concurrency (R=8 range):")
    for conc in sorted(by_conc_real):
        if 5 <= conc <= 50:
            lats = by_conc_real[conc]
            med = statistics.median(lats)
            print(f"  conc={conc:>3}: n={len(lats):>5}, median={med/1000:.1f}ms, "
                  f"min={min(lats)/1000:.1f}ms, max={max(lats)/1000:.1f}ms")

    # What total_tokens do we see at R=8 concurrency?
    print(f"\nTotal tokens distribution at conc=10-30:")
    r8_like = [r for r in decode_steps if 10 <= r["num_decode_seqs"] <= 30]
    if r8_like:
        tts = [r["total_tokens"] for r in r8_like]
        cycles = [r["step_cycle_us"]/1000 for r in r8_like]
        print(f"  n={len(r8_like)}, tt: min={min(tts)} max={max(tts)} median={statistics.median(tts):.0f}")
        print(f"  step_cycle: min={min(cycles):.1f}ms max={max(cycles):.1f}ms median={statistics.median(cycles):.1f}ms")

# Compare: what does the oracle predict at R=8 operating point?
print(f"\n=== Oracle Prediction at R=8 Operating Point ===")
# R=8 with input=256, output=128: decode steps have tt=10-30, conc=10-30
CONC_BOUNDARIES = [1, 3, 5, 10, 20, 50, 100, 200, 300]
def conc_bucket(n):
    for i in range(len(CONC_BOUNDARIES) - 1):
        if CONC_BOUNDARIES[i] <= n < CONC_BOUNDARIES[i + 1]:
            return (CONC_BOUNDARIES[i] + CONC_BOUNDARIES[i + 1]) // 2
    return CONC_BOUNDARIES[-1]

# Build decode 2D lookup
decode_map = {}
for e in decode_2d:
    decode_map[(e["tt"], e["conc"])] = e["latency_us"]

for tt in [2, 7, 12, 17, 22]:
    for conc in [10, 15, 20, 25, 30]:
        cb = conc_bucket(conc)
        val = decode_map.get((tt, cb))
        if val:
            print(f"  tt={tt:>3} conc={conc:>3} (bucket={cb:>3}): predict={val/1000:.1f}ms")

# Check the real benchmark baselines
print(f"\n=== Real Baselines ===")
for rate in [1, 4, 8]:
    f = f"/workspace/eval_results/RTX-3060-12GB/online/split2d_r{rate}_real.json"
    if os.path.exists(f):
        d = json.load(open(f))
        print(f"  R={rate}: TPOT={d['mean_tpot_ms']:.1f}ms TTFT={d['mean_ttft_ms']:.1f}ms "
              f"E2E={d.get('mean_e2el_ms',0):.1f}ms max_conc={d.get('max_concurrent_requests','?')}")
PYEOF
