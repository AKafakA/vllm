#!/bin/bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

python3 << 'PYEOF'
import json
import statistics
from collections import defaultdict

recs = []
for line in open('results/RTX-8000-quick/diag/real_step_cycle_r4.jsonl'):
    r = json.loads(line)
    if r.get('_header') or 'step_cycle_us' not in r:
        continue
    recs.append(r)
recs = recs[200:]

by_tt = defaultdict(list)
for r in recs:
    tt = r['total_tokens']
    by_tt[tt].append(r['step_cycle_us'])

print(f"{'tt':>5} {'p50':>8} {'mean':>8} {'std':>8} {'p90':>8} {'p99':>8} {'n':>5}")
for tt in sorted(by_tt.keys())[:30]:
    vals = sorted(by_tt[tt])
    if len(vals) < 10:
        continue
    p50 = statistics.median(vals)
    p90 = vals[int(len(vals)*0.9)]
    p99 = vals[int(len(vals)*0.99)]
    mean = statistics.mean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0
    print(f"{tt:>5} {p50/1000:>7.1f} {mean/1000:>7.1f} {std/1000:>7.1f} {p90/1000:>7.1f} {p99/1000:>7.1f} {len(vals):>5}")

print()
print("Concurrency-level analysis:")
# Group by concurrency
by_n = defaultdict(list)
for r in recs:
    n = r.get('num_new_reqs', 0) + r.get('num_decode_seqs', 0)
    by_n[n].append(r['step_cycle_us'])

print(f"{'N':>4} {'p50':>8} {'mean':>8} {'std':>8} {'p90':>8} {'p99':>8} {'n':>5}")
for n in sorted(by_n.keys()):
    vals = sorted(by_n[n])
    if len(vals) < 10:
        continue
    p50 = statistics.median(vals)
    p90 = vals[int(len(vals)*0.9)]
    p99 = vals[int(len(vals)*0.99)]
    mean = statistics.mean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0
    print(f"{n:>4} {p50/1000:>7.1f} {mean/1000:>7.1f} {std/1000:>7.1f} {p90/1000:>7.1f} {p99/1000:>7.1f} {len(vals):>5}")
PYEOF
