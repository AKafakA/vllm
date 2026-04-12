#!/bin/bash
# Analyze the emu step_cycle breakdown to find missing overhead.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

python3 -c "
import json, statistics
from collections import defaultdict

# Load emu step_cycle from diagnostic run
emu = []
for line in open('./results/RTX-8000-quick/diag/emu_step_cycle_r4.jsonl'):
    r = json.loads(line)
    if '_header' in r or '__marker__' in r:
        continue
    if 'step_cycle_us' in r:
        emu.append(r)

# Load real step_cycle
real = []
for line in open('./results/RTX-8000-quick/diag/real_step_cycle_r4.jsonl'):
    r = json.loads(line)
    if '_header' in r or '__marker__' in r:
        continue
    if 'step_cycle_us' in r:
        real.append(r)

# Skip warmup
emu = emu[200:]
real = real[200:]

print(f'Real: {len(real)} steps, Emu: {len(emu)} steps')

# Emu breakdown: what components make up step_cycle_us?
# Fields: step_cycle_us, wait_ms, update_ms, sched_ms, exec_ms, sample_ms, queue_ms
print('\n=== Emu step_cycle breakdown (decode-only) ===')
emu_decode = [r for r in emu if r.get('num_new_reqs', 0) == 0]
real_decode = [r for r in real if r.get('num_new_reqs', 0) == 0]

if emu_decode:
    fields = ['step_cycle_us', 'wait_ms', 'update_ms', 'sched_ms', 'exec_ms', 'sample_ms', 'queue_ms']
    print(f'Emu decode steps: {len(emu_decode)}')
    for f in fields:
        vals = [r.get(f, 0) for r in emu_decode]
        if f == 'step_cycle_us':
            vals_ms = [v/1000 for v in vals]
        else:
            vals_ms = vals
        if vals_ms:
            print(f'  {f:>15}: median={statistics.median(vals_ms):.2f}ms, '
                  f'mean={statistics.mean(vals_ms):.2f}ms')

    # Compute the 'other' time: step_cycle - (wait + update + sched + exec + sample + queue)
    others = []
    for r in emu_decode:
        sc = r['step_cycle_us'] / 1000  # convert to ms
        accounted = sum(r.get(f, 0) for f in ['wait_ms', 'update_ms', 'sched_ms', 'exec_ms', 'sample_ms', 'queue_ms'])
        others.append(sc - accounted)
    print(f'  {\"other (unaccounted)\":>15}: median={statistics.median(others):.2f}ms, '
          f'mean={statistics.mean(others):.2f}ms')

    # Total accounted vs step_cycle
    accounted_total = [sum(r.get(f, 0) for f in ['wait_ms', 'update_ms', 'sched_ms', 'exec_ms', 'sample_ms', 'queue_ms']) for r in emu_decode]
    print(f'  {\"accounted total\":>15}: median={statistics.median(accounted_total):.2f}ms')
    sc_total = [r['step_cycle_us']/1000 for r in emu_decode]
    print(f'  {\"step_cycle total\":>15}: median={statistics.median(sc_total):.2f}ms')

print(f'\nReal decode steps: {len(real_decode)}')
if real_decode:
    # Real step_cycle breakdown (if fields available)
    real_fields = ['step_cycle_us']
    if 'wait_ms' in real_decode[0]:
        real_fields += ['wait_ms', 'update_ms', 'sched_ms', 'exec_ms', 'sample_ms', 'queue_ms']
    for f in real_fields:
        vals = [r.get(f, 0) for r in real_decode]
        if f == 'step_cycle_us':
            vals_ms = [v/1000 for v in vals]
        else:
            vals_ms = vals
        if vals_ms:
            print(f'  {f:>15}: median={statistics.median(vals_ms):.2f}ms, '
                  f'mean={statistics.mean(vals_ms):.2f}ms')

# Compare step_cycle by concurrency
print('\n=== Real vs Emu step_cycle by concurrency (decode-only) ===')
print(f'{\"N\":>4} {\"Real_sc\":>10} {\"Emu_sc\":>10} {\"Emu_wait\":>10} {\"Emu_exec\":>10} {\"Emu_other\":>10} {\"Gap\":>10}')
real_by_n = defaultdict(list)
emu_by_n = defaultdict(list)
for r in real_decode:
    n = r.get('num_decode_seqs', 0)
    real_by_n[n].append(r)
for r in emu_decode:
    n = r.get('num_decode_seqs', 0)
    emu_by_n[n].append(r)

for n in sorted(set(list(real_by_n.keys()) + list(emu_by_n.keys()))):
    rr = real_by_n.get(n, [])
    ee = emu_by_n.get(n, [])
    if len(rr) < 3 or len(ee) < 3:
        continue
    r_sc = statistics.median([r['step_cycle_us']/1000 for r in rr])
    e_sc = statistics.median([r['step_cycle_us']/1000 for r in ee])
    e_wait = statistics.median([r.get('wait_ms', 0) for r in ee])
    e_exec = statistics.median([r.get('exec_ms', 0) for r in ee])
    e_other = e_sc - e_wait - e_exec - statistics.median([r.get('sched_ms', 0) + r.get('update_ms', 0) + r.get('queue_ms', 0) for r in ee])
    gap = e_sc - r_sc
    print(f'{n:>4} {r_sc:>10.1f} {e_sc:>10.1f} {e_wait:>10.1f} {e_exec:>10.1f} {e_other:>10.1f} {gap:>10.1f}')
" 2>&1 | tee /tmp/vllm_step_breakdown.log
