#!/bin/bash
# Analyze the high-latency spike steps in the emu trace
source /workspace/vllm-v18-env/bin/activate
python3 << 'PYEOF'
import csv, statistics

rows = []
with open("/workspace/emu_r8_1k_hook_trace.csv") as f:
    for r in csv.DictReader(f):
        rows.append({k: float(v) for k, v in r.items()})

bench = rows[len(rows)//5:]
print(f"Benchmark steps: {len(bench)}")

# Find spike steps (timer > 30ms)
spikes = [r for r in bench if r["total_latency_us"]/1000 > 30]
normal = [r for r in bench if r["total_latency_us"]/1000 <= 30]

print(f"\nNormal steps (timer <= 30ms): {len(normal)}")
print(f"Spike steps (timer > 30ms):   {len(spikes)} ({len(spikes)/len(bench)*100:.1f}%)")

if spikes:
    print(f"\n=== Spike step details ===")
    print(f"{'step':>6} {'tt':>5} {'reqs':>5} {'new':>4} {'prefill':>8} {'oracle':>8} {'hybrid':>8} {'total':>8}")
    for s in spikes[:20]:
        print(f"{int(s['step']):>6} {int(s['tt']):>5} {int(s['n_reqs']):>5} "
              f"{int(s['n_new']):>4} {int(s['has_prefill']):>8} "
              f"{s['oracle_us']/1000:>7.1f}ms {s['hybrid_overhead_us']/1000:>7.1f}ms "
              f"{s['total_latency_us']/1000:>7.1f}ms")

    # Are all spikes prefill steps?
    prefill_spikes = sum(1 for s in spikes if s["has_prefill"] > 0)
    decode_spikes = len(spikes) - prefill_spikes
    print(f"\n  Prefill spikes: {prefill_spikes}")
    print(f"  Decode-only spikes: {decode_spikes}")

    # tt distribution of spikes
    spike_tts = [int(s["tt"]) for s in spikes]
    print(f"\n  Spike tt: min={min(spike_tts)}, max={max(spike_tts)}, median={statistics.median(spike_tts):.0f}")

    # Oracle prediction for spike tt values
    print(f"\n  Oracle predictions at spike tt values:")
    for tt in sorted(set(spike_tts))[:10]:
        matching = [s for s in spikes if int(s["tt"]) == tt]
        oracle = matching[0]["oracle_us"]/1000
        print(f"    tt={tt}: oracle={oracle:.1f}ms (n={len(matching)})")

# Compare: what does real GPU step-cycle look like at same tt values?
print(f"\n=== Real GPU at spike tt values ===")
import json
real_records = []
for line in open("/workspace/real_r8_1k_trace.jsonl"):
    r = json.loads(line)
    if "total_tokens" in r:
        real_records.append(r)

real_bench = real_records[len(real_records)//5:]
for tt in sorted(set(spike_tts))[:10]:
    matching_real = [r for r in real_bench if r["total_tokens"] == tt]
    if matching_real:
        med = statistics.median([r["step_cycle_us"] for r in matching_real])
        print(f"  tt={tt}: real_step_cycle={med/1000:.1f}ms (n={len(matching_real)})")
    else:
        # find nearest
        nearest = min(real_bench, key=lambda r: abs(r["total_tokens"] - tt))
        print(f"  tt={tt}: no exact match, nearest tt={nearest['total_tokens']} cycle={nearest['step_cycle_us']/1000:.1f}ms")
PYEOF
