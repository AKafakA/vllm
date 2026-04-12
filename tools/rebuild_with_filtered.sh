#!/bin/bash
# Rebuild profile from existing adaptive trace using FILTERED builder.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

LOG=/tmp/vllm_rebuild_filtered.log
echo "=== Rebuild with filtered builder at $(date) ===" > "$LOG"

TRACE="./results/RTX-8000-adaptive/profiles/step_cycle_adaptive.jsonl"
PROFILE_FILTERED="./results/RTX-8000-adaptive/profiles/serving-Qwen3-8B-adaptive-filtered.json"

echo "Building with filtered builder..." >> "$LOG"
python3 vllm_emulator/profile/build_serving_profile_filtered.py \
    "$TRACE" "/dev/null" "$PROFILE_FILTERED" >> "$LOG" 2>&1

# Verify new profile has proper fields
python3 -c "
import json
p = json.load(open('$PROFILE_FILTERED'))
print(f'Profile size: {len(json.dumps(p))} bytes')
print(f'decode_forward_pass: {len(p.get(\"decode_forward_pass\", []))} buckets')
print(f'prefill_forward_pass: {len(p.get(\"prefill_forward_pass\", []))} buckets')
print(f'step_cycle_2d_table: {len(p.get(\"step_cycle_2d_table\", []))} cells')
print(f'decode_2d_table: {len(p.get(\"decode_2d_table\", []))} cells')
print(f'prefill_2d_table: {len(p.get(\"prefill_2d_table\", []))} cells')
print(f'decode_2d_distribution: {len(p.get(\"decode_2d_distribution\", []))} cells')
# Show conc buckets
conc = sorted(set(e['conc'] for e in p.get('step_cycle_2d_table', [])))
print(f'Conc buckets: {conc}')
# Sample check
for bucket in p.get('decode_forward_pass', [])[:3]:
    n = bucket.get('num_samples', 0)
    samples = bucket.get('samples', [])
    print(f'  tt={bucket[\"total_tokens\"]}: p50={bucket[\"latency_us\"]/1000:.1f}, p99={bucket.get(\"p99_us\", 0)/1000:.1f if bucket.get(\"p99_us\") else 0}ms, samples={len(samples)}, n={n}')
" >> "$LOG" 2>&1

echo "" >> "$LOG"
echo "=== Done at $(date) ===" >> "$LOG"
cat "$LOG"
