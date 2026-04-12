#!/bin/bash
# Rebuild profile from adaptive trace using filtered builder.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

LOG=/tmp/vllm_filtered_rebuild.log
echo "=== Filtered rebuild at $(date) ===" > "$LOG"

TRACE="./results/RTX-8000-adaptive/profiles/step_cycle_adaptive.jsonl"
PROFILE="./results/RTX-8000-adaptive/profiles/serving-Qwen3-8B-adaptive-filtered.json"

python3 vllm_emulator/profile/build_serving_profile_filtered.py \
    "$TRACE" "/dev/null" "$PROFILE" 2>&1 | tee -a "$LOG"

echo "" >> "$LOG"
echo "=== Profile verification ===" >> "$LOG"
python3 -c "
import json
p = json.load(open('$PROFILE'))
print('Top keys:', sorted(p.keys()))
print(f'decode_forward_pass: {len(p.get(\"decode_forward_pass\", []))} buckets')
print(f'step_cycle_2d_table: {len(p.get(\"step_cycle_2d_table\", []))} cells')
print(f'decode_2d_table: {len(p.get(\"decode_2d_table\", []))} cells')
print(f'prefill_2d_table: {len(p.get(\"prefill_2d_table\", []))} cells')
conc = sorted(set(e['conc'] for e in p.get('step_cycle_2d_table', [])))
print(f'Conc buckets: {conc}')
# Sample bucket check
for b in p.get('decode_forward_pass', [])[:3]:
    n = b.get('num_samples', 0)
    p99 = b.get('p99_us')
    print(f'  tt={b[\"total_tokens\"]}: p50={b[\"latency_us\"]/1000:.1f}, p99={p99/1000:.1f if p99 else 0}, n={n}')
" 2>&1 | tee -a "$LOG"

echo "=== Done at $(date) ===" >> "$LOG"
