#!/bin/bash
# Build a serving profile for a given model/tp config.
# Usage: ./build_serving_profile.sh <model> <tp> <label>
# Example: ./build_serving_profile.sh Qwen/Qwen2.5-3B-Instruct 2 3b-tp2

set -e
source /workspace/vllm-v18-env/bin/activate

MODEL="${1:-Qwen/Qwen2.5-1.5B-Instruct}"
TP="${2:-1}"
LABEL="${3:-1.5b-tp1}"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"
STEP_CYCLE_FILE="${RESULT_DIR}/step_cycle_${LABEL}.jsonl"
SWEEP_PROFILE="${RESULT_DIR}/profiles/sweep-${LABEL}.json"
SERVING_PROFILE="${RESULT_DIR}/profiles/serving-${LABEL}-step-cycle.json"
PORT=8100

export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub

echo "Building serving profile: model=${MODEL}, tp=${TP}, label=${LABEL}"

# Clean up
pkill -9 -f EngineCore 2>/dev/null || true
pkill -9 -f api_server 2>/dev/null || true
sleep 3
rm -f "${STEP_CYCLE_FILE}"

# Start server with step-cycle tracing
echo "Starting server with step-cycle tracing..."
VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
VLLM_EMULATOR_STEP_TRACE_OUTPUT="${STEP_CYCLE_FILE}" \
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 4096 \
    --port ${PORT} --trust-remote-code \
    --tensor-parallel-size ${TP} \
    > /workspace/serving_trace_${LABEL}_server.log 2>&1 &
SERVER_PID=$!

# Wait for server
echo "Waiting for server..."
for i in $(seq 1 180); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
        echo "Server ready after ${i}s"
        break
    fi
    sleep 1
done

if ! curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
    echo "FAILED: server didn't start"
    kill -9 ${SERVER_PID} 2>/dev/null
    pkill -9 -f EngineCore 2>/dev/null
    exit 1
fi

# Run benchmarks at multiple rates
for rate in 1 2 4; do
    echo "  Tracing at rate=${rate}..."
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "${MODEL}" --base-url http://localhost:${PORT} \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 30 --request-rate ${rate} \
        > /dev/null 2>&1
done

# Stop server
kill ${SERVER_PID} 2>/dev/null
wait ${SERVER_PID} 2>/dev/null || true
pkill -9 -f EngineCore 2>/dev/null || true
sleep 5

echo "Step cycle trace: $(wc -l < ${STEP_CYCLE_FILE}) records"

# Convert to serving profile
python3 -c "
import json, statistics
from collections import defaultdict

records = []
for line in open('${STEP_CYCLE_FILE}'):
    r = json.loads(line)
    if 'total_tokens' in r:
        records.append(r)

print(f'Records with batch info: {len(records)}')

by_tt = defaultdict(list)
for r in records:
    by_tt[r['total_tokens']].append(r['step_cycle_us'])

forward_pass = []
for tt in sorted(by_tt):
    lats = by_tt[tt]
    if len(lats) >= 2:
        forward_pass.append({
            'total_tokens': tt,
            'latency_us': round(statistics.median(lats), 1),
            'num_samples': len(lats),
        })

# Merge with sweep profile for large tt
if len(forward_pass) > 0:
    max_tt = max(e['total_tokens'] for e in forward_pass)
    try:
        sweep = json.load(open('${SWEEP_PROFILE}'))
        for e in sweep['forward_pass']:
            if e['total_tokens'] > max_tt:
                forward_pass.append(e)
    except FileNotFoundError:
        print('Warning: no sweep profile to merge')

profile = {
    'gpu_model': 'RTX-3060-12GB',
    'model_name': '${MODEL}',
    'profile_type': 'serving_step_cycle',
    'forward_pass': sorted(forward_pass, key=lambda e: e['total_tokens']),
}
json.dump(profile, open('${SERVING_PROFILE}', 'w'), indent=2)
print(f'Serving profile saved: {len(forward_pass)} buckets')
for e in profile['forward_pass'][:10]:
    print(f'  tt={e[\"total_tokens\"]:>4}: {e[\"latency_us\"]/1000:.1f}ms (n={e.get(\"num_samples\",0)})')
"

echo "DONE: ${SERVING_PROFILE}"
