#!/bin/bash
# Profile cold-start overhead: first prefill vs warm prefill
set -e
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"
PORT=8100

pkill -9 -f EngineCore 2>/dev/null || true; pkill -9 -f api_server 2>/dev/null || true; sleep 5

echo "=== Profile cold-start overhead ==="

# Start server with step-cycle tracing
rm -f "${RESULT_DIR}/cold_start_trace.jsonl"
VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
VLLM_EMULATOR_STEP_TRACE_OUTPUT="${RESULT_DIR}/cold_start_trace.jsonl" \
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 4096 --port ${PORT} --trust-remote-code \
    > /workspace/cold_start_server.log 2>&1 &
for i in $(seq 1 120); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then break; fi; sleep 1
done

# Send single requests one at a time (rate=0.5) to see cold vs warm
echo "  Sending 20 single requests at rate=0.5 (one at a time)..."
python3 -m vllm.entrypoints.cli.main bench serve --model "${MODEL}" --base-url http://localhost:${PORT} \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 20 --request-rate 0.5 \
    --percentile-metrics ttft --metric-percentiles 50,99 2>&1 | grep TTFT

pkill -f api_server 2>/dev/null || true; sleep 2; pkill -9 -f EngineCore 2>/dev/null || true

echo ""
echo "=== Cold-start analysis ==="
python3 -c "
import json

records = []
for line in open('${RESULT_DIR}/cold_start_trace.jsonl'):
    r = json.loads(line)
    if 'total_tokens' in r and r.get('num_new_reqs', 0) > 0:
        records.append(r)

print(f'Prefill steps: {len(records)}')
print()
print('First 10 prefill steps (first=cold, rest=warm):')
for i, r in enumerate(records[:10]):
    tt = r['total_tokens']
    lat = r['step_cycle_us'] / 1000
    print(f'  Step {i+1}: tt={tt}, latency={lat:.1f}ms')

if len(records) >= 5:
    cold = records[0]['step_cycle_us']
    warm_lats = [r['step_cycle_us'] for r in records[1:10]]
    import statistics
    warm = statistics.median(warm_lats)
    delta = cold - warm
    print(f'')
    print(f'Cold (first prefill): {cold/1000:.1f}ms')
    print(f'Warm (median of next 9): {warm/1000:.1f}ms')
    print(f'Cold-start overhead: {delta/1000:.1f}ms')
    print(f'')
    print(f'Recommendation: VLLM_EMULATOR_COLD_START_US={int(delta)}')
"
echo "DONE"
