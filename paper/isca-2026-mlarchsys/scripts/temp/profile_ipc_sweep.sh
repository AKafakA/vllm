#!/bin/bash
# Profile IPC scheduling overhead: sweep N=1..50 concurrent requests
# Measures TTFT at each N, computes overhead = TTFT - prefill_step_time
set -e
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"
PROFILE="${RESULT_DIR}/profiles/serving-1.5b-tp1-calibrated.json"
PORT=8100
MAX_N=50

pkill -9 -f EngineCore 2>/dev/null || true
pkill -9 -f api_server 2>/dev/null || true
sleep 5

echo "=== IPC overhead profiling (N=1..${MAX_N}) ==="

# Start real server
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > /workspace/ipc_profile_server.log 2>&1 &
for i in $(seq 1 120); do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "Server ready ($i s)"
        break
    fi
    sleep 1
done

# Warmup
echo "Warming up..."
for i in $(seq 1 10); do
    curl -s --max-time 10 http://localhost:$PORT/v1/completions \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$MODEL\",\"prompt\":\"warmup\",\"max_tokens\":3,\"temperature\":0}" > /dev/null
    sleep 0.2
done
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 30 --request-rate 2 > /dev/null 2>&1
sleep 3
echo "Warmup done"

# Run the sweep
python3 /workspace/vllm-emulator-v18/paper/isca-2026-mlarchsys/scripts/temp/profile_ipc_overhead.py \
    $PORT "$MODEL" "$PROFILE" "${RESULT_DIR}/profiles/ipc_overhead.json" $MAX_N

pkill -f api_server 2>/dev/null || true
sleep 2
pkill -9 -f EngineCore 2>/dev/null || true

echo ""
echo "=== Summary ==="
python3 -c "
import json
data = json.load(open('${RESULT_DIR}/profiles/ipc_overhead.json'))
print(f'Entries: {len(data)}')
for d in data:
    if d['num_reqs'] in [1, 2, 3, 5, 10, 20, 30, 50]:
        print(f'  N={d[\"num_reqs\"]:3d}: overhead={d[\"overhead_us\"]/1000:.1f}ms')
"
echo "DONE"
