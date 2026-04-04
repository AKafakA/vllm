#!/bin/bash
# Test mock path at rate=2 ONLY (skip rate=1 to isolate the bug)
set -e
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub

MODEL="Qwen/Qwen2.5-0.5B-Instruct"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"
PORT=8100

pkill -9 -f EngineCore 2>/dev/null || true
pkill -9 -f api_server 2>/dev/null || true
sleep 5

VLLM_EMULATOR_MOCK_CUDA=1 \
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="${RESULT_DIR}/profiles/serving-0.5b-tp1-step-cycle.json" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
VLLM_EMULATOR_MEMORY=$((80*1024*1024*1024)) \
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 4096 \
    --port ${PORT} --trust-remote-code \
    --load-format dummy --enforce-eager \
    > /workspace/mock_rate2_server.log 2>&1 &

for i in $(seq 1 60); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
        echo "Server ready after ${i}s"
        break
    fi
    sleep 1
done

echo "Testing rate=2 directly..."
python3 -m vllm.entrypoints.cli.main bench serve \
    --model "${MODEL}" --base-url http://localhost:${PORT} \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 10 --request-rate 2 \
    --percentile-metrics ttft,tpot --metric-percentiles 50,99 2>&1 | grep -E "TTFT|TPOT|completed|failed"

echo ""
echo "Checking server health after benchmark..."
curl -s http://localhost:${PORT}/health && echo " Still UP" || echo " DOWN"

echo ""
echo "Debug logs:"
grep 'ExecutorHook.*step=\|unexpected\|queue_len\|_resolve error' /workspace/mock_rate2_server.log 2>/dev/null | tail -10

pkill -f api_server 2>/dev/null || true
pkill -9 -f EngineCore 2>/dev/null || true
echo "DONE"
