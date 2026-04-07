#!/bin/bash
# Quick diagnostic: start emu server, send 2 requests, check logs
set -e
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"
PROFILE="${RESULT_DIR}/profiles/serving-1.5b-tp1-full.json"
PORT=8100

pkill -9 -f EngineCore 2>/dev/null || true; pkill -9 -f api_server 2>/dev/null || true; sleep 5

VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > /workspace/diag_single.log 2>&1 &

for i in $(seq 1 120); do if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then echo "Ready ($i s)"; break; fi; sleep 1; done

echo "=== Sending 2 requests ==="
curl -s --max-time 60 http://localhost:$PORT/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"Qwen/Qwen2.5-1.5B-Instruct","prompt":"Hello world","max_tokens":5,"temperature":0}' && echo ""

sleep 1

curl -s --max-time 60 http://localhost:$PORT/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"Qwen/Qwen2.5-1.5B-Instruct","prompt":"Testing emulator","max_tokens":5,"temperature":0}' && echo ""

sleep 2

echo ""
echo "=== Engine Log ==="
grep -E "ExecutorHook|SampleTokens|FALLTHROUGH" /workspace/diag_single.log

echo ""
echo "=== Throughput ==="
grep "throughput" /workspace/diag_single.log | tail -5

pkill -f api_server 2>/dev/null || true; sleep 2; pkill -9 -f EngineCore 2>/dev/null || true
echo "DONE"
