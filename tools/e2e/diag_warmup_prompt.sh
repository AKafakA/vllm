#!/bin/bash
# Test if longer prompts cause timeout with sample_tokens blocking
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
    > /workspace/diag_warmup.log 2>&1 &

for i in $(seq 1 120); do if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then echo "Ready ($i s)"; break; fi; sleep 1; done

echo "=== Test 1: short prompt (should work) ==="
time curl -s --max-time 10 http://localhost:$PORT/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"Qwen/Qwen2.5-1.5B-Instruct","prompt":"Hello","max_tokens":3,"temperature":0}' && echo ""

echo "=== Test 2: medium prompt (40 words) ==="
# Generate the prompt
PROMPT=$(python3 -c "print('warmup ' * 40)")
time curl -s --max-time 60 http://localhost:$PORT/v1/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"Qwen/Qwen2.5-1.5B-Instruct\",\"prompt\":\"${PROMPT}\",\"max_tokens\":5,\"temperature\":0}" && echo ""

echo "=== Test 3: long prompt (256 tokens) ==="
PROMPT256=$(python3 -c "print('test word ' * 90)")
time curl -s --max-time 60 http://localhost:$PORT/v1/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"Qwen/Qwen2.5-1.5B-Instruct\",\"prompt\":\"${PROMPT256}\",\"max_tokens\":5,\"temperature\":0}" && echo ""

echo ""
echo "=== Engine Log ==="
grep -E "ExecutorHook|SampleTokens|FALLTHROUGH|throughput" /workspace/diag_warmup.log | head -30

pkill -f api_server 2>/dev/null || true; sleep 2; pkill -9 -f EngineCore 2>/dev/null || true
echo "DONE"
