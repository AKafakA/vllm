#!/bin/bash
# Apples-to-apples rate=1 test: identical warmup for both real and emu
set -e
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"
PROFILE="${RESULT_DIR}/profiles/serving-1.5b-tp1-full.json"
PORT=8100
ONLINE_DIR="${RESULT_DIR}/online"

heavy_warmup() {
    # 5 basic curl requests
    for i in $(seq 1 5); do
        curl -s --max-time 10 http://localhost:$PORT/v1/completions \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"$MODEL\",\"prompt\":\"warmup test\",\"max_tokens\":3,\"temperature\":0}" > /dev/null
        sleep 0.2
    done
    # bench-serve 30 prompts at rate=1
    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 30 --request-rate 1 > /dev/null 2>&1
    sleep 2
    echo "  Warmup done"
}

pkill -9 -f EngineCore 2>/dev/null || true; pkill -9 -f api_server 2>/dev/null || true; sleep 5

# === REAL ===
echo "=== Real rate=1 ==="
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > /workspace/a2a_real.log 2>&1 &
for i in $(seq 1 120); do if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then break; fi; sleep 1; done
heavy_warmup

python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 50 --request-rate 1 --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
    --save-result --result-dir "$ONLINE_DIR" --result-filename "a2a_real_rate1.json" 2>&1 | grep -E "Mean|Median|P50|P99"

pkill -f api_server 2>/dev/null || true; sleep 2; pkill -9 -f EngineCore 2>/dev/null || true; sleep 5

# === EMULATOR ===
echo ""
echo "=== Emu rate=1 ==="
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
VLLM_EMULATOR_CUDA_GRAPH_WARMUP_US=80000 \
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > /workspace/a2a_emu.log 2>&1 &
for i in $(seq 1 120); do if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then break; fi; sleep 1; done
heavy_warmup

python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 50 --request-rate 1 --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
    --save-result --result-dir "$ONLINE_DIR" --result-filename "a2a_emu_rate1.json" 2>&1 | grep -E "Mean|Median|P50|P99"

pkill -f api_server 2>/dev/null || true; sleep 2; pkill -9 -f EngineCore 2>/dev/null || true

echo ""
echo "=== COMPARE ==="
python3 /workspace/vllm-emulator-v18/tools/e2e/compare_results.py \
    "$ONLINE_DIR/a2a_real_rate1.json" "$ONLINE_DIR/a2a_emu_rate1.json"
echo "DONE"
