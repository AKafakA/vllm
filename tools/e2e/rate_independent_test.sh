#!/bin/bash
# Test a SINGLE rate with fresh server instances (no sequential contamination)
# Usage: rate_independent_test.sh <rate> [num_prompts]
set -e
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub

RATE=${1:?Usage: $0 <rate> [num_prompts]}
NUM_PROMPTS=${2:-50}
MODEL="Qwen/Qwen2.5-1.5B-Instruct"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"
PROFILE="${RESULT_DIR}/profiles/serving-1.5b-tp1-calibrated.json"
PORT=8100
ONLINE_DIR="${RESULT_DIR}/online"
SCRIPT_DIR="/workspace/vllm-emulator-v18/tools/e2e"

# Rebuild profile (ensures latest calibration)
python3 /workspace/vllm-emulator-v18/vllm_emulator/profile/build_serving_profile.py \
    "${RESULT_DIR}/step_cycle_1.5b_full.jsonl" \
    "${RESULT_DIR}/profiles/sweep-1.5b-tp1-v14.json" \
    "$PROFILE" "$MODEL" "RTX-3060-12GB" > /dev/null 2>&1

heavy_warmup() {
    for i in $(seq 1 5); do
        curl -s --max-time 10 http://localhost:$PORT/v1/completions \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"$MODEL\",\"prompt\":\"warmup test\",\"max_tokens\":3,\"temperature\":0}" > /dev/null
        sleep 0.2
    done
    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 30 --request-rate $RATE > /dev/null 2>&1
    sleep 2
}

echo "=== Independent test: rate=$RATE, prompts=$NUM_PROMPTS ==="

# --- REAL (fresh) ---
pkill -9 -f EngineCore 2>/dev/null || true; pkill -9 -f api_server 2>/dev/null || true; sleep 5

echo "--- Real server ---"
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > /workspace/ind_real_${RATE}.log 2>&1 &
for i in $(seq 1 120); do if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then break; fi; sleep 1; done
heavy_warmup
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts $NUM_PROMPTS --request-rate $RATE --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
    --save-result --result-dir "$ONLINE_DIR" --result-filename "ind_real_rate${RATE}.json" > /dev/null 2>&1

pkill -f api_server 2>/dev/null || true; sleep 3; pkill -9 -f EngineCore 2>/dev/null || true; sleep 5

# --- EMU (fresh) ---
echo "--- Emu server ---"
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > /workspace/ind_emu_${RATE}.log 2>&1 &
for i in $(seq 1 120); do if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then break; fi; sleep 1; done
grep -m1 ExecutorEmulatorHook /workspace/ind_emu_${RATE}.log
heavy_warmup
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts $NUM_PROMPTS --request-rate $RATE --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
    --save-result --result-dir "$ONLINE_DIR" --result-filename "ind_emu_rate${RATE}.json" > /dev/null 2>&1

pkill -f api_server 2>/dev/null || true; sleep 3; pkill -9 -f EngineCore 2>/dev/null || true

echo ""
echo "=== Results rate=$RATE ==="
python3 "$SCRIPT_DIR/compare_results.py" \
    "$ONLINE_DIR/ind_real_rate${RATE}.json" "$ONLINE_DIR/ind_emu_rate${RATE}.json"
echo "DONE"
