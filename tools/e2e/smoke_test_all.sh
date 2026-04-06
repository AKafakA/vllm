#!/bin/bash
# Comprehensive smoke test: re-run ALL experiments with corrected code
# Previous results were invalid (silent fallback to real GPU due to missing 'version' field)
#
# Tests on Vast GPU host (RTX 3060):
#   1. Rebuild calibrated profile from existing trace
#   2. Online serving: 1.5B at rates 1,2,3,4 (50 prompts each)
#   3. Offline throughput: 1.5B
#   4. Online serving: 0.5B at rates 1,2,4 (50 prompts each)
#
# Each test: identical heavy warmup for real vs emu, A2A comparison
set -e
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub

PORT=8100
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"
ONLINE_DIR="${RESULT_DIR}/online"
SCRIPT_DIR="/workspace/vllm-emulator-v18/tools/e2e"
TIMESTAMP=$(date +%Y%m%d_%H%M)

mkdir -p "$ONLINE_DIR"

# === Helper functions ===
wait_for_server() {
    for i in $(seq 1 120); do
        if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
            echo "  Server ready ($i s)"
            return 0
        fi
        sleep 1
    done
    echo "  ERROR: Server not ready after 120s"
    return 1
}

heavy_warmup() {
    local rate=${1:-1}
    for i in $(seq 1 5); do
        curl -s --max-time 10 http://localhost:$PORT/v1/completions \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"$MODEL\",\"prompt\":\"warmup test\",\"max_tokens\":3,\"temperature\":0}" > /dev/null
        sleep 0.2
    done
    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 30 --request-rate $rate > /dev/null 2>&1
    sleep 2
    echo "  Warmup done (rate=$rate)"
}

kill_servers() {
    pkill -f api_server 2>/dev/null || true
    sleep 3
    pkill -9 -f EngineCore 2>/dev/null || true
    sleep 5
}

run_bench() {
    local rate=$1
    local num_prompts=$2
    local result_file=$3
    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts $num_prompts --request-rate $rate \
        --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
        --save-result --result-dir "$ONLINE_DIR" --result-filename "$result_file" > /dev/null 2>&1
}

# ==========================================================
# TEST 1: Rebuild calibrated profile for 1.5B
# ==========================================================
echo "============================================"
echo "=== TEST 1: Rebuild calibrated 1.5B profile ==="
echo "============================================"
python3 /workspace/vllm-emulator-v18/paper/isca-2026-mlarchsys/scripts/temp/build_serving_profile_2d.py \
    "${RESULT_DIR}/step_cycle_1.5b_full.jsonl" \
    "${RESULT_DIR}/profiles/sweep-1.5b-tp1-v14.json" \
    "${RESULT_DIR}/profiles/serving-1.5b-tp1-calibrated.json" \
    "Qwen/Qwen2.5-1.5B-Instruct" "RTX-3060-12GB"
PROFILE_1_5B="${RESULT_DIR}/profiles/serving-1.5b-tp1-calibrated.json"

# ==========================================================
# TEST 2: Online serving 1.5B at rates 1,2,3,4
# ==========================================================
MODEL="Qwen/Qwen2.5-1.5B-Instruct"

for RATE in 1 2 3 4; do
    # More prompts at low rates to ensure enough complete within duration
    if [ "$RATE" -le 1 ]; then NUM_PROMPTS=100; else NUM_PROMPTS=50; fi
    echo ""
    echo "============================================"
    echo "=== TEST 2.$RATE: 1.5B online rate=$RATE ==="
    echo "============================================"

    # --- Real ---
    kill_servers
    echo "  Starting real server..."
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
        > /workspace/smoke_real_${RATE}.log 2>&1 &
    wait_for_server
    heavy_warmup $RATE
    echo "  Benchmarking real rate=$RATE..."
    run_bench $RATE $NUM_PROMPTS "smoke_real_1.5b_rate${RATE}.json"

    # --- Emulator ---
    kill_servers
    echo "  Starting emulator server..."
    VLLM_EMULATOR_ENABLE_ORACLE=1 \
    VLLM_EMULATOR_PROFILE_PACK="$PROFILE_1_5B" \
    VLLM_EMULATOR_MODE=realtime \
    VLLM_EMULATOR_EXECUTOR_HOOK=1 \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
        > /workspace/smoke_emu_${RATE}.log 2>&1 &
    wait_for_server
    # Verify hook is active
    grep -m1 "ExecutorEmulatorHook" /workspace/smoke_emu_${RATE}.log || echo "  WARNING: Hook not found in log!"
    heavy_warmup $RATE
    echo "  Benchmarking emu rate=$RATE..."
    run_bench $RATE $NUM_PROMPTS "smoke_emu_1.5b_rate${RATE}.json"

    # --- Compare ---
    echo "  --- Rate=$RATE Results ---"
    python3 "$SCRIPT_DIR/compare_results.py" \
        "$ONLINE_DIR/smoke_real_1.5b_rate${RATE}.json" \
        "$ONLINE_DIR/smoke_emu_1.5b_rate${RATE}.json"
done

# ==========================================================
# TEST 3: Offline throughput 1.5B
# ==========================================================
echo ""
echo "============================================"
echo "=== TEST 3: Offline throughput 1.5B ==="
echo "============================================"
kill_servers

# Real offline
echo "  Real offline..."
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > /workspace/smoke_offline_real.log 2>&1 &
wait_for_server
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 100 --request-rate inf \
    --save-result --result-dir "$ONLINE_DIR" --result-filename "smoke_offline_real_1.5b.json" > /dev/null 2>&1

# Emu offline
kill_servers
echo "  Emu offline..."
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="$PROFILE_1_5B" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > /workspace/smoke_offline_emu.log 2>&1 &
wait_for_server
grep -m1 "ExecutorEmulatorHook" /workspace/smoke_offline_emu.log || echo "  WARNING: Hook not found!"
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 100 --request-rate inf \
    --save-result --result-dir "$ONLINE_DIR" --result-filename "smoke_offline_emu_1.5b.json" > /dev/null 2>&1

echo "  --- Offline Results ---"
python3 "$SCRIPT_DIR/compare_results.py" \
    "$ONLINE_DIR/smoke_offline_real_1.5b.json" \
    "$ONLINE_DIR/smoke_offline_emu_1.5b.json"

kill_servers

# ==========================================================
# SUMMARY
# ==========================================================
echo ""
echo "============================================"
echo "=== SUMMARY: All smoke tests ($TIMESTAMP) ==="
echo "============================================"
for RATE in 1 2 3 4; do
    echo ""
    echo "--- 1.5B rate=$RATE ---"
    python3 "$SCRIPT_DIR/compare_results.py" \
        "$ONLINE_DIR/smoke_real_1.5b_rate${RATE}.json" \
        "$ONLINE_DIR/smoke_emu_1.5b_rate${RATE}.json" 2>/dev/null || echo "  MISSING"
done
echo ""
echo "--- 1.5B offline ---"
python3 "$SCRIPT_DIR/compare_results.py" \
    "$ONLINE_DIR/smoke_offline_real_1.5b.json" \
    "$ONLINE_DIR/smoke_offline_emu_1.5b.json" 2>/dev/null || echo "  MISSING"

echo ""
echo "ALL SMOKE TESTS DONE ($TIMESTAMP)"
