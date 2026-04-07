#!/bin/bash
# Full verification on clean codebase
# Online: fresh real + emu at rates 1, 2, 4, 8 (500 prompts each)
# Offline: bench throughput real + emu (200 prompts)
# Real-to-real variance check
set -e
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub
export CUDA_VISIBLE_DEVICES=1

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"
PORT=8100
SCRIPT_DIR="/workspace/vllm-emulator-v18/tools/e2e"
ONLINE_DIR="${RESULT_DIR}/online"
PROFILE="${RESULT_DIR}/profiles/serving-1.5b-tp1-v3.json"
NP=500

cleanup_gpu() {
    pkill -9 -f "python3.*api_server" 2>/dev/null || true
    pkill -9 -f "python3.*bench" 2>/dev/null || true
    fuser /dev/nvidia* 2>/dev/null | tr " " "\n" | sort -u | xargs kill -9 2>/dev/null || true
    sleep 5
}

wait_server() {
    for i in $(seq 1 120); do
        if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then return 0; fi
        sleep 1
    done
    echo "ERROR: Server failed to start"
    return 1
}

heavy_warmup() {
    echo "  Warmup (200 prompts, rate=4)..."
    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 200 --request-rate 4 > /dev/null 2>&1
    sleep 3
}

run_bench() {
    local RATE=$1 TAG=$2
    echo "  Bench rate=$RATE ($NP prompts)..."
    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts $NP --request-rate $RATE --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,99 \
        --save-result --result-dir "$ONLINE_DIR" --result-filename "${TAG}.json" > /dev/null 2>&1
}

# Verify profile exists
if [ ! -f "$PROFILE" ]; then
    echo "ERROR: Profile not found: $PROFILE"
    exit 1
fi

echo "============================================"
echo "=== CLEAN CODEBASE FULL VERIFICATION ==="
echo "=== $(date) ==="
echo "============================================"
echo ""
echo "Profile: $PROFILE"
echo "Prompts: $NP per rate"
echo "Rates: 1, 2, 4, 8"
echo ""

# =============================================
# PHASE 1: Online serving (bench serve)
# =============================================
echo "============================================"
echo "=== PHASE 1: Online serving ==="
echo "============================================"

for RATE in 1 2 4 8; do
    TAG="clean_rate${RATE}"
    echo ""
    echo "========== Rate=$RATE =========="

    # REAL
    cleanup_gpu
    echo "  Starting REAL server..."
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
        > /workspace/clean_real_${RATE}.log 2>&1 &
    wait_server
    heavy_warmup
    run_bench $RATE "${TAG}_real"

    # EMU
    cleanup_gpu
    echo "  Starting EMU server..."
    VLLM_EMULATOR_ENABLE_ORACLE=1 \
    VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
    VLLM_EMULATOR_MODE=realtime \
    VLLM_EMULATOR_EXECUTOR_HOOK=1 \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
        > /workspace/clean_emu_${RATE}.log 2>&1 &
    wait_server
    heavy_warmup
    run_bench $RATE "${TAG}_emu"

    cleanup_gpu
    echo "  --- Rate=$RATE results ---"
    python3 "$SCRIPT_DIR/compare_results.py" "$ONLINE_DIR/${TAG}_real.json" "$ONLINE_DIR/${TAG}_emu.json"
done

# =============================================
# PHASE 2: Real-to-real variance (rate=2)
# =============================================
echo ""
echo "============================================"
echo "=== PHASE 2: Real-to-real variance ==="
echo "============================================"

cleanup_gpu
echo "  Starting REAL server (run 2)..."
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > /workspace/clean_real_2b.log 2>&1 &
wait_server
heavy_warmup
run_bench 2 "clean_rate2_real_run2"
cleanup_gpu

echo "  --- Real run1 vs run2 ---"
python3 "$SCRIPT_DIR/compare_results.py" "$ONLINE_DIR/clean_rate2_real.json" "$ONLINE_DIR/clean_rate2_real_run2.json"

# =============================================
# PHASE 3: Offline throughput (bench throughput)
# =============================================
echo ""
echo "============================================"
echo "=== PHASE 3: Offline throughput ==="
echo "============================================"

cleanup_gpu

echo "  Real warmup..."
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "$MODEL" --max-model-len 4096 --trust-remote-code \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 > /dev/null 2>&1

echo "  Real benchmark (200 prompts)..."
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "$MODEL" --max-model-len 4096 --trust-remote-code \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 2>&1 | tee /workspace/clean_offline_real.txt | grep "Throughput:"

cleanup_gpu

echo "  Emu warmup..."
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_PROFILE_USAGE=online \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "$MODEL" --max-model-len 4096 --trust-remote-code \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 > /dev/null 2>&1

echo "  Emu benchmark (200 prompts)..."
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_PROFILE_USAGE=online \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "$MODEL" --max-model-len 4096 --trust-remote-code \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 2>&1 | tee /workspace/clean_offline_emu.txt | grep "Throughput:"

cleanup_gpu

# =============================================
# FULL SUMMARY
# =============================================
echo ""
echo "============================================"
echo "=== FULL SUMMARY ==="
echo "============================================"
echo ""
echo "--- Online Serving ---"
for RATE in 1 2 4 8; do
    TAG="clean_rate${RATE}"
    echo ""
    echo "Rate=$RATE:"
    python3 "$SCRIPT_DIR/compare_results.py" "$ONLINE_DIR/${TAG}_real.json" "$ONLINE_DIR/${TAG}_emu.json" 2>/dev/null || echo "  MISSING"
done

echo ""
echo "--- Real-to-Real Variance (rate=2) ---"
python3 "$SCRIPT_DIR/compare_results.py" "$ONLINE_DIR/clean_rate2_real.json" "$ONLINE_DIR/clean_rate2_real_run2.json" 2>/dev/null || echo "  MISSING"

echo ""
echo "--- Offline Throughput ---"
echo "  Real: $(grep 'Throughput:' /workspace/clean_offline_real.txt 2>/dev/null || echo MISSING)"
echo "  Emu:  $(grep 'Throughput:' /workspace/clean_offline_emu.txt 2>/dev/null || echo MISSING)"

echo ""
echo "============================================"
echo "=== DONE $(date) ==="
echo "============================================"
