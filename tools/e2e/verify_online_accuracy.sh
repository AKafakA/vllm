#!/bin/bash
# Verify online accuracy: fresh real + emu at rates 1, 2, 4, 8
# 500 prompts each, independent server starts, same warmup
# Addresses concerns:
#   - Are TPOT/E2E/throughput really identical across rates?
#   - Does TTFT error change at high rates (queueing)?
#   - Are results reproducible (run real twice)?
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
# Use the v3 profile (same as previous online tests) for apples-to-apples
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
        --num-prompts $NP --request-rate $RATE --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
        --save-result --result-dir "$ONLINE_DIR" --result-filename "${TAG}.json" > /dev/null 2>&1
}

echo "============================================"
echo "=== Online Accuracy Verification ==="
echo "=== $NP prompts, rates 1,2,4,8 ==="
echo "=== Fresh real + emu baselines ==="
echo "============================================"

# Check profile exists
if [ ! -f "$PROFILE" ]; then
    echo "ERROR: Profile not found at $PROFILE"
    exit 1
fi

for RATE in 1 2 4 8; do
    TAG="verify_rate${RATE}"
    echo ""
    echo "========== Rate=$RATE =========="

    # REAL (fresh server)
    cleanup_gpu
    echo "  Starting REAL server..."
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
        > /workspace/verify_real_${RATE}.log 2>&1 &
    wait_server
    heavy_warmup
    run_bench $RATE "${TAG}_real"

    # EMU (fresh server)
    cleanup_gpu
    echo "  Starting EMU server..."
    VLLM_EMULATOR_ENABLE_ORACLE=1 \
    VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
    VLLM_EMULATOR_MODE=realtime \
    VLLM_EMULATOR_EXECUTOR_HOOK=1 \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
        > /workspace/verify_emu_${RATE}.log 2>&1 &
    wait_server
    heavy_warmup
    run_bench $RATE "${TAG}_emu"

    cleanup_gpu
    echo "  --- Rate=$RATE results ---"
    python3 "$SCRIPT_DIR/compare_results.py" "$ONLINE_DIR/${TAG}_real.json" "$ONLINE_DIR/${TAG}_emu.json"
done

# REPRODUCIBILITY: run real rate=2 again to measure variance
echo ""
echo "========== Reproducibility: Real rate=2 (run 2) =========="
cleanup_gpu
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > /workspace/verify_real_2b.log 2>&1 &
wait_server
heavy_warmup
run_bench 2 "verify_rate2_real_run2"
cleanup_gpu

echo "  --- Rate=2 run1 vs run2 (real-to-real variance) ---"
python3 "$SCRIPT_DIR/compare_results.py" "$ONLINE_DIR/verify_rate2_real.json" "$ONLINE_DIR/verify_rate2_real_run2.json"

echo ""
echo "============================================"
echo "=== FULL SUMMARY ==="
echo "============================================"
for RATE in 1 2 4 8; do
    TAG="verify_rate${RATE}"
    echo "--- Rate=$RATE ---"
    python3 "$SCRIPT_DIR/compare_results.py" "$ONLINE_DIR/${TAG}_real.json" "$ONLINE_DIR/${TAG}_emu.json" 2>/dev/null || echo "  MISSING"
done
echo ""
echo "--- Real-to-Real variance (rate=2) ---"
python3 "$SCRIPT_DIR/compare_results.py" "$ONLINE_DIR/verify_rate2_real.json" "$ONLINE_DIR/verify_rate2_real_run2.json" 2>/dev/null || echo "  MISSING"

echo ""
echo "=== Key metrics to check ==="
echo "1. Does TPOT error change across rates? (should vary if real)"
echo "2. Does TTFT error decrease at rate=8? (queueing should dominate)"
echo "3. Does E2E error change at rate=8? (TTFT becomes larger share)"
echo "4. Real-to-real variance — what is the baseline noise?"

echo ""
echo "DONE $(date)"
