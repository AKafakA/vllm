#!/bin/bash
# Test hybrid timer: step_cycle + overhead_per_req
# Uses fresh profile already on Vast
set -e
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub
export CUDA_VISIBLE_DEVICES=1

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
PROFILE="/workspace/eval_results/RTX-3060-12GB/profiles/serving-1.5b-tp1-fresh.json"
PORT=8100
SCRIPT_DIR="/workspace/vllm-emulator-v18/tools/e2e"
ONLINE_DIR="/workspace/eval_results/RTX-3060-12GB/online"

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
    echo "ERROR: Server failed to start"; return 1
}

echo "============================================"
echo "=== Hybrid Timer Test ==="
echo "=== $(date) ==="
echo "============================================"

# Test with auto-calibrated overhead
for RATE in 1 4; do
    TAG="hybrid_rate${RATE}"
    echo ""
    echo "========== Rate=$RATE (hybrid, auto-calibrated) =========="

    cleanup_gpu
    echo "  Emu server (hybrid)..."
    VLLM_EMULATOR_ENABLE_ORACLE=1 \
    VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
    VLLM_EMULATOR_MODE=realtime \
    VLLM_EMULATOR_EXECUTOR_HOOK=1 \
    VLLM_EMULATOR_ORACLE_MODE=hybrid \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
        > /workspace/hybrid_emu_${RATE}.log 2>&1 &
    wait_server
    grep "ExecutorEmulatorHook" /workspace/hybrid_emu_${RATE}.log | head -1

    echo "  Warmup..."
    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 200 --request-rate 4 > /dev/null 2>&1
    sleep 3

    echo "  Bench (1000 prompts)..."
    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 1000 --request-rate $RATE --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,99 \
        --save-result --result-dir "$ONLINE_DIR" --result-filename "${TAG}_emu.json" > /dev/null 2>&1

    cleanup_gpu
    echo "  --- Hybrid rate=$RATE (vs fresh real baseline) ---"
    python3 "$SCRIPT_DIR/compare_results.py" "$ONLINE_DIR/fresh_rate${RATE}_real.json" "$ONLINE_DIR/${TAG}_emu.json"
done

echo ""
echo "============================================"
echo "=== Comparison: step_cycle vs hybrid ==="
echo "============================================"
echo ""
echo "--- Rate=1: step_cycle (previous) ---"
python3 "$SCRIPT_DIR/compare_results.py" "$ONLINE_DIR/fresh_rate1_real.json" "$ONLINE_DIR/fresh_rate1_emu.json" 2>/dev/null || echo "  MISSING"
echo ""
echo "--- Rate=1: hybrid ---"
python3 "$SCRIPT_DIR/compare_results.py" "$ONLINE_DIR/fresh_rate1_real.json" "$ONLINE_DIR/hybrid_rate1_emu.json" 2>/dev/null || echo "  MISSING"
echo ""
echo "--- Rate=4: step_cycle (previous) ---"
python3 "$SCRIPT_DIR/compare_results.py" "$ONLINE_DIR/fresh_rate4_real.json" "$ONLINE_DIR/fresh_rate4_emu.json" 2>/dev/null || echo "  MISSING"
echo ""
echo "--- Rate=4: hybrid ---"
python3 "$SCRIPT_DIR/compare_results.py" "$ONLINE_DIR/fresh_rate4_real.json" "$ONLINE_DIR/hybrid_rate4_emu.json" 2>/dev/null || echo "  MISSING"

echo ""
echo "=== DONE $(date) ==="
