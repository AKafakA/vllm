#!/bin/bash
# Test without gpu_free_time chaining — rate=1, 4, 8
set -e
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub
export HF_HUB_OFFLINE=1
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
echo "=== No-Chain Timer Test ==="
echo "=== $(date) ==="
echo "============================================"

for RATE in 1 4 8; do
    TAG="nochain_rate${RATE}"
    echo ""
    echo "========== Rate=$RATE =========="

    cleanup_gpu
    VLLM_EMULATOR_ENABLE_ORACLE=1 \
    VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
    VLLM_EMULATOR_MODE=realtime \
    VLLM_EMULATOR_EXECUTOR_HOOK=1 \
    VLLM_EMULATOR_ORACLE_MODE=hybrid \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
        > /workspace/nochain_emu_${RATE}.log 2>&1 &
    wait_server

    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 200 --request-rate 4 > /dev/null 2>&1
    sleep 3

    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 1000 --request-rate $RATE --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,99 \
        --save-result --result-dir "$ONLINE_DIR" --result-filename "${TAG}_emu.json" > /dev/null 2>&1

    cleanup_gpu

    # Compare with fresh real baselines (rate=1,4) or hybrid baselines (rate=8)
    REAL="$ONLINE_DIR/fresh_rate${RATE}_real.json"
    if [ ! -f "$REAL" ]; then
        REAL="$ONLINE_DIR/hybrid_rate${RATE}_real.json"
    fi
    if [ ! -f "$REAL" ]; then
        REAL="$ONLINE_DIR/ext_rate8_real.json"
    fi
    echo "  --- Rate=$RATE (no-chain vs real) ---"
    python3 "$SCRIPT_DIR/compare_results.py" "$REAL" "$ONLINE_DIR/${TAG}_emu.json" 2>/dev/null || echo "  No baseline"
done

echo ""
echo "=== SUMMARY ==="
for RATE in 1 4 8; do
    TAG="nochain_rate${RATE}"
    REAL="$ONLINE_DIR/fresh_rate${RATE}_real.json"
    if [ ! -f "$REAL" ]; then REAL="$ONLINE_DIR/hybrid_rate${RATE}_real.json"; fi
    if [ ! -f "$REAL" ]; then REAL="$ONLINE_DIR/ext_rate8_real.json"; fi
    echo "--- Rate=$RATE ---"
    python3 "$SCRIPT_DIR/compare_results.py" "$REAL" "$ONLINE_DIR/${TAG}_emu.json" 2>/dev/null || echo "  MISSING"
done

echo ""
echo "=== DONE $(date) ==="
