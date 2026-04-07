#!/bin/bash
# Ablation: compare step_cycle vs hybrid vs 2d oracle modes
# Rate=1 and Rate=4, 1000 prompts, reuses fresh real baselines
set -e
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub
export CUDA_VISIBLE_DEVICES=1

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
PORT=8100
SCRIPT_DIR="/workspace/vllm-emulator-v18/tools/e2e"
ONLINE_DIR="/workspace/eval_results/RTX-3060-12GB/online"

# First rebuild profile with 2D regression
echo "=== Rebuild profile with 2D overhead ==="
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"
python3 /workspace/vllm-emulator-v18/paper/isca-2026-mlarchsys/scripts/temp/build_serving_profile_2d.py \
    "${RESULT_DIR}/step_cycle_fresh.jsonl" \
    "${RESULT_DIR}/profiles/sweep-1.5b-tp1-v14.json" \
    "${RESULT_DIR}/profiles/serving-1.5b-tp1-ablation.json" \
    "$MODEL" "RTX-3060-12GB"
PROFILE="${RESULT_DIR}/profiles/serving-1.5b-tp1-ablation.json"

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

echo ""
echo "============================================"
echo "=== Oracle Mode Ablation ==="
echo "============================================"

for MODE in step_cycle hybrid 2d; do
    for RATE in 1 4; do
        TAG="ablation_${MODE}_rate${RATE}"
        echo ""
        echo "--- $MODE, rate=$RATE ---"

        cleanup_gpu
        VLLM_EMULATOR_ENABLE_ORACLE=1 \
        VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
        VLLM_EMULATOR_MODE=realtime \
        VLLM_EMULATOR_EXECUTOR_HOOK=1 \
        VLLM_EMULATOR_ORACLE_MODE=$MODE \
        python3 -m vllm.entrypoints.openai.api_server \
            --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
            > /workspace/ablation_${MODE}_${RATE}.log 2>&1 &
        wait_server
        grep "ExecutorEmulatorHook" /workspace/ablation_${MODE}_${RATE}.log | head -1

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
        python3 "$SCRIPT_DIR/compare_results.py" \
            "$ONLINE_DIR/fresh_rate${RATE}_real.json" "$ONLINE_DIR/${TAG}_emu.json"
    done
done

echo ""
echo "============================================"
echo "=== ABLATION SUMMARY ==="
echo "============================================"
for RATE in 1 4; do
    echo ""
    echo "=== Rate=$RATE ==="
    for MODE in step_cycle hybrid 2d; do
        TAG="ablation_${MODE}_rate${RATE}"
        echo "--- $MODE ---"
        python3 "$SCRIPT_DIR/compare_results.py" \
            "$ONLINE_DIR/fresh_rate${RATE}_real.json" "$ONLINE_DIR/${TAG}_emu.json" 2>/dev/null \
            || echo "  MISSING"
    done
done

echo ""
echo "=== DONE $(date) ==="
