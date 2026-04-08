#!/bin/bash
# Two experiments:
# 1. step_cycle + uncapped chain (on chain branch code - using env CHAIN_CAP=0)
# 2. hybrid + sqrt scaling + threadpool
# Both at rates 1, 4, 8 with 1000 prompts, using final real baselines
set -e
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=1

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
PROFILE="/workspace/eval_results/RTX-3060-12GB/profiles/serving-1.5b-tp1-final.json"
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
echo "=== Experiment 1: hybrid + sqrt scaling ==="
echo "=== $(date) ==="
echo "============================================"

for RATE in 1 4 8; do
    TAG="sqrt_rate${RATE}"
    echo ""
    echo "--- Rate=$RATE (hybrid + sqrt) ---"
    cleanup_gpu
    VLLM_EMULATOR_ENABLE_ORACLE=1 \
    VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
    VLLM_EMULATOR_MODE=realtime \
    VLLM_EMULATOR_EXECUTOR_HOOK=1 \
    VLLM_EMULATOR_ORACLE_MODE=hybrid \
    VLLM_EMULATOR_OVERHEAD_SCALING=sqrt \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
        > /workspace/sqrt_emu_${RATE}.log 2>&1 &
    wait_server
    if [ "$RATE" = "1" ]; then
        grep "ExecutorEmulatorHook" /workspace/sqrt_emu_${RATE}.log | head -1
    fi

    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 200 --request-rate 4 > /dev/null 2>&1
    sleep 3

    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 1000 --request-rate $RATE --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,99 \
        --save-result --result-dir "$ONLINE_DIR" --result-filename "${TAG}_emu.json" > /dev/null 2>&1

    cleanup_gpu
    python3 "$SCRIPT_DIR/compare_results.py" \
        "$ONLINE_DIR/final_rate${RATE}_real.json" "$ONLINE_DIR/${TAG}_emu.json" 2>/dev/null || echo "  No baseline"
done

echo ""
echo "============================================"
echo "=== Experiment 2: step_cycle only (no overhead) ==="
echo "============================================"

for RATE in 1 4 8; do
    TAG="pure_sc_rate${RATE}"
    echo ""
    echo "--- Rate=$RATE (step_cycle, no overhead) ---"
    cleanup_gpu
    VLLM_EMULATOR_ENABLE_ORACLE=1 \
    VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
    VLLM_EMULATOR_MODE=realtime \
    VLLM_EMULATOR_EXECUTOR_HOOK=1 \
    VLLM_EMULATOR_ORACLE_MODE=step_cycle \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
        > /workspace/pure_sc_emu_${RATE}.log 2>&1 &
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
    python3 "$SCRIPT_DIR/compare_results.py" \
        "$ONLINE_DIR/final_rate${RATE}_real.json" "$ONLINE_DIR/${TAG}_emu.json" 2>/dev/null || echo "  No baseline"
done

echo ""
echo "============================================"
echo "=== SUMMARY ==="
echo "============================================"
for RATE in 1 4 8; do
    echo ""
    echo "--- Rate=$RATE ---"
    for TAG_PREFIX in "final_step_cycle" "final_hybrid" "final_2d" "sqrt" "pure_sc"; do
        F="$ONLINE_DIR/${TAG_PREFIX}_rate${RATE}_emu.json"
        LABEL=$(echo $TAG_PREFIX | sed 's/final_//')
        if [ -f "$F" ]; then
            python3 -c "
import json
e=json.load(open('$F')); r=json.load(open('$ONLINE_DIR/final_rate${RATE}_real.json'))
tpot=(e['mean_tpot_ms']-r['mean_tpot_ms'])/r['mean_tpot_ms']*100
e2e=(e.get('mean_e2el_ms',0)-r.get('mean_e2el_ms',0))/r.get('mean_e2el_ms',1)*100 if r.get('mean_e2el_ms') else 0
ttft=(e['mean_ttft_ms']-r['mean_ttft_ms'])/r['mean_ttft_ms']*100
print(f'  {\"$LABEL\":>14}: TPOT={tpot:+6.1f}% E2E={e2e:+6.1f}% TTFT={ttft:+6.1f}%')
" 2>/dev/null || echo "  ${LABEL}: error"
        fi
    done
done

echo ""
echo "=== DONE $(date) ==="
