#!/bin/bash
# Back-to-back TP=2 eval for 3B model on 2×RTX 3060
set -e
source /workspace/vllm-v18-env/bin/activate

export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub

MODEL="Qwen/Qwen2.5-3B-Instruct"
LABEL="3b-tp2"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"
SERVING_PROFILE="${RESULT_DIR}/profiles/serving-3b-tp2-step-cycle.json"
PORT=8100
NUM_PROMPTS=50

echo "============================================================"
echo "TP=2 Back-to-back: ${MODEL}"
echo "============================================================"

pkill -9 -f EngineCore 2>/dev/null || true
pkill -9 -f api_server 2>/dev/null || true
sleep 5

# Phase 1: Real baseline
echo ""
echo "PHASE 1: Real baseline (TP=2, enforce-eager)"
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 2048 \
    --port ${PORT} --trust-remote-code \
    --tensor-parallel-size 2 --enforce-eager \
    --gpu-memory-utilization 0.95 \
    > /workspace/b2b_${LABEL}_real_server.log 2>&1 &

for i in $(seq 1 180); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
        echo "  Server ready after ${i}s"
        break
    fi
    sleep 1
done

if ! curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
    echo "FAILED: real server didn't start"
    pkill -9 -f EngineCore 2>/dev/null || true
    exit 1
fi

for rate in 1 2 4; do
    echo "  --- Real rate=${rate} ---"
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "${MODEL}" --base-url http://localhost:${PORT} \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts ${NUM_PROMPTS} --request-rate ${rate} \
        --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
        --save-result --result-dir "${RESULT_DIR}/online" \
        --result-filename "b2b_real_${LABEL}_rate${rate}.json" 2>&1 | grep -E "TTFT|TPOT"
done

pkill -f api_server 2>/dev/null || true
sleep 2
pkill -9 -f EngineCore 2>/dev/null || true
sleep 5

# Phase 2: Emulator
echo ""
echo "PHASE 2: Emulator (TP=2, serving profile, executor hook)"
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="${SERVING_PROFILE}" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 2048 \
    --port ${PORT} --trust-remote-code \
    --tensor-parallel-size 2 --enforce-eager \
    --gpu-memory-utilization 0.95 \
    > /workspace/b2b_${LABEL}_emu_server.log 2>&1 &

for i in $(seq 1 180); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
        echo "  Server ready after ${i}s"
        break
    fi
    sleep 1
done

if ! curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
    echo "FAILED: emulator server didn't start"
    pkill -9 -f EngineCore 2>/dev/null || true
    exit 1
fi

grep ExecutorEmulatorHook /workspace/b2b_${LABEL}_emu_server.log 2>/dev/null | head -1

for rate in 1 2 4; do
    echo "  --- Emu rate=${rate} ---"
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "${MODEL}" --base-url http://localhost:${PORT} \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts ${NUM_PROMPTS} --request-rate ${rate} \
        --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
        --save-result --result-dir "${RESULT_DIR}/online" \
        --result-filename "b2b_emu_${LABEL}_rate${rate}.json" 2>&1 | grep -E "TTFT|TPOT"
done

pkill -f api_server 2>/dev/null || true
sleep 2
pkill -9 -f EngineCore 2>/dev/null || true

# Phase 3: Error analysis
echo ""
echo "============================================================"
echo "TP=2 ERROR ANALYSIS"
echo "============================================================"
python3 /workspace/vllm-emulator-v18/paper/isca-2026-mlarchsys/scripts/temp/compare_results.py "${LABEL}"

echo "DONE"
