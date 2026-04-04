#!/bin/bash
# GPU-portable evaluation script
# Works on any GPU: RTX 3060, A100, A30, H100, RTX 8000, etc.
#
# Usage: ./portable_eval.sh <model> <gpu_label> [tp] [max_model_len]
# Example: ./portable_eval.sh Qwen/Qwen2.5-7B-Instruct A100-40GB 1 4096
#
# Steps:
# 1. Sweep profiling (data-independent)
# 2. Serving trace (step-cycle)
# 3. Build serving profile
# 4. Back-to-back online serving eval (rates 1/2/4)
# 5. Error analysis
#
# Prerequisites: vLLM installed in active venv, model accessible

set -e

MODEL="${1:?Usage: $0 <model> <gpu_label> [tp] [max_model_len]}"
GPU_LABEL="${2:?Usage: $0 <model> <gpu_label> [tp] [max_model_len]}"
TP="${3:-1}"
MAX_MODEL_LEN="${4:-4096}"

# Derive label from model name
MODEL_SHORT=$(echo "${MODEL}" | sed 's/.*\///' | tr '[:upper:]' '[:lower:]' | sed 's/-instruct//')
LABEL="${MODEL_SHORT}-tp${TP}"

RESULT_DIR="${RESULT_DIR:-/workspace/eval_results/${GPU_LABEL}}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
PORT="${PORT:-8100}"
NUM_PROMPTS="${NUM_PROMPTS:-50}"

mkdir -p "${RESULT_DIR}/profiles" "${RESULT_DIR}/online" "${RESULT_DIR}/offline"

echo "============================================================"
echo "Portable Evaluation"
echo "  Model: ${MODEL}"
echo "  GPU: ${GPU_LABEL}"
echo "  TP: ${TP}"
echo "  Label: ${LABEL}"
echo "  Results: ${RESULT_DIR}"
echo "============================================================"

cleanup() {
    pkill -9 -f EngineCore 2>/dev/null || true
    pkill -9 -f api_server 2>/dev/null || true
    sleep 5
}

wait_server() {
    local timeout=${1:-180}
    for i in $(seq 1 ${timeout}); do
        if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
            echo "  Server ready after ${i}s"
            return 0
        fi
        sleep 1
    done
    echo "  TIMEOUT after ${timeout}s"
    return 1
}

SWEEP_PROFILE="${RESULT_DIR}/profiles/sweep-${LABEL}.json"
SWEEP_TRACE="${RESULT_DIR}/sweep_trace_${LABEL}.jsonl"
STEP_CYCLE="${RESULT_DIR}/step_cycle_${LABEL}.jsonl"
SERVING_PROFILE="${RESULT_DIR}/profiles/serving-${LABEL}-step-cycle.json"

EXTRA_ARGS=""
if [ "${TP}" -gt 1 ]; then
    # Check if CUDA graphs cause OOM on smaller GPUs
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    if [ "${GPU_MEM}" -lt 16000 ]; then
        EXTRA_ARGS="--enforce-eager --gpu-memory-utilization 0.95"
        echo "  Note: Using enforce-eager for ${GPU_MEM}MB GPU with TP=${TP}"
    fi
fi

cleanup

# ============================================================
# Step 1: Sweep profiling
# ============================================================
if [ ! -f "${SWEEP_PROFILE}" ]; then
    echo ""
    echo "STEP 1: Sweep profiling..."
    timeout 1800 python3 "${REPO_DIR}/paper/isca-2026-mlarchsys/scripts/shape_sweep_profiler.py" \
        --model "${MODEL}" --gpu-model "${GPU_LABEL}" \
        --output "${SWEEP_PROFILE}" \
        --max-model-len "${MAX_MODEL_LEN}" --max-num-seqs 64 \
        --max-output-len 128 --tp "${TP}" \
        --trace-output "${SWEEP_TRACE}" || echo "  Sweep timed out (using partial data)"
    cleanup
    echo "  Sweep done: $(wc -l < ${SWEEP_TRACE} 2>/dev/null || echo 0) trace records"
else
    echo ""
    echo "STEP 1: Sweep profile exists, skipping"
fi

# ============================================================
# Step 2: Serving trace
# ============================================================
if [ ! -f "${SERVING_PROFILE}" ]; then
    echo ""
    echo "STEP 2: Serving trace (step-cycle)..."
    rm -f "${STEP_CYCLE}"

    VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
    VLLM_EMULATOR_STEP_TRACE_OUTPUT="${STEP_CYCLE}" \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "${MODEL}" --max-model-len "${MAX_MODEL_LEN}" \
        --port ${PORT} --trust-remote-code \
        --tensor-parallel-size ${TP} ${EXTRA_ARGS} \
        > "${RESULT_DIR}/serving_trace_server.log" 2>&1 &

    if ! wait_server; then
        echo "  FAILED: trace server"
        tail -10 "${RESULT_DIR}/serving_trace_server.log"
        cleanup
        exit 1
    fi

    for rate in 1 2 4; do
        echo "  Tracing rate=${rate} (${NUM_PROMPTS} prompts)..."
        python3 -m vllm.entrypoints.cli.main bench serve \
            --model "${MODEL}" --base-url http://localhost:${PORT} \
            --dataset-name random --random-input-len 256 --random-output-len 128 \
            --num-prompts ${NUM_PROMPTS} --request-rate ${rate} > /dev/null 2>&1
    done

    cleanup
    echo "  Trace: $(wc -l < ${STEP_CYCLE} 2>/dev/null || echo 0) records"

    # Build serving profile
    echo "  Building serving profile..."
    python3 "${REPO_DIR}/paper/isca-2026-mlarchsys/scripts/temp/build_serving_profile_generic.py" \
        "${STEP_CYCLE}" "${SWEEP_PROFILE}" "${SERVING_PROFILE}" \
        "${MODEL}" "${GPU_LABEL}"
else
    echo ""
    echo "STEP 2: Serving profile exists, skipping"
fi

# ============================================================
# Step 3: Back-to-back eval
# ============================================================
echo ""
echo "STEP 3: Back-to-back online serving eval"

# Real baseline
echo ""
echo "  --- REAL BASELINE ---"
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len "${MAX_MODEL_LEN}" \
    --port ${PORT} --trust-remote-code \
    --tensor-parallel-size ${TP} ${EXTRA_ARGS} \
    > "${RESULT_DIR}/b2b_real_server.log" 2>&1 &

if ! wait_server; then
    echo "  FAILED: real server"
    cleanup; exit 1
fi

for rate in 1 2 4; do
    echo "  Real rate=${rate}:"
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "${MODEL}" --base-url http://localhost:${PORT} \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts ${NUM_PROMPTS} --request-rate ${rate} \
        --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
        --save-result --result-dir "${RESULT_DIR}/online" \
        --result-filename "b2b_real_${LABEL}_rate${rate}.json" 2>&1 | grep -E "TTFT|TPOT"
done

cleanup

# Emulator
echo ""
echo "  --- EMULATOR ---"
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="${SERVING_PROFILE}" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len "${MAX_MODEL_LEN}" \
    --port ${PORT} --trust-remote-code \
    --tensor-parallel-size ${TP} ${EXTRA_ARGS} \
    > "${RESULT_DIR}/b2b_emu_server.log" 2>&1 &

if ! wait_server; then
    echo "  FAILED: emu server"
    cleanup; exit 1
fi

grep ExecutorEmulatorHook "${RESULT_DIR}/b2b_emu_server.log" 2>/dev/null | head -1

for rate in 1 2 4; do
    echo "  Emu rate=${rate}:"
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "${MODEL}" --base-url http://localhost:${PORT} \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts ${NUM_PROMPTS} --request-rate ${rate} \
        --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
        --save-result --result-dir "${RESULT_DIR}/online" \
        --result-filename "b2b_emu_${LABEL}_rate${rate}.json" 2>&1 | grep -E "TTFT|TPOT"
done

cleanup

# ============================================================
# Step 4: Error analysis
# ============================================================
echo ""
echo "============================================================"
echo "ERROR ANALYSIS: ${LABEL} on ${GPU_LABEL}"
echo "============================================================"
python3 "${REPO_DIR}/paper/isca-2026-mlarchsys/scripts/temp/compare_results.py" "${LABEL}"

echo ""
echo "ALL DONE: ${MODEL} on ${GPU_LABEL}"
echo "Results: ${RESULT_DIR}"
