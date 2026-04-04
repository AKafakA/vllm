#!/bin/bash
# CSD3 A100 full evaluation pipeline
# Usage: ./run_eval.sh <model> <label> [tp]
# Example: ./run_eval.sh Qwen/Qwen2.5-7B-Instruct 7b-tp1 1
set -e

MODEL="${1:?Usage: $0 <model> <label> [tp]}"
LABEL="${2:?Usage: $0 <model> <label> [tp]}"
TP="${3:-1}"

WORKDIR="/rds/user/wd312/hpc-work/vllm-emulator"
RESULT_DIR="${WORKDIR}/eval_results/A100-80GB"
REPO="/home/wd312/Code/llm/vllm-emulator"
PORT=8100

source "${WORKDIR}/venv/bin/activate"
mkdir -p "${RESULT_DIR}/profiles" "${RESULT_DIR}/online" "${RESULT_DIR}/offline"

echo "============================================================"
echo "CSD3 A100 Eval: model=${MODEL}, label=${LABEL}, tp=${TP}"
echo "============================================================"

pkill -9 -f EngineCore 2>/dev/null || true
pkill -9 -f api_server 2>/dev/null || true
sleep 5

# Step 1: Sweep profiling
SWEEP_PROFILE="${RESULT_DIR}/profiles/sweep-${LABEL}.json"
TRACE_FILE="${RESULT_DIR}/sweep_trace_${LABEL}.jsonl"

if [ ! -f "${SWEEP_PROFILE}" ]; then
    echo ""
    echo "Step 1: Sweep profiling..."
    timeout 600 python3 "${REPO}/paper/isca-2026-mlarchsys/scripts/shape_sweep_profiler.py" \
        --model "${MODEL}" --gpu-model A100-80GB \
        --output "${SWEEP_PROFILE}" \
        --max-model-len 4096 --max-num-seqs 64 \
        --max-output-len 128 --tp "${TP}" \
        --trace-output "${TRACE_FILE}" || echo "Sweep timed out (partial profile)"
    pkill -9 -f EngineCore 2>/dev/null || true; sleep 5
else
    echo "Sweep profile exists: ${SWEEP_PROFILE}"
fi

# Step 2: Serving trace
STEP_CYCLE_FILE="${RESULT_DIR}/step_cycle_${LABEL}.jsonl"
SERVING_PROFILE="${RESULT_DIR}/profiles/serving-${LABEL}-step-cycle.json"

if [ ! -f "${SERVING_PROFILE}" ]; then
    echo ""
    echo "Step 2: Serving trace..."
    rm -f "${STEP_CYCLE_FILE}"

    VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
    VLLM_EMULATOR_STEP_TRACE_OUTPUT="${STEP_CYCLE_FILE}" \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "${MODEL}" --max-model-len 4096 \
        --port ${PORT} --trust-remote-code \
        --tensor-parallel-size ${TP} \
        > "${WORKDIR}/serving_trace_${LABEL}_server.log" 2>&1 &

    for i in $(seq 1 180); do
        if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
            echo "  Server ready after ${i}s"
            break
        fi
        sleep 1
    done

    for rate in 1 2 4; do
        echo "  Tracing rate=${rate}..."
        python3 -m vllm.entrypoints.cli.main bench serve \
            --model "${MODEL}" --base-url http://localhost:${PORT} \
            --dataset-name random --random-input-len 256 --random-output-len 128 \
            --num-prompts 50 --request-rate ${rate} > /dev/null 2>&1
    done

    pkill -f api_server 2>/dev/null || true; sleep 2
    pkill -9 -f EngineCore 2>/dev/null || true; sleep 5

    # Build serving profile
    python3 "${REPO}/paper/isca-2026-mlarchsys/scripts/temp/build_serving_profile_generic.py" \
        "${STEP_CYCLE_FILE}" "${SWEEP_PROFILE}" "${SERVING_PROFILE}" \
        "${MODEL}" "A100-80GB" || echo "Profile build failed"
else
    echo "Serving profile exists: ${SERVING_PROFILE}"
fi

# Step 3: Back-to-back eval
echo ""
echo "Step 3: Back-to-back online serving eval..."
bash "${REPO}/paper/isca-2026-mlarchsys/scripts/temp/eval_online_b2b.sh" \
    "${MODEL}" "${TP}" "${LABEL}" "${SERVING_PROFILE}"

echo ""
echo "ALL DONE: ${LABEL}"
echo "Results in: ${RESULT_DIR}"
