#!/bin/bash
# 0.5B model: full serving profile + back-to-back eval
set -e
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub

MODEL="Qwen/Qwen2.5-0.5B-Instruct"
LABEL="0.5b-tp1"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"
PORT=8100

pkill -9 -f EngineCore 2>/dev/null || true
pkill -9 -f api_server 2>/dev/null || true
sleep 5

echo "============================================================"
echo "0.5B Model: Serving Profile + B2B Eval"
echo "============================================================"

# Step 1: Serving trace
echo ""
echo "Step 1: Serving trace..."
rm -f "${RESULT_DIR}/step_cycle_${LABEL}.jsonl"

VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
VLLM_EMULATOR_STEP_TRACE_OUTPUT="${RESULT_DIR}/step_cycle_${LABEL}.jsonl" \
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 4096 \
    --port ${PORT} --trust-remote-code \
    > /workspace/serving_trace_${LABEL}_server.log 2>&1 &

for i in $(seq 1 120); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then echo "  Server ready after ${i}s"; break; fi
    sleep 1
done

if ! curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
    echo "FAILED"; pkill -9 -f EngineCore 2>/dev/null || true; exit 1
fi

for rate in 1 2 4 8; do
    echo "  Tracing rate=${rate}..."
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "${MODEL}" --base-url http://localhost:${PORT} \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 50 --request-rate ${rate} > /dev/null 2>&1
done

pkill -f api_server 2>/dev/null || true; sleep 2; pkill -9 -f EngineCore 2>/dev/null || true; sleep 5

echo "  Trace: $(wc -l < ${RESULT_DIR}/step_cycle_${LABEL}.jsonl) records"

# Step 2: Build serving profile
echo ""
echo "Step 2: Build serving profile..."
python3 /workspace/vllm-emulator-v18/paper/isca-2026-mlarchsys/scripts/temp/build_serving_profile_generic.py \
    "${RESULT_DIR}/step_cycle_${LABEL}.jsonl" \
    "${RESULT_DIR}/profiles/sweep-0.5b.json" \
    "${RESULT_DIR}/profiles/serving-${LABEL}-step-cycle.json" \
    "${MODEL}" "RTX-3060-12GB"

SERVING_PROFILE="${RESULT_DIR}/profiles/serving-${LABEL}-step-cycle.json"

# Step 3: Real baseline
echo ""
echo "Step 3: Real baseline..."
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 4096 \
    --port ${PORT} --trust-remote-code \
    > /workspace/b2b_${LABEL}_real_server.log 2>&1 &

for i in $(seq 1 120); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then echo "  Server ready after ${i}s"; break; fi
    sleep 1
done

for rate in 1 2 4; do
    echo "  --- Real rate=${rate} ---"
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "${MODEL}" --base-url http://localhost:${PORT} \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 50 --request-rate ${rate} \
        --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
        --save-result --result-dir "${RESULT_DIR}/online" \
        --result-filename "b2b_real_${LABEL}_rate${rate}.json" 2>&1 | grep -E "TTFT|TPOT"
done

pkill -f api_server 2>/dev/null || true; sleep 2; pkill -9 -f EngineCore 2>/dev/null || true; sleep 5

# Step 4: Emulator
echo ""
echo "Step 4: Emulator..."
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="${SERVING_PROFILE}" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 4096 \
    --port ${PORT} --trust-remote-code \
    > /workspace/b2b_${LABEL}_emu_server.log 2>&1 &

for i in $(seq 1 120); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then echo "  Server ready after ${i}s"; break; fi
    sleep 1
done

grep ExecutorEmulatorHook /workspace/b2b_${LABEL}_emu_server.log 2>/dev/null | head -1

for rate in 1 2 4; do
    echo "  --- Emu rate=${rate} ---"
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "${MODEL}" --base-url http://localhost:${PORT} \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 50 --request-rate ${rate} \
        --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
        --save-result --result-dir "${RESULT_DIR}/online" \
        --result-filename "b2b_emu_${LABEL}_rate${rate}.json" 2>&1 | grep -E "TTFT|TPOT"
done

pkill -f api_server 2>/dev/null || true; sleep 2; pkill -9 -f EngineCore 2>/dev/null || true

# Results
echo ""
echo "============================================================"
echo "0.5B RESULTS"
echo "============================================================"
python3 /workspace/vllm-emulator-v18/paper/isca-2026-mlarchsys/scripts/temp/compare_results.py "${LABEL}"

echo ""
echo "DONE"
