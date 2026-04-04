#!/bin/bash
# Back-to-back real vs emulator online serving evaluation.
# Usage: ./eval_online_b2b.sh <model> <tp> <label> <serving_profile>
# Example: ./eval_online_b2b.sh Qwen/Qwen2.5-3B-Instruct 2 3b-tp2 profiles/serving-3b-tp2-step-cycle.json

set -e
source /workspace/vllm-v18-env/bin/activate

MODEL="${1:-Qwen/Qwen2.5-1.5B-Instruct}"
TP="${2:-1}"
LABEL="${3:-1.5b-tp1}"
SERVING_PROFILE="${4:-/workspace/eval_results/RTX-3060-12GB/profiles/serving-${LABEL}-step-cycle.json}"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"
PORT=8100
NUM_PROMPTS=50

export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub

echo "============================================================"
echo "Back-to-back eval: model=${MODEL}, tp=${TP}, label=${LABEL}"
echo "Serving profile: ${SERVING_PROFILE}"
echo "============================================================"

# Clean up
pkill -9 -f EngineCore 2>/dev/null || true
pkill -9 -f api_server 2>/dev/null || true
sleep 5

# Phase 1: Real baseline
echo ""
echo "PHASE 1: Real baseline"
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 4096 \
    --port ${PORT} --trust-remote-code \
    --tensor-parallel-size ${TP} \
    > /workspace/b2b_${LABEL}_real_server.log 2>&1 &

for i in $(seq 1 180); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
        echo "  Server ready after ${i}s"
        break
    fi
    sleep 1
done

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

# Phase 2: Emulator with serving profile
echo ""
echo "PHASE 2: Emulator (serving profile, executor hook)"
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="${SERVING_PROFILE}" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 4096 \
    --port ${PORT} --trust-remote-code \
    --tensor-parallel-size ${TP} \
    > /workspace/b2b_${LABEL}_emu_server.log 2>&1 &

for i in $(seq 1 180); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
        echo "  Server ready after ${i}s"
        break
    fi
    sleep 1
done
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
echo "ERROR ANALYSIS: ${LABEL}"
echo "============================================================"
python3 -c "
import json, os
rd = '${RESULT_DIR}/online'
label = '${LABEL}'
print(f\"{'Config':<25} {'TTFT':>8} {'P99TTFT':>9} {'TPOT':>8} {'P99TPOT':>9}\")
print('-' * 62)
for rate in [1, 2, 4]:
    for pfx in ['real', 'emu']:
        f = f'{rd}/b2b_{pfx}_{label}_rate{rate}.json'
        if os.path.exists(f):
            d = json.load(open(f))
            print(f\"{pfx+' rate='+str(rate):<25} {d['mean_ttft_ms']:>8.1f} {d['p99_ttft_ms']:>9.1f} {d['mean_tpot_ms']:>8.1f} {d['p99_tpot_ms']:>9.1f}\")

print()
for rate in [1, 2, 4]:
    rf = f'{rd}/b2b_real_{label}_rate{rate}.json'
    ef = f'{rd}/b2b_emu_{label}_rate{rate}.json'
    if os.path.exists(rf) and os.path.exists(ef):
        r, e = json.load(open(rf)), json.load(open(ef))
        te = (e['mean_ttft_ms']-r['mean_ttft_ms'])/r['mean_ttft_ms']*100
        pe = (e['mean_tpot_ms']-r['mean_tpot_ms'])/r['mean_tpot_ms']*100
        tok = '✓' if abs(te)<10 else '✗'
        pok = '✓' if abs(pe)<10 else '✗'
        print(f'  rate={rate}: TTFT {te:>+6.1f}% {tok}  TPOT {pe:>+6.1f}% {pok}')
"

echo ""
echo "DONE: ${LABEL}"
