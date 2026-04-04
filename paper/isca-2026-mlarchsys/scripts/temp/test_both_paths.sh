#!/bin/bash
# Test BOTH paths after revert: Path A (GPU emu) + Path B (mock)
set -e
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub

MODEL="Qwen/Qwen2.5-0.5B-Instruct"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"
SERVING_PROFILE="${RESULT_DIR}/profiles/serving-0.5b-tp1-step-cycle.json"
PORT=8100

pkill -9 -f EngineCore 2>/dev/null || true; pkill -9 -f api_server 2>/dev/null || true; sleep 5

# Phase 1: Real baseline
echo "=== REAL BASELINE ==="
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 4096 --port ${PORT} --trust-remote-code \
    > /workspace/both_real.log 2>&1 &
for i in $(seq 1 120); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then break; fi; sleep 1
done
for rate in 1 2 4; do
    python3 -m vllm.entrypoints.cli.main bench serve --model "${MODEL}" --base-url http://localhost:${PORT} \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 30 --request-rate ${rate} --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
        --save-result --result-dir "${RESULT_DIR}/online" --result-filename "both_real_rate${rate}.json" > /dev/null 2>&1
done
pkill -f api_server 2>/dev/null || true; sleep 2; pkill -9 -f EngineCore 2>/dev/null || true; sleep 5

# Phase 2: Path A (GPU emulator, NO mock)
echo "=== PATH A (GPU Emulator) ==="
VLLM_EMULATOR_ENABLE_ORACLE=1 VLLM_EMULATOR_PROFILE_PACK="${SERVING_PROFILE}" \
VLLM_EMULATOR_MODE=realtime VLLM_EMULATOR_EXECUTOR_HOOK=1 \
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 4096 --port ${PORT} --trust-remote-code \
    > /workspace/both_patha.log 2>&1 &
for i in $(seq 1 120); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then break; fi; sleep 1
done
for rate in 1 2 4; do
    python3 -m vllm.entrypoints.cli.main bench serve --model "${MODEL}" --base-url http://localhost:${PORT} \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 30 --request-rate ${rate} --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
        --save-result --result-dir "${RESULT_DIR}/online" --result-filename "both_patha_rate${rate}.json" > /dev/null 2>&1
done
pkill -f api_server 2>/dev/null || true; sleep 2; pkill -9 -f EngineCore 2>/dev/null || true; sleep 5

# Phase 3: Path B (Mock, blocking sleep)
echo "=== PATH B (Mock + Blocking Sleep) ==="
VLLM_EMULATOR_MOCK_CUDA=1 VLLM_EMULATOR_ENABLE_ORACLE=1 VLLM_EMULATOR_PROFILE_PACK="${SERVING_PROFILE}" \
VLLM_EMULATOR_MODE=realtime VLLM_EMULATOR_EXECUTOR_HOOK=1 VLLM_EMULATOR_MEMORY=$((80*1024*1024*1024)) \
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 4096 --port ${PORT} --trust-remote-code \
    --load-format dummy --enforce-eager \
    > /workspace/both_pathb.log 2>&1 &
for i in $(seq 1 60); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then break; fi; sleep 1
done
for rate in 1 2 4; do
    python3 -m vllm.entrypoints.cli.main bench serve --model "${MODEL}" --base-url http://localhost:${PORT} \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 30 --request-rate ${rate} --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
        --save-result --result-dir "${RESULT_DIR}/online" --result-filename "both_pathb_rate${rate}.json" > /dev/null 2>&1
done
pkill -f api_server 2>/dev/null || true; sleep 2; pkill -9 -f EngineCore 2>/dev/null || true

echo ""
echo "============================================================"
echo "RESULTS"
echo "============================================================"
python3 -c "
import json, os
rd = '${RESULT_DIR}/online'
print(f'{\"Path\":>8} {\"Rate\":>4} {\"TTFT_err\":>9} {\"TPOT_err\":>9}')
print('-' * 35)
for path_name, prefix in [('A (GPU)', 'both_patha'), ('B (Mock)', 'both_pathb')]:
    for rate in [1, 2, 4]:
        rf = f'{rd}/both_real_rate{rate}.json'
        ef = f'{rd}/{prefix}_rate{rate}.json'
        if os.path.exists(rf) and os.path.exists(ef):
            r, e = json.load(open(rf)), json.load(open(ef))
            te = (e['mean_ttft_ms']-r['mean_ttft_ms'])/r['mean_ttft_ms']*100
            pe = (e['mean_tpot_ms']-r['mean_tpot_ms'])/r['mean_tpot_ms']*100
            tok = '✓' if abs(te)<=5 else ('~' if abs(te)<=6 else '✗')
            pok = '✓' if abs(pe)<=5 else ('~' if abs(pe)<=6 else '✗')
            print(f'{path_name:>8} {rate:>4} {te:>+8.1f}%{tok} {pe:>+8.1f}%{pok}')
    print()
"
echo "DONE"
