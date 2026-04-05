#!/bin/bash
# Test step_overhead to close rate=1 TTFT gap
set -e
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"
PROFILE="${RESULT_DIR}/profiles/serving-1.5b-tp1-2d.json"
PORT=8100

pkill -9 -f EngineCore 2>/dev/null || true; pkill -9 -f api_server 2>/dev/null || true; sleep 5

# Real baseline
echo "=== Real ==="
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 4096 --port ${PORT} --trust-remote-code \
    > /workspace/soh_real.log 2>&1 &
for i in $(seq 1 120); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then break; fi; sleep 1
done
for rate in 1 2 4; do
    python3 -m vllm.entrypoints.cli.main bench serve --model "${MODEL}" --base-url http://localhost:${PORT} \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 50 --request-rate ${rate} --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
        --save-result --result-dir "${RESULT_DIR}/online" --result-filename "soh_real_rate${rate}.json" > /dev/null 2>&1
done
pkill -f api_server 2>/dev/null || true; sleep 2; pkill -9 -f EngineCore 2>/dev/null || true; sleep 5

for SOH in 1000 3000; do
    echo "=== Emu (step_overhead=${SOH}us, pfoh=5000, cold=280000) ==="
    VLLM_EMULATOR_ENABLE_ORACLE=1 \
    VLLM_EMULATOR_PROFILE_PACK="${PROFILE}" \
    VLLM_EMULATOR_MODE=realtime \
    VLLM_EMULATOR_EXECUTOR_HOOK=1 \
    VLLM_EMULATOR_STEP_OVERHEAD_US=${SOH} \
    VLLM_EMULATOR_PREFILL_OVERHEAD_US=5000 \
    VLLM_EMULATOR_COLD_START_US=280000 \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "${MODEL}" --max-model-len 4096 --port ${PORT} --trust-remote-code \
        > /workspace/soh_emu_${SOH}.log 2>&1 &
    for i in $(seq 1 120); do
        if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then break; fi; sleep 1
    done
    for rate in 1 2 4; do
        python3 -m vllm.entrypoints.cli.main bench serve --model "${MODEL}" --base-url http://localhost:${PORT} \
            --dataset-name random --random-input-len 256 --random-output-len 128 \
            --num-prompts 50 --request-rate ${rate} --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
            --save-result --result-dir "${RESULT_DIR}/online" --result-filename "soh_emu_${SOH}_rate${rate}.json" > /dev/null 2>&1
    done
    pkill -f api_server 2>/dev/null || true; sleep 2; pkill -9 -f EngineCore 2>/dev/null || true; sleep 5
done

echo ""
echo "=== RESULTS ==="
python3 -c "
import json, os
rd = '${RESULT_DIR}/online'
print(f'{\"SOH\":>5} {\"Rate\":>4} {\"TTFT_err\":>9} {\"TPOT_err\":>9}')
for soh in [1000, 3000]:
    for rate in [1, 2, 4]:
        rf = f'{rd}/soh_real_rate{rate}.json'
        ef = f'{rd}/soh_emu_{soh}_rate{rate}.json'
        if os.path.exists(rf) and os.path.exists(ef):
            r, e = json.load(open(rf)), json.load(open(ef))
            te = (e['mean_ttft_ms']-r['mean_ttft_ms'])/r['mean_ttft_ms']*100
            pe = (e['mean_tpot_ms']-r['mean_tpot_ms'])/r['mean_tpot_ms']*100
            tok = '✓' if abs(te)<=5 else ('~' if abs(te)<=6 else '✗')
            pok = '✓' if abs(pe)<=5 else ('~' if abs(pe)<=6 else '✗')
            print(f'{soh:>5} {rate:>4} {te:>+8.1f}%{tok} {pe:>+8.1f}%{pok}')
    print()
"
echo "DONE"
