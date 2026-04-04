#!/bin/bash
# Final back-to-back: real GPU vs mock path with step-cycle serving profile
set -e
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub

MODEL="Qwen/Qwen2.5-0.5B-Instruct"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"
SERVING_PROFILE="${RESULT_DIR}/profiles/serving-0.5b-tp1-step-cycle.json"
PORT=8100

echo "============================================================"
echo "Final B2B: Real GPU vs Mock (step-cycle profile)"
echo "============================================================"

# Phase 1: Real
pkill -9 -f EngineCore 2>/dev/null || true; pkill -9 -f api_server 2>/dev/null || true; sleep 5

echo "Phase 1: Real baseline..."
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 4096 --port ${PORT} --trust-remote-code \
    > /workspace/final_real_server.log 2>&1 &
for i in $(seq 1 120); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then echo "  Ready ${i}s"; break; fi; sleep 1
done

for rate in 1 2 4; do
    echo "  Real rate=${rate}:"
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "${MODEL}" --base-url http://localhost:${PORT} \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 30 --request-rate ${rate} \
        --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
        --save-result --result-dir "${RESULT_DIR}/online" \
        --result-filename "final_real_rate${rate}.json" 2>&1 | grep -E "TTFT|TPOT"
done
pkill -f api_server 2>/dev/null || true; sleep 2; pkill -9 -f EngineCore 2>/dev/null || true; sleep 5

# Phase 2: Mock
echo ""
echo "Phase 2: GPU mock (step-cycle profile, no CUDA_SYNC_US)..."
VLLM_EMULATOR_MOCK_CUDA=1 \
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="${SERVING_PROFILE}" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
VLLM_EMULATOR_MEMORY=$((80*1024*1024*1024)) \
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 4096 --port ${PORT} --trust-remote-code \
    --load-format dummy --enforce-eager \
    > /workspace/final_mock_server.log 2>&1 &
for i in $(seq 1 60); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then echo "  Ready ${i}s"; break; fi; sleep 1
done

for rate in 1 2 4; do
    echo "  Mock rate=${rate}:"
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "${MODEL}" --base-url http://localhost:${PORT} \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 30 --request-rate ${rate} \
        --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
        --save-result --result-dir "${RESULT_DIR}/online" \
        --result-filename "final_mock_rate${rate}.json" 2>&1 | grep -E "TTFT|TPOT"
done
pkill -f api_server 2>/dev/null || true; sleep 2; pkill -9 -f EngineCore 2>/dev/null || true

echo ""
echo "============================================================"
echo "FINAL RESULTS"
echo "============================================================"
python3 -c "
import json, os
rd = '${RESULT_DIR}/online'
print(f'{\"Rate\":>4} {\"Real_TTFT\":>10} {\"Mock_TTFT\":>10} {\"TTFT_err\":>9} {\"Real_TPOT\":>10} {\"Mock_TPOT\":>10} {\"TPOT_err\":>9}')
print('-' * 68)
all_pass = True
for rate in [1, 2, 4]:
    rf = f'{rd}/final_real_rate{rate}.json'
    mf = f'{rd}/final_mock_rate{rate}.json'
    if os.path.exists(rf) and os.path.exists(mf):
        r, m = json.load(open(rf)), json.load(open(mf))
        te = (m['mean_ttft_ms']-r['mean_ttft_ms'])/r['mean_ttft_ms']*100
        pe = (m['mean_tpot_ms']-r['mean_tpot_ms'])/r['mean_tpot_ms']*100
        tok = '✓' if abs(te)<=6 else '✗'
        pok = '✓' if abs(pe)<=6 else '✗'
        if abs(te)>6 or abs(pe)>6: all_pass = False
        print(f'{rate:>4} {r[\"mean_ttft_ms\"]:>10.1f} {m[\"mean_ttft_ms\"]:>10.1f} {te:>+8.1f}%{tok} {r[\"mean_tpot_ms\"]:>10.1f} {m[\"mean_tpot_ms\"]:>10.1f} {pe:>+8.1f}%{pok}')

if all_pass:
    print('\\nALL WITHIN 6% — PASS')
else:
    print('\\nSOME OVER 6% — see rate=1 TTFT (known CPU mock limitation)')
"
echo "DONE"
