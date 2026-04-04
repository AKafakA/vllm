#!/bin/bash
# Test prefill-only CUDA sync overhead
set -e
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub

MODEL="Qwen/Qwen2.5-0.5B-Instruct"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"
PORT=8100

for SYNC_US in 20000 25000 30000; do
    echo "=== PREFILL_SYNC_US=${SYNC_US} ==="

    pkill -9 -f EngineCore 2>/dev/null || true
    pkill -9 -f api_server 2>/dev/null || true
    sleep 5

    VLLM_EMULATOR_MOCK_CUDA=1 \
    VLLM_EMULATOR_ENABLE_ORACLE=1 \
    VLLM_EMULATOR_PROFILE_PACK="${RESULT_DIR}/profiles/serving-0.5b-tp1-step-cycle.json" \
    VLLM_EMULATOR_MODE=realtime \
    VLLM_EMULATOR_EXECUTOR_HOOK=1 \
    VLLM_EMULATOR_MEMORY=$((80*1024*1024*1024)) \
    VLLM_EMULATOR_CUDA_SYNC_US=${SYNC_US} \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "${MODEL}" --max-model-len 4096 \
        --port ${PORT} --trust-remote-code \
        --load-format dummy --enforce-eager \
        > /workspace/pfill_sync_${SYNC_US}_server.log 2>&1 &

    for i in $(seq 1 60); do
        if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then break; fi
        sleep 1
    done

    for rate in 1 2 4; do
        python3 -m vllm.entrypoints.cli.main bench serve \
            --model "${MODEL}" --base-url http://localhost:${PORT} \
            --dataset-name random --random-input-len 256 --random-output-len 128 \
            --num-prompts 20 --request-rate ${rate} \
            --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
            --save-result --result-dir "${RESULT_DIR}/online" \
            --result-filename "pfill_sync_${SYNC_US}_rate${rate}.json" > /dev/null 2>&1
    done

    pkill -f api_server 2>/dev/null || true; sleep 2; pkill -9 -f EngineCore 2>/dev/null || true
done

echo ""
python3 -c "
import json, os
rd = '${RESULT_DIR}/online'
print(f'{\"Sync\":>6} {\"Rate\":>4} {\"TTFT_err\":>9} {\"TPOT_err\":>9}')
for sync in [20000, 25000, 30000]:
    for rate in [1, 2, 4]:
        rf = f'{rd}/cluster_real_rate{rate}.json'
        mf = f'{rd}/pfill_sync_{sync}_rate{rate}.json'
        if os.path.exists(rf) and os.path.exists(mf):
            r, m = json.load(open(rf)), json.load(open(mf))
            te = (m['mean_ttft_ms']-r['mean_ttft_ms'])/r['mean_ttft_ms']*100
            pe = (m['mean_tpot_ms']-r['mean_tpot_ms'])/r['mean_tpot_ms']*100
            tok = '✓' if abs(te)<5 else ''
            pok = '✓' if abs(pe)<5 else ''
            print(f'{sync:>6} {rate:>4} {te:>+8.1f}%{tok} {pe:>+8.1f}%{pok}')
    print()
"
echo "DONE"
