#!/bin/bash
# PD Disaggregation b2b eval: real vs emulator (0.5B, NIXL)
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub

MODEL="Qwen/Qwen2.5-0.5B-Instruct"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"
PORT_P=8100
PORT_D=8200
NUM_PROMPTS=30

cleanup() {
    pkill -9 -f EngineCore 2>/dev/null
    pkill -9 -f api_server 2>/dev/null
    sleep 5
}

wait_both() {
    for i in $(seq 1 120); do
        p=$(curl -s http://localhost:${PORT_P}/health 2>/dev/null && echo 1 || echo 0)
        d=$(curl -s http://localhost:${PORT_D}/health 2>/dev/null && echo 1 || echo 0)
        if [ "$p" = "1" ] && [ "$d" = "1" ]; then
            echo "  Both ready after ${i}s"
            return 0
        fi
        sleep 1
    done
    echo "  TIMEOUT"
    return 1
}

echo "============================================================"
echo "PD Disaggregation B2B (0.5B, NIXL)"
echo "============================================================"

cleanup

# ============================================================
# Phase 1: Real PD
# ============================================================
echo ""
echo "PHASE 1: Real PD disagg"

CUDA_VISIBLE_DEVICES=0 VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 2048 \
    --host 0.0.0.0 --port ${PORT_P} --trust-remote-code \
    --gpu-memory-utilization 0.9 --enforce-eager \
    --kv-transfer-config \
    '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_parallel_size":2,"kv_buffer_size":"5e8"}' \
    > /workspace/pd_b2b_real_p.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 VLLM_NIXL_SIDE_CHANNEL_PORT=5601 \
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 2048 \
    --host 0.0.0.0 --port ${PORT_D} --trust-remote-code \
    --gpu-memory-utilization 0.9 --enforce-eager \
    --kv-transfer-config \
    '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_parallel_size":2,"kv_buffer_size":"5e8"}' \
    > /workspace/pd_b2b_real_d.log 2>&1 &

if wait_both; then
    echo "  Benching prefill instance..."
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "${MODEL}" --base-url http://localhost:${PORT_P} \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts ${NUM_PROMPTS} --request-rate 2 \
        --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
        --save-result --result-dir "${RESULT_DIR}/online" \
        --result-filename "pd_real_instance.json" 2>&1 | grep -E "TTFT|TPOT"
fi

cleanup

# ============================================================
# Phase 2: Build serving profile from PD traces (already collected)
# ============================================================
echo ""
echo "PHASE 2: Build PD serving profile"

# Use the traces collected earlier (step_cycle_pd_prefill.jsonl)
TRACE_FILE="${RESULT_DIR}/step_cycle_pd_prefill.jsonl"
PD_PROFILE="${RESULT_DIR}/profiles/serving-0.5b-pd-step-cycle.json"

if [ -f "${TRACE_FILE}" ] && [ $(wc -l < "${TRACE_FILE}") -gt 100 ]; then
    python3 /workspace/vllm-emulator-v18/paper/isca-2026-mlarchsys/scripts/temp/build_serving_profile_generic.py \
        "${TRACE_FILE}" \
        "${RESULT_DIR}/profiles/sweep-0.5b.json" \
        "${PD_PROFILE}" \
        "${MODEL}" "RTX-3060-12GB"
else
    echo "  Not enough trace data. Building profile from fresh trace..."
    # Fall back to the 0.5B serving profile (non-PD)
    PD_PROFILE="${RESULT_DIR}/profiles/serving-0.5b-tp1-step-cycle.json"
    echo "  Using non-PD serving profile: ${PD_PROFILE}"
fi

# ============================================================
# Phase 3: Emulator PD
# ============================================================
echo ""
echo "PHASE 3: Emulator PD disagg"

VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="${PD_PROFILE}" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
CUDA_VISIBLE_DEVICES=0 VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 2048 \
    --host 0.0.0.0 --port ${PORT_P} --trust-remote-code \
    --gpu-memory-utilization 0.9 --enforce-eager \
    --kv-transfer-config \
    '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_parallel_size":2,"kv_buffer_size":"5e8"}' \
    > /workspace/pd_b2b_emu_p.log 2>&1 &

VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="${PD_PROFILE}" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
CUDA_VISIBLE_DEVICES=1 VLLM_NIXL_SIDE_CHANNEL_PORT=5601 \
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 2048 \
    --host 0.0.0.0 --port ${PORT_D} --trust-remote-code \
    --gpu-memory-utilization 0.9 --enforce-eager \
    --kv-transfer-config \
    '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_parallel_size":2,"kv_buffer_size":"5e8"}' \
    > /workspace/pd_b2b_emu_d.log 2>&1 &

if wait_both; then
    grep ExecutorEmulatorHook /workspace/pd_b2b_emu_p.log 2>/dev/null | head -1
    echo "  Benching emulator PD instance..."
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "${MODEL}" --base-url http://localhost:${PORT_P} \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts ${NUM_PROMPTS} --request-rate 2 \
        --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
        --save-result --result-dir "${RESULT_DIR}/online" \
        --result-filename "pd_emu_instance.json" 2>&1 | grep -E "TTFT|TPOT"
fi

cleanup

# ============================================================
# Results
# ============================================================
echo ""
echo "============================================================"
echo "PD DISAGG RESULTS"
echo "============================================================"
python3 -c "
import json, os
rd = '/workspace/eval_results/RTX-3060-12GB/online'
for pfx in ['real', 'emu']:
    f = f'{rd}/pd_{pfx}_instance.json'
    if os.path.exists(f):
        d = json.load(open(f))
        print(f'PD {pfx}: TTFT={d[\"mean_ttft_ms\"]:.1f}ms  TPOT={d[\"mean_tpot_ms\"]:.1f}ms')

rf, ef = f'{rd}/pd_real_instance.json', f'{rd}/pd_emu_instance.json'
if os.path.exists(rf) and os.path.exists(ef):
    r, e = json.load(open(rf)), json.load(open(ef))
    te = (e['mean_ttft_ms']-r['mean_ttft_ms'])/r['mean_ttft_ms']*100
    pe = (e['mean_tpot_ms']-r['mean_tpot_ms'])/r['mean_tpot_ms']*100
    tok = '✓' if abs(te)<5 else '✗'
    pok = '✓' if abs(pe)<5 else '✗'
    print(f'Error: TTFT {te:+.1f}% {tok}  TPOT {pe:+.1f}% {pok}')
"
echo ""
echo "DONE"
