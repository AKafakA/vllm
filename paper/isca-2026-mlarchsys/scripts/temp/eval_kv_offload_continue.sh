#!/bin/bash
# Continue KV offload eval from existing trace data
set -e
source /workspace/vllm-v18-env/bin/activate

export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"
PORT=8100

pkill -9 -f EngineCore 2>/dev/null || true
pkill -9 -f api_server 2>/dev/null || true
sleep 5

echo "Step 2: Build KV offload serving profile..."
python3 /workspace/vllm-emulator-v18/paper/isca-2026-mlarchsys/scripts/temp/build_serving_profile_generic.py \
    "${RESULT_DIR}/step_cycle_1.5b_kvoffload.jsonl" \
    "${RESULT_DIR}/profiles/sweep-1.5b-tp1-dense-v2.json" \
    "${RESULT_DIR}/profiles/serving-1.5b-tp1-kvoffload.json" \
    "${MODEL}" "RTX-3060-12GB"

KVOFFLOAD_PROFILE="${RESULT_DIR}/profiles/serving-1.5b-tp1-kvoffload.json"

echo ""
echo "Step 3: Real baseline with KV offloading..."
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 4096 \
    --port ${PORT} --trust-remote-code \
    --kv-offloading-size 2 --kv-offloading-backend native \
    --disable-hybrid-kv-cache-manager \
    > /workspace/kvoffload_real_server.log 2>&1 &

for i in $(seq 1 180); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then echo "  Server ready after ${i}s"; break; fi
    sleep 1
done

python3 -m vllm.entrypoints.cli.main bench serve \
    --model "${MODEL}" --base-url http://localhost:${PORT} \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 50 --request-rate 2 \
    --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
    --save-result --result-dir "${RESULT_DIR}/online" \
    --result-filename "kvoffload_real.json" 2>&1 | grep -E "TTFT|TPOT"

pkill -f api_server 2>/dev/null || true; sleep 2; pkill -9 -f EngineCore 2>/dev/null || true; sleep 5

echo ""
echo "Step 4: Emulator with KV offload profile..."
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="${KVOFFLOAD_PROFILE}" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 4096 \
    --port ${PORT} --trust-remote-code \
    --kv-offloading-size 2 --kv-offloading-backend native \
    --disable-hybrid-kv-cache-manager \
    > /workspace/kvoffload_emu_server.log 2>&1 &

for i in $(seq 1 180); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then echo "  Server ready after ${i}s"; break; fi
    sleep 1
done

grep ExecutorEmulatorHook /workspace/kvoffload_emu_server.log 2>/dev/null | head -1

python3 -m vllm.entrypoints.cli.main bench serve \
    --model "${MODEL}" --base-url http://localhost:${PORT} \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 50 --request-rate 2 \
    --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
    --save-result --result-dir "${RESULT_DIR}/online" \
    --result-filename "kvoffload_emu.json" 2>&1 | grep -E "TTFT|TPOT"

pkill -f api_server 2>/dev/null || true; sleep 2; pkill -9 -f EngineCore 2>/dev/null || true

echo ""
echo "============================================================"
echo "KV OFFLOADING RESULTS"
echo "============================================================"
python3 -c "
import json, os
rd = '/workspace/eval_results/RTX-3060-12GB/online'
for pfx in ['real', 'emu']:
    f = f'{rd}/kvoffload_{pfx}.json'
    if os.path.exists(f):
        d = json.load(open(f))
        print(f'{pfx}: TTFT={d[\"mean_ttft_ms\"]:.1f}ms  TPOT={d[\"mean_tpot_ms\"]:.1f}ms')
rf, ef = f'{rd}/kvoffload_real.json', f'{rd}/kvoffload_emu.json'
if os.path.exists(rf) and os.path.exists(ef):
    r, e = json.load(open(rf)), json.load(open(ef))
    te = (e['mean_ttft_ms']-r['mean_ttft_ms'])/r['mean_ttft_ms']*100
    pe = (e['mean_tpot_ms']-r['mean_tpot_ms'])/r['mean_tpot_ms']*100
    tok = '✓' if abs(te)<5 else '✗'
    pok = '✓' if abs(pe)<5 else '✗'
    print(f'Error: TTFT {te:+.1f}% {tok}  TPOT {pe:+.1f}% {pok}')
"
echo "DONE"
