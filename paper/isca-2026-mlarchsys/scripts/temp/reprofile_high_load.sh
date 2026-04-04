#!/bin/bash
# Re-profile serving step-cycle at higher load to cover tt=17-30
set -e
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"
PORT=8100

pkill -9 -f EngineCore 2>/dev/null || true; pkill -9 -f api_server 2>/dev/null || true; sleep 5

echo "=== Re-profile with higher load (rates 1-8, 100 prompts) ==="
rm -f "${RESULT_DIR}/step_cycle_1.5b_highload.jsonl"

VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
VLLM_EMULATOR_STEP_TRACE_OUTPUT="${RESULT_DIR}/step_cycle_1.5b_highload.jsonl" \
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 4096 --port ${PORT} --trust-remote-code \
    > /workspace/highload_trace_server.log 2>&1 &

for i in $(seq 1 120); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then echo "  Ready ${i}s"; break; fi; sleep 1
done

# Profile at multiple rates including high load
for rate in 1 2 4 8; do
    echo "  Tracing rate=${rate} (100 prompts)..."
    python3 -m vllm.entrypoints.cli.main bench serve --model "${MODEL}" --base-url http://localhost:${PORT} \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 100 --request-rate ${rate} > /dev/null 2>&1
done

pkill -f api_server 2>/dev/null || true; sleep 2; pkill -9 -f EngineCore 2>/dev/null || true; sleep 5

echo "  Trace: $(wc -l < ${RESULT_DIR}/step_cycle_1.5b_highload.jsonl) records"

echo ""
echo "=== Build new serving profile ==="
python3 /workspace/vllm-emulator-v18/paper/isca-2026-mlarchsys/scripts/temp/build_serving_profile_generic.py \
    "${RESULT_DIR}/step_cycle_1.5b_highload.jsonl" \
    "${RESULT_DIR}/profiles/sweep-1.5b-tp1-v14.json" \
    "${RESULT_DIR}/profiles/serving-1.5b-tp1-v3.json" \
    "${MODEL}" "RTX-3060-12GB"

NEW_PROFILE="${RESULT_DIR}/profiles/serving-1.5b-tp1-v3.json"

# Verify profile loads and covers tt=17-30
python3 -c "
import json
p = json.load(open('${NEW_PROFILE}'))
print(f'Buckets: {len(p[\"forward_pass\"])}')
print('Coverage at tt=15-30:')
for e in p['forward_pass']:
    if 15 <= e['total_tokens'] <= 35:
        print(f'  tt={e[\"total_tokens\"]:>3}: {e[\"latency_us\"]/1000:.1f}ms (n={e.get(\"num_samples\",0)})')
"

echo ""
echo "=== B2B test with new profile ==="

# Real baseline
echo "Real..."
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 4096 --port ${PORT} --trust-remote-code \
    > /workspace/highload_real.log 2>&1 &
for i in $(seq 1 120); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then break; fi; sleep 1
done
for rate in 1 2 4; do
    python3 -m vllm.entrypoints.cli.main bench serve --model "${MODEL}" --base-url http://localhost:${PORT} \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 50 --request-rate ${rate} --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
        --save-result --result-dir "${RESULT_DIR}/online" --result-filename "highload_real_rate${rate}.json" > /dev/null 2>&1
done
pkill -f api_server 2>/dev/null || true; sleep 2; pkill -9 -f EngineCore 2>/dev/null || true; sleep 5

# Emulator with new profile (executor hook)
echo "Emu (executor hook, new profile)..."
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="${NEW_PROFILE}" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 4096 --port ${PORT} --trust-remote-code \
    > /workspace/highload_emu.log 2>&1 &
for i in $(seq 1 120); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then break; fi; sleep 1
done
grep ExecutorEmulatorHook /workspace/highload_emu.log | head -1
for rate in 1 2 4; do
    python3 -m vllm.entrypoints.cli.main bench serve --model "${MODEL}" --base-url http://localhost:${PORT} \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 50 --request-rate ${rate} --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
        --save-result --result-dir "${RESULT_DIR}/online" --result-filename "highload_emu_rate${rate}.json" > /dev/null 2>&1
done
pkill -f api_server 2>/dev/null || true; sleep 2; pkill -9 -f EngineCore 2>/dev/null || true

echo ""
echo "=== RESULTS ==="
python3 -c "
import json, os
rd = '${RESULT_DIR}/online'
print(f'{\"Rate\":>4} {\"Real_TTFT\":>10} {\"Emu_TTFT\":>10} {\"TTFT_err\":>9} {\"Real_TPOT\":>10} {\"Emu_TPOT\":>10} {\"TPOT_err\":>9}')
for rate in [1, 2, 4]:
    rf = f'{rd}/highload_real_rate{rate}.json'
    ef = f'{rd}/highload_emu_rate{rate}.json'
    if os.path.exists(rf) and os.path.exists(ef):
        r, e = json.load(open(rf)), json.load(open(ef))
        te = (e['mean_ttft_ms']-r['mean_ttft_ms'])/r['mean_ttft_ms']*100
        pe = (e['mean_tpot_ms']-r['mean_tpot_ms'])/r['mean_tpot_ms']*100
        tok = '✓' if abs(te)<=5 else ('~' if abs(te)<=6 else '✗')
        pok = '✓' if abs(pe)<=5 else ('~' if abs(pe)<=6 else '✗')
        print(f'{rate:>4} {r[\"mean_ttft_ms\"]:>10.1f} {e[\"mean_ttft_ms\"]:>10.1f} {te:>+8.1f}%{tok} {r[\"mean_tpot_ms\"]:>10.1f} {e[\"mean_tpot_ms\"]:>10.1f} {pe:>+8.1f}%{pok}')
"
echo "DONE"
