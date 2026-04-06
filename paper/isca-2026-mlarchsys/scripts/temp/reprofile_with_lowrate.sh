#!/bin/bash
# Re-profile with low-rate trace to capture isolated step times
# Low-rate (0.5) captures tt=1 without batching overhead: ~12ms
# High-rate (4-8) captures tt=10-30 with batching: ~18-30ms
set -e
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"
PORT=8100

pkill -9 -f EngineCore 2>/dev/null || true; pkill -9 -f api_server 2>/dev/null || true; sleep 5

echo "=== Re-profile with low + high rate ==="
rm -f "${RESULT_DIR}/step_cycle_1.5b_full.jsonl"

VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
VLLM_EMULATOR_STEP_TRACE_OUTPUT="${RESULT_DIR}/step_cycle_1.5b_full.jsonl" \
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 4096 --port ${PORT} --trust-remote-code \
    > /workspace/lowrate_trace_server.log 2>&1 &
for i in $(seq 1 120); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then break; fi; sleep 1
done

# Low rate: isolated requests (captures real tt=1 step time ~12ms)
# Also save bench serve results for TTFT calibration
echo "  rate=0.5 (50 prompts, isolated)..."
python3 -m vllm.entrypoints.cli.main bench serve --model "${MODEL}" --base-url http://localhost:${PORT} \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 50 --request-rate 0.5 \
    --save-result --result-dir "${RESULT_DIR}/profiles" --result-filename "bench_rate0.5.json" > /dev/null 2>&1

# Medium rates: save TTFT results for overhead calibration
for rate in 1 2 4 8; do
    echo "  rate=${rate} (100 prompts)..."
    python3 -m vllm.entrypoints.cli.main bench serve --model "${MODEL}" --base-url http://localhost:${PORT} \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 100 --request-rate ${rate} \
        --save-result --result-dir "${RESULT_DIR}/profiles" --result-filename "bench_rate${rate}.json" > /dev/null 2>&1
done

pkill -f api_server 2>/dev/null || true; sleep 2; pkill -9 -f EngineCore 2>/dev/null || true; sleep 5

echo "  Trace: $(wc -l < ${RESULT_DIR}/step_cycle_1.5b_full.jsonl) records"

echo ""
echo "=== Build 2D profile from full trace ==="
python3 /workspace/vllm-emulator-v18/paper/isca-2026-mlarchsys/scripts/temp/build_serving_profile_2d.py \
    "${RESULT_DIR}/step_cycle_1.5b_full.jsonl" \
    "${RESULT_DIR}/profiles/sweep-1.5b-tp1-v14.json" \
    "${RESULT_DIR}/profiles/serving-1.5b-tp1-full.json" \
    "${MODEL}" "RTX-3060-12GB"

PROFILE="${RESULT_DIR}/profiles/serving-1.5b-tp1-full.json"

# Show key values
python3 -c "
import json
p = json.load(open('${PROFILE}'))
dec = {e['total_tokens']: e['latency_us'] for e in p.get('decode_forward_pass', [])}
print('Decode profile at low tt:')
for tt in [1, 2, 3, 4, 5, 8, 10]:
    if tt in dec:
        print(f'  tt={tt}: {dec[tt]/1000:.1f}ms')
"

echo ""
echo "=== B2B test ==="

# Real
echo "Real..."
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 4096 --port ${PORT} --trust-remote-code \
    > /workspace/lowrate_real.log 2>&1 &
for i in $(seq 1 120); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then break; fi; sleep 1
done
for rate in 1 2 4; do
    python3 -m vllm.entrypoints.cli.main bench serve --model "${MODEL}" --base-url http://localhost:${PORT} \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 50 --request-rate ${rate} --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
        --save-result --result-dir "${RESULT_DIR}/online" --result-filename "lowrate_real_rate${rate}.json" > /dev/null 2>&1
done
pkill -f api_server 2>/dev/null || true; sleep 2; pkill -9 -f EngineCore 2>/dev/null || true; sleep 5

# Emu with full profile
echo "Emu (full profile, timer approach)..."
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="${PROFILE}" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 4096 --port ${PORT} --trust-remote-code \
    > /workspace/lowrate_emu.log 2>&1 &
for i in $(seq 1 120); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then break; fi; sleep 1
done
grep ExecutorEmulatorHook /workspace/lowrate_emu.log | head -1
for rate in 1 2 4; do
    python3 -m vllm.entrypoints.cli.main bench serve --model "${MODEL}" --base-url http://localhost:${PORT} \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 50 --request-rate ${rate} --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
        --save-result --result-dir "${RESULT_DIR}/online" --result-filename "lowrate_emu_rate${rate}.json" > /dev/null 2>&1
done
pkill -f api_server 2>/dev/null || true; sleep 2; pkill -9 -f EngineCore 2>/dev/null || true

echo ""
echo "=== RESULTS ==="
python3 -c "
import json, os
rd = '${RESULT_DIR}/online'
print(f'{\"Rate\":>4} {\"Real_TTFT\":>10} {\"Emu_TTFT\":>10} {\"TTFT_err\":>9} {\"Real_TPOT\":>10} {\"Emu_TPOT\":>10} {\"TPOT_err\":>9}')
for rate in [1, 2, 4]:
    rf = f'{rd}/lowrate_real_rate{rate}.json'
    ef = f'{rd}/lowrate_emu_rate{rate}.json'
    if os.path.exists(rf) and os.path.exists(ef):
        r, e = json.load(open(rf)), json.load(open(ef))
        te = (e['mean_ttft_ms']-r['mean_ttft_ms'])/r['mean_ttft_ms']*100
        pe = (e['mean_tpot_ms']-r['mean_tpot_ms'])/r['mean_tpot_ms']*100
        tok = '✓' if abs(te)<=5 else ('~' if abs(te)<=6 else '✗')
        pok = '✓' if abs(pe)<=5 else ('~' if abs(pe)<=6 else '✗')
        print(f'{rate:>4} {r[\"mean_ttft_ms\"]:>10.1f} {e[\"mean_ttft_ms\"]:>10.1f} {te:>+8.1f}%{tok} {r[\"mean_tpot_ms\"]:>10.1f} {e[\"mean_tpot_ms\"]:>10.1f} {pe:>+8.1f}%{pok}')
"
echo "DONE"
