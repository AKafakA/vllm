#!/bin/bash
# Profile AND evaluate in the SAME server session
# This ensures GPU thermal state and CUDA graphs are identical
set -e
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"
PORT=8100

pkill -9 -f EngineCore 2>/dev/null || true; pkill -9 -f api_server 2>/dev/null || true; sleep 5

echo "=== Phase 1: Start REAL server with tracing ==="
rm -f "${RESULT_DIR}/same_session_trace.jsonl"

VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
VLLM_EMULATOR_STEP_TRACE_OUTPUT="${RESULT_DIR}/same_session_trace.jsonl" \
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 4096 --port ${PORT} --trust-remote-code \
    > /workspace/same_session_server.log 2>&1 &

for i in $(seq 1 120); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then echo "  Ready ${i}s"; break; fi; sleep 1
done

echo ""
echo "=== Phase 2: Profile (same server, same GPU state) ==="
for rate in 1 2 4 8; do
    echo "  Profiling rate=${rate} (100 prompts)..."
    python3 -m vllm.entrypoints.cli.main bench serve --model "${MODEL}" --base-url http://localhost:${PORT} \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 100 --request-rate ${rate} > /dev/null 2>&1
done

echo "  Trace: $(wc -l < ${RESULT_DIR}/same_session_trace.jsonl) records"

echo ""
echo "=== Phase 3: Real baseline (same server, same GPU state) ==="
for rate in 1 2 4; do
    echo "  Real rate=${rate}..."
    python3 -m vllm.entrypoints.cli.main bench serve --model "${MODEL}" --base-url http://localhost:${PORT} \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 50 --request-rate ${rate} --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
        --save-result --result-dir "${RESULT_DIR}/online" --result-filename "same_real_rate${rate}.json" 2>&1 | grep -E "TTFT|TPOT"
done

pkill -f api_server 2>/dev/null || true; sleep 2; pkill -9 -f EngineCore 2>/dev/null || true; sleep 5

echo ""
echo "=== Phase 4: Build profile from SAME session trace ==="
python3 /workspace/vllm-emulator-v18/paper/isca-2026-mlarchsys/scripts/temp/build_serving_profile_generic.py \
    "${RESULT_DIR}/same_session_trace.jsonl" \
    "${RESULT_DIR}/profiles/sweep-1.5b-tp1-v14.json" \
    "${RESULT_DIR}/profiles/serving-1.5b-tp1-same.json" \
    "${MODEL}" "RTX-3060-12GB"

PROFILE="${RESULT_DIR}/profiles/serving-1.5b-tp1-same.json"

# Check coverage at CUDA graph boundary
python3 -c "
import json
p = json.load(open('${PROFILE}'))
print('Coverage at tt=15-30:')
for e in p['forward_pass']:
    if 15 <= e['total_tokens'] <= 30:
        print(f'  tt={e[\"total_tokens\"]:>3}: {e[\"latency_us\"]/1000:.1f}ms (n={e.get(\"num_samples\",0)})')
"

echo ""
echo "=== Phase 5: Emulator test (same profile, fresh server) ==="
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="${PROFILE}" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 4096 --port ${PORT} --trust-remote-code \
    > /workspace/same_session_emu.log 2>&1 &

for i in $(seq 1 120); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then echo "  Ready ${i}s"; break; fi; sleep 1
done
grep ExecutorEmulatorHook /workspace/same_session_emu.log | head -1

for rate in 1 2 4; do
    echo "  Emu rate=${rate}..."
    python3 -m vllm.entrypoints.cli.main bench serve --model "${MODEL}" --base-url http://localhost:${PORT} \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 50 --request-rate ${rate} --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
        --save-result --result-dir "${RESULT_DIR}/online" --result-filename "same_emu_rate${rate}.json" 2>&1 | grep -E "TTFT|TPOT"
done

pkill -f api_server 2>/dev/null || true; sleep 2; pkill -9 -f EngineCore 2>/dev/null || true

echo ""
echo "=== RESULTS ==="
python3 -c "
import json, os
rd = '${RESULT_DIR}/online'
print(f'{\"Rate\":>4} {\"Real_TTFT\":>10} {\"Emu_TTFT\":>10} {\"TTFT_err\":>9} {\"Real_TPOT\":>10} {\"Emu_TPOT\":>10} {\"TPOT_err\":>9}')
for rate in [1, 2, 4]:
    rf = f'{rd}/same_real_rate{rate}.json'
    ef = f'{rd}/same_emu_rate{rate}.json'
    if os.path.exists(rf) and os.path.exists(ef):
        r, e = json.load(open(rf)), json.load(open(ef))
        te = (e['mean_ttft_ms']-r['mean_ttft_ms'])/r['mean_ttft_ms']*100
        pe = (e['mean_tpot_ms']-r['mean_tpot_ms'])/r['mean_tpot_ms']*100
        tok = '✓' if abs(te)<=5 else ('~' if abs(te)<=6 else '✗')
        pok = '✓' if abs(pe)<=5 else ('~' if abs(pe)<=6 else '✗')
        print(f'{rate:>4} {r[\"mean_ttft_ms\"]:>10.1f} {e[\"mean_ttft_ms\"]:>10.1f} {te:>+8.1f}%{tok} {r[\"mean_tpot_ms\"]:>10.1f} {e[\"mean_tpot_ms\"]:>10.1f} {pe:>+8.1f}%{pok}')
"
echo "DONE"
