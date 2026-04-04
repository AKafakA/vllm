#!/bin/bash
# Re-profile with loop-cycle tracer, build serving profile, test mock path
set -e
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub

MODEL="Qwen/Qwen2.5-0.5B-Instruct"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"
PORT=8100

echo "============================================================"
echo "Phase 1: Re-profile with LOOP-CYCLE tracer"
echo "============================================================"

pkill -9 -f EngineCore 2>/dev/null || true
pkill -9 -f api_server 2>/dev/null || true
sleep 5
rm -f "${RESULT_DIR}/loop_cycle_0.5b.jsonl"

VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
VLLM_EMULATOR_STEP_TRACE_OUTPUT="${RESULT_DIR}/loop_cycle_0.5b.jsonl" \
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 4096 \
    --port ${PORT} --trust-remote-code \
    > /workspace/loop_trace_server.log 2>&1 &

for i in $(seq 1 120); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then echo "  Ready ${i}s"; break; fi
    sleep 1
done

for rate in 1 2 4; do
    echo "  Tracing rate=${rate} (50 prompts)..."
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "${MODEL}" --base-url http://localhost:${PORT} \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 50 --request-rate ${rate} > /dev/null 2>&1
done

pkill -f api_server 2>/dev/null || true; sleep 2; pkill -9 -f EngineCore 2>/dev/null || true; sleep 5

echo "  Loop-cycle trace: $(wc -l < ${RESULT_DIR}/loop_cycle_0.5b.jsonl 2>/dev/null || echo 0) records"

echo ""
echo "============================================================"
echo "Phase 2: Build loop-cycle serving profile"
echo "============================================================"

python3 /workspace/vllm-emulator-v18/paper/isca-2026-mlarchsys/scripts/temp/build_serving_profile_generic.py \
    "${RESULT_DIR}/loop_cycle_0.5b.jsonl" \
    "${RESULT_DIR}/profiles/sweep-0.5b.json" \
    "${RESULT_DIR}/profiles/loop-0.5b-tp1.json" \
    "${MODEL}" "RTX-3060-12GB"

LOOP_PROFILE="${RESULT_DIR}/profiles/loop-0.5b-tp1.json"

# Compare step-cycle vs loop-cycle profiles
echo ""
echo "  Step-cycle vs Loop-cycle comparison:"
python3 -c "
import json
step = json.load(open('${RESULT_DIR}/profiles/serving-0.5b-tp1-step-cycle.json'))
loop = json.load(open('${LOOP_PROFILE}'))

step_map = {e['total_tokens']: e['latency_us'] for e in step['forward_pass']}
loop_map = {e['total_tokens']: e['latency_us'] for e in loop['forward_pass']}

print(f'{\"tt\":>4} {\"step_ms\":>10} {\"loop_ms\":>10} {\"delta_ms\":>10}')
for tt in sorted(set(step_map) & set(loop_map)):
    if tt <= 20:
        s, l = step_map[tt]/1000, loop_map[tt]/1000
        print(f'{tt:>4} {s:>10.1f} {l:>10.1f} {l-s:>+10.1f}')
"

echo ""
echo "============================================================"
echo "Phase 3: Real baseline + GPU mock test with loop-cycle profile"
echo "============================================================"

# Real baseline
echo "  Real baseline..."
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 4096 \
    --port ${PORT} --trust-remote-code \
    > /workspace/loop_real_server.log 2>&1 &

for i in $(seq 1 120); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then echo "  Ready ${i}s"; break; fi
    sleep 1
done

for rate in 1 2 4; do
    echo "  Real rate=${rate}..."
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "${MODEL}" --base-url http://localhost:${PORT} \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 30 --request-rate ${rate} \
        --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
        --save-result --result-dir "${RESULT_DIR}/online" \
        --result-filename "loop_real_rate${rate}.json" 2>&1 | grep -E "TTFT|TPOT"
done

pkill -f api_server 2>/dev/null || true; sleep 2; pkill -9 -f EngineCore 2>/dev/null || true; sleep 5

# GPU mock test with loop-cycle profile
echo ""
echo "  GPU mock with loop-cycle profile..."
VLLM_EMULATOR_MOCK_CUDA=1 \
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="${LOOP_PROFILE}" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
VLLM_EMULATOR_MEMORY=$((80*1024*1024*1024)) \
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 4096 \
    --port ${PORT} --trust-remote-code \
    --load-format dummy --enforce-eager \
    > /workspace/loop_mock_server.log 2>&1 &

for i in $(seq 1 60); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then echo "  Ready ${i}s"; break; fi
    sleep 1
done

for rate in 1 2 4; do
    echo "  Mock rate=${rate}..."
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "${MODEL}" --base-url http://localhost:${PORT} \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 30 --request-rate ${rate} \
        --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
        --save-result --result-dir "${RESULT_DIR}/online" \
        --result-filename "loop_mock_rate${rate}.json" 2>&1 | grep -E "TTFT|TPOT"
done

pkill -f api_server 2>/dev/null || true; sleep 2; pkill -9 -f EngineCore 2>/dev/null || true

echo ""
echo "============================================================"
echo "RESULTS"
echo "============================================================"
python3 -c "
import json, os
rd = '${RESULT_DIR}/online'
print(f'{\"Rate\":>4} {\"Real TTFT\":>10} {\"Mock TTFT\":>10} {\"TTFT err\":>9} {\"Real TPOT\":>10} {\"Mock TPOT\":>10} {\"TPOT err\":>9}')
for rate in [1, 2, 4]:
    rf = f'{rd}/loop_real_rate{rate}.json'
    mf = f'{rd}/loop_mock_rate{rate}.json'
    if os.path.exists(rf) and os.path.exists(mf):
        r, m = json.load(open(rf)), json.load(open(mf))
        te = (m['mean_ttft_ms']-r['mean_ttft_ms'])/r['mean_ttft_ms']*100
        pe = (m['mean_tpot_ms']-r['mean_tpot_ms'])/r['mean_tpot_ms']*100
        tok = '✓' if abs(te)<=6 else '✗'
        pok = '✓' if abs(pe)<=6 else '✗'
        print(f'{rate:>4} {r[\"mean_ttft_ms\"]:>10.1f} {m[\"mean_ttft_ms\"]:>10.1f} {te:>+8.1f}%{tok} {r[\"mean_tpot_ms\"]:>10.1f} {m[\"mean_tpot_ms\"]:>10.1f} {pe:>+8.1f}%{pok}')
"
echo ""
echo "DONE"
