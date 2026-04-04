#!/bin/bash
# Retrace enforce-eager with more prompts for better profile, then re-eval
set -e
source /workspace/vllm-v18-env/bin/activate

export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"
PORT=8100
NUM_PROMPTS=50

echo "============================================================"
echo "Re-trace enforce-eager with more prompts"
echo "============================================================"

pkill -9 -f EngineCore 2>/dev/null || true
pkill -9 -f api_server 2>/dev/null || true
sleep 5
rm -f "${RESULT_DIR}/step_cycle_1.5b_eager_v2.jsonl"

VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
VLLM_EMULATOR_STEP_TRACE_OUTPUT="${RESULT_DIR}/step_cycle_1.5b_eager_v2.jsonl" \
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 4096 \
    --port ${PORT} --trust-remote-code --enforce-eager \
    > /workspace/eager_retrace_server.log 2>&1 &

for i in $(seq 1 120); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then echo "  Server ready after ${i}s"; break; fi
    sleep 1
done

if ! curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
    echo "FAILED: server didn't start"
    pkill -9 -f EngineCore 2>/dev/null || true
    exit 1
fi

# More prompts for better statistics
for rate in 1 2 4; do
    echo "  Tracing rate=${rate} (50 prompts)..."
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "${MODEL}" --base-url http://localhost:${PORT} \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 50 --request-rate ${rate} \
        > /dev/null 2>&1
done

pkill -f api_server 2>/dev/null || true; sleep 2; pkill -9 -f EngineCore 2>/dev/null || true; sleep 5

echo "  Trace: $(wc -l < ${RESULT_DIR}/step_cycle_1.5b_eager_v2.jsonl) records"

# Build filtered profile
python3 -c "
import json, statistics
from collections import defaultdict

records = []
for line in open('${RESULT_DIR}/step_cycle_1.5b_eager_v2.jsonl'):
    r = json.loads(line)
    if 'total_tokens' in r:
        records.append(r)

print(f'Records: {len(records)}')

by_tt = defaultdict(list)
for r in records:
    by_tt[r['total_tokens']].append(r['step_cycle_us'])

forward_pass = []
for tt in sorted(by_tt):
    lats = by_tt[tt]
    if len(lats) < 2: continue
    med = statistics.median(lats)
    filtered = [v for v in lats if v > 5000 and v < med * 3]
    if len(filtered) >= 2:
        forward_pass.append({'total_tokens': tt, 'latency_us': round(statistics.median(filtered), 1), 'num_samples': len(filtered)})

profile = {'gpu_model': 'RTX-3060-12GB', 'model_name': '${MODEL}', 'profile_type': 'serving_step_cycle_eager', 'forward_pass': sorted(forward_pass, key=lambda e: e['total_tokens'])}
out = '${RESULT_DIR}/profiles/serving-1.5b-tp1-eager-step-cycle.json'
json.dump(profile, open(out, 'w'), indent=2)
print(f'Profile: {len(forward_pass)} buckets')
for e in profile['forward_pass'][:10]:
    print(f'  tt={e[\"total_tokens\"]:>4}: {e[\"latency_us\"]/1000:.1f}ms (n={e.get(\"num_samples\",0)})')
"

EAGER_PROFILE="${RESULT_DIR}/profiles/serving-1.5b-tp1-eager-step-cycle.json"

echo ""
echo "============================================================"
echo "Back-to-back: Real vs Emu (enforce-eager)"
echo "============================================================"

# Real
echo "  Real (enforce-eager)..."
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 4096 \
    --port ${PORT} --trust-remote-code --enforce-eager \
    > /workspace/eager_b2b_real_server.log 2>&1 &

for i in $(seq 1 120); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then echo "  Server ready after ${i}s"; break; fi
    sleep 1
done

python3 -m vllm.entrypoints.cli.main bench serve \
    --model "${MODEL}" --base-url http://localhost:${PORT} \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts ${NUM_PROMPTS} --request-rate 2 \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
    --save-result --result-dir "${RESULT_DIR}/online" \
    --result-filename "ablation_real_eager.json" 2>&1 | grep -E "TTFT|TPOT"

pkill -f api_server 2>/dev/null || true; sleep 2; pkill -9 -f EngineCore 2>/dev/null || true; sleep 5

# Emu
echo ""
echo "  Emu (enforce-eager)..."
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="${EAGER_PROFILE}" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 4096 \
    --port ${PORT} --trust-remote-code --enforce-eager \
    > /workspace/eager_b2b_emu_server.log 2>&1 &

for i in $(seq 1 120); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then echo "  Server ready after ${i}s"; break; fi
    sleep 1
done

python3 -m vllm.entrypoints.cli.main bench serve \
    --model "${MODEL}" --base-url http://localhost:${PORT} \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts ${NUM_PROMPTS} --request-rate 2 \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
    --save-result --result-dir "${RESULT_DIR}/online" \
    --result-filename "ablation_emu_eager.json" 2>&1 | grep -E "TTFT|TPOT"

pkill -f api_server 2>/dev/null || true; sleep 2; pkill -9 -f EngineCore 2>/dev/null || true

echo ""
echo "============================================================"
echo "RESULTS"
echo "============================================================"
python3 -c "
import json, os
rd = '${RESULT_DIR}/online'
r = json.load(open(f'{rd}/ablation_real_eager.json'))
e = json.load(open(f'{rd}/ablation_emu_eager.json'))
print(f'Real: TTFT={r[\"mean_ttft_ms\"]:.1f}ms  TPOT={r[\"mean_tpot_ms\"]:.1f}ms')
print(f'Emu:  TTFT={e[\"mean_ttft_ms\"]:.1f}ms  TPOT={e[\"mean_tpot_ms\"]:.1f}ms')
te = (e['mean_ttft_ms']-r['mean_ttft_ms'])/r['mean_ttft_ms']*100
pe = (e['mean_tpot_ms']-r['mean_tpot_ms'])/r['mean_tpot_ms']*100
tok = '✓' if abs(te)<5 else '✗'
pok = '✓' if abs(pe)<5 else '✗'
print(f'Error: TTFT {te:+.1f}% {tok}  TPOT {pe:+.1f}% {pok}')
"

echo "DONE"
