#!/bin/bash
# Re-trace TP=2 with more prompts, filter outliers, rebuild profile, and eval
set -e
source /workspace/vllm-v18-env/bin/activate

export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub

MODEL="Qwen/Qwen2.5-3B-Instruct"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"
LABEL="3b-tp2"
STEP_CYCLE_FILE="${RESULT_DIR}/step_cycle_${LABEL}_v2.jsonl"
SWEEP_PROFILE="${RESULT_DIR}/profiles/sweep-${LABEL}.json"
SERVING_PROFILE="${RESULT_DIR}/profiles/serving-${LABEL}-step-cycle.json"
PORT=8100

echo "============================================================"
echo "PHASE 0: Re-trace TP=2 with more prompts"
echo "============================================================"

pkill -9 -f EngineCore 2>/dev/null || true
pkill -9 -f api_server 2>/dev/null || true
sleep 5
rm -f "${STEP_CYCLE_FILE}"

VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
VLLM_EMULATOR_STEP_TRACE_OUTPUT="${STEP_CYCLE_FILE}" \
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 2048 \
    --port ${PORT} --trust-remote-code \
    --tensor-parallel-size 2 --enforce-eager \
    --gpu-memory-utilization 0.95 \
    > /workspace/tp2_retrace_server.log 2>&1 &

for i in $(seq 1 180); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
        echo "  Server ready after ${i}s"
        break
    fi
    sleep 1
done

if ! curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
    echo "FAILED: server didn't start"
    pkill -9 -f EngineCore 2>/dev/null || true
    exit 1
fi

# Run more prompts at each rate for better statistics
for rate in 1 2 4; do
    echo "  Tracing rate=${rate} (50 prompts)..."
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "${MODEL}" --base-url http://localhost:${PORT} \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 50 --request-rate ${rate} \
        > /dev/null 2>&1
done

pkill -f api_server 2>/dev/null || true
sleep 2
pkill -9 -f EngineCore 2>/dev/null || true
sleep 5

echo "  Step cycle trace: $(wc -l < ${STEP_CYCLE_FILE}) records"

echo ""
echo "============================================================"
echo "PHASE 1: Build filtered serving profile"
echo "============================================================"

python3 -c "
import json, statistics
from collections import defaultdict

records = []
for line in open('${STEP_CYCLE_FILE}'):
    r = json.loads(line)
    if 'total_tokens' in r:
        records.append(r)

print(f'Records with batch info: {len(records)}')

by_tt = defaultdict(list)
for r in records:
    by_tt[r['total_tokens']].append(r['step_cycle_us'])

# Filter outliers: remove values < 5ms or > 3x median per bucket
forward_pass = []
for tt in sorted(by_tt):
    lats = by_tt[tt]
    if len(lats) < 2:
        continue
    med = statistics.median(lats)
    # Filter: keep values between 5ms and 3x median
    filtered = [v for v in lats if v > 5000 and v < med * 3]
    if len(filtered) >= 2:
        forward_pass.append({
            'total_tokens': tt,
            'latency_us': round(statistics.median(filtered), 1),
            'num_samples': len(filtered),
        })
    elif len(lats) >= 2:
        # Fall back to unfiltered if too many filtered out
        forward_pass.append({
            'total_tokens': tt,
            'latency_us': round(med, 1),
            'num_samples': len(lats),
        })

# Merge with sweep for large tt
max_tt = max(e['total_tokens'] for e in forward_pass) if forward_pass else 0
try:
    sweep = json.load(open('${SWEEP_PROFILE}'))
    for e in sweep['forward_pass']:
        if e['total_tokens'] > max_tt:
            forward_pass.append(e)
except:
    pass

profile = {
    'gpu_model': 'RTX-3060-12GB-TP2',
    'model_name': '${MODEL}',
    'profile_type': 'serving_step_cycle',
    'forward_pass': sorted(forward_pass, key=lambda e: e['total_tokens']),
}
json.dump(profile, open('${SERVING_PROFILE}', 'w'), indent=2)
print(f'Serving profile: {len(forward_pass)} buckets')
for e in profile['forward_pass'][:15]:
    print(f'  tt={e[\"total_tokens\"]:>4}: {e[\"latency_us\"]/1000:.1f}ms (n={e.get(\"num_samples\",0)})')
"

echo ""
echo "============================================================"
echo "PHASE 2: Real baseline"
echo "============================================================"

pkill -9 -f EngineCore 2>/dev/null || true
sleep 3

python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 2048 \
    --port ${PORT} --trust-remote-code \
    --tensor-parallel-size 2 --enforce-eager \
    --gpu-memory-utilization 0.95 \
    > /workspace/b2b_${LABEL}_real_server.log 2>&1 &

for i in $(seq 1 180); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
        echo "  Server ready after ${i}s"
        break
    fi
    sleep 1
done

for rate in 1 2 4; do
    echo "  --- Real rate=${rate} ---"
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "${MODEL}" --base-url http://localhost:${PORT} \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 50 --request-rate ${rate} \
        --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
        --save-result --result-dir "${RESULT_DIR}/online" \
        --result-filename "b2b_real_${LABEL}_rate${rate}.json" 2>&1 | grep -E "TTFT|TPOT"
done

pkill -f api_server 2>/dev/null || true
sleep 2
pkill -9 -f EngineCore 2>/dev/null || true
sleep 5

echo ""
echo "============================================================"
echo "PHASE 3: Emulator"
echo "============================================================"

VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="${SERVING_PROFILE}" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 2048 \
    --port ${PORT} --trust-remote-code \
    --tensor-parallel-size 2 --enforce-eager \
    --gpu-memory-utilization 0.95 \
    > /workspace/b2b_${LABEL}_emu_server.log 2>&1 &

for i in $(seq 1 180); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
        echo "  Server ready after ${i}s"
        break
    fi
    sleep 1
done

grep ExecutorEmulatorHook /workspace/b2b_${LABEL}_emu_server.log 2>/dev/null | head -1

for rate in 1 2 4; do
    echo "  --- Emu rate=${rate} ---"
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "${MODEL}" --base-url http://localhost:${PORT} \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 50 --request-rate ${rate} \
        --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
        --save-result --result-dir "${RESULT_DIR}/online" \
        --result-filename "b2b_emu_${LABEL}_rate${rate}.json" 2>&1 | grep -E "TTFT|TPOT"
done

pkill -f api_server 2>/dev/null || true
sleep 2
pkill -9 -f EngineCore 2>/dev/null || true

echo ""
echo "============================================================"
echo "ERROR ANALYSIS"
echo "============================================================"
python3 /workspace/vllm-emulator-v18/paper/isca-2026-mlarchsys/scripts/temp/compare_results.py "${LABEL}"

echo ""
echo "ALL DONE"
