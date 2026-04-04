#!/bin/bash
# Feature ablation evaluation: default vs enforce-eager (no CUDA graphs)
# Uses 1.5B model, TP=1, serving profile
set -e
source /workspace/vllm-v18-env/bin/activate

export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"
SERVING_PROFILE="${RESULT_DIR}/profiles/serving-1.5b-tp1-step-cycle.json"
PORT=8100
NUM_PROMPTS=50
RATE=2

echo "============================================================"
echo "Feature Ablation: Default (chunked prefill + CUDA graphs)"
echo "============================================================"

pkill -9 -f EngineCore 2>/dev/null || true
pkill -9 -f api_server 2>/dev/null || true
sleep 5

# --- Default config: real ---
echo ""
echo "  Real (default config)..."
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 4096 \
    --port ${PORT} --trust-remote-code \
    > /workspace/ablation_default_real_server.log 2>&1 &

for i in $(seq 1 120); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then echo "  Server ready after ${i}s"; break; fi
    sleep 1
done

python3 -m vllm.entrypoints.cli.main bench serve \
    --model "${MODEL}" --base-url http://localhost:${PORT} \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts ${NUM_PROMPTS} --request-rate ${RATE} \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
    --save-result --result-dir "${RESULT_DIR}/online" \
    --result-filename "ablation_real_default.json" 2>&1 | grep -E "TTFT|TPOT"

pkill -f api_server 2>/dev/null || true; sleep 2; pkill -9 -f EngineCore 2>/dev/null || true; sleep 5

# --- Default config: emulator ---
echo ""
echo "  Emu (default config)..."
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="${SERVING_PROFILE}" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 4096 \
    --port ${PORT} --trust-remote-code \
    > /workspace/ablation_default_emu_server.log 2>&1 &

for i in $(seq 1 120); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then echo "  Server ready after ${i}s"; break; fi
    sleep 1
done

python3 -m vllm.entrypoints.cli.main bench serve \
    --model "${MODEL}" --base-url http://localhost:${PORT} \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts ${NUM_PROMPTS} --request-rate ${RATE} \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
    --save-result --result-dir "${RESULT_DIR}/online" \
    --result-filename "ablation_emu_default.json" 2>&1 | grep -E "TTFT|TPOT"

pkill -f api_server 2>/dev/null || true; sleep 2; pkill -9 -f EngineCore 2>/dev/null || true; sleep 5

echo ""
echo "============================================================"
echo "Feature Ablation: enforce-eager (no CUDA graphs)"
echo "============================================================"

# --- First need a serving profile for enforce-eager config ---
echo ""
echo "  Building enforce-eager serving profile..."
rm -f "${RESULT_DIR}/step_cycle_1.5b_eager.jsonl"

VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
VLLM_EMULATOR_STEP_TRACE_OUTPUT="${RESULT_DIR}/step_cycle_1.5b_eager.jsonl" \
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 4096 \
    --port ${PORT} --trust-remote-code --enforce-eager \
    > /workspace/ablation_eager_trace_server.log 2>&1 &

for i in $(seq 1 120); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then echo "  Server ready after ${i}s"; break; fi
    sleep 1
done

for rate in 1 2 4; do
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "${MODEL}" --base-url http://localhost:${PORT} \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 30 --request-rate ${rate} > /dev/null 2>&1
done

pkill -f api_server 2>/dev/null || true; sleep 2; pkill -9 -f EngineCore 2>/dev/null || true; sleep 5

# Build eager serving profile
python3 -c "
import json, statistics
from collections import defaultdict

records = []
for line in open('${RESULT_DIR}/step_cycle_1.5b_eager.jsonl'):
    r = json.loads(line)
    if 'total_tokens' in r:
        records.append(r)

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
print(f'Eager serving profile: {len(forward_pass)} buckets')
"

EAGER_PROFILE="${RESULT_DIR}/profiles/serving-1.5b-tp1-eager-step-cycle.json"

# --- Enforce-eager: real ---
echo ""
echo "  Real (enforce-eager)..."
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 4096 \
    --port ${PORT} --trust-remote-code --enforce-eager \
    > /workspace/ablation_eager_real_server.log 2>&1 &

for i in $(seq 1 120); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then echo "  Server ready after ${i}s"; break; fi
    sleep 1
done

python3 -m vllm.entrypoints.cli.main bench serve \
    --model "${MODEL}" --base-url http://localhost:${PORT} \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts ${NUM_PROMPTS} --request-rate ${RATE} \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
    --save-result --result-dir "${RESULT_DIR}/online" \
    --result-filename "ablation_real_eager.json" 2>&1 | grep -E "TTFT|TPOT"

pkill -f api_server 2>/dev/null || true; sleep 2; pkill -9 -f EngineCore 2>/dev/null || true; sleep 5

# --- Enforce-eager: emulator ---
echo ""
echo "  Emu (enforce-eager)..."
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="${EAGER_PROFILE}" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 4096 \
    --port ${PORT} --trust-remote-code --enforce-eager \
    > /workspace/ablation_eager_emu_server.log 2>&1 &

for i in $(seq 1 120); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then echo "  Server ready after ${i}s"; break; fi
    sleep 1
done

python3 -m vllm.entrypoints.cli.main bench serve \
    --model "${MODEL}" --base-url http://localhost:${PORT} \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts ${NUM_PROMPTS} --request-rate ${RATE} \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
    --save-result --result-dir "${RESULT_DIR}/online" \
    --result-filename "ablation_emu_eager.json" 2>&1 | grep -E "TTFT|TPOT"

pkill -f api_server 2>/dev/null || true; sleep 2; pkill -9 -f EngineCore 2>/dev/null || true

# --- Results ---
echo ""
echo "============================================================"
echo "ABLATION RESULTS"
echo "============================================================"

python3 -c "
import json, os
rd = '${RESULT_DIR}/online'

print('Default (chunked prefill + CUDA graphs):')
for pfx in ['real', 'emu']:
    f = f'{rd}/ablation_{pfx}_default.json'
    if os.path.exists(f):
        d = json.load(open(f))
        print(f'  {pfx}: TTFT={d[\"mean_ttft_ms\"]:.1f}ms  TPOT={d[\"mean_tpot_ms\"]:.1f}ms')

r = json.load(open(f'{rd}/ablation_real_default.json'))
e = json.load(open(f'{rd}/ablation_emu_default.json'))
te = (e['mean_ttft_ms']-r['mean_ttft_ms'])/r['mean_ttft_ms']*100
pe = (e['mean_tpot_ms']-r['mean_tpot_ms'])/r['mean_tpot_ms']*100
print(f'  Error: TTFT {te:+.1f}%, TPOT {pe:+.1f}%')

print()
print('Enforce-eager (no CUDA graphs):')
for pfx in ['real', 'emu']:
    f = f'{rd}/ablation_{pfx}_eager.json'
    if os.path.exists(f):
        d = json.load(open(f))
        print(f'  {pfx}: TTFT={d[\"mean_ttft_ms\"]:.1f}ms  TPOT={d[\"mean_tpot_ms\"]:.1f}ms')

r = json.load(open(f'{rd}/ablation_real_eager.json'))
e = json.load(open(f'{rd}/ablation_emu_eager.json'))
te = (e['mean_ttft_ms']-r['mean_ttft_ms'])/r['mean_ttft_ms']*100
pe = (e['mean_tpot_ms']-r['mean_tpot_ms'])/r['mean_tpot_ms']*100
print(f'  Error: TTFT {te:+.1f}%, TPOT {pe:+.1f}%')

print()
print('Feature impact (real):')
rd_default = json.load(open(f'{rd}/ablation_real_default.json'))
rd_eager = json.load(open(f'{rd}/ablation_real_eager.json'))
print(f'  CUDA graphs effect on TPOT: {rd_default[\"mean_tpot_ms\"]:.1f}ms (with) vs {rd_eager[\"mean_tpot_ms\"]:.1f}ms (without)')
print(f'  Speedup: {rd_eager[\"mean_tpot_ms\"]/rd_default[\"mean_tpot_ms\"]:.2f}x slower without CUDA graphs')

print()
print('Feature impact (emulator):')
ed_default = json.load(open(f'{rd}/ablation_emu_default.json'))
ed_eager = json.load(open(f'{rd}/ablation_emu_eager.json'))
print(f'  CUDA graphs effect on TPOT: {ed_default[\"mean_tpot_ms\"]:.1f}ms (with) vs {ed_eager[\"mean_tpot_ms\"]:.1f}ms (without)')
print(f'  Speedup: {ed_eager[\"mean_tpot_ms\"]/ed_default[\"mean_tpot_ms\"]:.2f}x slower without CUDA graphs')
"

echo ""
echo "ABLATION DONE"
