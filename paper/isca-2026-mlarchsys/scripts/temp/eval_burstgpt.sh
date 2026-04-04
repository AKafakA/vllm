#!/bin/bash
# BurstGPT trace replay: online serving with real-world arrival patterns
# Back-to-back real vs emulator
set -e
source /workspace/vllm-v18-env/bin/activate

export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"
SERVING_PROFILE="${RESULT_DIR}/profiles/serving-1.5b-tp1-step-cycle.json"
BURSTGPT_CSV="/workspace/BurstGPT_1.csv"
PORT=8100
NUM_PROMPTS=100

echo "============================================================"
echo "BurstGPT Trace Replay: Online Serving"
echo "============================================================"

pkill -9 -f EngineCore 2>/dev/null || true
pkill -9 -f api_server 2>/dev/null || true
sleep 5

# Phase 1: Real baseline with BurstGPT trace
echo ""
echo "PHASE 1: Real baseline (BurstGPT trace)"
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 4096 \
    --port ${PORT} --trust-remote-code \
    > /workspace/burstgpt_real_server.log 2>&1 &

for i in $(seq 1 120); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then echo "  Server ready after ${i}s"; break; fi
    sleep 1
done

if ! curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
    echo "FAILED: server didn't start"
    pkill -9 -f EngineCore 2>/dev/null || true
    exit 1
fi

# BurstGPT uses real arrival timestamps from the trace
for rate in 2 4; do
    echo "  --- Real BurstGPT rate=${rate} ---"
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "${MODEL}" --base-url http://localhost:${PORT} \
        --dataset-name burstgpt --dataset-path "${BURSTGPT_CSV}" \
        --num-prompts ${NUM_PROMPTS} --request-rate ${rate} \
        --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
        --save-result --result-dir "${RESULT_DIR}/online" \
        --result-filename "burstgpt_real_rate${rate}.json" 2>&1 | grep -E "TTFT|TPOT|Throughput"
done

# Also run with random for comparison
echo "  --- Real Random rate=2 (control) ---"
python3 -m vllm.entrypoints.cli.main bench serve \
    --model "${MODEL}" --base-url http://localhost:${PORT} \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts ${NUM_PROMPTS} --request-rate 2 \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
    --save-result --result-dir "${RESULT_DIR}/online" \
    --result-filename "burstgpt_real_random_rate2.json" 2>&1 | grep -E "TTFT|TPOT|Throughput"

pkill -f api_server 2>/dev/null || true; sleep 2; pkill -9 -f EngineCore 2>/dev/null || true; sleep 5

# Phase 2: Emulator with serving profile
echo ""
echo "PHASE 2: Emulator (BurstGPT trace)"
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="${SERVING_PROFILE}" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 4096 \
    --port ${PORT} --trust-remote-code \
    > /workspace/burstgpt_emu_server.log 2>&1 &

for i in $(seq 1 120); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then echo "  Server ready after ${i}s"; break; fi
    sleep 1
done

if ! curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
    echo "FAILED: emulator server didn't start"
    pkill -9 -f EngineCore 2>/dev/null || true
    exit 1
fi

grep ExecutorEmulatorHook /workspace/burstgpt_emu_server.log 2>/dev/null | head -1

for rate in 2 4; do
    echo "  --- Emu BurstGPT rate=${rate} ---"
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "${MODEL}" --base-url http://localhost:${PORT} \
        --dataset-name burstgpt --dataset-path "${BURSTGPT_CSV}" \
        --num-prompts ${NUM_PROMPTS} --request-rate ${rate} \
        --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
        --save-result --result-dir "${RESULT_DIR}/online" \
        --result-filename "burstgpt_emu_rate${rate}.json" 2>&1 | grep -E "TTFT|TPOT|Throughput"
done

# Random control
echo "  --- Emu Random rate=2 (control) ---"
python3 -m vllm.entrypoints.cli.main bench serve \
    --model "${MODEL}" --base-url http://localhost:${PORT} \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts ${NUM_PROMPTS} --request-rate 2 \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
    --save-result --result-dir "${RESULT_DIR}/online" \
    --result-filename "burstgpt_emu_random_rate2.json" 2>&1 | grep -E "TTFT|TPOT|Throughput"

pkill -f api_server 2>/dev/null || true; sleep 2; pkill -9 -f EngineCore 2>/dev/null || true

# Phase 3: Results
echo ""
echo "============================================================"
echo "BURSTGPT RESULTS"
echo "============================================================"

python3 -c "
import json, os
rd = '${RESULT_DIR}/online'

print(f\"{'Config':<35} {'TTFT':>8} {'P99TTFT':>9} {'TPOT':>8} {'P99TPOT':>9}\")
print('-' * 72)

for rate in [2, 4]:
    for pfx in ['real', 'emu']:
        f = f'{rd}/burstgpt_{pfx}_rate{rate}.json'
        if os.path.exists(f):
            d = json.load(open(f))
            print(f\"{pfx+' BurstGPT rate='+str(rate):<35} {d['mean_ttft_ms']:>8.1f} {d['p99_ttft_ms']:>9.1f} {d['mean_tpot_ms']:>8.1f} {d['p99_tpot_ms']:>9.1f}\")

for pfx in ['real', 'emu']:
    f = f'{rd}/burstgpt_{pfx}_random_rate2.json'
    if os.path.exists(f):
        d = json.load(open(f))
        print(f\"{pfx+' Random rate=2 (ctrl)':<35} {d['mean_ttft_ms']:>8.1f} {d['p99_ttft_ms']:>9.1f} {d['mean_tpot_ms']:>8.1f} {d['p99_tpot_ms']:>9.1f}\")

print()
print('Error analysis:')
for rate in [2, 4]:
    rf = f'{rd}/burstgpt_real_rate{rate}.json'
    ef = f'{rd}/burstgpt_emu_rate{rate}.json'
    if os.path.exists(rf) and os.path.exists(ef):
        r, e = json.load(open(rf)), json.load(open(ef))
        te = (e['mean_ttft_ms']-r['mean_ttft_ms'])/r['mean_ttft_ms']*100
        pe = (e['mean_tpot_ms']-r['mean_tpot_ms'])/r['mean_tpot_ms']*100
        tok = '✓' if abs(te)<5 else '✗'
        pok = '✓' if abs(pe)<5 else '✗'
        print(f'  BurstGPT rate={rate}: TTFT {te:>+6.1f}% {tok}  TPOT {pe:>+6.1f}% {pok}')

# Random control
rf = f'{rd}/burstgpt_real_random_rate2.json'
ef = f'{rd}/burstgpt_emu_random_rate2.json'
if os.path.exists(rf) and os.path.exists(ef):
    r, e = json.load(open(rf)), json.load(open(ef))
    te = (e['mean_ttft_ms']-r['mean_ttft_ms'])/r['mean_ttft_ms']*100
    pe = (e['mean_tpot_ms']-r['mean_tpot_ms'])/r['mean_tpot_ms']*100
    tok = '✓' if abs(te)<5 else '✗'
    pok = '✓' if abs(pe)<5 else '✗'
    print(f'  Random rate=2 (ctrl): TTFT {te:>+6.1f}% {tok}  TPOT {pe:>+6.1f}% {pok}')
"

echo ""
echo "BURSTGPT DONE"
