#!/bin/bash
# Test worker hook (NO executor hook) with correctly formatted serving profile
# The worker hook blocks the worker thread with time.sleep — may deadlock at high rate
set -e
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"
PROFILE="${RESULT_DIR}/profiles/serving-1.5b-tp1-same.json"
PORT=8100

pkill -9 -f EngineCore 2>/dev/null || true; pkill -9 -f api_server 2>/dev/null || true; sleep 5

# Real baseline (from same_session results)
echo "=== Emulator (worker hook, correct profile) ==="

# NO EXECUTOR_HOOK — worker hook only
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="${PROFILE}" \
VLLM_EMULATOR_MODE=realtime \
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 4096 --port ${PORT} --trust-remote-code \
    > /workspace/worker_correct_server.log 2>&1 &

for i in $(seq 1 120); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then echo "  Ready ${i}s"; break; fi; sleep 1
done

# Test at rate=1 first (should work)
echo "  Rate=1..."
timeout 120 python3 -m vllm.entrypoints.cli.main bench serve --model "${MODEL}" --base-url http://localhost:${PORT} \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 30 --request-rate 1 --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
    --save-result --result-dir "${RESULT_DIR}/online" --result-filename "worker_correct_rate1.json" 2>&1 | grep -E "TTFT|TPOT" || echo "  TIMEOUT/FAILED at rate=1"

# Test at rate=2 (may deadlock with 30 prompts)
echo "  Rate=2..."
timeout 120 python3 -m vllm.entrypoints.cli.main bench serve --model "${MODEL}" --base-url http://localhost:${PORT} \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 30 --request-rate 2 --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
    --save-result --result-dir "${RESULT_DIR}/online" --result-filename "worker_correct_rate2.json" 2>&1 | grep -E "TTFT|TPOT" || echo "  TIMEOUT/FAILED at rate=2"

# Test at rate=4 (likely deadlock)
echo "  Rate=4..."
timeout 120 python3 -m vllm.entrypoints.cli.main bench serve --model "${MODEL}" --base-url http://localhost:${PORT} \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 30 --request-rate 4 --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
    --save-result --result-dir "${RESULT_DIR}/online" --result-filename "worker_correct_rate4.json" 2>&1 | grep -E "TTFT|TPOT" || echo "  TIMEOUT/FAILED at rate=4"

pkill -f api_server 2>/dev/null || true; sleep 2; pkill -9 -f EngineCore 2>/dev/null || true

echo ""
echo "=== Comparison (using same-session real baseline) ==="
python3 -c "
import json, os
rd = '${RESULT_DIR}/online'
for rate in [1, 2, 4]:
    rf = f'{rd}/same_real_rate{rate}.json'
    ef = f'{rd}/worker_correct_rate{rate}.json'
    if os.path.exists(rf) and os.path.exists(ef):
        r, e = json.load(open(rf)), json.load(open(ef))
        te = (e['mean_ttft_ms']-r['mean_ttft_ms'])/r['mean_ttft_ms']*100
        pe = (e['mean_tpot_ms']-r['mean_tpot_ms'])/r['mean_tpot_ms']*100
        print(f'rate={rate}: TTFT {te:+.1f}%  TPOT {pe:+.1f}%')
    else:
        print(f'rate={rate}: MISSING (deadlock?)')
"
echo "DONE"
