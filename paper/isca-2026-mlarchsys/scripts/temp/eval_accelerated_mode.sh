#!/bin/bash
# Accelerated mode validation: verify virtual time predicts real throughput
set -e
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"
SWEEP_PROFILE="${RESULT_DIR}/profiles/sweep-1.5b-tp1-v13.json"

pkill -9 -f EngineCore 2>/dev/null || true
sleep 5

echo "============================================================"
echo "Accelerated Mode Validation"
echo "============================================================"

# Real baseline
echo ""
echo "=== Real throughput ==="
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "${MODEL}" --max-model-len 4096 \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 100 \
    --output-json "${RESULT_DIR}/offline/accel_real.json" 2>&1 | grep -E "Throughput|elapsed"

pkill -9 -f EngineCore 2>/dev/null || true; sleep 5

# Accelerated mode
echo ""
echo "=== Accelerated mode (no sleep, virtual time) ==="
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="${SWEEP_PROFILE}" \
VLLM_EMULATOR_MODE=accelerated \
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "${MODEL}" --max-model-len 4096 \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 100 \
    --output-json "${RESULT_DIR}/offline/accel_emu.json" 2>&1 | grep -E "Throughput|elapsed"

pkill -9 -f EngineCore 2>/dev/null || true; sleep 5

# Also test with online serving in accelerated mode
echo ""
echo "=== Accelerated mode online serving ==="
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="${RESULT_DIR}/profiles/serving-1.5b-tp1-step-cycle.json" \
VLLM_EMULATOR_MODE=accelerated \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 4096 \
    --port 8100 --trust-remote-code \
    > /workspace/accel_online_server.log 2>&1 &

for i in $(seq 1 120); do
    if curl -s http://localhost:8100/health > /dev/null 2>&1; then echo "  Server ready after ${i}s"; break; fi
    sleep 1
done

if curl -s http://localhost:8100/health > /dev/null 2>&1; then
    echo "  Running accelerated online benchmark (rate=inf, all at once)..."
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "${MODEL}" --base-url http://localhost:8100 \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 50 --request-rate inf \
        --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
        --save-result --result-dir "${RESULT_DIR}/online" \
        --result-filename "accel_online.json" 2>&1 | grep -E "TTFT|TPOT|Throughput"
fi

pkill -f api_server 2>/dev/null || true; sleep 2; pkill -9 -f EngineCore 2>/dev/null || true

echo ""
echo "============================================================"
echo "RESULTS"
echo "============================================================"
python3 -c "
import json, os
rd = '${RESULT_DIR}/offline'

r = json.load(open(f'{rd}/accel_real.json'))
e = json.load(open(f'{rd}/accel_emu.json'))

real_tps = r.get('tokens_per_second', r.get('generation_tokens_per_second', 0))
emu_tps = e.get('tokens_per_second', e.get('generation_tokens_per_second', 0))
real_elapsed = r.get('elapsed_time', 0)
emu_elapsed = e.get('elapsed_time', 0)
speedup = real_elapsed / emu_elapsed if emu_elapsed > 0 else 0

print(f'Real:        {real_tps:.0f} tok/s ({real_elapsed:.1f}s)')
print(f'Accelerated: {emu_tps:.0f} tok/s ({emu_elapsed:.1f}s)')
print(f'Wall-time speedup: {speedup:.1f}x')
print()

# The key metric: does predicted throughput match real?
# In accelerated mode, the reported throughput is based on wall time
# (which is fast because no sleep). The virtual time throughput is
# what we want to compare: total_tokens / virtual_gpu_time
total_tokens = 100 * (256 + 128)  # num_prompts * (input + output)
print(f'Total tokens: {total_tokens}')
print(f'Real throughput: {real_tps:.0f} tok/s')
print(f'Accelerated wall throughput: {emu_tps:.0f} tok/s (not meaningful - no sleep)')
print()
print('For accelerated mode, the meaningful metric is the VIRTUAL throughput:')
print('total_tokens / virtual_gpu_time. This is reported by the emulator hook')
print('via get_virtual_time_summary() and should match real throughput.')
"
echo ""
echo "DONE"
