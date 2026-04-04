#!/bin/bash
# Offline throughput back-to-back + accelerated mode test
set -e
source /workspace/vllm-v18-env/bin/activate

export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"
SWEEP_PROFILE="${RESULT_DIR}/profiles/sweep-1.5b-tp1-dense-v2.json"

pkill -9 -f EngineCore 2>/dev/null || true
pkill -9 -f api_server 2>/dev/null || true
sleep 5

echo "============================================================"
echo "PHASE 1: Offline throughput (realtime mode, sweep profile)"
echo "============================================================"

mkdir -p "${RESULT_DIR}/offline"

# Real baseline
echo "  Real offline throughput..."
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "${MODEL}" --max-model-len 4096 \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 100 \
    --output-json "${RESULT_DIR}/offline/b2b_real_offline_1.5b.json" 2>&1 | grep -E "Throughput|tok"

pkill -9 -f EngineCore 2>/dev/null || true
sleep 5

# Emulator (realtime mode - should match real throughput)
echo ""
echo "  Emulator offline throughput (realtime)..."
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="${SWEEP_PROFILE}" \
VLLM_EMULATOR_MODE=realtime \
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "${MODEL}" --max-model-len 4096 \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 100 \
    --output-json "${RESULT_DIR}/offline/b2b_emu_offline_1.5b.json" 2>&1 | grep -E "Throughput|tok"

pkill -9 -f EngineCore 2>/dev/null || true
sleep 5

echo ""
echo "============================================================"
echo "PHASE 2: Accelerated mode (virtual time, no blocking)"
echo "============================================================"

# Accelerated mode - should run much faster, same total virtual time
echo "  Emulator offline throughput (accelerated)..."
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="${SWEEP_PROFILE}" \
VLLM_EMULATOR_MODE=accelerated \
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "${MODEL}" --max-model-len 4096 \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 100 \
    --output-json "${RESULT_DIR}/offline/b2b_accel_offline_1.5b.json" 2>&1 | grep -E "Throughput|tok"

pkill -9 -f EngineCore 2>/dev/null || true
sleep 3

echo ""
echo "============================================================"
echo "RESULTS"
echo "============================================================"

python3 -c "
import json, os
rd = '${RESULT_DIR}/offline'

results = {}
for label, fname in [('Real', 'b2b_real_offline_1.5b'), ('Emu realtime', 'b2b_emu_offline_1.5b'), ('Emu accelerated', 'b2b_accel_offline_1.5b')]:
    f = f'{rd}/{fname}.json'
    if os.path.exists(f):
        d = json.load(open(f))
        tps = d.get('tokens_per_second', d.get('generation_tokens_per_second', 0))
        elapsed = d.get('elapsed_time', 0)
        results[label] = {'tps': tps, 'elapsed': elapsed}
        print(f'{label:<20}: {tps:>8.0f} tok/s  (elapsed: {elapsed:.1f}s)')

if 'Real' in results and 'Emu realtime' in results:
    err = (results['Emu realtime']['tps'] - results['Real']['tps']) / results['Real']['tps'] * 100
    ok = '✓' if abs(err) < 5 else '✗'
    print(f'  Realtime error: {err:+.1f}% {ok}')

if 'Real' in results and 'Emu accelerated' in results:
    speedup = results['Real']['elapsed'] / results['Emu accelerated']['elapsed'] if results['Emu accelerated']['elapsed'] > 0 else 0
    print(f'  Accelerated speedup: {speedup:.1f}x faster than real')
"

echo ""
echo "DONE"
