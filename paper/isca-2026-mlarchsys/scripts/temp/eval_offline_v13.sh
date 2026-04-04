#!/bin/bash
# Quick offline throughput test with v13 profile
set -e
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"

pkill -9 -f EngineCore 2>/dev/null || true
sleep 5

echo "=== Real ==="
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "${MODEL}" --max-model-len 4096 \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 100 \
    --output-json "${RESULT_DIR}/offline/real_offline_v3.json" 2>&1 | grep Throughput

pkill -9 -f EngineCore 2>/dev/null || true
sleep 5

echo "=== Emu v13 ==="
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="${RESULT_DIR}/profiles/sweep-1.5b-tp1-v13.json" \
VLLM_EMULATOR_MODE=realtime \
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "${MODEL}" --max-model-len 4096 \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 100 \
    --output-json "${RESULT_DIR}/offline/emu_offline_v13.json" 2>&1 | grep Throughput

pkill -9 -f EngineCore 2>/dev/null || true
sleep 5

echo "=== Emu dense-v2 ==="
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="${RESULT_DIR}/profiles/sweep-1.5b-tp1-dense-v2.json" \
VLLM_EMULATOR_MODE=realtime \
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "${MODEL}" --max-model-len 4096 \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 100 \
    --output-json "${RESULT_DIR}/offline/emu_offline_densev2.json" 2>&1 | grep Throughput

pkill -9 -f EngineCore 2>/dev/null || true

echo ""
echo "=== COMPARISON ==="
python3 -c "
import json, os
rd = '${RESULT_DIR}/offline'
for label, fname in [('Real', 'real_offline_v3'), ('Emu v13', 'emu_offline_v13'), ('Emu dense-v2', 'emu_offline_densev2')]:
    f = f'{rd}/{fname}.json'
    if os.path.exists(f):
        d = json.load(open(f))
        tps = d.get('tokens_per_second', d.get('generation_tokens_per_second', 0))
        print(f'{label:<15}: {tps:.0f} tok/s')

r = json.load(open(f'{rd}/real_offline_v3.json'))
real_tps = r.get('tokens_per_second', r.get('generation_tokens_per_second', 0))
for label, fname in [('v13', 'emu_offline_v13'), ('dense-v2', 'emu_offline_densev2')]:
    f = f'{rd}/{fname}.json'
    if os.path.exists(f):
        d = json.load(open(f))
        tps = d.get('tokens_per_second', d.get('generation_tokens_per_second', 0))
        err = (tps - real_tps) / real_tps * 100
        ok = '✓' if abs(err) < 5 else '✗'
        print(f'  {label} error: {err:+.1f}% {ok}')
"
echo "DONE"
