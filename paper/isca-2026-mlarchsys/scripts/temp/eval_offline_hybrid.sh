#!/bin/bash
# Test offline throughput with hybrid sleep (sleep + busywait)
set -e
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"
SWEEP_PROFILE="${RESULT_DIR}/profiles/sweep-1.5b-tp1-v13.json"

pkill -9 -f EngineCore 2>/dev/null || true
sleep 5

echo "=== Real ==="
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "${MODEL}" --max-model-len 4096 \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 100 \
    --output-json "${RESULT_DIR}/offline/real_offline_hybrid.json" 2>&1 | grep Throughput

pkill -9 -f EngineCore 2>/dev/null || true; sleep 5

echo "=== Emu (hybrid sleep) ==="
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="${SWEEP_PROFILE}" \
VLLM_EMULATOR_MODE=realtime \
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "${MODEL}" --max-model-len 4096 \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 100 \
    --output-json "${RESULT_DIR}/offline/emu_offline_hybrid.json" 2>&1 | grep Throughput

pkill -9 -f EngineCore 2>/dev/null || true

echo ""
echo "=== COMPARISON ==="
python3 -c "
import json, os
rd = '${RESULT_DIR}/offline'
for label, fname in [('Real', 'real_offline_hybrid'), ('Emu hybrid', 'emu_offline_hybrid')]:
    f = f'{rd}/{fname}.json'
    if os.path.exists(f):
        d = json.load(open(f))
        tps = d.get('tokens_per_second', d.get('generation_tokens_per_second', 0))
        elapsed = d.get('elapsed_time', 0)
        print(f'{label:<15}: {tps:.0f} tok/s ({elapsed:.1f}s)')
r = json.load(open(f'{rd}/real_offline_hybrid.json'))
e = json.load(open(f'{rd}/emu_offline_hybrid.json'))
rt = r.get('tokens_per_second', r.get('generation_tokens_per_second', 0))
et = e.get('tokens_per_second', e.get('generation_tokens_per_second', 0))
err = (et - rt) / rt * 100
ok = '✓' if abs(err) < 5 else '✗'
print(f'Error: {err:+.1f}% {ok}')
"
echo "DONE"
