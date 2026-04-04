#!/bin/bash
# Re-profile with extended decode batch sizes (up to 128) and test offline
set -e
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"

pkill -9 -f EngineCore 2>/dev/null || true
sleep 5

echo "============================================================"
echo "Re-profile with extended decode batch sizes (up to 128)"
echo "============================================================"

SWEEP_OUT="${RESULT_DIR}/profiles/sweep-1.5b-tp1-v14.json"
TRACE_OUT="${RESULT_DIR}/sweep_trace_1.5b_v14.jsonl"

echo "  Running sweep profiler..."
timeout 1800 python3 /workspace/vllm-emulator-v18/paper/isca-2026-mlarchsys/scripts/shape_sweep_profiler.py \
    --model "${MODEL}" --gpu-model RTX-3060-12GB \
    --output "${SWEEP_OUT}" \
    --max-model-len 4096 --max-num-seqs 128 \
    --max-output-len 128 --tp 1 \
    --trace-output "${TRACE_OUT}" || echo "  Sweep timed out (partial)"

pkill -9 -f EngineCore 2>/dev/null || true; sleep 5
echo "  Trace: $(wc -l < ${TRACE_OUT} 2>/dev/null || echo 0) records"

# Check coverage at offline batch sizes
echo ""
echo "=== Profile Coverage Check ==="
python3 -c "
import json
p = json.load(open('${SWEEP_OUT}'))
print(f'Total buckets: {len(p[\"forward_pass\"])}')
print('Coverage at tt=30-128:')
for e in p['forward_pass']:
    tt = e['total_tokens']
    if 25 <= tt <= 130:
        print(f'  tt={tt:>4}: {e[\"latency_us\"]/1000:.1f}ms (n={e.get(\"num_samples\",0)})')
"

echo ""
echo "=== Test with v14 profile ==="
for n in 30 100; do
    echo ""
    echo "  Real N=${n}..."
    python3 -m vllm.entrypoints.cli.main bench throughput \
        --model "${MODEL}" --max-model-len 4096 \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts ${n} \
        --output-json "${RESULT_DIR}/offline/v14_real_${n}.json" 2>&1 | grep Throughput
    pkill -9 -f EngineCore 2>/dev/null || true; sleep 5

    echo "  Emu N=${n} (v14)..."
    VLLM_EMULATOR_ENABLE_ORACLE=1 \
    VLLM_EMULATOR_PROFILE_PACK="${SWEEP_OUT}" \
    VLLM_EMULATOR_MODE=realtime \
    python3 -m vllm.entrypoints.cli.main bench throughput \
        --model "${MODEL}" --max-model-len 4096 \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts ${n} \
        --output-json "${RESULT_DIR}/offline/v14_emu_${n}.json" 2>&1 | grep Throughput
    pkill -9 -f EngineCore 2>/dev/null || true; sleep 5
done

echo ""
echo "=== COMPARISON ==="
python3 -c "
import json, os
rd = '${RESULT_DIR}/offline'
for n in [30, 100]:
    rf = f'{rd}/v14_real_{n}.json'
    ef = f'{rd}/v14_emu_{n}.json'
    if os.path.exists(rf) and os.path.exists(ef):
        r, e = json.load(open(rf)), json.load(open(ef))
        rt = r.get('tokens_per_second', r.get('generation_tokens_per_second', 0))
        et = e.get('tokens_per_second', e.get('generation_tokens_per_second', 0))
        err = (et - rt) / rt * 100
        ok = '✓' if abs(err) < 5 else '✗'
        print(f'N={n}: Real={rt:.0f} Emu={et:.0f} Error={err:+.1f}% {ok}')
"
echo ""
echo "DONE"
