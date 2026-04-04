#!/bin/bash
# Chunked prefill OFF ablation: trace + b2b eval
# Default has chunked prefill ON (max_num_batched_tokens=2048)
# We disable by setting --max-num-batched-tokens to a very large value
# so all prefill tokens go in one step
set -e
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"
PORT=8100

pkill -9 -f EngineCore 2>/dev/null || true
pkill -9 -f api_server 2>/dev/null || true
sleep 5

echo "============================================================"
echo "Chunked Prefill OFF Ablation"
echo "============================================================"

# Step 1: Serving trace with chunked prefill disabled
echo ""
echo "Step 1: Serving trace (no chunked prefill)..."
rm -f "${RESULT_DIR}/step_cycle_1.5b_nochunk.jsonl"

VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
VLLM_EMULATOR_STEP_TRACE_OUTPUT="${RESULT_DIR}/step_cycle_1.5b_nochunk.jsonl" \
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 4096 \
    --port ${PORT} --trust-remote-code \
    --max-num-batched-tokens 32768 \
    > /workspace/nochunk_trace_server.log 2>&1 &

for i in $(seq 1 120); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then echo "  Server ready after ${i}s"; break; fi
    sleep 1
done

if ! curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
    echo "FAILED"; tail -10 /workspace/nochunk_trace_server.log; pkill -9 -f EngineCore 2>/dev/null || true; exit 1
fi

# Verify chunked prefill is off
grep -i "chunked" /workspace/nochunk_trace_server.log | head -1

for rate in 1 2 4; do
    echo "  Tracing rate=${rate}..."
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "${MODEL}" --base-url http://localhost:${PORT} \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 50 --request-rate ${rate} > /dev/null 2>&1
done

pkill -f api_server 2>/dev/null || true; sleep 2; pkill -9 -f EngineCore 2>/dev/null || true; sleep 5

echo "  Trace: $(wc -l < ${RESULT_DIR}/step_cycle_1.5b_nochunk.jsonl) records"

# Step 2: Build serving profile
echo ""
echo "Step 2: Build no-chunk serving profile..."
python3 /workspace/vllm-emulator-v18/paper/isca-2026-mlarchsys/scripts/temp/build_serving_profile_generic.py \
    "${RESULT_DIR}/step_cycle_1.5b_nochunk.jsonl" \
    "${RESULT_DIR}/profiles/sweep-1.5b-tp1-dense-v2.json" \
    "${RESULT_DIR}/profiles/serving-1.5b-tp1-nochunk.json" \
    "${MODEL}" "RTX-3060-12GB"

NOCHUNK_PROFILE="${RESULT_DIR}/profiles/serving-1.5b-tp1-nochunk.json"

# Step 3: B2B eval
echo ""
echo "Step 3: Real (no chunked prefill)..."
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 4096 \
    --port ${PORT} --trust-remote-code \
    --max-num-batched-tokens 32768 \
    > /workspace/nochunk_real_server.log 2>&1 &

for i in $(seq 1 120); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then echo "  Server ready after ${i}s"; break; fi
    sleep 1
done

python3 -m vllm.entrypoints.cli.main bench serve \
    --model "${MODEL}" --base-url http://localhost:${PORT} \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 50 --request-rate 2 \
    --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
    --save-result --result-dir "${RESULT_DIR}/online" \
    --result-filename "ablation_real_nochunk.json" 2>&1 | grep -E "TTFT|TPOT"

pkill -f api_server 2>/dev/null || true; sleep 2; pkill -9 -f EngineCore 2>/dev/null || true; sleep 5

echo ""
echo "Step 4: Emu (no chunked prefill)..."
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="${NOCHUNK_PROFILE}" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 4096 \
    --port ${PORT} --trust-remote-code \
    --max-num-batched-tokens 32768 \
    > /workspace/nochunk_emu_server.log 2>&1 &

for i in $(seq 1 120); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then echo "  Server ready after ${i}s"; break; fi
    sleep 1
done

python3 -m vllm.entrypoints.cli.main bench serve \
    --model "${MODEL}" --base-url http://localhost:${PORT} \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 50 --request-rate 2 \
    --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
    --save-result --result-dir "${RESULT_DIR}/online" \
    --result-filename "ablation_emu_nochunk.json" 2>&1 | grep -E "TTFT|TPOT"

pkill -f api_server 2>/dev/null || true; sleep 2; pkill -9 -f EngineCore 2>/dev/null || true

# Results
echo ""
echo "============================================================"
echo "CHUNKED PREFILL ABLATION"
echo "============================================================"
python3 -c "
import json, os
rd = '/workspace/eval_results/RTX-3060-12GB/online'

# Compare default (chunked) vs no-chunk
configs = [
    ('Default (chunked)', 'ablation_real_default', 'ablation_emu_default'),
    ('No chunk', 'ablation_real_nochunk', 'ablation_emu_nochunk'),
]

for name, rf, ef in configs:
    rp, ep = f'{rd}/{rf}.json', f'{rd}/{ef}.json'
    if os.path.exists(rp) and os.path.exists(ep):
        r, e = json.load(open(rp)), json.load(open(ep))
        te = (e['mean_ttft_ms']-r['mean_ttft_ms'])/r['mean_ttft_ms']*100
        pe = (e['mean_tpot_ms']-r['mean_tpot_ms'])/r['mean_tpot_ms']*100
        tok = '✓' if abs(te)<5 else '✗'
        pok = '✓' if abs(pe)<5 else '✗'
        print(f'{name}:')
        print(f'  Real: TTFT={r[\"mean_ttft_ms\"]:.1f}ms  TPOT={r[\"mean_tpot_ms\"]:.1f}ms')
        print(f'  Emu:  TTFT={e[\"mean_ttft_ms\"]:.1f}ms  TPOT={e[\"mean_tpot_ms\"]:.1f}ms')
        print(f'  Error: TTFT {te:+.1f}% {tok}  TPOT {pe:+.1f}% {pok}')
        print()

# Show impact of chunked prefill
for label in ['real', 'emu']:
    dp = f'{rd}/ablation_{label}_default.json'
    np = f'{rd}/ablation_{label}_nochunk.json'
    if os.path.exists(dp) and os.path.exists(np):
        d, n = json.load(open(dp)), json.load(open(np))
        print(f'Chunked prefill impact ({label}):')
        print(f'  Default: TTFT={d[\"mean_ttft_ms\"]:.1f}ms  TPOT={d[\"mean_tpot_ms\"]:.1f}ms')
        print(f'  No chunk: TTFT={n[\"mean_ttft_ms\"]:.1f}ms  TPOT={n[\"mean_tpot_ms\"]:.1f}ms')
"
echo ""
echo "DONE"
