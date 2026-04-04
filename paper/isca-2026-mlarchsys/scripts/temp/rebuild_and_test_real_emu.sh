#!/bin/bash
# Rebuild serving profile with correct format, then test REAL emulator
set -e
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"
PORT=8100

echo "============================================================"
echo "Step 1: Rebuild serving profile with version/prefill/decode"
echo "============================================================"

# Use existing step_cycle trace to rebuild with correct format
STEP_CYCLE="${RESULT_DIR}/step_cycle_serving.jsonl"
SWEEP="${RESULT_DIR}/profiles/sweep-1.5b-tp1-v13.json"
NEW_PROFILE="${RESULT_DIR}/profiles/serving-1.5b-tp1-v2.json"

python3 /workspace/vllm-emulator-v18/paper/isca-2026-mlarchsys/scripts/temp/build_serving_profile_generic.py \
    "${STEP_CYCLE}" "${SWEEP}" "${NEW_PROFILE}" \
    "${MODEL}" "RTX-3060-12GB"

# Verify it loads
python3 -c "
from vllm_emulator.profile.loader import load_profile_pack
p = load_profile_pack('${NEW_PROFILE}')
print(f'Profile loads OK: {len(p[\"forward_pass\"])} forward_pass buckets')
print(f'Keys: {list(p.keys())}')
"

echo ""
echo "============================================================"
echo "Step 2: B2B test — real vs REAL emulator (worker hook)"
echo "============================================================"

pkill -9 -f EngineCore 2>/dev/null || true; pkill -9 -f api_server 2>/dev/null || true; sleep 5

# Real baseline
echo "Real..."
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 4096 --port ${PORT} --trust-remote-code \
    > /workspace/real_emu_real.log 2>&1 &
for i in $(seq 1 120); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then break; fi; sleep 1
done
for rate in 1 2 4; do
    python3 -m vllm.entrypoints.cli.main bench serve --model "${MODEL}" --base-url http://localhost:${PORT} \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 50 --request-rate ${rate} --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
        --save-result --result-dir "${RESULT_DIR}/online" --result-filename "real_emu_real_rate${rate}.json" > /dev/null 2>&1
done
pkill -f api_server 2>/dev/null || true; sleep 2; pkill -9 -f EngineCore 2>/dev/null || true; sleep 5

# Emulator with NEW serving profile (worker hook, NO executor hook)
echo "Emulator (worker hook, new serving profile)..."
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="${NEW_PROFILE}" \
VLLM_EMULATOR_MODE=realtime \
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 4096 --port ${PORT} --trust-remote-code \
    > /workspace/real_emu_emu.log 2>&1 &
for i in $(seq 1 120); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then break; fi; sleep 1
done
# Verify hook is active
grep 'emulator_hook\|Emulator\|hook' /workspace/real_emu_emu.log | head -3
for rate in 1 2 4; do
    python3 -m vllm.entrypoints.cli.main bench serve --model "${MODEL}" --base-url http://localhost:${PORT} \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 50 --request-rate ${rate} --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
        --save-result --result-dir "${RESULT_DIR}/online" --result-filename "real_emu_emu_rate${rate}.json" > /dev/null 2>&1
done
pkill -f api_server 2>/dev/null || true; sleep 2; pkill -9 -f EngineCore 2>/dev/null || true

echo ""
echo "============================================================"
echo "RESULTS — FIRST REAL EMULATOR TEST WITH SERVING PROFILE"
echo "============================================================"
python3 -c "
import json, os
rd = '${RESULT_DIR}/online'
print(f'{\"Rate\":>4} {\"Real_TTFT\":>10} {\"Emu_TTFT\":>10} {\"TTFT_err\":>9} {\"Real_TPOT\":>10} {\"Emu_TPOT\":>10} {\"TPOT_err\":>9}')
print('-' * 68)
for rate in [1, 2, 4]:
    rf = f'{rd}/real_emu_real_rate{rate}.json'
    ef = f'{rd}/real_emu_emu_rate{rate}.json'
    if os.path.exists(rf) and os.path.exists(ef):
        r, e = json.load(open(rf)), json.load(open(ef))
        te = (e['mean_ttft_ms']-r['mean_ttft_ms'])/r['mean_ttft_ms']*100
        pe = (e['mean_tpot_ms']-r['mean_tpot_ms'])/r['mean_tpot_ms']*100
        tok = '✓' if abs(te)<=5 else ('~' if abs(te)<=6 else '✗')
        pok = '✓' if abs(pe)<=5 else ('~' if abs(pe)<=6 else '✗')
        print(f'{rate:>4} {r[\"mean_ttft_ms\"]:>10.1f} {e[\"mean_ttft_ms\"]:>10.1f} {te:>+8.1f}%{tok} {r[\"mean_tpot_ms\"]:>10.1f} {e[\"mean_tpot_ms\"]:>10.1f} {pe:>+8.1f}%{pok}')
"
echo ""
echo "This is the FIRST time the emulator is truly intercepting with"
echo "the serving profile. Previous <3% was real-vs-real (hooks failed)."
echo "DONE"
