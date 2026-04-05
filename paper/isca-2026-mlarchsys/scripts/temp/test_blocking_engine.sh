#!/bin/bash
# Test executor hook with engine-core-level blocking
# Uses same-session profile for accurate comparison
set -e
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"
PROFILE="${RESULT_DIR}/profiles/serving-1.5b-tp1-same.json"
PORT=8100

pkill -9 -f EngineCore 2>/dev/null || true; pkill -9 -f api_server 2>/dev/null || true; sleep 5

# Real baseline (reuse same_real from previous test)
echo "=== Emulator (executor hook + engine blocking) ==="

VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="${PROFILE}" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 4096 --port ${PORT} --trust-remote-code \
    > /workspace/blocking_engine_server.log 2>&1 &

for i in $(seq 1 120); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then echo "  Ready ${i}s"; break; fi; sleep 1
done

grep ExecutorEmulatorHook /workspace/blocking_engine_server.log | head -1

for rate in 1 2 4; do
    echo "  Rate=${rate}..."
    python3 -m vllm.entrypoints.cli.main bench serve --model "${MODEL}" --base-url http://localhost:${PORT} \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 50 --request-rate ${rate} --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
        --save-result --result-dir "${RESULT_DIR}/online" --result-filename "blocking_eng_rate${rate}.json" 2>&1 | grep -E "TTFT|TPOT"
done

pkill -f api_server 2>/dev/null || true; sleep 2; pkill -9 -f EngineCore 2>/dev/null || true

echo ""
echo "=== RESULTS (vs same-session baseline) ==="
python3 -c "
import json, os
rd = '${RESULT_DIR}/online'
print(f'{\"Rate\":>4} {\"Real_TTFT\":>10} {\"Emu_TTFT\":>10} {\"TTFT_err\":>9} {\"Real_TPOT\":>10} {\"Emu_TPOT\":>10} {\"TPOT_err\":>9}')
for rate in [1, 2, 4]:
    rf = f'{rd}/same_real_rate{rate}.json'
    ef = f'{rd}/blocking_eng_rate{rate}.json'
    if os.path.exists(rf) and os.path.exists(ef):
        r, e = json.load(open(rf)), json.load(open(ef))
        te = (e['mean_ttft_ms']-r['mean_ttft_ms'])/r['mean_ttft_ms']*100
        pe = (e['mean_tpot_ms']-r['mean_tpot_ms'])/r['mean_tpot_ms']*100
        tok = '✓' if abs(te)<=5 else ('~' if abs(te)<=6 else '✗')
        pok = '✓' if abs(pe)<=5 else ('~' if abs(pe)<=6 else '✗')
        print(f'{rate:>4} {r[\"mean_ttft_ms\"]:>10.1f} {e[\"mean_ttft_ms\"]:>10.1f} {te:>+8.1f}%{tok} {r[\"mean_tpot_ms\"]:>10.1f} {e[\"mean_tpot_ms\"]:>10.1f} {pe:>+8.1f}%{pok}')
"
echo "DONE"
