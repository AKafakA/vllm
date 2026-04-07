#!/bin/bash
# B2B accuracy test with proper CUDA graph warmup
set -e
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"
PROFILE="${RESULT_DIR}/profiles/serving-1.5b-tp1-full.json"
PORT=8100
BENCH_SCRIPT="/workspace/vllm-emulator-v18/tools/e2e/warmup_and_bench.py"
ONLINE_DIR="${RESULT_DIR}/online"
NUM_PROMPTS=100

pkill -9 -f EngineCore 2>/dev/null || true; pkill -9 -f api_server 2>/dev/null || true; sleep 5

# === REAL SERVER ===
echo "=== Real Server ==="
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 4096 --port ${PORT} --trust-remote-code \
    > /workspace/b2b_warmup_real.log 2>&1 &

python3 "${BENCH_SCRIPT}" --port ${PORT} --model "${MODEL}" \
    --result-dir "${ONLINE_DIR}" --prefix warmup_real \
    --rates 1,2,4 --num-prompts ${NUM_PROMPTS}

pkill -f api_server 2>/dev/null || true; sleep 2; pkill -9 -f EngineCore 2>/dev/null || true; sleep 5

# === EMULATOR SERVER ===
echo ""
echo "=== Emulator Server ==="
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="${PROFILE}" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 4096 --port ${PORT} --trust-remote-code \
    > /workspace/b2b_warmup_emu.log 2>&1 &

python3 "${BENCH_SCRIPT}" --port ${PORT} --model "${MODEL}" \
    --result-dir "${ONLINE_DIR}" --prefix warmup_emu \
    --rates 1,2,4 --num-prompts ${NUM_PROMPTS}

pkill -f api_server 2>/dev/null || true; sleep 2; pkill -9 -f EngineCore 2>/dev/null || true

# === COMPARE RESULTS ===
echo ""
echo "=== RESULTS ==="
python3 << 'PYEOF'
import json, os
rd = "/workspace/eval_results/RTX-3060-12GB/online"
hdr = "%4s %10s %10s %9s %10s %10s %9s" % ("Rate", "Real_TTFT", "Emu_TTFT", "TTFT_err", "Real_TPOT", "Emu_TPOT", "TPOT_err")
print(hdr)
for rate in [1, 2, 4]:
    rf = "%s/warmup_real_rate%d.json" % (rd, rate)
    ef = "%s/warmup_emu_rate%d.json" % (rd, rate)
    if os.path.exists(rf) and os.path.exists(ef):
        r = json.load(open(rf))
        e = json.load(open(ef))
        te = (e["mean_ttft_ms"]-r["mean_ttft_ms"])/r["mean_ttft_ms"]*100
        pe = (e["mean_tpot_ms"]-r["mean_tpot_ms"])/r["mean_tpot_ms"]*100
        tok = "pass" if abs(te)<=5 else ("~" if abs(te)<=6 else "FAIL")
        pok = "pass" if abs(pe)<=5 else ("~" if abs(pe)<=6 else "FAIL")
        print("%4d %10.1f %10.1f %+8.1f%%%s %10.1f %10.1f %+8.1f%%%s" % (
            rate, r["mean_ttft_ms"], e["mean_ttft_ms"], te, tok,
            r["mean_tpot_ms"], e["mean_tpot_ms"], pe, pok))
PYEOF
echo "DONE"
