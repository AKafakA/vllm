#!/bin/bash
# Run TTFT trace on both real and emulator servers
set -e
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"
PROFILE="${RESULT_DIR}/profiles/serving-1.5b-tp1-2d.json"
PORT=8100

pip install aiohttp > /dev/null 2>&1

pkill -9 -f EngineCore 2>/dev/null || true; pkill -9 -f api_server 2>/dev/null || true; sleep 5

echo "=== Real server TTFT trace ==="
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 4096 --port ${PORT} --trust-remote-code \
    > /workspace/ttft_trace_real_server.log 2>&1 &
for i in $(seq 1 120); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then echo "  Ready"; break; fi; sleep 1
done
python3 /workspace/vllm-emulator-v18/paper/isca-2026-mlarchsys/scripts/temp/trace_ttft_breakdown.py
cp /tmp/ttft_trace_results.json /workspace/ttft_trace_real.json
pkill -f api_server 2>/dev/null || true; sleep 2; pkill -9 -f EngineCore 2>/dev/null || true; sleep 5

echo ""
echo "=== Emulator server TTFT trace ==="
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="${PROFILE}" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
VLLM_EMULATOR_PREFILL_OVERHEAD_US=5000 \
VLLM_EMULATOR_COLD_START_US=280000 \
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 4096 --port ${PORT} --trust-remote-code \
    > /workspace/ttft_trace_emu_server.log 2>&1 &
for i in $(seq 1 120); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then echo "  Ready"; break; fi; sleep 1
done
python3 /workspace/vllm-emulator-v18/paper/isca-2026-mlarchsys/scripts/temp/trace_ttft_breakdown.py
cp /tmp/ttft_trace_results.json /workspace/ttft_trace_emu.json
pkill -f api_server 2>/dev/null || true; sleep 2; pkill -9 -f EngineCore 2>/dev/null || true

echo ""
echo "=== Comparison ==="
python3 -c "
import json
real = json.load(open('/workspace/ttft_trace_real.json'))
emu = json.load(open('/workspace/ttft_trace_emu.json'))

print(f'{\"Req\":>4} {\"Real_TTFT\":>10} {\"Emu_TTFT\":>10} {\"Gap\":>8} {\"Real_Total\":>11} {\"Emu_Total\":>11}')
for i in range(min(len(real), len(emu))):
    r, e = real[i], emu[i]
    gap = r['ttft_ms'] - e['ttft_ms']
    print(f'{i+1:>4} {r[\"ttft_ms\"]:>10.1f} {e[\"ttft_ms\"]:>10.1f} {gap:>+7.1f} {r[\"total_ms\"]:>11.1f} {e[\"total_ms\"]:>11.1f}')
"
echo "DONE"
