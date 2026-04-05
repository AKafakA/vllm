#!/bin/bash
# Deep diagnosis of rate=1 TTFT: log every prefill step timing
set -e
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"
PROFILE="${RESULT_DIR}/profiles/serving-1.5b-tp1-2d.json"
PORT=8100

pkill -9 -f EngineCore 2>/dev/null || true; pkill -9 -f api_server 2>/dev/null || true; sleep 5

# Step 1: Real rate=1 with detailed step tracing
echo "=== Real rate=1 with step trace ==="
rm -f "${RESULT_DIR}/diag_r1_detailed.jsonl"
VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
VLLM_EMULATOR_STEP_TRACE_OUTPUT="${RESULT_DIR}/diag_r1_detailed.jsonl" \
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 4096 --port ${PORT} --trust-remote-code \
    > /workspace/diag_r1_detail_server.log 2>&1 &
for i in $(seq 1 120); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then break; fi; sleep 1
done
echo "  Running 10 prompts at rate=1..."
python3 -m vllm.entrypoints.cli.main bench serve --model "${MODEL}" --base-url http://localhost:${PORT} \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 10 --request-rate 1 --save-detailed \
    --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
    --save-result --result-dir "${RESULT_DIR}/online" --result-filename "diag_r1_real.json" 2>&1 | grep -E "TTFT|TPOT"
pkill -f api_server 2>/dev/null || true; sleep 2; pkill -9 -f EngineCore 2>/dev/null || true; sleep 5

# Step 2: Emulator rate=1 with same settings
echo ""
echo "=== Emu rate=1 ==="
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="${PROFILE}" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
VLLM_EMULATOR_PREFILL_OVERHEAD_US=5000 \
VLLM_EMULATOR_COLD_START_US=280000 \
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 4096 --port ${PORT} --trust-remote-code \
    > /workspace/diag_r1_emu_server.log 2>&1 &
for i in $(seq 1 120); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then break; fi; sleep 1
done
echo "  Running 10 prompts at rate=1..."
python3 -m vllm.entrypoints.cli.main bench serve --model "${MODEL}" --base-url http://localhost:${PORT} \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 10 --request-rate 1 --save-detailed \
    --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
    --save-result --result-dir "${RESULT_DIR}/online" --result-filename "diag_r1_emu.json" 2>&1 | grep -E "TTFT|TPOT"
pkill -f api_server 2>/dev/null || true; sleep 2; pkill -9 -f EngineCore 2>/dev/null || true

echo ""
echo "=== Per-request TTFT analysis ==="
python3 -c "
import json, os

# Check executor hook logging
print('Executor hook prefill step logs:')
for line in open('/workspace/diag_r1_emu_server.log'):
    if 'ExecutorHook' in line and 'step=' in line:
        print(f'  {line.strip().split(\"] \")[-1] if \"] \" in line else line.strip()}')
        # Only show first 15 steps
        if 'step=15' in line:
            break

print()

# Load step-cycle traces
print('Real prefill step latencies:')
prefills = []
for line in open('${RESULT_DIR}/diag_r1_detailed.jsonl'):
    r = json.loads(line)
    if 'total_tokens' in r and r.get('num_new_reqs', 0) > 0:
        prefills.append(r)
for i, r in enumerate(prefills[:10]):
    print(f'  Request {i+1}: tt={r[\"total_tokens\"]}, step_cycle={r[\"step_cycle_us\"]/1000:.1f}ms')

# What oracle would estimate for these
profile = json.load(open('${PROFILE}'))
pfill_map = {e['total_tokens']: e['latency_us'] for e in profile.get('prefill_forward_pass', [])}
comb_map = {e['total_tokens']: e['latency_us'] for e in profile['forward_pass']}
print()
print('Profile estimates for these tt values:')
for r in prefills[:5]:
    tt = r['total_tokens']
    pfill = pfill_map.get(tt, 0)
    comb = comb_map.get(tt, 0)
    actual = r['step_cycle_us']
    print(f'  tt={tt}: actual={actual/1000:.1f}ms, prefill_profile={pfill/1000:.1f}ms, combined={comb/1000:.1f}ms')
"
echo "DONE"
