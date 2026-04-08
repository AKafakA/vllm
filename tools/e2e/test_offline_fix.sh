#!/bin/bash
# Test offline fix: rebuild profile with single-sample inclusion, test offline
set -e
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub
export CUDA_VISIBLE_DEVICES=1

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"
PORT=8100
SCRIPT_DIR="/workspace/vllm-emulator-v18/tools/e2e"
ONLINE_DIR="${RESULT_DIR}/online"

echo "=== Step 1: Rebuild profile with single-sample fix ==="
python3 /workspace/vllm-emulator-v18/vllm_emulator/profile/build_serving_profile.py \
    "${RESULT_DIR}/step_cycle_full.jsonl" \
    "${RESULT_DIR}/profiles/sweep-1.5b-tp1-v14.json" \
    "${RESULT_DIR}/profiles/serving-1.5b-tp1-v4.json" \
    "$MODEL" "RTX-3060-12GB"

PROFILE="${RESULT_DIR}/profiles/serving-1.5b-tp1-v4.json"

echo ""
echo "=== Step 2: Check oracle at key tt values ==="
python3 -c "
import sys; sys.path.insert(0, '/workspace/vllm-emulator-v18')
import json
from vllm_emulator.oracle import create_oracle_from_profile_pack
p = json.load(open('${PROFILE}'))
oracle = create_oracle_from_profile_pack(p)
fp = {e['total_tokens']: e['latency_us'] for e in p['forward_pass']}
print('forward_pass entries around boundary:')
for tt, lat in sorted(fp.items()):
    if 260 <= tt <= 530:
        print(f'  tt={tt:>5}: {lat/1000:.1f}ms')
print()
print('Oracle predictions:')
for tt in [1, 50, 100, 150, 200, 256, 300, 400, 500, 600, 1000]:
    lat = oracle.estimate_step_latency_us(tt, has_prefill=False)
    print(f'  oracle(tt={tt:>4}, decode): {lat/1000:.1f}ms')
"

echo ""
echo "=== Step 3: Test offline emu ==="
pkill -9 -f python3 2>/dev/null || true
fuser /dev/nvidia* 2>/dev/null | tr " " "\n" | sort -u | xargs kill -9 2>/dev/null || true
sleep 5

VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > /workspace/v4_offline_emu.log 2>&1 &
for i in $(seq 1 120); do if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then break; fi; sleep 1; done
grep -m1 ExecutorEmulatorHook /workspace/v4_offline_emu.log | strings || true

echo "  Heavy warmup..."
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 --request-rate 4 > /dev/null 2>&1
sleep 3

echo "  Benchmarking offline (rate=inf, 200 prompts)..."
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 --request-rate inf --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
    --save-result --result-dir "$ONLINE_DIR" --result-filename "v4_rateinf_emu.json" > /dev/null 2>&1

pkill -9 -f python3 2>/dev/null || true
fuser /dev/nvidia* 2>/dev/null | tr " " "\n" | sort -u | xargs kill -9 2>/dev/null || true
sleep 3

echo ""
echo "=== Offline results (vs real baseline) ==="
python3 "$SCRIPT_DIR/compare_results.py" "$ONLINE_DIR/fullv2_rateinf_real.json" "$ONLINE_DIR/v4_rateinf_emu.json"

echo ""
echo "=== Step 4: Quick online sanity check (rate=2, 200 prompts) ==="
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > /workspace/v4_online_emu.log 2>&1 &
for i in $(seq 1 120); do if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then break; fi; sleep 1; done

echo "  Warmup..."
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 --request-rate 4 > /dev/null 2>&1
sleep 3

echo "  Benchmarking rate=2..."
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 --request-rate 2 --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
    --save-result --result-dir "$ONLINE_DIR" --result-filename "v4_rate2_emu.json" > /dev/null 2>&1

pkill -9 -f python3 2>/dev/null || true
fuser /dev/nvidia* 2>/dev/null | tr " " "\n" | sort -u | xargs kill -9 2>/dev/null || true
sleep 3

echo "--- Online rate=2 results ---"
python3 "$SCRIPT_DIR/compare_results.py" "$ONLINE_DIR/fullv2_rate2_real.json" "$ONLINE_DIR/v4_rate2_emu.json"

echo ""
echo "=== Step 5: Debug log — batch sizes during offline ==="
grep "step=" /workspace/v4_offline_emu.log | head -25
echo "..."
grep "step=" /workspace/v4_offline_emu.log | tail -10

echo ""
echo "DONE"
