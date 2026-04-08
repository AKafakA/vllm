#!/bin/bash
# Test v6: offline profile integration + online sanity
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

echo "=== Step 1: Rebuild profile with offline section ==="
python3 /workspace/vllm-emulator-v18/vllm_emulator/profile/build_serving_profile.py \
    "${RESULT_DIR}/step_cycle_full.jsonl" \
    "${RESULT_DIR}/profiles/sweep-1.5b-tp1-v14.json" \
    "${RESULT_DIR}/profiles/serving-1.5b-tp1-v6.json" \
    "$MODEL" "RTX-3060-12GB"

PROFILE="${RESULT_DIR}/profiles/serving-1.5b-tp1-v6.json"

echo ""
echo "=== Step 2: Offline throughput test (200 prompts) ==="
pkill -9 -f python3 2>/dev/null || true
fuser /dev/nvidia* 2>/dev/null | tr " " "\n" | sort -u | xargs kill -9 2>/dev/null || true
sleep 5

echo "  Real GPU..."
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "$MODEL" --max-model-len 4096 --trust-remote-code \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 2>&1 | tee /workspace/v6_offline_real.txt | grep "Throughput:"

pkill -9 -f python3 2>/dev/null || true
fuser /dev/nvidia* 2>/dev/null | tr " " "\n" | sort -u | xargs kill -9 2>/dev/null || true
sleep 5

echo "  Emulator..."
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "$MODEL" --max-model-len 4096 --trust-remote-code \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 2>&1 | tee /workspace/v6_offline_emu.txt | grep "Throughput:"

pkill -9 -f python3 2>/dev/null || true
fuser /dev/nvidia* 2>/dev/null | tr " " "\n" | sort -u | xargs kill -9 2>/dev/null || true
sleep 5

echo ""
echo "--- Offline Comparison ---"
echo "  Real: $(grep 'Throughput:' /workspace/v6_offline_real.txt)"
echo "  Emu:  $(grep 'Throughput:' /workspace/v6_offline_emu.txt)"

echo ""
echo "=== Step 3: Online rate=2 sanity (200 prompts) ==="
# Start emu server for online test
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > /workspace/v6_online_emu.log 2>&1 &
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
    --save-result --result-dir "$ONLINE_DIR" --result-filename "v6_rate2_emu.json" > /dev/null 2>&1

pkill -9 -f python3 2>/dev/null || true
fuser /dev/nvidia* 2>/dev/null | tr " " "\n" | sort -u | xargs kill -9 2>/dev/null || true
sleep 3

echo "--- Online rate=2 results ---"
python3 "$SCRIPT_DIR/compare_results.py" "$ONLINE_DIR/fullv2_rate2_real.json" "$ONLINE_DIR/v6_rate2_emu.json"

echo ""
echo "=== SUMMARY ==="
echo "Offline:"
echo "  Real: $(grep 'Throughput:' /workspace/v6_offline_real.txt)"
echo "  Emu:  $(grep 'Throughput:' /workspace/v6_offline_emu.txt)"
echo ""
echo "Online rate=2:"
python3 "$SCRIPT_DIR/compare_results.py" "$ONLINE_DIR/fullv2_rate2_real.json" "$ONLINE_DIR/v6_rate2_emu.json" 2>/dev/null || echo "  MISSING"

echo ""
echo "DONE $(date)"
