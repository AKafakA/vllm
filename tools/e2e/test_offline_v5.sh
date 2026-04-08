#!/bin/bash
# Test offline fix v5: decode section gets high-tt entries + adaptive blocking
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

echo "=== Step 1: Rebuild profile v5 ==="
python3 /workspace/vllm-emulator-v18/vllm_emulator/profile/build_serving_profile.py \
    "${RESULT_DIR}/step_cycle_full.jsonl" \
    "${RESULT_DIR}/profiles/sweep-1.5b-tp1-v14.json" \
    "${RESULT_DIR}/profiles/serving-1.5b-tp1-v5.json" \
    "$MODEL" "RTX-3060-12GB"

PROFILE="${RESULT_DIR}/profiles/serving-1.5b-tp1-v5.json"

echo ""
echo "=== Step 2: Oracle predictions ==="
python3 << 'PYEOF'
import sys; sys.path.insert(0, '/workspace/vllm-emulator-v18')
import json
from vllm_emulator.oracle import create_oracle_from_profile_pack
p = json.load(open("/workspace/eval_results/RTX-3060-12GB/profiles/serving-1.5b-tp1-v5.json"))
oracle = create_oracle_from_profile_pack(p)
print("Decode oracle predictions:")
for tt in [1, 50, 100, 150, 200, 256, 300, 400, 500]:
    lat = oracle.estimate_step_latency_us(tt, has_prefill=False)
    print(f"  decode(tt={tt:>4}): {lat/1000:.1f}ms")
print("\nPrefill oracle predictions:")
for tt in [256, 260, 270, 300, 500]:
    lat = oracle.estimate_step_latency_us(tt, has_prefill=True)
    print(f"  prefill(tt={tt:>4}): {lat/1000:.1f}ms")
print(f"\nTarget: decode(tt=200) should be ~49ms (real GPU offline TPOT)")

# Show decode section coverage
dec = p.get("decode_forward_pass", [])
print(f"\nDecode section: {len(dec)} entries, max_tt={max(e['total_tokens'] for e in dec) if dec else 0}")
PYEOF

echo ""
echo "=== Step 3: Offline emu test ==="
pkill -9 -f "python3.*api_server" 2>/dev/null || true
fuser /dev/nvidia* 2>/dev/null | tr " " "\n" | sort -u | xargs kill -9 2>/dev/null || true
sleep 5

VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > /workspace/v5_offline_emu.log 2>&1 &
for i in $(seq 1 120); do if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then break; fi; sleep 1; done

echo "  Heavy warmup..."
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 --request-rate 4 > /dev/null 2>&1
sleep 3

echo "  Benchmarking offline..."
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 --request-rate inf --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
    --save-result --result-dir "$ONLINE_DIR" --result-filename "v5_rateinf_emu.json" > /dev/null 2>&1

pkill -9 -f "python3.*api_server" 2>/dev/null || true
fuser /dev/nvidia* 2>/dev/null | tr " " "\n" | sort -u | xargs kill -9 2>/dev/null || true
sleep 3

echo "--- Offline results ---"
python3 "$SCRIPT_DIR/compare_results.py" "$ONLINE_DIR/fullv2_rateinf_real.json" "$ONLINE_DIR/v5_rateinf_emu.json"

echo ""
echo "=== Step 4: Online rate=1 sanity ==="
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > /workspace/v5_rate1_emu.log 2>&1 &
for i in $(seq 1 120); do if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then break; fi; sleep 1; done

echo "  Warmup..."
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 --request-rate 4 > /dev/null 2>&1
sleep 3

echo "  Benchmarking rate=1..."
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 --request-rate 1 --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
    --save-result --result-dir "$ONLINE_DIR" --result-filename "v5_rate1_emu.json" > /dev/null 2>&1

pkill -9 -f "python3.*api_server" 2>/dev/null || true
fuser /dev/nvidia* 2>/dev/null | tr " " "\n" | sort -u | xargs kill -9 2>/dev/null || true
sleep 3

echo "--- Online rate=1 results ---"
python3 "$SCRIPT_DIR/compare_results.py" "$ONLINE_DIR/fullv2_rate1_real.json" "$ONLINE_DIR/v5_rate1_emu.json"

echo ""
echo "=== Step 5: Debug batch sizes ==="
echo "Offline steps:"
grep "step=" /workspace/v5_offline_emu.log | head -5
echo "..."
grep "step=" /workspace/v5_offline_emu.log | grep -E "reqs=(20|50|100|200)" | head -10

echo ""
echo "DONE $(date)"
