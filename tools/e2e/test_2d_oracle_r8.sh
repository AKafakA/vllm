#!/bin/bash
# Test 2D oracle at rate=8 — compare step_cycle vs hybrid vs 2d
set -e
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=1

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"
PORT=8100
SCRIPT_DIR="/workspace/vllm-emulator-v18/tools/e2e"
ONLINE_DIR="${RESULT_DIR}/online"

cleanup_gpu() {
    pkill -9 -f "python3.*api_server" 2>/dev/null || true
    pkill -9 -f "python3.*bench" 2>/dev/null || true
    fuser /dev/nvidia* 2>/dev/null | tr " " "\n" | sort -u | xargs kill -9 2>/dev/null || true
    sleep 5
}

wait_server() {
    for i in $(seq 1 120); do
        if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then return 0; fi
        sleep 1
    done
    echo "ERROR: Server failed to start"; return 1
}

echo "============================================"
echo "=== 2D Oracle Test ==="
echo "=== $(date) ==="
echo "============================================"

# Step 1: Rebuild profile with 2D regression
echo "=== Rebuild profile with overhead_per_request_us ==="
python3 /workspace/vllm-emulator-v18/vllm_emulator/profile/build_serving_profile.py \
    "${RESULT_DIR}/step_cycle_fresh.jsonl" \
    "${RESULT_DIR}/profiles/sweep-1.5b-tp1-v14.json" \
    "${RESULT_DIR}/profiles/serving-1.5b-tp1-2d.json" \
    "$MODEL" "RTX-3060-12GB"

PROFILE="${RESULT_DIR}/profiles/serving-1.5b-tp1-2d.json"

# Check what regression computed
echo ""
grep "overhead_per_request" "$PROFILE" | head -1
echo ""

# Step 2: Fresh real baseline at rate=8 (200 prompts for speed)
cleanup_gpu
echo "=== Real rate=8 (200 prompts) ==="
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > /workspace/2d_real_8.log 2>&1 &
wait_server
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 --request-rate 4 > /dev/null 2>&1
sleep 3
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 --request-rate 8 --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,99 \
    --save-result --result-dir "$ONLINE_DIR" --result-filename "2d_rate8_real.json" > /dev/null 2>&1
cleanup_gpu

# Step 3: Test all 3 oracle modes at rate=8
for MODE in step_cycle hybrid 2d; do
    TAG="2d_${MODE}_rate8"
    echo ""
    echo "=== Rate=8 oracle=$MODE ==="
    cleanup_gpu
    VLLM_EMULATOR_ENABLE_ORACLE=1 \
    VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
    VLLM_EMULATOR_MODE=realtime \
    VLLM_EMULATOR_EXECUTOR_HOOK=1 \
    VLLM_EMULATOR_ORACLE_MODE=$MODE \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
        > /workspace/2d_emu_${MODE}_8.log 2>&1 &
    wait_server
    grep "ExecutorEmulatorHook" /workspace/2d_emu_${MODE}_8.log | head -1

    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 200 --request-rate 4 > /dev/null 2>&1
    sleep 3

    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 200 --request-rate 8 --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,99 \
        --save-result --result-dir "$ONLINE_DIR" --result-filename "${TAG}_emu.json" > /dev/null 2>&1

    cleanup_gpu
    echo "  --- $MODE ---"
    python3 "$SCRIPT_DIR/compare_results.py" "$ONLINE_DIR/2d_rate8_real.json" "$ONLINE_DIR/${TAG}_emu.json"
done

# Also test rate=4 to check no regression
for MODE in 2d; do
    TAG="2d_${MODE}_rate4"
    echo ""
    echo "=== Rate=4 oracle=$MODE ==="
    cleanup_gpu
    VLLM_EMULATOR_ENABLE_ORACLE=1 \
    VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
    VLLM_EMULATOR_MODE=realtime \
    VLLM_EMULATOR_EXECUTOR_HOOK=1 \
    VLLM_EMULATOR_ORACLE_MODE=$MODE \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
        > /workspace/2d_emu_${MODE}_4.log 2>&1 &
    wait_server
    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 200 --request-rate 4 > /dev/null 2>&1
    sleep 3
    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 200 --request-rate 4 --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,99 \
        --save-result --result-dir "$ONLINE_DIR" --result-filename "${TAG}_emu.json" > /dev/null 2>&1
    cleanup_gpu
    python3 "$SCRIPT_DIR/compare_results.py" "$ONLINE_DIR/fresh_rate4_real.json" "$ONLINE_DIR/${TAG}_emu.json" 2>/dev/null || echo "  No rate=4 baseline"
done

echo ""
echo "=== SUMMARY ==="
echo "Rate=8:"
for MODE in step_cycle hybrid 2d; do
    TAG="2d_${MODE}_rate8"
    python3 -c "
import json
try:
    e=json.load(open('$ONLINE_DIR/${TAG}_emu.json'))
    r=json.load(open('$ONLINE_DIR/2d_rate8_real.json'))
    tpot=(e['mean_tpot_ms']-r['mean_tpot_ms'])/r['mean_tpot_ms']*100
    e2e=(e.get('mean_e2el_ms',0)-r.get('mean_e2el_ms',0))/r.get('mean_e2el_ms',1)*100 if r.get('mean_e2el_ms') else 0
    print(f'  {\"$MODE\":>12}: TPOT={tpot:+.1f}% E2E={e2e:+.1f}%')
except: print(f'  {\"$MODE\":>12}: N/A')
" 2>/dev/null
done

echo ""
echo "=== DONE $(date) ==="
