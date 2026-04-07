#!/bin/bash
# Rebuild profile with outlier fix, run emu-only at rates 1,2,4
# Uses existing real baselines from fullv2_rate*_real.json
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

echo "=== Rebuild profile with outlier fix ==="
python3 /workspace/vllm-emulator-v18/paper/isca-2026-mlarchsys/scripts/temp/build_serving_profile_2d.py \
    "${RESULT_DIR}/step_cycle_full.jsonl" \
    "${RESULT_DIR}/profiles/sweep-1.5b-tp1-v14.json" \
    "${RESULT_DIR}/profiles/serving-1.5b-tp1-v3.json" \
    "$MODEL" "RTX-3060-12GB"

PROFILE="${RESULT_DIR}/profiles/serving-1.5b-tp1-v3.json"

# Verify the fix
python3 -c "
import json
p = json.load(open('${PROFILE}'))
pfp = {e['total_tokens']: e['latency_us'] for e in p.get('prefill_forward_pass', [])}
for tt in [256, 258, 260]:
    if tt in pfp: print(f'  prefill tt={tt}: {pfp[tt]/1000:.1f}ms')
"

pkill -9 -f python3 2>/dev/null || true
fuser /dev/nvidia* 2>/dev/null | xargs kill -9 2>/dev/null || true
sleep 5

for RATE in 1 2 4; do
    NP=200
    TAG="v3_rate${RATE}"
    echo ""
    echo "=== Emu rate=$RATE ($NP prompts) ==="
    pkill -9 -f python3 2>/dev/null || true
    fuser /dev/nvidia* 2>/dev/null | xargs kill -9 2>/dev/null || true
    sleep 5

    VLLM_EMULATOR_ENABLE_ORACLE=1 \
    VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
    VLLM_EMULATOR_MODE=realtime \
    VLLM_EMULATOR_EXECUTOR_HOOK=1 \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
        > /workspace/v3_emu_${RATE}.log 2>&1 &
    for i in $(seq 1 120); do if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then break; fi; sleep 1; done
    grep -m1 ExecutorEmulatorHook /workspace/v3_emu_${RATE}.log | strings || true

    # Heavy warmup (same as real baseline)
    echo "  Heavy warmup..."
    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 200 --request-rate 4 > /dev/null 2>&1
    sleep 3

    echo "  Benchmarking..."
    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts $NP --request-rate $RATE --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
        --save-result --result-dir "$ONLINE_DIR" --result-filename "${TAG}_emu.json" > /dev/null 2>&1

    pkill -9 -f python3 2>/dev/null || true
    fuser /dev/nvidia* 2>/dev/null | xargs kill -9 2>/dev/null || true
    sleep 3

    echo "  --- vs fresh real baseline ---"
    python3 "$SCRIPT_DIR/compare_results.py" "$ONLINE_DIR/fullv2_rate${RATE}_real.json" "$ONLINE_DIR/${TAG}_emu.json"
done
echo ""
echo "ALL DONE"
