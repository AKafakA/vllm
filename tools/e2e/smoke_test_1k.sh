#!/bin/bash
# 1000-prompt A2A test at rates 1,2,4 + offline
# Uses measured IPC overhead from profile, CUDA graph warmup
set -e
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"
PORT=8100
ONLINE_DIR="${RESULT_DIR}/online"
SCRIPT_DIR="/workspace/vllm-emulator-v18/tools/e2e"

# Rebuild calibrated profile with IPC overhead table
echo "=== Rebuild profile ==="
python3 /workspace/vllm-emulator-v18/paper/isca-2026-mlarchsys/scripts/temp/build_serving_profile_2d.py \
    "${RESULT_DIR}/step_cycle_1.5b_full.jsonl" \
    "${RESULT_DIR}/profiles/sweep-1.5b-tp1-v14.json" \
    "${RESULT_DIR}/profiles/serving-1.5b-tp1-calibrated.json" \
    "${MODEL}" "RTX-3060-12GB"
PROFILE="${RESULT_DIR}/profiles/serving-1.5b-tp1-calibrated.json"

thorough_warmup() {
    local rate=$1
    for i in $(seq 1 10); do
        curl -s --max-time 10 http://localhost:$PORT/v1/completions \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"$MODEL\",\"prompt\":\"warmup $i\",\"max_tokens\":5,\"temperature\":0}" > /dev/null
        sleep 0.2
    done
    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 50 --request-rate $rate > /dev/null 2>&1
    sleep 3
    echo "  Warmup done"
}

kill_servers() {
    pkill -f api_server 2>/dev/null || true; sleep 3
    pkill -9 -f EngineCore 2>/dev/null || true; sleep 5
}

run_test() {
    local rate=$1
    local num_prompts=$2
    local tag=$3

    kill_servers

    # Real
    echo "  Real server..."
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
        > /workspace/1k_real_${tag}.log 2>&1 &
    for i in $(seq 1 120); do if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then break; fi; sleep 1; done
    thorough_warmup $rate
    echo "  Benchmarking real (${num_prompts} prompts, rate=${rate})..."
    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts $num_prompts --request-rate $rate \
        --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
        --save-result --result-dir "$ONLINE_DIR" --result-filename "1k_real_${tag}.json" > /dev/null 2>&1

    kill_servers

    # Emu
    echo "  Emu server..."
    VLLM_EMULATOR_ENABLE_ORACLE=1 \
    VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
    VLLM_EMULATOR_MODE=realtime \
    VLLM_EMULATOR_EXECUTOR_HOOK=1 \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
        > /workspace/1k_emu_${tag}.log 2>&1 &
    for i in $(seq 1 120); do if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then break; fi; sleep 1; done
    grep -m1 ExecutorEmulatorHook /workspace/1k_emu_${tag}.log || true
    thorough_warmup $rate
    echo "  Benchmarking emu (${num_prompts} prompts, rate=${rate})..."
    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts $num_prompts --request-rate $rate \
        --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
        --save-result --result-dir "$ONLINE_DIR" --result-filename "1k_emu_${tag}.json" > /dev/null 2>&1

    echo "  --- ${tag} Results ---"
    python3 "$SCRIPT_DIR/compare_results.py" \
        "$ONLINE_DIR/1k_real_${tag}.json" "$ONLINE_DIR/1k_emu_${tag}.json"
}

# === Tests ===
echo ""
echo "============================================"
echo "=== 1000-prompt rate=1 ==="
echo "============================================"
run_test 1 1000 "rate1"

echo ""
echo "============================================"
echo "=== 1000-prompt rate=2 ==="
echo "============================================"
run_test 2 1000 "rate2"

echo ""
echo "============================================"
echo "=== 1000-prompt rate=4 ==="
echo "============================================"
run_test 4 1000 "rate4"

echo ""
echo "============================================"
echo "=== Offline (200 prompts) ==="
echo "============================================"
run_test inf 200 "offline"

kill_servers
echo ""
echo "ALL 1K TESTS DONE"
