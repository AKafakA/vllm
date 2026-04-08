#!/bin/bash
# A2A test with BlockingFuture approach — rates 1,2,4
# Adequate timeouts for real-time speed emulation
set -e
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"
PROFILE="${RESULT_DIR}/profiles/serving-1.5b-tp1-calibrated.json"
PORT=8100
ONLINE_DIR="${RESULT_DIR}/online"
SCRIPT_DIR="/workspace/vllm-emulator-v18/tools/e2e"

# Rebuild profile
python3 /workspace/vllm-emulator-v18/vllm_emulator/profile/build_serving_profile.py \
    "${RESULT_DIR}/step_cycle_1.5b_full.jsonl" \
    "${RESULT_DIR}/profiles/sweep-1.5b-tp1-v14.json" \
    "$PROFILE" "$MODEL" "RTX-3060-12GB" > /dev/null 2>&1

warmup() {
    local rate=$1
    for i in $(seq 1 5); do
        curl -s --max-time 10 http://localhost:$PORT/v1/completions \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"$MODEL\",\"prompt\":\"warmup $i\",\"max_tokens\":5,\"temperature\":0}" > /dev/null
        sleep 0.2
    done
    # Lighter warmup: 10 prompts (not 30) to avoid long blocking wait
    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 10 --request-rate $rate > /dev/null 2>&1
    sleep 3
    echo "  Warmup done"
}

run_rate() {
    local rate=$1
    local num_prompts=50
    local tag="bf_rate${rate}"

    echo ""
    echo "============================================"
    echo "=== Rate=$rate ($num_prompts prompts) ==="
    echo "============================================"

    # Real
    pkill -9 -f EngineCore 2>/dev/null || true; pkill -9 -f api_server 2>/dev/null || true; sleep 5
    echo "  Real server..."
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
        > /workspace/bf_real_${rate}.log 2>&1 &
    for i in $(seq 1 120); do if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then break; fi; sleep 1; done
    warmup $rate
    echo "  Benchmarking real..."
    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts $num_prompts --request-rate $rate \
        --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
        --save-result --result-dir "$ONLINE_DIR" --result-filename "${tag}_real.json" > /dev/null 2>&1

    # Emu
    pkill -9 -f EngineCore 2>/dev/null || true; pkill -9 -f api_server 2>/dev/null || true; sleep 5
    echo "  Emu server..."
    VLLM_EMULATOR_ENABLE_ORACLE=1 \
    VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
    VLLM_EMULATOR_MODE=realtime \
    VLLM_EMULATOR_EXECUTOR_HOOK=1 \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
        > /workspace/bf_emu_${rate}.log 2>&1 &
    for i in $(seq 1 120); do if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then break; fi; sleep 1; done
    grep -m1 ExecutorEmulatorHook /workspace/bf_emu_${rate}.log || true
    warmup $rate
    echo "  Benchmarking emu..."
    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts $num_prompts --request-rate $rate \
        --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
        --save-result --result-dir "$ONLINE_DIR" --result-filename "${tag}_emu.json" > /dev/null 2>&1

    pkill -9 -f EngineCore 2>/dev/null || true; pkill -9 -f api_server 2>/dev/null || true

    echo "  --- Rate=$rate Results ---"
    python3 "$SCRIPT_DIR/compare_results.py" \
        "$ONLINE_DIR/${tag}_real.json" "$ONLINE_DIR/${tag}_emu.json"
}

echo "=== BlockingFuture A2A Test ==="
run_rate 1
run_rate 2
run_rate 4

echo ""
echo "ALL DONE"
