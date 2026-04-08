#!/bin/bash
# Full profiling + evaluation: heavy warmup (200 prompts), more rates, independent tests
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

cleanup_gpu() {
    pkill -9 -f python3 2>/dev/null || true
    fuser /dev/nvidia* 2>/dev/null | xargs kill -9 2>/dev/null || true
    sleep 5
}

heavy_warmup() {
    # 200 prompts at rate=4 → ~50s sustained GPU load
    # Exercises all batch shapes AND reaches thermal equilibrium
    echo "  Heavy warmup (200 prompts at rate=4)..."
    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 200 --request-rate 4 > /dev/null 2>&1
    sleep 3
    echo "  Warmup done"
}

cleanup_gpu

echo "============================================"
echo "=== STEP 1: Comprehensive profiling ==="
echo "============================================"

rm -f "${RESULT_DIR}/step_cycle_full.jsonl"
VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
VLLM_EMULATOR_STEP_TRACE_OUTPUT="${RESULT_DIR}/step_cycle_full.jsonl" \
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > /workspace/full_profile_server.log 2>&1 &
for i in $(seq 1 120); do if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then break; fi; sleep 1; done
echo "Profiling server ready"

# Heavy warmup BEFORE profiling (reach thermal steady state)
heavy_warmup

# Profile at many rates (online + offline)
# Low rates use fewer prompts (arrival time dominates), high rates use more
for rate in 0.5 1 2 3 4 6 8 12; do
    NP=200
    if [ "$rate" = "0.5" ]; then NP=50; fi
    echo "  Profiling rate=$rate ($NP prompts)..."
    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts $NP --request-rate $rate > /dev/null 2>&1
done
# Offline (rate=inf) for high-concurrency coverage (tt=50-100+)
echo "  Profiling offline (200 prompts, rate=inf)..."
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 --request-rate inf > /dev/null 2>&1
cleanup_gpu

RECORDS=$(wc -l < "${RESULT_DIR}/step_cycle_full.jsonl")
echo "Total trace records: $RECORDS"

# Build profile
echo "Building profile..."
python3 /workspace/vllm-emulator-v18/vllm_emulator/profile/build_serving_profile.py \
    "${RESULT_DIR}/step_cycle_full.jsonl" \
    "${RESULT_DIR}/profiles/sweep-1.5b-tp1-v14.json" \
    "${RESULT_DIR}/profiles/serving-1.5b-tp1-full-v2.json" \
    "$MODEL" "RTX-3060-12GB"

PROFILE="${RESULT_DIR}/profiles/serving-1.5b-tp1-full-v2.json"

echo ""
echo "============================================"
echo "=== STEP 2: Independent rate evaluation ==="
echo "============================================"

for RATE in 1 2 4 inf; do
    NP=200
    TAG="fullv2_rate${RATE}"
    echo ""
    echo "--- Rate=$RATE ($NP prompts, 200-prompt warmup) ---"

    # Real (independent, fresh start)
    cleanup_gpu
    echo "  Real server..."
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
        > /workspace/fullv2_real_${RATE}.log 2>&1 &
    for i in $(seq 1 120); do if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then break; fi; sleep 1; done
    heavy_warmup
    echo "  Benchmarking real..."
    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts $NP --request-rate $RATE --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
        --save-result --result-dir "$ONLINE_DIR" --result-filename "${TAG}_real.json" > /dev/null 2>&1

    # Emu (independent, fresh start)
    cleanup_gpu
    echo "  Emu server..."
    VLLM_EMULATOR_ENABLE_ORACLE=1 \
    VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
    VLLM_EMULATOR_MODE=realtime \
    VLLM_EMULATOR_EXECUTOR_HOOK=1 \
    VLLM_EMULATOR_DISABLE_DEFER_ADD=1 \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
        > /workspace/fullv2_emu_${RATE}.log 2>&1 &
    for i in $(seq 1 120); do if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then break; fi; sleep 1; done
    grep -m1 ExecutorEmulatorHook /workspace/fullv2_emu_${RATE}.log | strings || true
    heavy_warmup
    echo "  Benchmarking emu..."
    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts $NP --request-rate $RATE --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
        --save-result --result-dir "$ONLINE_DIR" --result-filename "${TAG}_emu.json" > /dev/null 2>&1

    cleanup_gpu
    echo "  Results:"
    python3 "$SCRIPT_DIR/compare_results.py" "$ONLINE_DIR/${TAG}_real.json" "$ONLINE_DIR/${TAG}_emu.json"
done

echo ""
echo "============================================"
echo "=== SUMMARY ==="
echo "============================================"
for RATE in 1 2 4 inf; do
    TAG="fullv2_rate${RATE}"
    echo "--- Rate=$RATE ---"
    python3 "$SCRIPT_DIR/compare_results.py" "$ONLINE_DIR/${TAG}_real.json" "$ONLINE_DIR/${TAG}_emu.json" 2>/dev/null || echo "  MISSING"
done
echo "ALL DONE"
