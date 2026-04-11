#!/bin/bash
# Sync scheduler ablation: test with --no-async-scheduling
# Compare real vs emulator with async scheduling disabled
# This validates our claim about async scheduling complexity
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1
MODEL="Qwen/Qwen2.5-1.5B-Instruct"
PORT=8600
PROFILE="/workspace/eval_results/RTX-3060-12GB/profiles/sweep-1.5b-tp1-v14.json"

cleanup_gpu() {
    pkill -9 -f "python3.*api_server" 2>/dev/null || true
    fuser /dev/nvidia* 2>/dev/null | tr " " "\n" | sort -u | xargs kill -9 2>/dev/null || true
    sleep 10
}
wait_server() {
    for i in $(seq 1 120); do curl -s http://localhost:$PORT/health > /dev/null 2>&1 && return 0; sleep 1; done
    echo "ERROR: Server failed to start"; return 1
}

echo "=== Sync Scheduler Ablation ==="
echo "=== $(date) ==="

# Test 1: Real server with sync scheduling (--no-async-scheduling)
echo ""
echo "=== Real Server (sync scheduling) ==="
cleanup_gpu
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    --no-async-scheduling \
    > /workspace/sync_sched_real.log 2>&1 &
wait_server || exit 1

# CUDA graph sweep
for SWEEP_NP in 1 2 4 8 16 24 32 48 64 96 128 160 192 224 256; do
    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 1 --random-output-len 1 \
        --num-prompts $SWEEP_NP --request-rate inf > /dev/null 2>&1 || true
done

# Warmup
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 --request-rate 4 > /dev/null 2>&1

for RATE in 1 4 8; do
    NP=$((RATE * 100))
    if [ "$RATE" -eq 4 ]; then NP=400; fi
    if [ "$RATE" -eq 8 ]; then NP=800; fi

    echo ""
    echo "--- Real-Sync R=$RATE (NP=$NP) ---"
    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts $NP --request-rate $RATE \
        --percentile-metrics ttft,tpot,e2el 2>&1 | grep -E "Mean TTFT|Mean TPOT|Mean E2EL"
done
cleanup_gpu

# Test 2: Emulator with sync scheduling
# Note: emulator uses the same profile (profiled with async scheduling)
# This tests how well the emulator handles sync scheduling mode
for RATE in 1 4 8; do
    NP=$((RATE * 100))
    if [ "$RATE" -eq 4 ]; then NP=400; fi
    if [ "$RATE" -eq 8 ]; then NP=800; fi

    echo ""
    echo "--- Emu-Sync R=$RATE (NP=$NP) ---"
    cleanup_gpu
    VLLM_EMULATOR_ENABLE_ORACLE=1 \
    VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
    VLLM_EMULATOR_MODE=realtime \
    VLLM_EMULATOR_EXECUTOR_HOOK=1 \
    VLLM_EMULATOR_ORACLE_MODE=2d \
    VLLM_EMULATOR_PREP_SURROGATE=1 \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
        --no-async-scheduling \
        > /workspace/sync_sched_emu_r${RATE}.log 2>&1 &
    wait_server || { echo "FAILED to start emu server R=$RATE"; continue; }

    # CUDA graph sweep for emulator
    for SWEEP_NP in 1 2 4 8 16 24 32 48 64 96 128 160 192 224 256; do
        python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
            --dataset-name random --random-input-len 1 --random-output-len 1 \
            --num-prompts $SWEEP_NP --request-rate inf > /dev/null 2>&1 || true
    done

    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts $NP --request-rate $RATE \
        --percentile-metrics ttft,tpot,e2el 2>&1 | grep -E "Mean TTFT|Mean TPOT|Mean E2EL"

    cleanup_gpu
done

echo ""
echo "=== DONE $(date) ==="
