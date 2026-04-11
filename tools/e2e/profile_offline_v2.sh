#!/bin/bash
# Separate offline profiling v2: with CUDA graph warmup + more rounds
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1
MODEL="Qwen/Qwen2.5-1.5B-Instruct"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"
TRACE="${RESULT_DIR}/step_cycle_offline.jsonl"
PROFILE="${RESULT_DIR}/profiles/serving-1.5b-tp1-offline.json"
PROFILE_BUILDER="/workspace/vllm-emulator-v18/vllm_emulator/profile/build_serving_profile_filtered.py"

cleanup_gpu() {
    pkill -9 -f python3 2>/dev/null
    fuser /dev/nvidia* 2>/dev/null | tr " " "\n" | sort -u | xargs kill -9 2>/dev/null
    sleep 10
}

echo "=== Offline Profiling v2 ==="
echo "=== $(date) ==="
rm -f "$TRACE"

# 5 rounds for stability, various batch sizes
# Each round: warmup with small batch first (compiles CUDA graphs),
# then profile at various sizes
for ROUND in 1 2 3 4 5; do
    echo ""
    echo "=== Round $ROUND/5 ==="

    # Warmup: small batch to compile CUDA graphs
    cleanup_gpu
    python3 -m vllm.entrypoints.cli.main bench throughput \
        --model "$MODEL" --max-model-len 4096 --trust-remote-code \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 10 > /dev/null 2>&1

    # Profile at many batch sizes for dense concurrency coverage
    # Need intermediate sizes (50-500) to fill conc=10-200 range
    for NP in 20 50 100 150 200 300 400 500 700 1000; do
        cleanup_gpu
        echo '{"__marker__": "profiling_start"}' >> "$TRACE"
        VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
        VLLM_EMULATOR_STEP_TRACE_OUTPUT="$TRACE" \
        python3 -m vllm.entrypoints.cli.main bench throughput \
            --model "$MODEL" --max-model-len 4096 --trust-remote-code \
            --dataset-name random --random-input-len 256 --random-output-len 128 \
            --num-prompts $NP 2>&1 | grep "Throughput:"
    done
done

TOTAL=$(wc -l < "$TRACE" 2>/dev/null)
echo ""
echo "Total offline trace records: $TOTAL"

echo ""
echo "=== Building offline profile ==="
python3 "$PROFILE_BUILDER" \
    "$TRACE" \
    "${RESULT_DIR}/profiles/sweep-1.5b-tp1-v14.json" \
    "$PROFILE" \
    "$MODEL" "RTX-3060-12GB"

echo ""
echo "=== Testing offline ==="
for NP in 500 2000; do
    echo ""
    echo "--- Real $NP ---"
    cleanup_gpu
    python3 -m vllm.entrypoints.cli.main bench throughput \
        --model "$MODEL" --max-model-len 4096 --trust-remote-code \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts $NP 2>&1 | grep "Throughput:"

    echo "--- Emu $NP ---"
    cleanup_gpu
    VLLM_EMULATOR_ENABLE_ORACLE=1 \
    VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
    VLLM_EMULATOR_MODE=realtime \
    VLLM_EMULATOR_EXECUTOR_HOOK=1 \
    VLLM_EMULATOR_ORACLE_MODE=2d \
    VLLM_EMULATOR_PREP_SURROGATE=1 \
    python3 -m vllm.entrypoints.cli.main bench throughput \
        --model "$MODEL" --max-model-len 4096 --trust-remote-code \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts $NP 2>&1 | grep "Throughput:"
done

echo ""
echo "=== DONE $(date) ==="
