#!/bin/bash
# Full profiling with worker subfunction timing
# Captures: step_cycle trace (engine) + worker timing trace (worker)
# Then builds profile including cpu_prep_ms data
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1
MODEL="Qwen/Qwen2.5-1.5B-Instruct"
PORT=8300
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"
STEP_TRACE="${RESULT_DIR}/step_cycle_subfunc.jsonl"
WORKER_TRACE="${RESULT_DIR}/worker_timing.jsonl"
PROFILE="${RESULT_DIR}/profiles/sweep-1.5b-tp1-v14.json"

cleanup_gpu() {
    pkill -9 -f "python3.*api_server" 2>/dev/null || true
    fuser /dev/nvidia* 2>/dev/null | tr " " "\n" | sort -u | xargs kill -9 2>/dev/null || true
    sleep 10
}
wait_server() {
    for i in $(seq 1 120); do curl -s http://localhost:$PORT/health > /dev/null 2>&1 && return 0; sleep 1; done
    echo "ERROR: Server failed to start"; return 1
}

echo "=== Profiling with Worker Subfunctions ==="
echo "=== $(date) ==="

rm -f "$STEP_TRACE" "$WORKER_TRACE"

# 5 rounds of adaptive profiling
for ROUND in 1 2 3 4 5; do
    echo ""
    echo "=== Round $ROUND/5 ==="
    cleanup_gpu

    # Start server with tracing enabled
    VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
    VLLM_EMULATOR_STEP_TRACE_OUTPUT="$STEP_TRACE" \
    VLLM_EMULATOR_WORKER_TIMING_TRACE="$WORKER_TRACE" \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
        > /workspace/subfunc_srv_r${ROUND}.log 2>&1 &
    wait_server || { echo "FAILED to start server round $ROUND"; continue; }

    # CUDA graph sweep warmup
    for SWEEP_NP in 1 2 4 8 16 24 32 48 64 96 128 160 192 224 256; do
        python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
            --dataset-name random --random-input-len 1 --random-output-len 1 \
            --num-prompts $SWEEP_NP --request-rate inf > /dev/null 2>&1 || true
    done

    # Write profiling start marker
    echo '{"__marker__": "profiling_start"}' >> "$STEP_TRACE"

    # Profile at various rates
    for RATE in 1 4 8; do
        NP=$((RATE * 100))
        if [ "$RATE" -eq 1 ]; then NP=100; fi
        if [ "$RATE" -eq 4 ]; then NP=300; fi
        if [ "$RATE" -eq 8 ]; then NP=800; fi

        echo "  Round $ROUND: R=$RATE NP=$NP"
        python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
            --dataset-name random --random-input-len 256 --random-output-len 128 \
            --num-prompts $NP --request-rate $RATE \
            --percentile-metrics ttft,tpot,e2el > /dev/null 2>&1
    done

    cleanup_gpu
done

STEP_LINES=$(wc -l < "$STEP_TRACE" 2>/dev/null)
WORKER_LINES=$(wc -l < "$WORKER_TRACE" 2>/dev/null)
echo ""
echo "Step-cycle trace: $STEP_LINES records"
echo "Worker timing trace: $WORKER_LINES records"

echo ""
echo "=== Worker timing summary ==="
# Quick summary using grep/awk
echo "Average cpu_prep_ms by step range:"
head -5 "$WORKER_TRACE" 2>/dev/null
echo "..."
echo "Total records: $WORKER_LINES"

# Compute average cpu_prep_ms
grep -oP '"cpu_prep_ms":\s*[0-9.]+' "$WORKER_TRACE" 2>/dev/null | grep -oP '[0-9.]+' | awk '
{sum+=$1; n++}
END {if(n>0) printf "Average cpu_prep_ms: %.3fms (n=%d)\n", sum/n, n}
'

echo ""
echo "=== DONE $(date) ==="
