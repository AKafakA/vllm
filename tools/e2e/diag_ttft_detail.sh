#!/bin/bash
# TTFT detailed diagnosis: compare prefill step timing real vs emu at R=4,8
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1
MODEL="Qwen/Qwen2.5-1.5B-Instruct"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"
PORT=8100

cleanup_gpu() {
    pkill -9 -f "python3.*api_server" 2>/dev/null || true
    fuser /dev/nvidia* 2>/dev/null | tr " " "\n" | sort -u | xargs kill -9 2>/dev/null || true
    sleep 10
}
wait_server() {
    for i in $(seq 1 120); do curl -s http://localhost:$PORT/health > /dev/null 2>&1 && return 0; sleep 1; done
    echo "ERROR"; return 1
}

for MODE in real emu; do
    TRACE="${RESULT_DIR}/diag_ttft_${MODE}.jsonl"
    echo "=== $MODE ==="
    cleanup_gpu
    rm -f "$TRACE"
    if [ "$MODE" = "emu" ]; then
        VLLM_EMULATOR_ENABLE_ORACLE=1 \
        VLLM_EMULATOR_PROFILE_PACK="${RESULT_DIR}/profiles/serving-1.5b-tp1-fresh.json" \
        VLLM_EMULATOR_MODE=realtime VLLM_EMULATOR_EXECUTOR_HOOK=1 \
        VLLM_EMULATOR_ORACLE_MODE=2d VLLM_EMULATOR_TIMER_MODE=chain \
        VLLM_EMULATOR_CHAIN_CAP=0 VLLM_EMULATOR_OUTPUT_OVERHEAD=0 \
        VLLM_EMULATOR_PREP_SURROGATE=1 \
        VLLM_EMULATOR_TRACE_STEP_CYCLE=1 VLLM_EMULATOR_STEP_TRACE_OUTPUT="$TRACE" \
        python3 -m vllm.entrypoints.openai.api_server \
            --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
            > /workspace/diag_ttft_${MODE}_srv.log 2>&1 &
    else
        VLLM_EMULATOR_TRACE_STEP_CYCLE=1 VLLM_EMULATOR_STEP_TRACE_OUTPUT="$TRACE" \
        python3 -m vllm.entrypoints.openai.api_server \
            --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
            > /workspace/diag_ttft_${MODE}_srv.log 2>&1 &
    fi
    wait_server || continue
    if [ "$MODE" = "real" ]; then
        for SWEEP_NP in 1 2 4 8 16 24 32 48 64 96 128 160 192 224 256; do
            python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
                --dataset-name random --random-input-len 1 --random-output-len 1 \
                --num-prompts $SWEEP_NP --request-rate inf > /dev/null 2>&1 || true
        done
    fi
    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 200 --request-rate 4 > /dev/null 2>&1
    echo '{"__marker__": "benchmark_start"}' >> "$TRACE"
    for RATE in 4 8; do
        python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
            --dataset-name random --random-input-len 256 --random-output-len 128 \
            --num-prompts 1000 --request-rate $RATE --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,99 \
            --save-result --result-dir "${RESULT_DIR}/online" --result-filename "ttft_diag_${MODE}_r${RATE}.json" > /dev/null 2>&1
        echo '{"__marker__": "rate_done", "rate": '$RATE'}' >> "$TRACE"
    done
    cleanup_gpu
done

echo ""
echo "=== TTFT Analysis ==="
python3 /workspace/vllm-emulator-v18/tools/e2e/analyze_ttft.py
echo ""
echo "=== DONE $(date) ==="
