#!/bin/bash
# Profile worker subfunctions timing at R=8 on real GPU
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1
MODEL="Qwen/Qwen2.5-1.5B-Instruct"
PORT=8100

cleanup_gpu() {
    pkill -9 -f "python3.*api_server" 2>/dev/null || true
    fuser /dev/nvidia* 2>/dev/null | tr " " "\n" | sort -u | xargs kill -9 2>/dev/null || true
    sleep 10
}
wait_server() {
    for i in $(seq 1 120); do curl -s http://localhost:$PORT/health > /dev/null 2>&1 && return 0; sleep 1; done
    echo "ERROR: Server failed to start"; return 1
}

echo "=== Worker Subfunction Profiling ==="
echo "=== $(date) ==="

cleanup_gpu
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > /workspace/worker_profile_srv.log 2>&1 &
wait_server || exit 1

# Sweep warmup
for SWEEP_NP in 1 2 4 8 16 24 32 48 64 96 128 160 192 224 256; do
    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 1 --random-output-len 1 \
        --num-prompts $SWEEP_NP --request-rate inf > /dev/null 2>&1 || true
done

# Warmup
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 --request-rate 4 > /dev/null 2>&1

# Profile at R=8
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 1000 --request-rate 8 > /dev/null 2>&1

cleanup_gpu

# Extract WorkerTiming lines
echo ""
echo "=== Worker Subfunction Timing ==="
grep "WorkerTiming" /workspace/worker_profile_srv.log 2>/dev/null

echo ""
echo "=== DONE $(date) ==="
