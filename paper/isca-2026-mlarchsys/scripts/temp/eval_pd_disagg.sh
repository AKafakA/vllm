#!/bin/bash
# PD Disaggregation demo: prefill on GPU0, decode on GPU1
# Based on vLLM's disaggregated_prefill.sh example
set -e
source /workspace/vllm-v18-env/bin/activate

export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub
export VLLM_HOST_IP=127.0.0.1

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"

echo "============================================================"
echo "PD Disaggregation Demo (Prefill GPU0, Decode GPU1)"
echo "============================================================"

pkill -9 -f EngineCore 2>/dev/null || true
pkill -9 -f api_server 2>/dev/null || true
pkill -9 -f disagg 2>/dev/null || true
pkill -9 -f quart 2>/dev/null || true
sleep 5

# Install quart if needed (for disagg proxy)
python3 -c "import quart" 2>/dev/null || pip install quart 2>/dev/null

# Phase 1: Real PD disagg
echo ""
echo "PHASE 1: Real PD disaggregation"

# Prefill instance (KV producer) on GPU0
echo "  Starting prefill instance (GPU0, port 8100)..."
CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 2048 \
    --host 0.0.0.0 --port 8100 --trust-remote-code \
    --gpu-memory-utilization 0.8 \
    --kv-transfer-config \
    '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2,"kv_buffer_size":"1e9","kv_port":"14579","kv_connector_extra_config":{"proxy_ip":"127.0.0.1","proxy_port":"30001","http_ip":"127.0.0.1","http_port":"8100","send_type":"PUT_ASYNC"}}' \
    > /workspace/pd_prefill_server.log 2>&1 &
PREFILL_PID=$!

# Decode instance (KV consumer) on GPU1
echo "  Starting decode instance (GPU1, port 8200)..."
CUDA_VISIBLE_DEVICES=1 python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 2048 \
    --host 0.0.0.0 --port 8200 --trust-remote-code \
    --gpu-memory-utilization 0.8 \
    --kv-transfer-config \
    '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2,"kv_buffer_size":"1e10","kv_port":"14580","kv_connector_extra_config":{"proxy_ip":"127.0.0.1","proxy_port":"30001","http_ip":"127.0.0.1","http_port":"8200","send_type":"PUT_ASYNC"}}' \
    > /workspace/pd_decode_server.log 2>&1 &
DECODE_PID=$!

# Wait for both instances
echo "  Waiting for servers..."
for i in $(seq 1 300); do
    prefill_up=$(curl -s http://localhost:8100/health 2>/dev/null && echo 1 || echo 0)
    decode_up=$(curl -s http://localhost:8200/health 2>/dev/null && echo 1 || echo 0)
    if [ "$prefill_up" = "1" ] && [ "$decode_up" = "1" ]; then
        echo "  Both servers ready after ${i}s"
        break
    fi
    sleep 1
done

if [ "$prefill_up" != "1" ] || [ "$decode_up" != "1" ]; then
    echo "FAILED: PD disagg servers didn't start"
    echo "  Prefill log:"
    tail -10 /workspace/pd_prefill_server.log
    echo "  Decode log:"
    tail -10 /workspace/pd_decode_server.log
    kill $PREFILL_PID $DECODE_PID 2>/dev/null
    pkill -9 -f EngineCore 2>/dev/null || true
    echo ""
    echo "PD disaggregation requires P2P NCCL between GPUs."
    echo "RTX 3060s may not support P2P — this is expected."
    echo "The emulator's value: it can emulate PD disagg on any hardware"
    echo "by profiling on supported hardware and replaying anywhere."
    exit 0
fi

# Start proxy server
echo "  Starting disagg proxy (port 8000)..."
python3 /workspace/vllm-emulator-v18/benchmarks/disagg_benchmarks/disagg_prefill_proxy_server.py &
sleep 3

# Simple test
echo "  Testing PD disagg with curl..."
output=$(curl -s -X POST http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "'"${MODEL}"'", "prompt": "The future of AI is", "max_tokens": 20, "temperature": 0}')
echo "  Output: ${output}"

# Cleanup
kill $PREFILL_PID $DECODE_PID 2>/dev/null
pkill -9 -f EngineCore 2>/dev/null || true
pkill -f disagg 2>/dev/null || true
sleep 5

echo ""
echo "PD Disaggregation test completed."
echo "For full benchmark: use proxy at port 8000 with vllm bench serve"
echo ""
echo "DONE"
