#!/bin/bash
# Debug version: no set -e, full output
source /workspace/vllm-v18-env/bin/activate

export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub
export VLLM_HOST_IP=127.0.0.1

MODEL="Qwen/Qwen2.5-1.5B-Instruct"

echo "=== Cleaning up ==="
pkill -9 -f EngineCore 2>/dev/null
pkill -9 -f api_server 2>/dev/null
sleep 5

echo "=== Starting prefill instance (GPU0) ==="
CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 2048 \
    --host 0.0.0.0 --port 8100 --trust-remote-code \
    --gpu-memory-utilization 0.8 \
    --kv-transfer-config \
    '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2,"kv_buffer_size":"1e9","kv_port":"14579","kv_connector_extra_config":{"proxy_ip":"127.0.0.1","proxy_port":"30001","http_ip":"127.0.0.1","http_port":"8100","send_type":"PUT_ASYNC"}}' \
    > /workspace/pd_prefill_debug.log 2>&1 &
echo "Prefill PID: $!"

echo "=== Starting decode instance (GPU1) ==="
CUDA_VISIBLE_DEVICES=1 python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 2048 \
    --host 0.0.0.0 --port 8200 --trust-remote-code \
    --gpu-memory-utilization 0.8 \
    --kv-transfer-config \
    '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2,"kv_buffer_size":"1e10","kv_port":"14580","kv_connector_extra_config":{"proxy_ip":"127.0.0.1","proxy_port":"30001","http_ip":"127.0.0.1","http_port":"8200","send_type":"PUT_ASYNC"}}' \
    > /workspace/pd_decode_debug.log 2>&1 &
echo "Decode PID: $!"

echo "=== Waiting for servers (5 min timeout) ==="
for i in $(seq 1 300); do
    p=$(curl -s http://localhost:8100/health 2>/dev/null && echo 1 || echo 0)
    d=$(curl -s http://localhost:8200/health 2>/dev/null && echo 1 || echo 0)
    if [ "$p" = "1" ] && [ "$d" = "1" ]; then
        echo "Both ready after ${i}s"
        break
    fi
    if [ $((i % 30)) -eq 0 ]; then
        echo "  Waiting... prefill=$p decode=$d (${i}s)"
    fi
    sleep 1
done

echo "=== Checking server status ==="
curl -s http://localhost:8100/health && echo " Prefill OK" || echo " Prefill FAILED"
curl -s http://localhost:8200/health && echo " Decode OK" || echo " Decode FAILED"

echo "=== Prefill server log (last 10 lines) ==="
tail -10 /workspace/pd_prefill_debug.log

echo "=== Decode server log (last 10 lines) ==="
tail -10 /workspace/pd_decode_debug.log

echo "=== Cleanup ==="
pkill -9 -f EngineCore 2>/dev/null
pkill -9 -f api_server 2>/dev/null

echo "DONE"
