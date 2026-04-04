#!/bin/bash
# PD Disaggregation with NIXL connector (works over PCIe, NVLink, or NIC)
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"

echo "============================================================"
echo "PD Disaggregation with NIXL Connector"
echo "============================================================"

pkill -9 -f EngineCore 2>/dev/null
pkill -9 -f api_server 2>/dev/null
sleep 5

# Prefill instance (GPU0)
echo "  Starting prefill instance (GPU0, port 8100)..."
CUDA_VISIBLE_DEVICES=0 VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 2048 \
    --host 0.0.0.0 --port 8100 --trust-remote-code \
    --gpu-memory-utilization 0.9 --enforce-eager \
    --kv-transfer-config \
    '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_parallel_size":2,"kv_buffer_size":"5e8"}' \
    > /workspace/pd_nixl_prefill.log 2>&1 &

# Decode instance (GPU1)
echo "  Starting decode instance (GPU1, port 8200)..."
CUDA_VISIBLE_DEVICES=1 VLLM_NIXL_SIDE_CHANNEL_PORT=5601 \
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 2048 \
    --host 0.0.0.0 --port 8200 --trust-remote-code \
    --gpu-memory-utilization 0.9 --enforce-eager \
    --kv-transfer-config \
    '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_parallel_size":2,"kv_buffer_size":"5e8"}' \
    > /workspace/pd_nixl_decode.log 2>&1 &

echo "  Waiting for servers (5 min)..."
for i in $(seq 1 300); do
    p=$(curl -s http://localhost:8100/health 2>/dev/null && echo 1 || echo 0)
    d=$(curl -s http://localhost:8200/health 2>/dev/null && echo 1 || echo 0)
    if [ "$p" = "1" ] && [ "$d" = "1" ]; then
        echo "  Both ready after ${i}s"
        break
    fi
    if [ $((i % 60)) -eq 0 ]; then
        echo "  Waiting... prefill=$p decode=$d (${i}s)"
    fi
    sleep 1
done

# Check status
echo ""
echo "  Server status:"
curl -s http://localhost:8100/health && echo " Prefill: OK" || echo " Prefill: FAILED"
curl -s http://localhost:8200/health && echo " Decode: OK" || echo " Decode: FAILED"

# Check for errors
echo ""
echo "  Prefill log (last errors):"
grep -i "error\|Error\|FAILED\|nixl" /workspace/pd_nixl_prefill.log 2>/dev/null | tail -5
echo ""
echo "  Decode log (last errors):"
grep -i "error\|Error\|FAILED\|nixl" /workspace/pd_nixl_decode.log 2>/dev/null | tail -5

# If both servers are up, try a simple request
if curl -s http://localhost:8100/health > /dev/null 2>&1; then
    echo ""
    echo "  Testing prefill instance directly..."
    curl -s -X POST http://localhost:8100/v1/completions \
        -H "Content-Type: application/json" \
        -d '{"model": "'"${MODEL}"'", "prompt": "Hello", "max_tokens": 10, "temperature": 0}' | python3 -m json.tool 2>/dev/null | head -10
fi

# Cleanup
pkill -9 -f EngineCore 2>/dev/null
pkill -9 -f api_server 2>/dev/null
sleep 5

echo ""
echo "DONE"
