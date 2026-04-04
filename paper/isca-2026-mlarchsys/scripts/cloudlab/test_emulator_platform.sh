#!/bin/bash
# Test EmulatorPlatform on CPU-only host
set -e
source ~/vllm-emulator/venv/bin/activate

MODEL="Qwen/Qwen2.5-0.5B-Instruct"
PORT=8100
PROFILE_DIR=~/vllm-emulator/profiles
mkdir -p "${PROFILE_DIR}"

echo "============================================================"
echo "EmulatorPlatform Test (CPU-only)"
echo "  Host: $(hostname)"
echo "  CPUs: $(nproc)"
echo "  GPU: none"
echo "============================================================"

# Step 1: Test basic import and platform activation
echo ""
echo "Step 1: Test EmulatorPlatform activation..."
VLLM_EMULATOR_ENABLE_ORACLE=1 \
python3 -c "
from vllm_emulator.platform import emulator_platform_plugin
result = emulator_platform_plugin()
print(f'Platform plugin: {result}')
assert result is not None, 'EmulatorPlatform should activate with ORACLE=1'
print('OK: EmulatorPlatform activates correctly')
"

# Step 2: Try starting vLLM with EmulatorPlatform + dummy weights
echo ""
echo "Step 2: Start vLLM server with EmulatorPlatform..."
# Note: This is experimental — vLLM has many CUDA code paths
# that may fail on CPU. The EmulatorPlatform sets device=cpu.

LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:${HOME}/vllm-emulator/venv/lib/libiomp5.so" \
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="${PROFILE_DIR}/serving-1.5b-tp1.json" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
VLLM_EMULATOR_MEMORY=$((12*1024*1024*1024)) \
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 2048 \
    --port ${PORT} --trust-remote-code \
    --load-format dummy \
    --enforce-eager \
    > ~/vllm-setup/emulator_platform_server.log 2>&1 &
SERVER_PID=$!

echo "  Server PID: ${SERVER_PID}"
echo "  Waiting for server..."

for i in $(seq 1 60); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
        echo "  Server ready after ${i}s!"
        break
    fi
    if ! kill -0 ${SERVER_PID} 2>/dev/null; then
        echo "  Server died"
        break
    fi
    sleep 1
done

if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
    echo ""
    echo "  Testing inference..."
    curl -s -X POST http://localhost:${PORT}/v1/completions \
        -H "Content-Type: application/json" \
        -d '{"model": "'"${MODEL}"'", "prompt": "Hello world", "max_tokens": 5, "temperature": 0}' | python3 -m json.tool 2>/dev/null | head -10
    echo ""
    echo "  SUCCESS: EmulatorPlatform works on CPU!"
else
    echo ""
    echo "  Server failed to start. Log:"
    tail -30 ~/vllm-setup/emulator_platform_server.log
    echo ""
    echo "  Checking for CUDA-specific errors..."
    grep -i "cuda\|CUDA\|RuntimeError\|ImportError" ~/vllm-setup/emulator_platform_server.log | head -5
fi

kill ${SERVER_PID} 2>/dev/null
wait ${SERVER_PID} 2>/dev/null

echo ""
echo "DONE"
