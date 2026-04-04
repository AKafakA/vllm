#!/bin/bash
# Test CUDA mock + init skip on GPU host (Vast)
# This proves we can emulate ANY model size on ANY GPU
# because we skip model loading and KV cache allocation
set -e
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub

MODEL="Qwen/Qwen2.5-0.5B-Instruct"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"
SERVING_PROFILE="${RESULT_DIR}/profiles/serving-0.5b-tp1-step-cycle.json"
PORT=8100

pkill -9 -f EngineCore 2>/dev/null || true
pkill -9 -f api_server 2>/dev/null || true
sleep 5

echo "=== Test CUDA Mock + Init Skip on GPU Host ==="
echo "This uses the GPU vLLM build but skips GPU memory allocation."
echo "Should work for any model size regardless of GPU VRAM."

# Use MOCK_CUDA to force init skip path (even though we have real GPU)
VLLM_EMULATOR_MOCK_CUDA=1 \
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="${SERVING_PROFILE}" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
VLLM_EMULATOR_MEMORY=$((80*1024*1024*1024)) \
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 4096 \
    --port ${PORT} --trust-remote-code \
    --load-format dummy --enforce-eager \
    > /workspace/mock_on_gpu_server.log 2>&1 &

for i in $(seq 1 60); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
        echo "  Server ready after ${i}s"
        break
    fi
    sleep 1
done

if ! curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
    echo "  FAILED"
    tail -20 /workspace/mock_on_gpu_server.log
    exit 1
fi

# Check GPU memory — should be minimal since we skipped allocation
nvidia-smi --query-gpu=memory.used --format=csv,noheader

grep -i 'Emulator mode\|mock model\|skipping' /workspace/mock_on_gpu_server.log | head -5

# Benchmark
for rate in 1 2; do
    echo "  Rate=${rate}..."
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "${MODEL}" --base-url http://localhost:${PORT} \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 30 --request-rate ${rate} \
        --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
        --save-result --result-dir "${RESULT_DIR}/online" \
        --result-filename "mock_gpu_rate${rate}.json" 2>&1 | grep -E "TTFT|TPOT"
done

pkill -f api_server 2>/dev/null || true; sleep 2; pkill -9 -f EngineCore 2>/dev/null || true

echo ""
echo "=== Comparison ==="
python3 -c "
import json, os
rd = '${RESULT_DIR}/online'
for rate in [1, 2]:
    for label, fname in [('Real GPU', f'cluster_real_rate{rate}'), ('Mock on GPU', f'mock_gpu_rate{rate}')]:
        f = f'{rd}/{fname}.json'
        if os.path.exists(f):
            d = json.load(open(f))
            print(f'{label} rate={rate}: TTFT={d[\"mean_ttft_ms\"]:.1f}ms  TPOT={d[\"mean_tpot_ms\"]:.1f}ms')
    print()
"
echo "DONE"
