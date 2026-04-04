#!/bin/bash
# Quick test: verify virtual time summary prints on shutdown
set -e
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"

pkill -9 -f EngineCore 2>/dev/null || true
sleep 5

echo "=== Accelerated mode with virtual time summary ==="
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="${RESULT_DIR}/profiles/sweep-1.5b-tp1-v13.json" \
VLLM_EMULATOR_MODE=accelerated \
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "${MODEL}" --max-model-len 4096 \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 50 2>&1 | grep -E "Throughput|Virtual|virtual|GpuWorkerHook"

pkill -9 -f EngineCore 2>/dev/null || true

echo "DONE"
