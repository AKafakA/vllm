#!/bin/bash
# Test offline throughput using vllm bench throughput (LLM() interface)
# This is the correct offline benchmark — no HTTP server, direct batch processing
set -e
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub
export CUDA_VISIBLE_DEVICES=1

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"
PROFILE="${RESULT_DIR}/profiles/serving-1.5b-tp1-v3.json"

echo "============================================"
echo "=== Offline Throughput: LLM() interface ==="
echo "============================================"

# Clean GPU
pkill -9 -f python3 2>/dev/null || true
fuser /dev/nvidia* 2>/dev/null | tr " " "\n" | sort -u | xargs kill -9 2>/dev/null || true
sleep 5

echo ""
echo "=== Real GPU (200 prompts) ==="
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "$MODEL" --max-model-len 4096 --trust-remote-code \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 \
    2>&1 | tee /workspace/offline_real_throughput.txt

# Clean GPU between runs
pkill -9 -f python3 2>/dev/null || true
fuser /dev/nvidia* 2>/dev/null | tr " " "\n" | sort -u | xargs kill -9 2>/dev/null || true
sleep 5

echo ""
echo "=== Emulator (200 prompts) ==="
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "$MODEL" --max-model-len 4096 --trust-remote-code \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 \
    2>&1 | tee /workspace/offline_emu_throughput.txt

echo ""
echo "=== Comparison ==="
echo "Real:"
grep "Throughput:" /workspace/offline_real_throughput.txt
echo "Emu:"
grep "Throughput:" /workspace/offline_emu_throughput.txt

# Also test with 1000 prompts if first test passes
echo ""
echo "=== Real GPU (1000 prompts) ==="
pkill -9 -f python3 2>/dev/null || true
fuser /dev/nvidia* 2>/dev/null | tr " " "\n" | sort -u | xargs kill -9 2>/dev/null || true
sleep 5

python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "$MODEL" --max-model-len 4096 --trust-remote-code \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 1000 \
    2>&1 | tee /workspace/offline_real_1k_throughput.txt

pkill -9 -f python3 2>/dev/null || true
fuser /dev/nvidia* 2>/dev/null | tr " " "\n" | sort -u | xargs kill -9 2>/dev/null || true
sleep 5

echo ""
echo "=== Emulator (1000 prompts) ==="
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "$MODEL" --max-model-len 4096 --trust-remote-code \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 1000 \
    2>&1 | tee /workspace/offline_emu_1k_throughput.txt

echo ""
echo "=== Final Comparison ==="
echo "200 prompts:"
echo "  Real: $(grep 'Throughput:' /workspace/offline_real_throughput.txt)"
echo "  Emu:  $(grep 'Throughput:' /workspace/offline_emu_throughput.txt)"
echo "1000 prompts:"
echo "  Real: $(grep 'Throughput:' /workspace/offline_real_1k_throughput.txt)"
echo "  Emu:  $(grep 'Throughput:' /workspace/offline_emu_1k_throughput.txt)"

echo ""
echo "DONE $(date)"
