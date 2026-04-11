#!/bin/bash
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1
MODEL="Qwen/Qwen2.5-1.5B-Instruct"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"
PROFILE="${RESULT_DIR}/profiles/serving-1.5b-tp1-fresh.json"

cleanup_gpu() {
    pkill -9 -f "python3.*api_server" 2>/dev/null || true
    pkill -9 -f "python3.*bench" 2>/dev/null || true
    fuser /dev/nvidia* 2>/dev/null | tr " " "\n" | sort -u | xargs kill -9 2>/dev/null || true
    sleep 10
}

echo "=== Offline Throughput Test ==="
echo "=== $(date) ==="

# Real GPU offline
echo ""
echo "=== Real GPU offline ==="
cleanup_gpu
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "$MODEL" --max-model-len 4096 --trust-remote-code \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 500 2>&1 | tee /workspace/offline_real.txt

# Emulator offline
echo ""
echo "=== Emulator offline ==="
cleanup_gpu
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
VLLM_EMULATOR_ORACLE_MODE=2d \
VLLM_EMULATOR_PREP_SURROGATE=1 \
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "$MODEL" --max-model-len 4096 --trust-remote-code \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 500 2>&1 | tee /workspace/offline_emu.txt

echo ""
echo "=== Comparison ==="
REAL_TPUT=$(grep "Throughput:" /workspace/offline_real.txt | grep -oP '[\d.]+(?= requests)')
EMU_TPUT=$(grep "Throughput:" /workspace/offline_emu.txt | grep -oP '[\d.]+(?= requests)')
echo "Real: $REAL_TPUT req/s"
echo "Emu:  $EMU_TPUT req/s"
if [ -n "$REAL_TPUT" ] && [ -n "$EMU_TPUT" ]; then
    python3 -c "
r=$REAL_TPUT; e=$EMU_TPUT
err=(e-r)/r*100
print(f'Error: {err:+.1f}%')
"
fi
echo ""
echo "=== DONE $(date) ==="
