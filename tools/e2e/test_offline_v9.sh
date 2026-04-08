#!/bin/bash
# v9: extensive offline profiling after warmup, then benchmark
set -e
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub
export CUDA_VISIBLE_DEVICES=1

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"
PORT=8100
SCRIPT_DIR="/workspace/vllm-emulator-v18/tools/e2e"
ONLINE_DIR="${RESULT_DIR}/online"
TRACE_FILE="${RESULT_DIR}/offline_step_cycle.jsonl"

pkill -9 -f python3 2>/dev/null || true
fuser /dev/nvidia* 2>/dev/null | tr " " "\n" | sort -u | xargs kill -9 2>/dev/null || true
sleep 5
rm -f "$TRACE_FILE"

echo "=== PHASE 1: Warmup GPU ==="
echo "  200 prompts warmup (throwaway)..."
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "$MODEL" --max-model-len 4096 --trust-remote-code \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 2>&1 | grep "Throughput:"

echo ""
echo "=== PHASE 2: Profile offline (warm GPU, many batch sizes) ==="

# Standard workload (input=256, output=128) at many sizes
for NP in 20 50 80 100 150 200 250 300 400 500; do
    echo "  Profile $NP prompts (256+128)..."
    VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
    VLLM_EMULATOR_STEP_TRACE_OUTPUT="$TRACE_FILE" \
    python3 -m vllm.entrypoints.cli.main bench throughput \
        --model "$MODEL" --max-model-len 4096 --trust-remote-code \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts $NP 2>&1 | grep "Throughput:"
done

# Pure decode (input=1) for clean decode-only coverage
for NP in 50 100 200 300; do
    echo "  Profile pure decode $NP prompts (1+256)..."
    VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
    VLLM_EMULATOR_STEP_TRACE_OUTPUT="$TRACE_FILE" \
    python3 -m vllm.entrypoints.cli.main bench throughput \
        --model "$MODEL" --max-model-len 4096 --trust-remote-code \
        --dataset-name random --random-input-len 1 --random-output-len 256 \
        --num-prompts $NP 2>&1 | grep "Throughput:"
done

RECORDS=$(wc -l < "$TRACE_FILE" 2>/dev/null || echo 0)
echo "  Total offline trace records: $RECORDS"

echo ""
echo "=== Build profile v9 ==="
python3 /workspace/vllm-emulator-v18/vllm_emulator/profile/build_serving_profile.py \
    "${RESULT_DIR}/step_cycle_full.jsonl" \
    "${RESULT_DIR}/profiles/sweep-1.5b-tp1-v14.json" \
    "${RESULT_DIR}/profiles/serving-1.5b-tp1-v9.json" \
    "$MODEL" "RTX-3060-12GB"

PROFILE="${RESULT_DIR}/profiles/serving-1.5b-tp1-v9.json"

echo ""
echo "=== PHASE 3: Benchmark offline (warm GPU) ==="
pkill -9 -f python3 2>/dev/null || true
fuser /dev/nvidia* 2>/dev/null | tr " " "\n" | sort -u | xargs kill -9 2>/dev/null || true
sleep 5

echo "  Real warmup..."
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "$MODEL" --max-model-len 4096 --trust-remote-code \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 > /dev/null 2>&1

echo "  Real benchmark (200)..."
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "$MODEL" --max-model-len 4096 --trust-remote-code \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 2>&1 | tee /workspace/v9_offline_real_200.txt | grep "Throughput:"

echo "  Real benchmark (1000)..."
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "$MODEL" --max-model-len 4096 --trust-remote-code \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 1000 2>&1 | tee /workspace/v9_offline_real_1k.txt | grep "Throughput:"

pkill -9 -f python3 2>/dev/null || true
fuser /dev/nvidia* 2>/dev/null | tr " " "\n" | sort -u | xargs kill -9 2>/dev/null || true
sleep 5

echo "  Emu warmup..."
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
VLLM_EMULATOR_MODE=offline \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "$MODEL" --max-model-len 4096 --trust-remote-code \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 > /dev/null 2>&1

echo "  Emu benchmark (200)..."
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
VLLM_EMULATOR_MODE=offline \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "$MODEL" --max-model-len 4096 --trust-remote-code \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 2>&1 | tee /workspace/v9_offline_emu_200.txt | grep "Throughput:"

echo "  Emu benchmark (1000)..."
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
VLLM_EMULATOR_MODE=offline \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "$MODEL" --max-model-len 4096 --trust-remote-code \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 1000 2>&1 | tee /workspace/v9_offline_emu_1k.txt | grep "Throughput:"

pkill -9 -f python3 2>/dev/null || true
fuser /dev/nvidia* 2>/dev/null | tr " " "\n" | sort -u | xargs kill -9 2>/dev/null || true
sleep 5

echo ""
echo "============================================"
echo "=== SUMMARY ==="
echo "============================================"
echo "Offline 200 (warm):"
echo "  Real: $(grep 'Throughput:' /workspace/v9_offline_real_200.txt)"
echo "  Emu:  $(grep 'Throughput:' /workspace/v9_offline_emu_200.txt)"
echo "Offline 1000 (warm):"
echo "  Real: $(grep 'Throughput:' /workspace/v9_offline_real_1k.txt)"
echo "  Emu:  $(grep 'Throughput:' /workspace/v9_offline_emu_1k.txt)"

echo ""
echo "DONE $(date)"
