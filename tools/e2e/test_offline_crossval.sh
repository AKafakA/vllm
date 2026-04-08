#!/bin/bash
# Cross-validation: profile with workload A, benchmark with workload B
# Tests that the offline profile generalizes to unseen workloads
set -e
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub
export CUDA_VISIBLE_DEVICES=1

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"
TRACE_FILE="${RESULT_DIR}/offline_step_cycle.jsonl"

pkill -9 -f python3 2>/dev/null || true
fuser /dev/nvidia* 2>/dev/null | tr " " "\n" | sort -u | xargs kill -9 2>/dev/null || true
sleep 5
rm -f "$TRACE_FILE"

echo "============================================"
echo "=== PHASE 1: Profile (workload A) ==="
echo "============================================"
echo "  Profile workloads: 100 prompts (128+64), 300 prompts (512+256), 50 prompts (64+32)"
echo "  These are DIFFERENT from benchmark workloads"

# Warmup
echo "  Warmup..."
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "$MODEL" --max-model-len 4096 --trust-remote-code \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 > /dev/null 2>&1

# Profile workload A: varied input/output lengths and prompt counts
VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
VLLM_EMULATOR_STEP_TRACE_OUTPUT="$TRACE_FILE" \
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "$MODEL" --max-model-len 4096 --trust-remote-code \
    --dataset-name random --random-input-len 128 --random-output-len 64 \
    --num-prompts 100 2>&1 | grep "Throughput:"

VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
VLLM_EMULATOR_STEP_TRACE_OUTPUT="$TRACE_FILE" \
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "$MODEL" --max-model-len 4096 --trust-remote-code \
    --dataset-name random --random-input-len 512 --random-output-len 256 \
    --num-prompts 300 2>&1 | grep "Throughput:"

VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
VLLM_EMULATOR_STEP_TRACE_OUTPUT="$TRACE_FILE" \
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "$MODEL" --max-model-len 4096 --trust-remote-code \
    --dataset-name random --random-input-len 64 --random-output-len 32 \
    --num-prompts 50 2>&1 | grep "Throughput:"

# Pure decode at various sizes
for NP in 50 150 300; do
    VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
    VLLM_EMULATOR_STEP_TRACE_OUTPUT="$TRACE_FILE" \
    python3 -m vllm.entrypoints.cli.main bench throughput \
        --model "$MODEL" --max-model-len 4096 --trust-remote-code \
        --dataset-name random --random-input-len 1 --random-output-len 256 \
        --num-prompts $NP 2>&1 | grep "Throughput:"
done

RECORDS=$(wc -l < "$TRACE_FILE" 2>/dev/null || echo 0)
echo "  Profile records: $RECORDS"

# Build profile
echo ""
echo "=== Build profile ==="
python3 /workspace/vllm-emulator-v18/vllm_emulator/profile/build_serving_profile.py \
    "${RESULT_DIR}/step_cycle_full.jsonl" \
    "${RESULT_DIR}/profiles/sweep-1.5b-tp1-v14.json" \
    "${RESULT_DIR}/profiles/serving-1.5b-tp1-crossval.json" \
    "$MODEL" "RTX-3060-12GB"

PROFILE="${RESULT_DIR}/profiles/serving-1.5b-tp1-crossval.json"

echo ""
echo "============================================"
echo "=== PHASE 2: Benchmark (workload B) ==="
echo "============================================"
echo "  Benchmark workloads: NONE of these were used for profiling"

pkill -9 -f python3 2>/dev/null || true
fuser /dev/nvidia* 2>/dev/null | tr " " "\n" | sort -u | xargs kill -9 2>/dev/null || true
sleep 5

# Benchmark B1: 200 prompts, input=256, output=128
echo ""
echo "--- B1: 200 prompts (256+128) ---"
echo "  Real warmup..."
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "$MODEL" --max-model-len 4096 --trust-remote-code \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 > /dev/null 2>&1
echo "  Real..."
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "$MODEL" --max-model-len 4096 --trust-remote-code \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 2>&1 | tee /workspace/crossval_b1_real.txt | grep "Throughput:"

pkill -9 -f python3 2>/dev/null || true
fuser /dev/nvidia* 2>/dev/null | tr " " "\n" | sort -u | xargs kill -9 2>/dev/null || true
sleep 5

echo "  Emu warmup..."
VLLM_EMULATOR_ENABLE_ORACLE=1 VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
VLLM_EMULATOR_MODE=offline VLLM_EMULATOR_EXECUTOR_HOOK=1 \
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "$MODEL" --max-model-len 4096 --trust-remote-code \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 > /dev/null 2>&1
echo "  Emu..."
VLLM_EMULATOR_ENABLE_ORACLE=1 VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
VLLM_EMULATOR_MODE=offline VLLM_EMULATOR_EXECUTOR_HOOK=1 \
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "$MODEL" --max-model-len 4096 --trust-remote-code \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 2>&1 | tee /workspace/crossval_b1_emu.txt | grep "Throughput:"

pkill -9 -f python3 2>/dev/null || true
fuser /dev/nvidia* 2>/dev/null | tr " " "\n" | sort -u | xargs kill -9 2>/dev/null || true
sleep 5

# Benchmark B2: 150 prompts, input=192, output=96 (unseen workload)
echo ""
echo "--- B2: 150 prompts (192+96) ---"
echo "  Real warmup..."
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "$MODEL" --max-model-len 4096 --trust-remote-code \
    --dataset-name random --random-input-len 192 --random-output-len 96 \
    --num-prompts 150 > /dev/null 2>&1
echo "  Real..."
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "$MODEL" --max-model-len 4096 --trust-remote-code \
    --dataset-name random --random-input-len 192 --random-output-len 96 \
    --num-prompts 150 2>&1 | tee /workspace/crossval_b2_real.txt | grep "Throughput:"

pkill -9 -f python3 2>/dev/null || true
fuser /dev/nvidia* 2>/dev/null | tr " " "\n" | sort -u | xargs kill -9 2>/dev/null || true
sleep 5

echo "  Emu warmup..."
VLLM_EMULATOR_ENABLE_ORACLE=1 VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
VLLM_EMULATOR_MODE=offline VLLM_EMULATOR_EXECUTOR_HOOK=1 \
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "$MODEL" --max-model-len 4096 --trust-remote-code \
    --dataset-name random --random-input-len 192 --random-output-len 96 \
    --num-prompts 150 > /dev/null 2>&1
echo "  Emu..."
VLLM_EMULATOR_ENABLE_ORACLE=1 VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
VLLM_EMULATOR_MODE=offline VLLM_EMULATOR_EXECUTOR_HOOK=1 \
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "$MODEL" --max-model-len 4096 --trust-remote-code \
    --dataset-name random --random-input-len 192 --random-output-len 96 \
    --num-prompts 150 2>&1 | tee /workspace/crossval_b2_emu.txt | grep "Throughput:"

pkill -9 -f python3 2>/dev/null || true
fuser /dev/nvidia* 2>/dev/null | tr " " "\n" | sort -u | xargs kill -9 2>/dev/null || true
sleep 5

# Benchmark B3: 400 prompts, input=384, output=64 (large batch, short output)
echo ""
echo "--- B3: 400 prompts (384+64) ---"
echo "  Real warmup..."
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "$MODEL" --max-model-len 4096 --trust-remote-code \
    --dataset-name random --random-input-len 384 --random-output-len 64 \
    --num-prompts 400 > /dev/null 2>&1
echo "  Real..."
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "$MODEL" --max-model-len 4096 --trust-remote-code \
    --dataset-name random --random-input-len 384 --random-output-len 64 \
    --num-prompts 400 2>&1 | tee /workspace/crossval_b3_real.txt | grep "Throughput:"

pkill -9 -f python3 2>/dev/null || true
fuser /dev/nvidia* 2>/dev/null | tr " " "\n" | sort -u | xargs kill -9 2>/dev/null || true
sleep 5

echo "  Emu warmup..."
VLLM_EMULATOR_ENABLE_ORACLE=1 VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
VLLM_EMULATOR_MODE=offline VLLM_EMULATOR_EXECUTOR_HOOK=1 \
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "$MODEL" --max-model-len 4096 --trust-remote-code \
    --dataset-name random --random-input-len 384 --random-output-len 64 \
    --num-prompts 400 > /dev/null 2>&1
echo "  Emu..."
VLLM_EMULATOR_ENABLE_ORACLE=1 VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
VLLM_EMULATOR_MODE=offline VLLM_EMULATOR_EXECUTOR_HOOK=1 \
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "$MODEL" --max-model-len 4096 --trust-remote-code \
    --dataset-name random --random-input-len 384 --random-output-len 64 \
    --num-prompts 400 2>&1 | tee /workspace/crossval_b3_emu.txt | grep "Throughput:"

pkill -9 -f python3 2>/dev/null || true
fuser /dev/nvidia* 2>/dev/null | tr " " "\n" | sort -u | xargs kill -9 2>/dev/null || true
sleep 5

echo ""
echo "============================================"
echo "=== CROSS-VALIDATION RESULTS ==="
echo "============================================"
echo "Profile workloads: 100(128+64), 300(512+256), 50(64+32), pure_decode(50,150,300)"
echo ""
echo "B1: 200 prompts (256+128) — UNSEEN"
echo "  Real: $(grep 'Throughput:' /workspace/crossval_b1_real.txt)"
echo "  Emu:  $(grep 'Throughput:' /workspace/crossval_b1_emu.txt)"
echo ""
echo "B2: 150 prompts (192+96) — UNSEEN"
echo "  Real: $(grep 'Throughput:' /workspace/crossval_b2_real.txt)"
echo "  Emu:  $(grep 'Throughput:' /workspace/crossval_b2_emu.txt)"
echo ""
echo "B3: 400 prompts (384+64) — UNSEEN"
echo "  Real: $(grep 'Throughput:' /workspace/crossval_b3_real.txt)"
echo "  Emu:  $(grep 'Throughput:' /workspace/crossval_b3_emu.txt)"

echo ""
echo "DONE $(date)"
