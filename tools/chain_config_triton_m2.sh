#!/bin/bash
# Config ablation: attention backend = TRITON_ATTN instead of the
# default (FlashInfer on RTX 8000 Turing). Tests that the
# profile-driven oracle is backend-agnostic by construction (the
# claim we make in §2 related work).
#
# Full 3-stage cycle: real baseline, dense profile, validate — all
# with VLLM_ATTENTION_BACKEND=TRITON_ATTN. Separate profile
# required because step-cycle distribution differs with kernel.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

export BENCH_MODEL="Qwen/Qwen3-8B"
export BENCH_PORT=8100
export BENCH_MAX_MODEL_LEN=4096
export EXTRA_SERVER_ARGS=""
export VLLM_ATTENTION_BACKEND="TRITON_ATTN"

export VLLM_EMULATOR_SCHEDULER_HOOK=0
export VLLM_EMULATOR_IPC_POSITION=disabled
export VLLM_EMULATOR_PREP_SURROGATE=0

TAG="apr24-config-triton-m2"
MASTER_MARKER="/tmp/vllm_overnight_${TAG}"
MASTER_LOG="/tmp/vllm_overnight_${TAG}.log"

touch "${MASTER_MARKER}.started"
echo "=== redo-${TAG} start $(date -u) ===" > "$MASTER_LOG"
echo "VLLM_ATTENTION_BACKEND=$VLLM_ATTENTION_BACKEND" >> "$MASTER_LOG"

if [ ! -d ./results/m2-triton-5rate-real ]; then
    echo "[$(date +%T)] STAGE 1: real baseline (TRITON_ATTN)" >> "$MASTER_LOG"
    bash tools/chain_sharegpt_real_baseline.sh >> "$MASTER_LOG" 2>&1 || \
        echo "  stage1 partial" >> "$MASTER_LOG"
    if [ -d ./results/sharegpt-5rate-real ] && [ ! -d ./results/m2-triton-5rate-real ]; then
        mv ./results/sharegpt-5rate-real ./results/m2-triton-5rate-real
    fi
else
    echo "[$(date +%T)] STAGE 1: skipped (real baseline exists)" >> "$MASTER_LOG"
fi

echo "" >> "$MASTER_LOG"
echo "[$(date +%T)] STAGE 2: dense profile (TRITON_ATTN)" >> "$MASTER_LOG"
ROUNDS=2 TAG="${TAG}" bash tools/adaptive_profile_sharegpt_dense.sh >> "$MASTER_LOG" 2>&1
PROFILE="./results/RTX-8000-adaptive-${TAG}/serving-full.json"
if [ ! -f "$PROFILE" ]; then
    echo "STAGE 2 FAIL -- profile missing" >> "$MASTER_LOG"
    touch "${MASTER_MARKER}.done"; exit 1
fi

echo "" >> "$MASTER_LOG"
echo "[$(date +%T)] STAGE 3: validate" >> "$MASTER_LOG"
PROFILE="$PROFILE" ORACLE_K=auto OUT_TAG="validate-${TAG}" \
  bash tools/chain_validate_sharegpt.sh >> "$MASTER_LOG" 2>&1

echo "=== redo-${TAG} DONE $(date -u) ===" >> "$MASTER_LOG"
touch "${MASTER_MARKER}.done"
