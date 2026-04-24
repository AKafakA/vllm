#!/bin/bash
# M5 redo with DENSE profile (replaces v2-profile run that failed r=32).
# Qwen3-14B. Reuses existing real baseline if present.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

export BENCH_MODEL="Qwen/Qwen3-14B"
export BENCH_PORT=8100
export BENCH_MAX_MODEL_LEN=4096
export EXTRA_SERVER_ARGS=""

export VLLM_EMULATOR_SCHEDULER_HOOK=0
export VLLM_EMULATOR_IPC_POSITION=disabled
export VLLM_EMULATOR_PREP_SURROGATE=0

TAG="apr24-m5-dense"
MASTER_MARKER="/tmp/vllm_overnight_${TAG}"
MASTER_LOG="/tmp/vllm_overnight_${TAG}.log"

touch "${MASTER_MARKER}.started"
echo "=== redo-${TAG} start $(date -u) ===" > "$MASTER_LOG"
echo "EXTRA_SERVER_ARGS=$EXTRA_SERVER_ARGS" >> "$MASTER_LOG"

# Stage 1 only if real baseline absent.
if [ ! -d ./results/m5-qwen3-14b-5rate-real ]; then
    echo "[$(date +%T)] STAGE 1: real baseline (Qwen3-14B)" >> "$MASTER_LOG"
    bash tools/chain_sharegpt_real_baseline.sh >> "$MASTER_LOG" 2>&1 || \
        echo "  stage1 partial" >> "$MASTER_LOG"
    if [ -d ./results/sharegpt-5rate-real ] && [ ! -d ./results/m5-qwen3-14b-5rate-real ]; then
        mv ./results/sharegpt-5rate-real ./results/m5-qwen3-14b-5rate-real
    fi
else
    echo "[$(date +%T)] STAGE 1: skipped (real baseline exists)" >> "$MASTER_LOG"
fi

# Stage 2: dense profile.
echo "" >> "$MASTER_LOG"
echo "[$(date +%T)] STAGE 2: dense profile" >> "$MASTER_LOG"
ROUNDS=2 TAG="${TAG}" bash tools/adaptive_profile_sharegpt_dense.sh >> "$MASTER_LOG" 2>&1
PROFILE="./results/RTX-8000-adaptive-${TAG}/serving-full.json"
if [ ! -f "$PROFILE" ]; then
    echo "STAGE 2 FAIL -- profile missing" >> "$MASTER_LOG"
    touch "${MASTER_MARKER}.done"; exit 1
fi

# Stage 3: validate.
echo "" >> "$MASTER_LOG"
echo "[$(date +%T)] STAGE 3: validate" >> "$MASTER_LOG"
PROFILE="$PROFILE" ORACLE_K=auto OUT_TAG="validate-${TAG}" \
  bash tools/chain_validate_sharegpt.sh >> "$MASTER_LOG" 2>&1

echo "=== redo-${TAG} DONE $(date -u) ===" >> "$MASTER_LOG"
touch "${MASTER_MARKER}.done"
