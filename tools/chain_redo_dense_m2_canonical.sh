#!/bin/bash
# M2 (Qwen3-8B) on vLLM's canonical ShareGPT file with roofline-corrected
# dense profile. Full 3-stage cycle + BW slope fit + 3-way validate.
#
# Headline cell for the paper post-branching to
# exp/apr24-roofline-correction.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

export BENCH_MODEL="Qwen/Qwen3-8B"
export BENCH_PORT=8100
export BENCH_MAX_MODEL_LEN=4096
export EXTRA_SERVER_ARGS=""
export SHAREGPT="./results/ShareGPT_V3_unfiltered_cleaned_split.json"

export VLLM_EMULATOR_SCHEDULER_HOOK=0
export VLLM_EMULATOR_IPC_POSITION=disabled
export VLLM_EMULATOR_PREP_SURROGATE=0
export VLLM_EMULATOR_ORACLE_AGG=sample

TAG="apr24-m2-canonical-dense"
MASTER_MARKER="/tmp/vllm_overnight_${TAG}"
MASTER_LOG="/tmp/vllm_overnight_${TAG}.log"

touch "${MASTER_MARKER}.started"
echo "=== redo-${TAG} start $(date -u) ===" > "$MASTER_LOG"

# Stage 1: real baseline on canonical ShareGPT.
if [ ! -d ./results/m2-canonical-5rate-real ]; then
    echo "[$(date +%T)] STAGE 1: real baseline (canonical ShareGPT)" >> "$MASTER_LOG"
    bash tools/chain_sharegpt_real_baseline.sh >> "$MASTER_LOG" 2>&1 || \
        echo "  stage1 partial" >> "$MASTER_LOG"
    if [ -d ./results/sharegpt-5rate-real ] && [ ! -d ./results/m2-canonical-5rate-real ]; then
        mv ./results/sharegpt-5rate-real ./results/m2-canonical-5rate-real
    fi
else
    echo "[$(date +%T)] STAGE 1: skipped (real baseline exists)" >> "$MASTER_LOG"
fi

# Stage 2: dense profile.
echo "" >> "$MASTER_LOG"
echo "[$(date +%T)] STAGE 2: dense profile" >> "$MASTER_LOG"
ROUNDS=2 TAG="${TAG}" bash tools/adaptive_profile_sharegpt_dense.sh >> "$MASTER_LOG" 2>&1
BASE_PROFILE="./results/RTX-8000-adaptive-${TAG}/serving-full.json"
PROFILE_TRACE="./results/RTX-8000-adaptive-${TAG}/step_cycle_trace.jsonl"
if [ ! -f "$BASE_PROFILE" ]; then
    echo "STAGE 2 FAIL -- profile missing" >> "$MASTER_LOG"
    touch "${MASTER_MARKER}.done"; exit 1
fi

# Stage 2b: fit BW slope from the fresh profile trace.
# RTX 8000 HBM is ~600 GB/s peak, typical sustained ~400 GB/s.
echo "" >> "$MASTER_LOG"
echo "[$(date +%T)] STAGE 2b: fit BW slope from profile trace" >> "$MASTER_LOG"
CALIB="./results/bw_calibration_${TAG}.json"
python3 tools/fit_bw_slope_from_profile.py \
    --trace-path "$PROFILE_TRACE" \
    --out-json "$CALIB" \
    --hw-bw-gbs 400 \
    >> "$MASTER_LOG" 2>&1

# Stage 2c: decorate two profile copies (measured + constant).
MEASURED_PROFILE="./results/RTX-8000-adaptive-${TAG}/serving-full-roofline-measured.json"
CONSTANT_PROFILE="./results/RTX-8000-adaptive-${TAG}/serving-full-roofline-constant.json"
cp "$BASE_PROFILE" "$MEASURED_PROFILE"
cp "$BASE_PROFILE" "$CONSTANT_PROFILE"
python3 tools/merge_bw_calibration_into_profile.py \
    --profile "$MEASURED_PROFILE" --calibration "$CALIB" >> "$MASTER_LOG" 2>&1
python3 tools/merge_bw_calibration_into_profile.py \
    --profile "$CONSTANT_PROFILE" --calibration "$CALIB" >> "$MASTER_LOG" 2>&1

# Stage 3: three validations.
run_validate () {
    local mode="$1"
    local profile="$2"
    local out_tag="$3"
    echo "" >> "$MASTER_LOG"
    echo "[$(date +%T)] STAGE 3 ($mode): validate $out_tag" >> "$MASTER_LOG"
    VLLM_EMULATOR_BW_SLOPE_SOURCE="$mode" \
    PROFILE="$profile" ORACLE_K=auto OUT_TAG="$out_tag" \
        bash tools/chain_validate_sharegpt.sh >> "$MASTER_LOG" 2>&1
}

run_validate "disabled" "$BASE_PROFILE" "validate-${TAG}-off"
run_validate "measured" "$MEASURED_PROFILE" "validate-${TAG}-measured"
run_validate "constant" "$CONSTANT_PROFILE" "validate-${TAG}-constant"

echo "" >> "$MASTER_LOG"
echo "=== redo-${TAG} DONE $(date -u) ===" >> "$MASTER_LOG"
touch "${MASTER_MARKER}.done"
