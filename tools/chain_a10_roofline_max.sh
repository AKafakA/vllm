#!/bin/bash
# A10 roofline test with MAX-ROOFLINE combine mode (ReLU one-sided
# correction) and MULTIVARIATE slope (conc-controlled regression).
# Tests 3 slope variants × max_roofline combine at A10-envelope rates.
set -uo pipefail
REPO=/workspace/vllm-emulator
cd "$REPO"
source "$REPO/.venv/bin/activate"

export HF_HOME=/dev/shm/hf_home
export BENCH_MODEL="Qwen/Qwen3-8B"
export BENCH_PORT=8100
export BENCH_MAX_MODEL_LEN=4096
export EXTRA_SERVER_ARGS="--max-num-seqs 64"
export SHAREGPT="$REPO/results/sharegpt_full_4096.json"
export VLLM_EMULATOR_SCHEDULER_HOOK=0
export VLLM_EMULATOR_IPC_POSITION=disabled
export VLLM_EMULATOR_PREP_SURROGATE=0
export VLLM_EMULATOR_ORACLE_AGG=sample
# A10 delivery envelope: 3 sub-ceiling + 2 past-ceiling
export RATES="1 2 3 4 8"

TAG="apr24-a10-roofline-max"
MARKER="/tmp/vllm_${TAG}"
LOG="/tmp/vllm_${TAG}.log"
CALIB="$REPO/results/bw_calibration_a10_qwen3-8b_max.json"

PROFILE_TRACE="$REPO/results/A10-adaptive-apr24-a10-sat-band/step_cycle_trace.jsonl"
BASE_PROFILE="$REPO/results/A10-adaptive-apr24-a10-sat-band/serving-full.json"

touch "${MARKER}.started"
echo "=== ${TAG} start $(date -u) ===" > "$LOG"

# Stage 1: fit with multivariate slope emitted.
echo "[$(date +%T)] STAGE 1: fit (with multivariate slope)" >> "$LOG"
python3 "$REPO/tools/fit_bw_slope_from_profile.py" \
    --trace-path "$PROFILE_TRACE" \
    --out-json "$CALIB" \
    --hw-bw-gbs 480 \
    >> "$LOG" 2>&1

# Stage 2: one profile copy (all 3 slopes selected at query time).
ROOFLINE_PROFILE="$REPO/results/A10-adaptive-apr24-a10-sat-band/serving-full-roofline-max.json"
cp "$BASE_PROFILE" "$ROOFLINE_PROFILE"
python3 "$REPO/tools/merge_bw_calibration_into_profile.py" \
    --profile "$ROOFLINE_PROFILE" --calibration "$CALIB" >> "$LOG" 2>&1

# Stage 3: 3-way validate (measured/constant/multivariate × max_roofline)
run_validate () {
    local slope_src="$1"
    local out_tag="$2"
    echo "" >> "$LOG"
    echo "[$(date +%T)] STAGE 3 (slope=$slope_src, combine=max_roofline, per_conc): validate $out_tag" >> "$LOG"
    VLLM_EMULATOR_BW_SLOPE_SOURCE="$slope_src" \
    VLLM_EMULATOR_BW_REF_MODE="per_conc" \
    VLLM_EMULATOR_BW_COMBINE_MODE="max_roofline" \
    PROFILE="$ROOFLINE_PROFILE" ORACLE_K=auto OUT_TAG="$out_tag" \
        bash "$REPO/tools/chain_validate_sharegpt.sh" >> "$LOG" 2>&1
}

run_validate "multivariate" "validate-${TAG}-mv"
run_validate "constant"     "validate-${TAG}-const"
run_validate "measured"     "validate-${TAG}-meas"

echo "" >> "$LOG"
echo "=== ${TAG} DONE $(date -u) ===" >> "$LOG"
touch "${MARKER}.done"
