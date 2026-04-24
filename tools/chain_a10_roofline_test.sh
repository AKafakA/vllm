#!/bin/bash
# A10 roofline-correction validation cycle.
#   1. Start vLLM server with trace enabled.
#   2. Run BW calibration (single long-sequence decode).
#   3. Stop server.
#   4. Merge calibration into two profile copies (measured + constant).
#   5. Run validate with roofline OFF (baseline — reproduces Test A).
#   6. Run validate with roofline MEASURED.
#   7. Run validate with roofline CONSTANT.
# All runs hit the existing a10-qwen38b-full-5rate-real baseline.
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

TAG="apr24-a10-roofline"
MARKER="/tmp/vllm_${TAG}"
LOG="/tmp/vllm_${TAG}.log"
CALIB="$REPO/results/bw_calibration_a10_qwen3-8b.json"
TRACE="$REPO/results/bw_calibration_trace.jsonl"

BASE_PROFILE="$REPO/results/A10-adaptive-apr24-a10-sat-band/serving-full.json"
MEASURED_PROFILE="$REPO/results/A10-adaptive-apr24-a10-sat-band/serving-full-roofline-measured.json"
CONSTANT_PROFILE="$REPO/results/A10-adaptive-apr24-a10-sat-band/serving-full-roofline-constant.json"

touch "${MARKER}.started"
echo "=== ${TAG} start $(date -u) ===" > "$LOG"

# ------------------------------------------------------------------
# Stage 1: fit BW slope from the EXISTING profile's trace.
# No separate GPU run needed — regresses step_cycle_us across the
# profile's conc buckets, using each bucket's mean sum_kv as the x
# axis. The profile already spans sum_kv ≈ 400 → 21 000, which
# dwarfs what a single-sequence synthetic request can sweep.
# ------------------------------------------------------------------
echo "[$(date +%T)] STAGE 1: fit BW slope from profile trace" >> "$LOG"
PROFILE_TRACE="$REPO/results/A10-adaptive-apr24-a10-sat-band/step_cycle_trace.jsonl"
if [ ! -f "$PROFILE_TRACE" ]; then
    echo "profile trace missing at $PROFILE_TRACE — aborting" >> "$LOG"
    touch "${MARKER}.done"; exit 1
fi
python3 "$REPO/tools/fit_bw_slope_from_profile.py" \
    --trace-path "$PROFILE_TRACE" \
    --out-json "$CALIB" \
    --hw-bw-gbs 480 \
    >> "$LOG" 2>&1 || echo "  fit FAIL" >> "$LOG"

if [ ! -f "$CALIB" ] || grep -q '"error"' "$CALIB"; then
    echo "calibration did not produce valid JSON — aborting" >> "$LOG"
    touch "${MARKER}.done"; exit 1
fi

# ------------------------------------------------------------------
# Stage 2: decorate two profile copies.
# ------------------------------------------------------------------
echo "" >> "$LOG"
echo "[$(date +%T)] STAGE 2: merge calibration into profiles" >> "$LOG"
cp "$BASE_PROFILE" "$MEASURED_PROFILE"
cp "$BASE_PROFILE" "$CONSTANT_PROFILE"

# Use mean of the calibration's sum_kv_range as reference — makes the
# correction symmetric around the calibration operating point.
python3 "$REPO/tools/merge_bw_calibration_into_profile.py" \
    --profile "$MEASURED_PROFILE" --calibration "$CALIB" \
    >> "$LOG" 2>&1
python3 "$REPO/tools/merge_bw_calibration_into_profile.py" \
    --profile "$CONSTANT_PROFILE" --calibration "$CALIB" \
    >> "$LOG" 2>&1

# ------------------------------------------------------------------
# Stage 3: three validation runs (OFF, MEASURED, CONSTANT).
# ------------------------------------------------------------------
run_validate () {
    local mode="$1"
    local profile="$2"
    local out_tag="$3"
    echo "" >> "$LOG"
    echo "[$(date +%T)] STAGE 3 ($mode): validate $out_tag" >> "$LOG"
    VLLM_EMULATOR_BW_SLOPE_SOURCE="$mode" \
    PROFILE="$profile" ORACLE_K=auto OUT_TAG="$out_tag" \
        bash "$REPO/tools/chain_validate_sharegpt.sh" >> "$LOG" 2>&1
}

# Skip "off" — Test A earlier this session already produced it
# (validate-apr24-a10-sat-band-sample). Run measured + constant only.
run_validate "measured" "$MEASURED_PROFILE" "validate-${TAG}-measured"
run_validate "constant" "$CONSTANT_PROFILE" "validate-${TAG}-constant"

echo "" >> "$LOG"
echo "=== ${TAG} DONE $(date -u) ===" >> "$LOG"
touch "${MARKER}.done"
