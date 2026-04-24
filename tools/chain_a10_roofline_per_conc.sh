#!/bin/bash
# A10 roofline-per_conc-reference validation.
# Runs after the global-ref test completes. Re-fits calibration with
# per-conc references, then validates with
# VLLM_EMULATOR_BW_REF_MODE=per_conc. Expected: eliminates the
# feedback trap at the saturation knee (r=4) that global-ref
# exhibits.
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

# A10 delivery ceiling is ~3.5 r/s. Rates past 4 converge to the
# same delivered ~3.5 r/s with only the queue depth differing, so
# they measure queue-wait not real throughput. Use rates that
# exercise the envelope meaningfully: sub-ceiling + just past.
export RATES="1 2 3 4 8"

TAG="apr24-a10-roofline-per_conc"
MARKER="/tmp/vllm_${TAG}"
LOG="/tmp/vllm_${TAG}.log"
CALIB="$REPO/results/bw_calibration_a10_qwen3-8b_per_conc.json"

PROFILE_TRACE="$REPO/results/A10-adaptive-apr24-a10-sat-band/step_cycle_trace.jsonl"
BASE_PROFILE="$REPO/results/A10-adaptive-apr24-a10-sat-band/serving-full.json"
MEASURED_PROFILE="$REPO/results/A10-adaptive-apr24-a10-sat-band/serving-full-roofline-measured-percon.json"
CONSTANT_PROFILE="$REPO/results/A10-adaptive-apr24-a10-sat-band/serving-full-roofline-constant-percon.json"

touch "${MARKER}.started"
echo "=== ${TAG} start $(date -u) ===" > "$LOG"

# Stage 1: fresh fit that emits per-conc references.
echo "[$(date +%T)] STAGE 1: refit (with per-conc refs)" >> "$LOG"
python3 "$REPO/tools/fit_bw_slope_from_profile.py" \
    --trace-path "$PROFILE_TRACE" \
    --out-json "$CALIB" \
    --hw-bw-gbs 480 \
    >> "$LOG" 2>&1

# Stage 2: decorate profile copies (per-conc refs now included).
cp "$BASE_PROFILE" "$MEASURED_PROFILE"
cp "$BASE_PROFILE" "$CONSTANT_PROFILE"
python3 "$REPO/tools/merge_bw_calibration_into_profile.py" \
    --profile "$MEASURED_PROFILE" --calibration "$CALIB" >> "$LOG" 2>&1
python3 "$REPO/tools/merge_bw_calibration_into_profile.py" \
    --profile "$CONSTANT_PROFILE" --calibration "$CALIB" >> "$LOG" 2>&1

# Stage 3: 2-way validate (measured + constant, both per_conc mode).
run_validate () {
    local mode="$1"
    local profile="$2"
    local out_tag="$3"
    echo "" >> "$LOG"
    echo "[$(date +%T)] STAGE 3 ($mode, per_conc): validate $out_tag" >> "$LOG"
    VLLM_EMULATOR_BW_SLOPE_SOURCE="$mode" \
    VLLM_EMULATOR_BW_REF_MODE="per_conc" \
    PROFILE="$profile" ORACLE_K=auto OUT_TAG="$out_tag" \
        bash "$REPO/tools/chain_validate_sharegpt.sh" >> "$LOG" 2>&1
}

run_validate "measured" "$MEASURED_PROFILE" "validate-${TAG}-measured"
run_validate "constant" "$CONSTANT_PROFILE" "validate-${TAG}-constant"

echo "" >> "$LOG"
echo "=== ${TAG} DONE $(date -u) ===" >> "$LOG"
touch "${MARKER}.done"
