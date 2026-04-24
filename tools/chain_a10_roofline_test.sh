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
# Stage 1: start server with trace on, run calibration.
# ------------------------------------------------------------------
echo "[$(date +%T)] STAGE 1: BW calibration" >> "$LOG"

# Kill any stale server/bench first.
pkill -9 -f 'vllm.entrypoints' 2>/dev/null || true
pkill -9 -f 'bench serve' 2>/dev/null || true
sleep 2

env \
    VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
    VLLM_EMULATOR_STEP_TRACE_OUTPUT="$TRACE" \
python3 -m vllm.entrypoints.openai.api_server \
    --model "$BENCH_MODEL" --max-model-len "$BENCH_MAX_MODEL_LEN" \
    --port "$BENCH_PORT" --trust-remote-code $EXTRA_SERVER_ARGS \
    >> "$LOG" 2>&1 &
SERVER_PID=$!
echo "  server pid=$SERVER_PID" >> "$LOG"

# Wait for /health.
for i in $(seq 1 60); do
    if curl -sf "http://localhost:${BENCH_PORT}/health" >/dev/null 2>&1; then
        echo "  server healthy at t=$i" >> "$LOG"
        break
    fi
    sleep 2
done

python3 "$REPO/tools/profile_bw_calibration.py" \
    --model "$BENCH_MODEL" \
    --base-url "http://localhost:${BENCH_PORT}" \
    --trace-path "$TRACE" \
    --prompt-len 3500 --output-len 500 \
    --out-json "$CALIB" \
    --hw-bw-gbs 480 \
    >> "$LOG" 2>&1 || echo "  calib FAIL" >> "$LOG"

# Stop server.
kill -9 "$SERVER_PID" 2>/dev/null || true
pkill -9 -f 'vllm.entrypoints' 2>/dev/null || true
sleep 3

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

run_validate "disabled" "$BASE_PROFILE" "validate-${TAG}-off"
run_validate "measured" "$MEASURED_PROFILE" "validate-${TAG}-measured"
run_validate "constant" "$CONSTANT_PROFILE" "validate-${TAG}-constant"

echo "" >> "$LOG"
echo "=== ${TAG} DONE $(date -u) ===" >> "$LOG"
touch "${MARKER}.done"
