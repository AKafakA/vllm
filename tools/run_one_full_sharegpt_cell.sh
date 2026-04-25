#!/bin/bash
# Generic 3-stage full-ShareGPT cell runner.
#
# Required env:
#   CELL_TAG         - unique tag for this cell, e.g. "apr24-r3-prefix-off"
#   BENCH_MODEL      - HF model id, e.g. "Qwen/Qwen3-8B"
# Optional:
#   EXTRA_SERVER_ARGS - appended to "--max-num-seqs 64" (default "")
#   EXTRA_EMU_ENV_VARS - space-separated KEY=VAL pairs (rarely needed now that
#                        --attention-backend is a CLI arg; prefer EXTRA_SERVER_ARGS)
#   REUSE_PROFILE    - if set, path to existing profile pack; skip stage 2
#   VALIDATE_ONLY    - if set, skip stage 1 real and use existing real baseline
#   REAL_BASELINE_DIR - required if VALIDATE_ONLY; dir with real_r*.json
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator
source tools/_bench_common.sh

: "${CELL_TAG:?required}"
: "${BENCH_MODEL:?required}"
EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS:-}"
EXTRA_EMU_ENV_VARS="${EXTRA_EMU_ENV_VARS:-}"
REUSE_PROFILE="${REUSE_PROFILE:-}"
VALIDATE_ONLY="${VALIDATE_ONLY:-}"
REAL_BASELINE_DIR="${REAL_BASELINE_DIR:-}"

# Mandatory parity flag unless explicitly overridden.
FULL_SERVER_ARGS="--max-num-seqs 64 $EXTRA_SERVER_ARGS"

export BENCH_PORT=8100
export BENCH_MAX_MODEL_LEN=4096
export SHAREGPT="./results/sharegpt_full_4096.json"

OUT="./results/${CELL_TAG}"
mkdir -p "$OUT"
MASTER_LOG="$OUT/run.log"
MARKER="/tmp/vllm_${CELL_TAG}"
touch "${MARKER}.started"
echo "=== ${CELL_TAG} start $(date -u) ===" > "$MASTER_LOG"
echo "MODEL=$BENCH_MODEL" >> "$MASTER_LOG"
echo "EXTRA_SERVER_ARGS=$FULL_SERVER_ARGS" >> "$MASTER_LOG"
echo "EXTRA_EMU_ENV_VARS=$EXTRA_EMU_ENV_VARS" >> "$MASTER_LOG"
echo "REUSE_PROFILE=$REUSE_PROFILE" >> "$MASTER_LOG"
echo "VALIDATE_ONLY=$VALIDATE_ONLY" >> "$MASTER_LOG"

# Shared profile-time CSV (single row per cell).
SUMMARY_CSV="./results/profile_time_summary.csv"
if [ ! -f "$SUMMARY_CSV" ]; then
    echo "cell_tag,model,stage1_real_s,stage2_profile_s,stage3_validate_s,total_s,date_utc" > "$SUMMARY_CSV"
fi

RATES=(2 4 8 16 32)
PROMPTS=2000
PARITY_REAL=""
PARITY_EMU=""
S1_SECS=0
S2_SECS=0
S3_SECS=0

start_real_server() {
    local SRV="$1"
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$BENCH_MODEL" --max-model-len "$BENCH_MAX_MODEL_LEN" \
        --port "$BENCH_PORT" --trust-remote-code $FULL_SERVER_ARGS \
        > "$SRV" 2>&1 &
}

start_emu_server() {
    local SRV="$1" PROFILE="$2"
    env VLLM_EMULATOR_ENABLE_ORACLE=1 VLLM_EMULATOR_MODE=realtime \
        VLLM_EMULATOR_EXECUTOR_HOOK=1 VLLM_EMULATOR_SCHEDULER_HOOK=0 \
        VLLM_EMULATOR_IPC_POSITION=disabled VLLM_EMULATOR_PREP_SURROGATE=0 \
        VLLM_EMULATOR_ORACLE_AGG=sample VLLM_EMULATOR_ORACLE_K=auto \
        VLLM_EMULATOR_ORACLE_MIN_SAMPLES=30 VLLM_EMULATOR_BW_SLOPE_SOURCE=disabled \
        VLLM_EMULATOR_PROFILE_PACK="$PROFILE" $EXTRA_EMU_ENV_VARS \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$BENCH_MODEL" --max-model-len "$BENCH_MAX_MODEL_LEN" \
        --port "$BENCH_PORT" --trust-remote-code $FULL_SERVER_ARGS \
        > "$SRV" 2>&1 &
}

run_bench_r() {
    local TAG="$1" R="$2"
    timeout 1800 python3 -m vllm.entrypoints.cli.main bench serve \
        --model "$BENCH_MODEL" --base-url "http://localhost:${BENCH_PORT}" \
        --dataset-name sharegpt --dataset-path "$SHAREGPT" \
        --num-prompts "$PROMPTS" --request-rate "$R" --seed 0 \
        --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
        --save-result --result-dir "$OUT" --result-filename "${TAG}_r${R}.json" \
        > "$OUT/bench_${TAG}_r${R}.log" 2>&1 || true
}

# ---------------- STAGE 1: real baseline ----------------
if [ -n "$VALIDATE_ONLY" ] && [ -n "$REAL_BASELINE_DIR" ]; then
    echo "[$(date -u +%T)] STAGE 1: skipped (VALIDATE_ONLY, using $REAL_BASELINE_DIR)" >> "$MASTER_LOG"
    S1_SECS=0
    for R in "${RATES[@]}"; do
        [ -f "$REAL_BASELINE_DIR/real_r${R}.json" ] && cp "$REAL_BASELINE_DIR/real_r${R}.json" "$OUT/real_r${R}.json"
    done
else
    echo "" >> "$MASTER_LOG"
    T0=$(date +%s)
    echo "[$(date -u +%T)] STAGE 1: real 5-rate baseline" >> "$MASTER_LOG"
    for R in "${RATES[@]}"; do
        SRV="$OUT/server_real_r${R}.log"
        common_cleanup
        common_preflight || { echo "preflight FAIL real r=$R" >> "$MASTER_LOG"; continue; }
        start_real_server "$SRV"
        common_wait_server || { echo "server FAIL real r=$R" >> "$MASTER_LOG"; common_cleanup; continue; }
        common_warmup
        if [ -z "$PARITY_REAL" ]; then
            PARITY_REAL=$(grep -m1 'non-default args' "$SRV" 2>/dev/null || echo NONE)
            echo "PARITY REAL: $PARITY_REAL" >> "$MASTER_LOG"
        fi
        echo "  [$(date +%T)] bench real r=$R" >> "$MASTER_LOG"
        run_bench_r "real" "$R"
        common_cleanup
    done
    T1=$(date +%s)
    S1_SECS=$((T1-T0))
    echo "[$(date -u +%T)] STAGE 1 duration: ${S1_SECS}s ($(awk "BEGIN{printf \"%.2f\",$S1_SECS/3600}")h)" >> "$MASTER_LOG"
fi

# ---------------- STAGE 2: dense profile ----------------
if [ -n "$REUSE_PROFILE" ]; then
    echo "" >> "$MASTER_LOG"
    echo "[$(date -u +%T)] STAGE 2: skipped (REUSE_PROFILE=$REUSE_PROFILE)" >> "$MASTER_LOG"
    PROFILE="$REUSE_PROFILE"
    S2_SECS=0
else
    echo "" >> "$MASTER_LOG"
    T0=$(date +%s)
    echo "[$(date -u +%T)] STAGE 2: dense profile capture" >> "$MASTER_LOG"
    ROUNDS=2 TAG="${CELL_TAG}" \
        bash tools/adaptive_profile_sharegpt_dense.sh >> "$MASTER_LOG" 2>&1
    T1=$(date +%s)
    S2_SECS=$((T1-T0))
    echo "[$(date -u +%T)] STAGE 2 duration: ${S2_SECS}s ($(awk "BEGIN{printf \"%.2f\",$S2_SECS/3600}")h)" >> "$MASTER_LOG"

    # Find the produced profile (RTX-8000 or A10 prefix, depending on platform).
    PROFILE=""
    for PREFIX in RTX-8000 A10 L40S; do
        CAND="./results/${PREFIX}-adaptive-${CELL_TAG}/serving-full.json"
        [ -f "$CAND" ] && PROFILE="$CAND" && break
    done
    if [ -z "$PROFILE" ]; then
        echo "STAGE 2 FAIL — profile missing" >> "$MASTER_LOG"
        touch "${MARKER}.done"
        exit 1
    fi
    echo "PROFILE=$PROFILE ($(stat -c%s "$PROFILE") bytes)" >> "$MASTER_LOG"
fi

# ---------------- STAGE 3: emu validate ----------------
echo "" >> "$MASTER_LOG"
T0=$(date +%s)
echo "[$(date -u +%T)] STAGE 3: emu 5-rate validate" >> "$MASTER_LOG"
for R in "${RATES[@]}"; do
    SRV="$OUT/server_emu_r${R}.log"
    common_cleanup
    common_preflight || { echo "preflight FAIL emu r=$R" >> "$MASTER_LOG"; continue; }
    start_emu_server "$SRV" "$PROFILE"
    common_wait_server || { echo "server FAIL emu r=$R" >> "$MASTER_LOG"; common_cleanup; continue; }
    common_warmup
    if [ -z "$PARITY_EMU" ]; then
        PARITY_EMU=$(grep -m1 'non-default args' "$SRV" 2>/dev/null || echo NONE)
        echo "PARITY EMU : $PARITY_EMU" >> "$MASTER_LOG"
    fi
    echo "  [$(date +%T)] bench emu r=$R" >> "$MASTER_LOG"
    run_bench_r "emu" "$R"
    common_cleanup
done
T1=$(date +%s)
S3_SECS=$((T1-T0))
echo "[$(date -u +%T)] STAGE 3 duration: ${S3_SECS}s" >> "$MASTER_LOG"

# ---------------- Deltas ----------------
echo "" >> "$MASTER_LOG"
echo "=== PER-RATE DELTAS ===" >> "$MASTER_LOG"
for R in "${RATES[@]}"; do
    python3 - <<PY >> "$MASTER_LOG" 2>&1
import json
R=$R
out="$OUT"
try:
    r=json.load(open(f"{out}/real_r{R}.json"))
    e=json.load(open(f"{out}/emu_r{R}.json"))
    keys=["mean_tpot_ms","mean_ttft_ms","mean_itl_ms","mean_e2el_ms"]
    parts=[f"r={R:>2}"]
    for k in keys:
        rv=r.get(k); ev=e.get(k)
        if rv and ev and rv!=0:
            d=(ev-rv)/rv*100
            parts.append(f"{k.replace('mean_','').replace('_ms',''):<4} delta={d:+6.2f}%")
    print(" | ".join(parts))
except Exception as ex:
    print(f"r={R}: ERROR {ex}")
PY
done

# ---------------- Profile-time summary row ----------------
TOTAL=$((S1_SECS + S2_SECS + S3_SECS))
echo "${CELL_TAG},${BENCH_MODEL},${S1_SECS},${S2_SECS},${S3_SECS},${TOTAL},$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$SUMMARY_CSV"

echo "" >> "$MASTER_LOG"
echo "[$(date -u +%T)] DONE (total=${TOTAL}s)" >> "$MASTER_LOG"
touch "${MARKER}.done"
