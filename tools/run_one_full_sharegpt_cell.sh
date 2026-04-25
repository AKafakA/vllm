#!/bin/bash
# Generic 3-stage full-ShareGPT cell runner (v4 + DEFAULT max-num-seqs).
#
# v4 invariants (parity-asserted by Apr 25 quick test):
#   - All 3 stages (real bench, profile capture, emu validate) run with
#     DEFAULT max-num-seqs (no --max-num-seqs flag passed to api_server).
#     Mismatch in this flag was the root cause of the Apr 22-24 fake
#     "A10 saturation gap" — see MEMORY.md / feedback_config_parity.
#   - Emu stage uses CUDA-invisible v4 path (CUDA_VISIBLE_DEVICES=""
#     + LD_LIBRARY_PATH stub + NCCL→gloo). Emu sees no GPU; oracle
#     samples from profile pack and the executor hook returns fake
#     output before any kernel dispatches.
#   - parity_audit.sh runs after each stage; cell aborts on violation.
#
# Required env:
#   CELL_TAG         - unique tag for this cell, e.g. "apr26-r3-prefix-off"
#   BENCH_MODEL      - HF model id, e.g. "Qwen/Qwen3-8B"
# Optional:
#   EXTRA_SERVER_ARGS - appended to api_server (default ""). Use this for
#                       cell-specific server config like --no-prefix-caching
#                       or --attention-backend TRITON_ATTN. NEVER include
#                       --max-num-seqs here — it must remain DEFAULT.
#   REUSE_PROFILE    - if set, path to existing profile pack; skip stage 2
#   VALIDATE_ONLY    - if set, skip stage 1 real and use existing real baseline
#   REAL_BASELINE_DIR - required if VALIDATE_ONLY; dir with real_r*.json
#   STUB_DIR         - cuda stub dir (default ~/cuda_stubs); must contain
#                       libcuda.so.1 + libcudart.so.* for v4 emu mode
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator
source tools/_bench_common.sh

: "${CELL_TAG:?required}"
: "${BENCH_MODEL:?required}"
EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS:-}"
REUSE_PROFILE="${REUSE_PROFILE:-}"
VALIDATE_ONLY="${VALIDATE_ONLY:-}"
REAL_BASELINE_DIR="${REAL_BASELINE_DIR:-}"
STUB_DIR="${STUB_DIR:-$HOME/cuda_stubs}"

# Refuse known-bad parity violation: --max-num-seqs in EXTRA_SERVER_ARGS.
if echo "$EXTRA_SERVER_ARGS" | grep -qE '\-\-max-num-seqs|\-\-max_num_seqs'; then
    echo "FATAL: EXTRA_SERVER_ARGS contains --max-num-seqs. The v4 path requires" >&2
    echo "       DEFAULT max-num-seqs across all 3 stages (see Apr 25 parity bug" >&2
    echo "       in MEMORY.md / feedback_config_parity)." >&2
    exit 1
fi

# v4 mandatory parity: ALL 3 stages run at DEFAULT max-num-seqs. Cell-specific
# server args (TRITON, prefix-caching off, etc.) flow through EXTRA_SERVER_ARGS.
FULL_SERVER_ARGS="$EXTRA_SERVER_ARGS"

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
echo "EXTRA_SERVER_ARGS=$FULL_SERVER_ARGS  (max-num-seqs=DEFAULT enforced)" >> "$MASTER_LOG"
echo "REUSE_PROFILE=$REUSE_PROFILE" >> "$MASTER_LOG"
echo "VALIDATE_ONLY=$VALIDATE_ONLY" >> "$MASTER_LOG"
echo "STUB_DIR=$STUB_DIR" >> "$MASTER_LOG"

# Shared profile-time CSV (single row per cell).
SUMMARY_CSV="./results/profile_time_summary.csv"
if [ ! -f "$SUMMARY_CSV" ]; then
    echo "cell_tag,model,stage1_real_s,stage2_profile_s,stage3_validate_s,total_s,date_utc" > "$SUMMARY_CSV"
fi

# Per-cell per-rate delta CSV (parsed by correctness cron).
DELTA_CSV="$OUT/per_rate_deltas.csv"
echo "rate,ttft_mean_pct,tpot_mean_pct,itl_mean_pct,e2e_mean_pct,tput_pct" > "$DELTA_CSV"

RATES=(2 4 8 16 32)
PROMPTS=2000
PARITY_REAL=""
PARITY_EMU=""
PARITY_PROFILE=""
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

# v4 emu server: CUDA invisible, NCCL→gloo, no kernels dispatched.
# Profile pack drives oracle sampling via the executor hook.
start_emu_server() {
    local SRV="$1" PROFILE="$2"
    env LD_LIBRARY_PATH="$STUB_DIR:${LD_LIBRARY_PATH:-}" \
        CUDA_VISIBLE_DEVICES="" \
        VLLM_EMULATOR_ENABLE_ORACLE=1 \
        VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
        VLLM_EMULATOR_MODE=realtime \
        VLLM_EMULATOR_EXECUTOR_HOOK=1 \
        VLLM_EMULATOR_SCHEDULER_HOOK=0 \
        VLLM_EMULATOR_IPC_POSITION=disabled \
        VLLM_EMULATOR_PREP_SURROGATE=0 \
        VLLM_EMULATOR_ORACLE_AGG=sample \
        VLLM_EMULATOR_ORACLE_K=auto \
        VLLM_EMULATOR_ORACLE_MIN_SAMPLES=30 \
        VLLM_EMULATOR_BW_SLOPE_SOURCE=disabled \
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
    echo "[$(date -u +%T)] STAGE 1: real 5-rate baseline (DEFAULT max-num-seqs)" >> "$MASTER_LOG"
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
    # Capture parity from profile pack's first server log (if present)
    PROFILE_DIR="$(dirname "$PROFILE")"
    PROFILE_LOGS=$(ls "$PROFILE_DIR"/logs/server_*.log 2>/dev/null | head -1)
    if [ -n "$PROFILE_LOGS" ]; then
        PARITY_PROFILE=$(grep -m1 'non-default args' "$PROFILE_LOGS" 2>/dev/null || echo NONE)
        echo "PARITY PROFILE: $PARITY_PROFILE" >> "$MASTER_LOG"
    fi
else
    echo "" >> "$MASTER_LOG"
    T0=$(date +%s)
    echo "[$(date -u +%T)] STAGE 2: dense profile capture (DEFAULT max-num-seqs)" >> "$MASTER_LOG"
    # Profile capture script — adaptive 5-rate sweep to populate the 2D table.
    # Reads BENCH_MODEL, BENCH_MAX_MODEL_LEN, BENCH_PORT, EXTRA_SERVER_ARGS from env.
    ROUNDS=2 TAG="${CELL_TAG}" EXTRA_SERVER_ARGS="$FULL_SERVER_ARGS" \
        bash tools/adaptive_profile_capture.sh >> "$MASTER_LOG" 2>&1
    T1=$(date +%s)
    S2_SECS=$((T1-T0))
    echo "[$(date -u +%T)] STAGE 2 duration: ${S2_SECS}s ($(awk "BEGIN{printf \"%.2f\",$S2_SECS/3600}")h)" >> "$MASTER_LOG"

    # Find the produced profile (RTX-8000 or A10 prefix, depending on platform).
    PROFILE=""
    for PREFIX in RTX-8000 A10 L40S H100; do
        CAND="./results/${PREFIX}-adaptive-${CELL_TAG}/serving-full.json"
        [ -f "$CAND" ] && PROFILE="$CAND" && break
    done
    if [ -z "$PROFILE" ]; then
        echo "STAGE 2 FAIL — profile missing" >> "$MASTER_LOG"
        touch "${MARKER}.done"
        exit 1
    fi
    echo "PROFILE=$PROFILE ($(stat -c%s "$PROFILE") bytes)" >> "$MASTER_LOG"
    PROFILE_DIR="$(dirname "$PROFILE")"
    PROFILE_LOGS=$(ls "$PROFILE_DIR"/logs/server_*.log 2>/dev/null | head -1)
    if [ -n "$PROFILE_LOGS" ]; then
        PARITY_PROFILE=$(grep -m1 'non-default args' "$PROFILE_LOGS" 2>/dev/null || echo NONE)
        echo "PARITY PROFILE: $PARITY_PROFILE" >> "$MASTER_LOG"
    fi
fi

# Parity gate after Stage 2: real-bench args MUST match profile-capture args.
if [ -n "$PARITY_REAL" ] && [ -n "$PARITY_PROFILE" ] && \
   [ "$PARITY_REAL" != "$PARITY_PROFILE" ]; then
    NORM_REAL=$(echo "$PARITY_REAL" | sed -E "s/'port':[^,]*, ?//")
    NORM_PROFILE=$(echo "$PARITY_PROFILE" | sed -E "s/'port':[^,]*, ?//")
    if [ "$NORM_REAL" != "$NORM_PROFILE" ]; then
        echo "PARITY VIOLATION (real vs profile, ignoring port):" >> "$MASTER_LOG"
        echo "  real:    $NORM_REAL" >> "$MASTER_LOG"
        echo "  profile: $NORM_PROFILE" >> "$MASTER_LOG"
        echo "ABORT — fix scripts before continuing" >> "$MASTER_LOG"
        touch "${MARKER}.done"
        exit 2
    fi
fi

# ---------------- STAGE 3: emu validate ----------------
echo "" >> "$MASTER_LOG"
T0=$(date +%s)
echo "[$(date -u +%T)] STAGE 3: emu 5-rate validate (v4: CUDA invisible, NCCL→gloo)" >> "$MASTER_LOG"
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

# Parity gate after Stage 3: emu-bench args MUST match real-bench args.
if [ -n "$PARITY_REAL" ] && [ -n "$PARITY_EMU" ] && [ "$PARITY_REAL" != "$PARITY_EMU" ]; then
    NORM_REAL=$(echo "$PARITY_REAL" | sed -E "s/'port':[^,]*, ?//")
    NORM_EMU=$(echo "$PARITY_EMU" | sed -E "s/'port':[^,]*, ?//")
    if [ "$NORM_REAL" != "$NORM_EMU" ]; then
        echo "PARITY VIOLATION (real vs emu, ignoring port):" >> "$MASTER_LOG"
        echo "  real: $NORM_REAL" >> "$MASTER_LOG"
        echo "  emu : $NORM_EMU" >> "$MASTER_LOG"
        echo "WARNING — Stage 3 results may be unusable" >> "$MASTER_LOG"
    fi
fi

# ---------------- Per-rate MEAN deltas (paper-grade) ----------------
echo "" >> "$MASTER_LOG"
echo "=== PER-RATE DELTAS (mean) ===" >> "$MASTER_LOG"
for R in "${RATES[@]}"; do
    python3 - "$R" "$OUT" "$DELTA_CSV" <<'PY' >> "$MASTER_LOG" 2>&1
import json, sys
R = int(sys.argv[1]); out = sys.argv[2]; delta_csv = sys.argv[3]
try:
    r = json.load(open(f"{out}/real_r{R}.json"))
    e = json.load(open(f"{out}/emu_r{R}.json"))
    pct = lambda a, b: (b/a-1)*100 if a > 0 else 0.0
    d_ttft = pct(r["mean_ttft_ms"], e["mean_ttft_ms"])
    d_tpot = pct(r["mean_tpot_ms"], e["mean_tpot_ms"])
    d_itl  = pct(r["mean_itl_ms"],  e["mean_itl_ms"])
    d_e2e  = pct(r.get("mean_e2el_ms", 0), e.get("mean_e2el_ms", 0))
    d_tput = pct(r["output_throughput"], e["output_throughput"])
    print(f"r={R:>2} | TTFT_mean {d_ttft:+7.2f}% | TPOT_mean {d_tpot:+6.2f}% | "
          f"ITL_mean {d_itl:+6.2f}% | E2E_mean {d_e2e:+6.2f}% | tput {d_tput:+5.2f}%")
    with open(delta_csv, "a") as f:
        f.write(f"{R},{d_ttft:.4f},{d_tpot:.4f},{d_itl:.4f},{d_e2e:.4f},{d_tput:.4f}\n")
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
