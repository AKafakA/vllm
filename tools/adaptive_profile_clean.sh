#!/bin/bash
# Fresh-server-per-rate adaptive step-cycle profile builder (clean mode).
#
# Eliminates the state-carry confound present in adaptive_profile_archive_5r.sh:
# per-rate cleanup → start server (trace on) → common_warmup → bench at rate → cleanup.
# Every rate's step-cycle samples are collected on a server that has just been
# warmed identically — same warmup as validation benches source from _bench_common.sh.
# Profile↔bench state matches; bench deltas no longer include methodology drift.
#
# Per-rate traces are concatenated into one step_cycle_trace.jsonl at the end,
# then build_serving_profile_filtered.py builds the profile pack.
#
# Runtime impact vs archive-5r single-session:
#   added per-rate restart ~90s × 14 rates × 5 rounds ≈ 105 min added.
#   expected total ~5-6h (vs archive-5r's 3-4h).
#
# Usage:
#   ROUNDS=5 TAG=apr20-clean ./adaptive_profile_clean.sh
#   ROUNDS=1 TAG=apr20-clean-1r ./adaptive_profile_clean.sh  (fast dev iteration)
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

source "$(dirname "$0")/_bench_common.sh"

TAG="${TAG:-apr20-clean}"
ROUNDS="${ROUNDS:-5}"
# Archive's proven rate + prompt list (from adaptive_profile_archive_5r.sh).
RATES_AND_PROMPTS="1:1000 2:2000 3:2000 4:2000 6:2000 8:2000 10:1500 12:1500 16:1500 20:1000 24:1000 32:1000 inf:500"

OUT_DIR="./results/RTX-8000-adaptive-${TAG}"
TRACE_DIR="$OUT_DIR/per_rate_traces"
FINAL_TRACE="$OUT_DIR/step_cycle_trace.jsonl"
PROFILE="$OUT_DIR/serving-full.json"
LOG="$OUT_DIR/run.log"
MARKER="/tmp/vllm_adaptive_${TAG}"

mkdir -p "$TRACE_DIR" "$OUT_DIR/logs"
rm -f "$FINAL_TRACE"
touch "${MARKER}.started"
echo "=== adaptive-clean tag=$TAG rounds=$ROUNDS start $(date) ===" > "$LOG"
echo "Warmup: " >> "$LOG"
common_warmup_config_json >> "$LOG"
echo "" >> "$LOG"

for ROUND in $(seq 1 $ROUNDS); do
    echo "" >> "$LOG"
    echo "=== ROUND $ROUND / $ROUNDS at $(date) ===" >> "$LOG"

    for entry in $RATES_AND_PROMPTS; do
        RATE=$(echo "$entry" | cut -d: -f1)
        PROMPTS=$(echo "$entry" | cut -d: -f2)
        PER_TRACE="$TRACE_DIR/round${ROUND}_r${RATE}.jsonl"

        echo "  [$(date +%T)] round=$ROUND rate=$RATE prompts=$PROMPTS" >> "$LOG"
        common_cleanup
        if ! common_preflight; then
            echo "    preflight FAIL — aborting round" >> "$LOG"
            break
        fi

        # Fresh server with step trace enabled → fresh state per rate.
        rm -f "$PER_TRACE"
        env \
            VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
            VLLM_EMULATOR_STEP_TRACE_OUTPUT="$PER_TRACE" \
        python3 -m vllm.entrypoints.openai.api_server \
            --model "$BENCH_MODEL" --max-model-len "$BENCH_MAX_MODEL_LEN" \
            --port "$BENCH_PORT" --trust-remote-code \
            > "$OUT_DIR/logs/server_round${ROUND}_r${RATE}.log" 2>&1 &
        if ! common_wait_server; then
            echo "    server FAIL" >> "$LOG"
            common_cleanup
            continue
        fi

        common_warmup

        # Profiling bench at this rate. Markers bracket the section of the
        # trace that is actually the profiling workload (vs warmup leftovers).
        echo '{"__marker__": "profiling_start"}' >> "$PER_TRACE"
        python3 -m vllm.entrypoints.cli.main bench serve \
            --model "$BENCH_MODEL" --base-url "http://localhost:${BENCH_PORT}" \
            --dataset-name random --random-input-len 256 --random-output-len 128 \
            --num-prompts "$PROMPTS" --request-rate "$RATE" \
            --seed 0 > /dev/null 2>&1 || true
        echo '{"__marker__": "profiling_stop"}' >> "$PER_TRACE"

        common_cleanup
        if [ -f "$PER_TRACE" ]; then
            cat "$PER_TRACE" >> "$FINAL_TRACE"
            echo "    lines=$(wc -l < "$PER_TRACE")" >> "$LOG"
        else
            echo "    trace MISSING" >> "$LOG"
        fi
    done
done

echo "" >> "$LOG"
echo "=== Building profile $(date) ===" >> "$LOG"
python3 vllm_emulator/profile/build_serving_profile_filtered.py \
    "$FINAL_TRACE" "$PROFILE" \
    --tt-bucket-width 1 --conc-bucket-width 5 >> "$LOG" 2>&1

echo "=== DONE $(date) ===" >> "$LOG"
touch "${MARKER}.done"
