#!/bin/bash
# Fresh-server-per-rate adaptive step-cycle profile — v2 rate list.
#
# Motivation: inspection of the apr20-clean profile showed conc 112-237
# is sparse (3-18 samples per bucket). Root cause: batching efficiency at
# rate=16+ produces ~20× fewer trace lines per prompt than rate=1, so the
# transit-region concurrency buckets are undersampled.
#
# v2 rate list: adds rates 14, 18 (dwell in conc 100-220) and bumps prompts
# at rates 10, 12, 16 from 1500 → 3000 to multiply transit-region samples.
# Runtime +~30 min vs v1 (~3.5h total).
#
# Usage:
#   ROUNDS=2 TAG=apr21-dense ./adaptive_profile_clean_v2.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

source "$(dirname "$0")/_bench_common.sh"

TAG="${TAG:-apr21-dense}"
ROUNDS="${ROUNDS:-2}"

# v2 rebalanced list. Added rates 14, 18. Bumped 10/12/16 to 3000.
RATES_AND_PROMPTS="1:1000 2:2000 3:2000 4:2000 6:2000 8:2000 10:3000 12:3000 14:3000 16:3000 18:3000 20:2000 24:2000 32:1000 inf:500"

OUT_DIR="./results/RTX-8000-adaptive-${TAG}"
TRACE_DIR="$OUT_DIR/per_rate_traces"
FINAL_TRACE="$OUT_DIR/step_cycle_trace.jsonl"
PROFILE="$OUT_DIR/serving-full.json"
LOG="$OUT_DIR/run.log"
MARKER="/tmp/vllm_adaptive_${TAG}"

mkdir -p "$TRACE_DIR" "$OUT_DIR/logs"
rm -f "$FINAL_TRACE"
touch "${MARKER}.started"
echo "=== adaptive-clean-v2 tag=$TAG rounds=$ROUNDS start $(date) ===" > "$LOG"
echo "Warmup:" >> "$LOG"
common_warmup_config_json >> "$LOG"
echo "" >> "$LOG"
echo "Rate list: $RATES_AND_PROMPTS" >> "$LOG"
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
