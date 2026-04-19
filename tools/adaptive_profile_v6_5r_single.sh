#!/bin/bash
# v6: single-session 5-round adaptive profiling. Starts ONE server, keeps it
# alive across all rounds, emits profiling_start/stop markers around each
# round's workload. Archive's single-session methodology + v4's 5-round
# density. Expected runtime: 3.5-4h. Output: ~300k+ raw records.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

MODEL="Qwen/Qwen3-8B"
PORT=8100
OUT_DIR="./results/RTX-8000-adaptive-v6-5r-single"
TRACE="$OUT_DIR/step_cycle_trace.jsonl"
PROFILE="$OUT_DIR/serving-full.json"
MAX_ROUNDS=5

mkdir -p "$OUT_DIR/logs"
rm -f "$TRACE"
touch "$OUT_DIR/.started"

LOG="$OUT_DIR/run.log"
echo "=== v6 single-session 5-round start $(date) ===" > "$LOG"

cleanup() {
    pkill -9 -f "VLLM::EngineCore" 2>/dev/null
    pkill -9 -f "vllm.entrypoints" 2>/dev/null
    fuser ${PORT}/tcp 2>/dev/null | xargs -r kill -9 2>/dev/null
    sleep 5
}
wait_server() {
    for i in $(seq 1 300); do
        curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1 && return 0
        sleep 1
    done
    return 1
}

RATES="1 2 3 4 5 6 8 10 12 16 24 32 0.5 inf"

get_num_prompts() {
    local rate=$1 round=$2
    local base=300
    [[ "$round" -gt 1 ]] && base=500
    [[ "$round" -gt 3 ]] && base=800
    case $rate in
        0.5) echo $((base / 3)) ;;
        *)   echo $base ;;
    esac
}

# Launch server ONCE for the entire 5-round run.
cleanup
echo "  [$(date +%T)] starting single server session for all $MAX_ROUNDS rounds" >> "$LOG"
env \
    VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
    VLLM_EMULATOR_STEP_TRACE_OUTPUT="$TRACE" \
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > "$OUT_DIR/logs/server.log" 2>&1 &
if ! wait_server; then
    echo "SERVER_NEVER_READY" >> "$LOG"
    touch "$OUT_DIR/.failed"
    cleanup
    exit 1
fi
echo "  [$(date +%T)] server ready" >> "$LOG"

# Single warmup at the top. Subsequent rounds start with a hot server.
echo "  [$(date +%T)] warmup 200 @ rate=4..." >> "$LOG"
python3 -m vllm.entrypoints.cli.main bench serve \
    --model "$MODEL" --base-url "http://localhost:${PORT}" \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 --request-rate 4 > /dev/null 2>&1 || true

for ROUND in $(seq 1 $MAX_ROUNDS); do
    echo "" >> "$LOG"
    echo "=== ROUND $ROUND / $MAX_ROUNDS at $(date) ===" >> "$LOG"
    touch "$OUT_DIR/.round_${ROUND}_started"

    # Begin profile window for this round.
    echo '{"__marker__": "profiling_start"}' >> "$TRACE"

    for rate in $RATES; do
        NP=$(get_num_prompts $rate $ROUND)
        echo "  [$(date +%T)] r=$rate n=$NP" >> "$LOG"
        python3 -m vllm.entrypoints.cli.main bench serve \
            --model "$MODEL" --base-url "http://localhost:${PORT}" \
            --dataset-name random --random-input-len 256 --random-output-len 128 \
            --num-prompts $NP --request-rate $rate > /dev/null 2>&1 || true
    done

    # Variable-shape workloads.
    echo "  [$(date +%T)] variable 64/32..." >> "$LOG"
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "$MODEL" --base-url "http://localhost:${PORT}" \
        --dataset-name random --random-input-len 64 --random-output-len 32 \
        --num-prompts 300 --request-rate 8 > /dev/null 2>&1 || true

    echo "  [$(date +%T)] variable 512/256..." >> "$LOG"
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "$MODEL" --base-url "http://localhost:${PORT}" \
        --dataset-name random --random-input-len 512 --random-output-len 256 \
        --num-prompts 200 --request-rate 4 > /dev/null 2>&1 || true

    echo "  [$(date +%T)] variable 128/64..." >> "$LOG"
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "$MODEL" --base-url "http://localhost:${PORT}" \
        --dataset-name random --random-input-len 128 --random-output-len 64 \
        --num-prompts 300 --request-rate 16 > /dev/null 2>&1 || true

    echo '{"__marker__": "profiling_stop"}' >> "$TRACE"
    touch "$OUT_DIR/.round_${ROUND}_done"

    TOTAL=$(wc -l < "$TRACE" 2>/dev/null || echo 0)
    echo "  [$(date +%T)] round $ROUND done, total records: $TOTAL" >> "$LOG"
done

# Only cleanup at the end.
cleanup

echo "" >> "$LOG"
echo "=== Building profile $(date) ===" >> "$LOG"
python3 vllm_emulator/profile/build_serving_profile_filtered.py \
    "$TRACE" "$PROFILE" \
    --tt-bucket-width 1 --conc-bucket-width 5 >> "$LOG" 2>&1

echo "" >> "$LOG"
echo "=== DONE $(date) ===" >> "$LOG"
touch "$OUT_DIR/.all_done"
touch /tmp/vllm_v6_profile.done
