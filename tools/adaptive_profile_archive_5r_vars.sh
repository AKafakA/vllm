#!/bin/bash
# Archive's recipe WITH v6's variable shapes added per round. 5 rounds,
# single session, otherwise identical to archive_5r. Isolates variable-
# shape effect: compare emu accuracy vs archive_5r (no vars).
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

MODEL="Qwen/Qwen3-8B"
PORT=8100
OUT_DIR="./results/RTX-8000-adaptive-archive-5r-vars"
TRACE="$OUT_DIR/step_cycle_trace.jsonl"
PROFILE="$OUT_DIR/serving-full.json"
MAX_ROUNDS=5

mkdir -p "$OUT_DIR/logs"
rm -f "$TRACE"

LOG="$OUT_DIR/run.log"
echo "=== archive-5r-vars start $(date) ===" > "$LOG"

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

# Archive's rate + prompt list
RATES_AND_PROMPTS="
1:1000
2:2000
3:2000
4:2000
6:2000
8:2000
10:1500
12:1500
16:1500
20:1000
24:1000
32:1000
inf:500
"

cleanup
env \
    VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
    VLLM_EMULATOR_STEP_TRACE_OUTPUT="$TRACE" \
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > "$OUT_DIR/logs/server.log" 2>&1 &
if ! wait_server; then
    echo "SERVER_NEVER_READY" >> "$LOG"; cleanup; exit 1
fi
echo "  [$(date +%T)] server ready" >> "$LOG"

python3 -m vllm.entrypoints.cli.main bench serve \
    --model "$MODEL" --base-url "http://localhost:${PORT}" \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 500 --request-rate 4 > /dev/null 2>&1 || true
sleep 3

for ROUND in $(seq 1 $MAX_ROUNDS); do
    echo "" >> "$LOG"
    echo "=== ROUND $ROUND / $MAX_ROUNDS at $(date) ===" >> "$LOG"
    echo '{"__marker__": "profiling_start"}' >> "$TRACE"

    for entry in $RATES_AND_PROMPTS; do
        RATE=$(echo "$entry" | cut -d: -f1)
        PROMPTS=$(echo "$entry" | cut -d: -f2)
        echo "  [$(date +%T)] r=${RATE} n=${PROMPTS}" >> "$LOG"
        python3 -m vllm.entrypoints.cli.main bench serve \
            --model "$MODEL" --base-url "http://localhost:${PORT}" \
            --dataset-name random --random-input-len 256 --random-output-len 128 \
            --num-prompts $PROMPTS --request-rate $RATE > /dev/null 2>&1 || true
        sleep 2
    done

    # v6's variable-shape benches (the added variable)
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
    echo "  [$(date +%T)] round $ROUND done" >> "$LOG"
done

cleanup
echo "" >> "$LOG"
echo "=== Building profile $(date) ===" >> "$LOG"
python3 vllm_emulator/profile/build_serving_profile_filtered.py \
    "$TRACE" "$PROFILE" \
    --tt-bucket-width 1 --conc-bucket-width 5 >> "$LOG" 2>&1

echo "=== DONE $(date) ===" >> "$LOG"
touch "$OUT_DIR/.all_done"
touch /tmp/vllm_archive_5r_vars.done
