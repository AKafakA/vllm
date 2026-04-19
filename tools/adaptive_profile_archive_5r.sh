#!/bin/bash
# Archive's recipe (no variable shapes, archive's rate+prompt list) wrapped
# in 5 rounds, single session. Tests whether archive's proven methodology
# scales gracefully with more rounds, or replicates v6's degradation.
#
# Comparison points:
#   archive (1 round, ~108k samples)  → reference
#   this   (5 rounds, ~500k samples)  → "archive scaled up"
#   v6     (5 rounds, ~309k samples)  → current failing baseline with vars
#
# Expected runtime: 3-4 hours.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

MODEL="Qwen/Qwen3-8B"
PORT=8100
OUT_DIR="./results/RTX-8000-adaptive-archive-5r"
TRACE="$OUT_DIR/step_cycle_trace.jsonl"
PROFILE="$OUT_DIR/serving-full.json"
MAX_ROUNDS=5

mkdir -p "$OUT_DIR/logs"
rm -f "$TRACE"

LOG="$OUT_DIR/run.log"
echo "=== archive-5r single-session start $(date) ===" > "$LOG"

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

# Archive's rate + prompt list, exactly
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
echo "  [$(date +%T)] starting single server session for $MAX_ROUNDS rounds" >> "$LOG"
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

# Archive warmup: 500 prompts @ rate=4
echo "  [$(date +%T)] warmup (500 @ rate=4)" >> "$LOG"
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

    echo '{"__marker__": "profiling_stop"}' >> "$TRACE"
    TOTAL=$(wc -l < "$TRACE")
    echo "  [$(date +%T)] round $ROUND done, total records: $TOTAL" >> "$LOG"
done

cleanup

echo "" >> "$LOG"
echo "=== Building profile $(date) ===" >> "$LOG"
python3 vllm_emulator/profile/build_serving_profile_filtered.py \
    "$TRACE" "$PROFILE" \
    --tt-bucket-width 1 --conc-bucket-width 5 >> "$LOG" 2>&1

echo "=== DONE $(date) ===" >> "$LOG"
touch "$OUT_DIR/.all_done"
touch /tmp/vllm_archive_5r.done
