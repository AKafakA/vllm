#!/bin/bash
# Profile A: 2 rounds × single session × fixed-mixed shapes
#   {256/128, 128/64, 512/256} in equal proportion.
# Replaces archive-r2's single-shape recipe with a mixed-shape one.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

MODEL="Qwen/Qwen3-8B"
PORT=8100
OUT_DIR="./results/RTX-8000-profile-fixedmix-2r"
TRACE="$OUT_DIR/step_cycle_trace.jsonl"
PROFILE="$OUT_DIR/serving-full.json"
MAX_ROUNDS=2

mkdir -p "$OUT_DIR/logs"
rm -f "$TRACE"

LOG="$OUT_DIR/run.log"
echo "=== fixedmix-2r start $(date) ===" > "$LOG"

cleanup() {
    pkill -9 -f "VLLM::EngineCore" 2>/dev/null
    pkill -9 -f "vllm.entrypoints" 2>/dev/null
    pkill -9 -f "bench serve" 2>/dev/null
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

# Archive's rate+prompts pattern, per shape (total prompts split across shapes).
RATES_AND_PROMPTS="
1:400
2:700
3:700
4:700
6:700
8:700
10:500
12:500
16:500
24:400
32:400
inf:200
"

cleanup
env \
    VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
    VLLM_EMULATOR_STEP_TRACE_OUTPUT="$TRACE" \
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > "$OUT_DIR/logs/server.log" 2>&1 &
wait_server || { echo "SERVER_NEVER_READY" >> "$LOG"; cleanup; exit 1; }
echo "  [$(date +%T)] server ready" >> "$LOG"

# Warmup (mixed shapes, small).
for SHAPE_IN_OUT in "256 128" "128 64" "512 256"; do
    IN=$(echo $SHAPE_IN_OUT | cut -d' ' -f1)
    OUT=$(echo $SHAPE_IN_OUT | cut -d' ' -f2)
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "$MODEL" --base-url "http://localhost:${PORT}" \
        --dataset-name random --random-input-len $IN --random-output-len $OUT \
        --num-prompts 100 --request-rate 4 > /dev/null 2>&1 || true
done
sleep 3

for ROUND in $(seq 1 $MAX_ROUNDS); do
    echo "" >> "$LOG"
    echo "=== ROUND $ROUND / $MAX_ROUNDS at $(date) ===" >> "$LOG"
    echo '{"__marker__": "profiling_start"}' >> "$TRACE"

    for SHAPE_IN_OUT in "256 128" "128 64" "512 256"; do
        IN=$(echo $SHAPE_IN_OUT | cut -d' ' -f1)
        OUT=$(echo $SHAPE_IN_OUT | cut -d' ' -f2)
        echo "  [$(date +%T)] round $ROUND shape ${IN}/${OUT}" >> "$LOG"
        for entry in $RATES_AND_PROMPTS; do
            RATE=$(echo "$entry" | cut -d: -f1)
            PROMPTS=$(echo "$entry" | cut -d: -f2)
            python3 -m vllm.entrypoints.cli.main bench serve \
                --model "$MODEL" --base-url "http://localhost:${PORT}" \
                --dataset-name random --random-input-len $IN --random-output-len $OUT \
                --num-prompts $PROMPTS --request-rate $RATE > /dev/null 2>&1 || true
            sleep 1
        done
    done

    echo '{"__marker__": "profiling_stop"}' >> "$TRACE"
    echo "  [$(date +%T)] round $ROUND done" >> "$LOG"
done

cleanup
echo "" >> "$LOG"
echo "=== Building profile $(date) ===" >> "$LOG"
python3 vllm_emulator/profile/build_serving_profile_filtered.py \
    "$TRACE" "$PROFILE" \
    --tt-bucket-width 1 --conc-bucket-width 5 >> "$LOG" 2>&1

# Carry sched_overhead_table across (same hardware/model).
python3 - << PYEOF >> "$LOG" 2>&1
import json
src = json.load(open("./results/RTX-8000-adaptive-archive-5r/serving-r2.json"))
dst = json.load(open("$PROFILE"))
if "sched_overhead_table" in src:
    dst["sched_overhead_table"] = src["sched_overhead_table"]
if "sched_overhead_table_v2" in src:
    dst["sched_overhead_table_v2"] = src["sched_overhead_table_v2"]
json.dump(dst, open("$PROFILE", "w"), indent=2)
print("Carried sched_overhead_table forward from archive-r2")
PYEOF

echo "=== DONE $(date) ===" >> "$LOG"
touch "$OUT_DIR/.all_done"
touch /tmp/vllm_profile_fixedmix.done
