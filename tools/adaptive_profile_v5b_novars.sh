#!/bin/bash
# v5b: identical to v5_oneround EXCEPT variable-shape workloads are
# removed. Single server, single rate sweep at 256/128 only, no CUDA
# warmup sweep, no high-conc burst, no variable shapes. This is the
# control for isolating whether variable shapes explain the residual
# v4/v5 bias against archive.
#
# If v5b matches archive better than v5 → variable shapes ARE the cause.
# If v5b ≈ v5 → variable shapes are NOT the cause; the bias is something
# else (e.g. rate-sweep rate list, per-rate prompt counts, etc.).
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

MODEL="Qwen/Qwen3-8B"
PORT=8100
OUT_DIR="./results/RTX-8000-adaptive-v5b-novars"
TRACE="$OUT_DIR/step_cycle_trace.jsonl"
PROFILE="$OUT_DIR/serving-full.json"

mkdir -p "$OUT_DIR/logs"
rm -f "$TRACE"
touch "$OUT_DIR/.started"

LOG="$OUT_DIR/run.log"
echo "=== Adaptive profile v5b (no variable shapes) start $(date) ===" > "$LOG"

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

cleanup
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

echo "  [$(date +%T)] standard warmup (200 @ rate=4)..." >> "$LOG"
python3 -m vllm.entrypoints.cli.main bench serve \
    --model "$MODEL" --base-url "http://localhost:${PORT}" \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 --request-rate 4 > /dev/null 2>&1 || true

echo '{"__marker__": "profiling_start"}' >> "$TRACE"

RATES="1 2 3 4 5 6 8 10 12 16 24 32 0.5 inf"
get_num_prompts() {
    local rate=$1
    case $rate in
        0.5) echo 200 ;;
        1)   echo 500 ;;
        inf) echo 500 ;;
        *)   echo 800 ;;
    esac
}

for rate in $RATES; do
    NP=$(get_num_prompts $rate)
    echo "  [$(date +%T)] r=$rate n=$NP" >> "$LOG"
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "$MODEL" --base-url "http://localhost:${PORT}" \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts $NP --request-rate $rate > /dev/null 2>&1 || true
done

# NO variable-shape workloads — this is the control run.

echo '{"__marker__": "profiling_stop"}' >> "$TRACE"
cleanup

echo "" >> "$LOG"
echo "=== Building profile $(date) ===" >> "$LOG"
python3 vllm_emulator/profile/build_serving_profile_filtered.py \
    "$TRACE" "$PROFILE" \
    --tt-bucket-width 1 --conc-bucket-width 5 >> "$LOG" 2>&1

echo "=== DONE $(date) ===" >> "$LOG"
touch "$OUT_DIR/.all_done"
touch "/tmp/vllm_v5b_novars.done"
