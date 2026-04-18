#!/bin/bash
# v5 one-round single-session profiler. Tests whether archive's per-bucket
# heavy tails (missing in v4) come from sustained-state accumulation within
# a single long-lived server, as opposed to v3/v4's pattern of restarting
# the server between rounds.
#
# Structure matches archive (adaptive_profiling.sh): one server, one
# warmup, one full rate sweep, one pass of variable shapes. If this
# produces archive-matching p90/p95 tails, the next iteration scales
# to 5 rounds inside the SAME server session (no restarts) for more
# sample density.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

MODEL="Qwen/Qwen3-8B"
PORT=8100
OUT_DIR="./results/RTX-8000-adaptive-v5-oneround"
TRACE="$OUT_DIR/step_cycle_trace.jsonl"
PROFILE="$OUT_DIR/serving-full.json"

mkdir -p "$OUT_DIR/logs"
rm -f "$TRACE"
touch "$OUT_DIR/.started"

LOG="$OUT_DIR/run.log"
echo "=== Adaptive profile v5 (one round, single session) start $(date) ===" > "$LOG"

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
echo "  [$(date +%T)] server ready (one-session, no restarts)" >> "$LOG"

# Standard warmup (only warmup phase - no CUDA sweep, no burst).
echo "  [$(date +%T)] standard warmup (200 @ rate=4)..." >> "$LOG"
python3 -m vllm.entrypoints.cli.main bench serve \
    --model "$MODEL" --base-url "http://localhost:${PORT}" \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 --request-rate 4 > /dev/null 2>&1 || true

# Begin profile window.
echo '{"__marker__": "profiling_start"}' >> "$TRACE"

# Rate sweep: matches v4's 14-rate list, with a generous per-rate size
# chosen to roughly match archive's total workload (archive used
# ~10-20k total prompts across 13 rates).
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

# Variable-shape workloads (matches v4).
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
cleanup

# Build profile.
echo "" >> "$LOG"
echo "=== Building profile $(date) ===" >> "$LOG"
python3 vllm_emulator/profile/build_serving_profile_filtered.py \
    "$TRACE" "$PROFILE" \
    --tt-bucket-width 1 --conc-bucket-width 5 >> "$LOG" 2>&1

echo "=== DONE $(date) ===" >> "$LOG"
touch "$OUT_DIR/.all_done"
touch "/tmp/vllm_v5_oneround.done"
