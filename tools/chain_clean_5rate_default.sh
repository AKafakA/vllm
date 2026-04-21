#!/bin/bash
# CLEAN 5-rate baseline with DEFAULT seed and FRESH server per bench.
# Purpose: establish unbiased baseline after discovering two bugs:
#   1) Previous runs reused a single server session for multiple rates →
#      state carry (oracle RNG drift, scheduler internal state) polluted deltas.
#   2) Previous r=16 "variance test" used --seed 1001/1002/1003 for emu but
#      --seed 0 (default) for real → workload mismatch invalidated the comparison.
# This chain fixes both: each (config × rate) cell gets a fresh server + warmup,
# and all benches use --seed 0 (default) so emu and real see identical arrivals.
#
# 15 benches: 3 configs × 5 rates × seed=0, fresh server per bench.
# Runtime estimate: ~2h wall.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

MASTER_LOG="/tmp/vllm_clean_5rate.log"
echo "=== clean 5-rate baseline start $(date) ===" > "$MASTER_LOG"
touch /tmp/vllm_clean_5rate.started

MODEL="Qwen/Qwen3-8B"
PORT=8100
PROFILE="./results/RTX-8000-adaptive-archive-5r/serving-r2.json"
DIR="./results/clean-5rate-default"
mkdir -p "$DIR"

cleanup() {
    pkill -TERM -f "vllm.entrypoints" 2>/dev/null
    sleep 3
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

# Run one bench with fresh server start.
# Args: config (real|hookOn|hookOff), rate
run_bench() {
    local CONFIG=$1
    local RATE=$2
    local LABEL="${CONFIG}_r${RATE}"

    cleanup

    if [ "$CONFIG" = "real" ]; then
        python3 -m vllm.entrypoints.openai.api_server \
            --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
            > "$DIR/server_${LABEL}.log" 2>&1 &
    elif [ "$CONFIG" = "hookOn" ]; then
        env \
            VLLM_EMULATOR_ENABLE_ORACLE=1 \
            VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
            VLLM_EMULATOR_MODE=realtime \
            VLLM_EMULATOR_EXECUTOR_HOOK=1 \
            VLLM_EMULATOR_SCHEDULER_HOOK=1 \
            VLLM_IPC_OVERHEAD_AGG=median \
            VLLM_EMULATOR_PREP_SURROGATE=1 \
        python3 -m vllm.entrypoints.openai.api_server \
            --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
            > "$DIR/server_${LABEL}.log" 2>&1 &
    elif [ "$CONFIG" = "hookOff" ]; then
        env \
            VLLM_EMULATOR_ENABLE_ORACLE=1 \
            VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
            VLLM_EMULATOR_MODE=realtime \
            VLLM_EMULATOR_EXECUTOR_HOOK=1 \
            VLLM_EMULATOR_SCHEDULER_HOOK=0 \
            VLLM_IPC_OVERHEAD_AGG=median \
            VLLM_EMULATOR_PREP_SURROGATE=1 \
        python3 -m vllm.entrypoints.openai.api_server \
            --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
            > "$DIR/server_${LABEL}.log" 2>&1 &
    fi

    if ! wait_server; then
        echo "    SERVER FAIL ${LABEL}" >> "$MASTER_LOG"
        cleanup
        return 1
    fi

    # Warmup — short, deterministic.
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "$MODEL" --base-url "http://localhost:${PORT}" \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 100 --request-rate 4 --seed 0 > /dev/null 2>&1 || true
    sleep 2

    # Actual bench: 2000 prompts, default seed (no --seed flag → seed=0).
    echo "  [$(date +%T)] bench ${LABEL}" >> "$MASTER_LOG"
    timeout 1800 python3 -m vllm.entrypoints.cli.main bench serve \
        --model "$MODEL" --base-url "http://localhost:${PORT}" \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 2000 --request-rate $RATE \
        --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
        --save-result --result-dir "$DIR" \
        --result-filename "${LABEL}.json" > /dev/null 2>&1 \
        && echo "    ${LABEL} done" >> "$MASTER_LOG" \
        || echo "    ${LABEL} FAIL" >> "$MASTER_LOG"

    cleanup
}

# 15 benches: 3 configs × 5 rates, fresh server per bench, default seed.
for CONFIG in real hookOn hookOff; do
    for RATE in 2 4 8 16 32; do
        run_bench "$CONFIG" "$RATE"
    done
done

echo "=== clean 5-rate baseline DONE $(date) ===" >> "$MASTER_LOG"
touch /tmp/vllm_clean_5rate.done
