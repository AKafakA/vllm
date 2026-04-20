#!/bin/bash
# Phase 3 — sharegpt-first ordering. All 4 profiles × sharegpt × 5 rates ×
# 2000p FIRST (the decisive axis — shape comparison shows most here).
# Then all 4 profiles × random × 5 rates × 2000p.
#
# Split so that if we interrupt partway, we still have the sharegpt matrix
# (the answer to "does shape variation help dynamic workload").
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

MASTER_LOG="/tmp/vllm_apr20_matrix_sharegpt_first.log"
echo "=== sharegpt-first matrix start $(date) ===" > "$MASTER_LOG"
touch /tmp/vllm_apr20_matrix_sharegpt_first.started

MODEL="Qwen/Qwen3-8B"
PORT=8100
SHAREGPT="./results/sharegpt_filtered_256_128.json"

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

declare -A PROFILES
PROFILES[archive-r2]="./results/RTX-8000-adaptive-archive-5r/serving-r2.json"
PROFILES[fixedmix]="./results/RTX-8000-profile-fixedmix-2r/serving-full.json"
PROFILES[shareptsampled]="./results/RTX-8000-profile-shareptsampled-2r/serving-full.json"
PROFILES[archiver2ext]="./results/RTX-8000-profile-archiver2ext-2r/serving-full.json"

run_cell() {
    local PROF_KEY=$1
    local WORKLOAD=$2
    local PROF_PATH="${PROFILES[$PROF_KEY]}"
    if [ ! -f "$PROF_PATH" ]; then
        echo "    SKIP $PROF_KEY (missing profile)" >> "$MASTER_LOG"
        return
    fi
    local CELL_DIR="./results/workload-emu-${PROF_KEY}-${WORKLOAD}-2000p5r"
    mkdir -p "$CELL_DIR"
    if [ "$WORKLOAD" = "random" ]; then
        for R in 2 4 8 16 32; do
            cp "./results/ttft-variant-v3-arrival-delay/r${R}_real.json" \
                "$CELL_DIR/r${R}_real.json" 2>/dev/null || true
        done
    else
        for R in 2 4 8 16 32; do
            cp "./results/workload-real-sharegpt/r${R}_real.json" \
                "$CELL_DIR/r${R}_real.json" 2>/dev/null || true
        done
    fi

    echo "  [$(date +%T)] cell: $PROF_KEY × $WORKLOAD" >> "$MASTER_LOG"
    cleanup
    env \
        VLLM_EMULATOR_ENABLE_ORACLE=1 \
        VLLM_EMULATOR_PROFILE_PACK="$PROF_PATH" \
        VLLM_EMULATOR_MODE=realtime \
        VLLM_EMULATOR_EXECUTOR_HOOK=1 \
        VLLM_EMULATOR_SCHEDULER_HOOK=1 \
        VLLM_IPC_OVERHEAD_AGG=median \
        VLLM_EMULATOR_PREP_SURROGATE=1 \
        VLLM_EMULATOR_SAMPLE_TRIM="0,100" \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
        > "$CELL_DIR/server.log" 2>&1 &
    wait_server || { echo "    server fail $PROF_KEY $WORKLOAD" >> "$MASTER_LOG"; cleanup; return; }

    if [ "$WORKLOAD" = "random" ]; then
        python3 -m vllm.entrypoints.cli.main bench serve \
            --model "$MODEL" --base-url "http://localhost:${PORT}" \
            --dataset-name random --random-input-len 256 --random-output-len 128 \
            --num-prompts 100 --request-rate 4 > /dev/null 2>&1 || true
    else
        python3 -m vllm.entrypoints.cli.main bench serve \
            --model "$MODEL" --base-url "http://localhost:${PORT}" \
            --dataset-name sharegpt --dataset-path "$SHAREGPT" \
            --num-prompts 100 --request-rate 4 > /dev/null 2>&1 || true
    fi
    sleep 2

    for RATE in 2 4 8 16 32; do
        if [ "$WORKLOAD" = "random" ]; then
            DS_ARGS="--dataset-name random --random-input-len 256 --random-output-len 128"
        else
            DS_ARGS="--dataset-name sharegpt --dataset-path $SHAREGPT"
        fi
        echo "    [$(date +%T)] r=$RATE" >> "$MASTER_LOG"
        timeout 1800 python3 -m vllm.entrypoints.cli.main bench serve \
            --model "$MODEL" --base-url "http://localhost:${PORT}" \
            $DS_ARGS \
            --num-prompts 2000 --request-rate $RATE \
            --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
            --save-result --result-dir "$CELL_DIR" \
            --result-filename "r${RATE}_emu.json" > /dev/null 2>&1 \
            && echo "      r=${RATE} done" >> "$MASTER_LOG" \
            || echo "      r=${RATE} FAIL" >> "$MASTER_LOG"
        sleep 2
    done
    cleanup
}

# === SHAREGPT PASS FIRST (decisive axis) ===
echo "" >> "$MASTER_LOG"
echo "=== PASS 1: sharegpt $(date) ===" >> "$MASTER_LOG"
for PROF_KEY in archive-r2 fixedmix shareptsampled archiver2ext; do
    run_cell "$PROF_KEY" "sharegpt"
done
touch /tmp/vllm_apr20_sharegpt_pass.done

# === RANDOM PASS SECOND ===
echo "" >> "$MASTER_LOG"
echo "=== PASS 2: random $(date) ===" >> "$MASTER_LOG"
for PROF_KEY in archive-r2 fixedmix shareptsampled archiver2ext; do
    run_cell "$PROF_KEY" "random"
done
touch /tmp/vllm_apr20_random_pass.done

echo "=== matrix sharegpt-first DONE $(date) ===" >> "$MASTER_LOG"
touch /tmp/vllm_apr20_full_matrix.done
