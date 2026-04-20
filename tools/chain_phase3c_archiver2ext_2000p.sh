#!/bin/bash
# Corrective rerun: archiver2ext × {random, sharegpt} at 2000p × 3 rates.
# Original Phase 3b used 1500p to save time, violating the always-2000p
# debug rule. This rerun gives apples-to-apples vs archive-r2 baseline.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

MASTER_LOG="/tmp/vllm_apr20_phase3c.log"
echo "=== phase3c archiver2ext @ 2000p start $(date) ===" > "$MASTER_LOG"
touch /tmp/vllm_apr20_phase3c.started

MODEL="Qwen/Qwen3-8B"
PORT=8100
SHAREGPT="./results/sharegpt_filtered_256_128.json"
PROFILE="./results/RTX-8000-profile-archiver2ext-2r/serving-full.json"

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

for WORKLOAD in random sharegpt; do
    CELL_DIR="./results/workload-emu-archiver2ext-${WORKLOAD}-2000p"
    mkdir -p "$CELL_DIR"
    if [ "$WORKLOAD" = "random" ]; then
        for R in 2 8 32; do
            cp "./results/workload-emu-archive-r2-random/r${R}_real.json" \
                "$CELL_DIR/r${R}_real.json" 2>/dev/null || true
        done
    else
        for R in 2 8 32; do
            cp "./results/workload-real-sharegpt/r${R}_real.json" \
                "$CELL_DIR/r${R}_real.json" 2>/dev/null || true
        done
    fi

    echo "  [$(date +%T)] 2000p cell: archiver2ext × $WORKLOAD" >> "$MASTER_LOG"
    cleanup
    env \
        VLLM_EMULATOR_ENABLE_ORACLE=1 \
        VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
        VLLM_EMULATOR_MODE=realtime \
        VLLM_EMULATOR_EXECUTOR_HOOK=1 \
        VLLM_EMULATOR_SCHEDULER_HOOK=1 \
        VLLM_IPC_OVERHEAD_AGG=median \
        VLLM_EMULATOR_PREP_SURROGATE=1 \
        VLLM_EMULATOR_SAMPLE_TRIM="0,100" \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
        > "$CELL_DIR/server.log" 2>&1 &
    wait_server || { echo "    server fail $WORKLOAD" >> "$MASTER_LOG"; cleanup; continue; }

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

    for RATE in 2 8 32; do
        if [ "$WORKLOAD" = "random" ]; then
            DS_ARGS="--dataset-name random --random-input-len 256 --random-output-len 128"
        else
            DS_ARGS="--dataset-name sharegpt --dataset-path $SHAREGPT"
        fi
        timeout 1500 python3 -m vllm.entrypoints.cli.main bench serve \
            --model "$MODEL" --base-url "http://localhost:${PORT}" \
            $DS_ARGS \
            --num-prompts 2000 --request-rate $RATE \
            --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
            --save-result --result-dir "$CELL_DIR" \
            --result-filename "r${RATE}_emu.json" > /dev/null 2>&1 \
            && echo "    r=${RATE} done" >> "$MASTER_LOG" \
            || echo "    r=${RATE} FAIL" >> "$MASTER_LOG"
        sleep 2
    done
    cleanup
done

echo "=== phase3c DONE $(date) ===" >> "$MASTER_LOG"
touch /tmp/vllm_apr20_phase3c.done
