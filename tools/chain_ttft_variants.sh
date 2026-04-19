#!/bin/bash
# Overnight TTFT variant chain: re-validate archive-r2 + IPC sched_overhead
# with the CORRECTED architectural design (synchronous sleep in engine
# thread, not additive in oracle latency), matching Apr 6 commit 4d9983a0c.
#
# Waits for supplementary chain to free the GPU, then runs full 5-rate
# × 2000p emu-validation.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

MASTER_LOG="/tmp/vllm_ttft_variants.log"
echo "=== ttft variants chain start $(date) ===" > "$MASTER_LOG"
touch /tmp/vllm_ttft_variants.started

# Wait for supplementary chain to finish (or timeout 60 min).
echo "  [$(date +%T)] waiting for supplementary chain done..." >> "$MASTER_LOG"
for i in $(seq 1 120); do
    [ -f /tmp/vllm_supp_fullrates.done ] && break
    sleep 30
done

MODEL="Qwen/Qwen3-8B"
PORT=8100
PROFILE="./results/RTX-8000-adaptive-archive-5r/serving-r2.json"

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

run_variant() {
    local NAME="$1"
    local DIR="./results/ttft-variant-${NAME}"
    echo "" >> "$MASTER_LOG"
    echo "=== Variant ${NAME} start $(date) ===" >> "$MASTER_LOG"
    mkdir -p "$DIR"
    for R in 2 4 8 16 32; do
        cp "./results/RTX-8000-v31-2000p/r${R}_real.json" "$DIR/r${R}_real.json" 2>/dev/null || true
    done
    cleanup
    env \
        VLLM_EMULATOR_ENABLE_ORACLE=1 \
        VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
        VLLM_EMULATOR_MODE=realtime \
        VLLM_EMULATOR_EXECUTOR_HOOK=1 \
        VLLM_EMULATOR_PREP_SURROGATE=1 \
        VLLM_EMULATOR_SAMPLE_TRIM="2,98" \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
        > "$DIR/server.log" 2>&1 &
    wait_server || { echo "SERVER_NEVER_READY" >> "$MASTER_LOG"; cleanup; return 1; }
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "$MODEL" --base-url "http://localhost:${PORT}" \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 200 --request-rate 4 > /dev/null 2>&1 || true
    sleep 3
    for RATE in 2 4 8 16 32; do
        echo "  [$(date +%T)] ${NAME} r=${RATE}" >> "$MASTER_LOG"
        timeout 1500 python3 -m vllm.entrypoints.cli.main bench serve \
            --model "$MODEL" --base-url "http://localhost:${PORT}" \
            --dataset-name random --random-input-len 256 --random-output-len 128 \
            --num-prompts 2000 --request-rate $RATE \
            --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
            --save-result --result-dir "$DIR" \
            --result-filename "r${RATE}_emu.json" > /dev/null 2>&1 \
            && echo "    r=${RATE} done" >> "$MASTER_LOG" \
            || echo "    r=${RATE} FAIL" >> "$MASTER_LOG"
        pkill -9 -f "bench serve.*request-rate $RATE " 2>/dev/null || true
        sleep 3
    done
    cleanup
    python3 tools/summarize_matrix.py "$DIR" 2>&1 | tee "$DIR/summary.txt" >> "$MASTER_LOG"
}

# Variant: additive IPC overhead in oracle latency, divided by num_new_reqs
# (burst amortisation). Async chain-timer architecture preserved.
run_variant "v2-additive-divide"

echo "" >> "$MASTER_LOG"
echo "=== ttft variants chain DONE $(date) ===" >> "$MASTER_LOG"
touch /tmp/vllm_ttft_variants.done
