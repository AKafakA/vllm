#!/bin/bash
# A/B test reservoir-capped v6 profile vs full v6 vs archive at r=2,8 × 500 prompts.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

MODEL="Qwen/Qwen3-8B"
PORT=8100
RATES="2 8"
NUM_PROMPTS=500

V6_FULL="./results/RTX-8000-adaptive-v6-5r-single/serving-full.json"
V6_RES="./results/overnight_v6_reservoir7009_profile.json"
MASTER_LOG="/tmp/vllm_reservoir.log"
echo "=== reservoir A/B start $(date) ===" > "$MASTER_LOG"

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

run_pass() {
    local PROF="$1"
    local TAG="$2"
    local DIR="./results/RTX-8000-reservoir-${TAG}-apr19"
    local LOG="$DIR/run.log"
    mkdir -p "$DIR"
    echo "=== PASS ${TAG} start $(date) ===" > "$LOG"
    for R in $RATES; do
        cp "./results/RTX-8000-v31-2000p/r${R}_real.json" "$DIR/r${R}_real.json" 2>/dev/null || true
    done
    cleanup
    env \
        VLLM_EMULATOR_ENABLE_ORACLE=1 \
        VLLM_EMULATOR_PROFILE_PACK="$PROF" \
        VLLM_EMULATOR_MODE=realtime \
        VLLM_EMULATOR_EXECUTOR_HOOK=1 \
        VLLM_EMULATOR_PREP_SURROGATE=1 \
        VLLM_EMULATOR_SAMPLE_TRIM="2,98" \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
        > "$DIR/emu_server.log" 2>&1 &
    if ! wait_server; then
        echo "SERVER_NEVER_READY" >> "$LOG"; cleanup; return 1
    fi
    echo "  [$(date +%T)] server ready" >> "$LOG"
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "$MODEL" --base-url "http://localhost:${PORT}" \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 200 --request-rate 4 > /dev/null 2>&1 || true
    sleep 3
    for RATE in $RATES; do
        echo "  [$(date +%T)] r=${RATE}" >> "$LOG"
        python3 -m vllm.entrypoints.cli.main bench serve \
            --model "$MODEL" --base-url "http://localhost:${PORT}" \
            --dataset-name random --random-input-len 256 --random-output-len 128 \
            --num-prompts $NUM_PROMPTS --request-rate $RATE \
            --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
            --save-result --result-dir "$DIR" \
            --result-filename "r${RATE}_emu.json" > /dev/null 2>&1
    done
    cleanup
    python3 tools/summarize_matrix.py "$DIR" 2>&1 | tee "$DIR/summary.txt" >> "$LOG"
    echo "  [PASS ${TAG} done]" >> "$MASTER_LOG"
}

run_pass "$V6_RES"  "v6_res7009"
# V6_FULL baseline already covered by agg-v6_sample from previous A/B

echo "=== reservoir A/B DONE $(date) ===" >> "$MASTER_LOG"
touch /tmp/vllm_reservoir.done
