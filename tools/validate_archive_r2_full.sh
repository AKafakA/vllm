#!/bin/bash
# Full archive-r2 fixed-shape validation: 5 rates × 2000 prompts, 256/128.
# SINGLE SESSION across all rates — matches profile's single-session recipe
# and real-baseline capture methodology. 20-min per-bench timeout prevents
# hang from stalling the whole run.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

MODEL="Qwen/Qwen3-8B"
PORT=8100
PROFILE="./results/RTX-8000-adaptive-archive-5r/serving-r2.json"
RATES="2 4 8 16 32"
DIR="./results/RTX-8000-archive-r2-full-v2"

mkdir -p "$DIR"
for R in $RATES; do
    cp "./results/RTX-8000-v31-2000p/r${R}_real.json" "$DIR/r${R}_real.json" 2>/dev/null || true
done

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
preflight() {
    if pgrep -f "VLLM::EngineCore|bench serve|vllm.entrypoints" > /dev/null; then
        echo "FATAL: vllm/bench processes survived cleanup. Halting."
        exit 1
    fi
}

cleanup
preflight
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
wait_server || { echo "SERVER_NEVER_READY"; cleanup; exit 1; }
echo "server ready"

# Single warmup at the top.
python3 -m vllm.entrypoints.cli.main bench serve \
    --model "$MODEL" --base-url "http://localhost:${PORT}" \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 --request-rate 4 > /dev/null 2>&1 || true
sleep 3

for RATE in $RATES; do
    echo "=== r=${RATE} === $(date)"
    # 20-min timeout on bench client — if hang, kill client only, server stays.
    timeout 1200 python3 -m vllm.entrypoints.cli.main bench serve \
        --model "$MODEL" --base-url "http://localhost:${PORT}" \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 2000 --request-rate $RATE \
        --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
        --save-result --result-dir "$DIR" \
        --result-filename "r${RATE}_emu.json" > /dev/null 2>&1 \
        && echo "r=${RATE} done" \
        || echo "r=${RATE} FAILED (timeout or error)"
    # Kill any zombie bench client; server stays alive.
    pkill -9 -f "bench serve.*request-rate $RATE" 2>/dev/null || true
    sleep 3
done
cleanup

python3 tools/summarize_matrix.py "$DIR" 2>&1 | tee "$DIR/summary.txt"
touch /tmp/vllm_archive_r2_full.done
