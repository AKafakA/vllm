#!/bin/bash
# Supplementary experiments to fill the 5-rate coverage gap:
#  Sharegpt at r=4 and r=16 (baseline Exp 2 only did r=2/8/32).
# Waits for ipc_chain done. ~40 min total.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

MODEL="Qwen/Qwen3-8B"
PORT=8100
PROFILE="./results/RTX-8000-adaptive-archive-5r/serving-r2.json"
SHAREGPT="./results/sharegpt_filtered_256_128.json"
MASTER_LOG="/tmp/vllm_supp_fullrates.log"

echo "=== supplementary full-rates start $(date) ===" > "$MASTER_LOG"

echo "  [$(date +%T)] waiting for ipc_chain done..." >> "$MASTER_LOG"
for i in $(seq 1 480); do
    [ -f /tmp/vllm_ipc_chain.done ] && break
    sleep 30
done
echo "  [$(date +%T)] ipc_chain done, proceeding" >> "$MASTER_LOG"

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
start_real() {
    cleanup
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
        > "$1/server.log" 2>&1 &
    wait_server || return 1
}
start_emu() {
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
        > "$1/server.log" 2>&1 &
    wait_server || return 1
}

# Sharegpt at r=4 and r=16 to complete the 5-rate matrix.
if [ -f "$SHAREGPT" ]; then
    for RATE in 4 16; do
        for MODE in real emu; do
            DIR="./results/supp-sharegpt-r${RATE}-${MODE}"
            mkdir -p "$DIR"
            echo "[$(date +%T)] sharegpt r=$RATE $MODE" >> "$MASTER_LOG"
            if [ "$MODE" = real ]; then start_real "$DIR"; else start_emu "$DIR"; fi
            python3 -m vllm.entrypoints.cli.main bench serve \
                --model "$MODEL" --base-url "http://localhost:${PORT}" \
                --dataset-name random --random-input-len 256 --random-output-len 128 \
                --num-prompts 200 --request-rate 4 > /dev/null 2>&1 || true
            sleep 2
            timeout 1500 python3 -m vllm.entrypoints.cli.main bench serve \
                --model "$MODEL" --base-url "http://localhost:${PORT}" \
                --dataset-name sharegpt --dataset-path "$SHAREGPT" \
                --num-prompts 2000 --request-rate $RATE \
                --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
                --save-result --result-dir "$DIR" \
                --result-filename "r${RATE}_${MODE}.json" > /dev/null 2>&1 \
                || echo "    FAIL sharegpt r=$RATE $MODE" >> "$MASTER_LOG"
            cleanup
        done
    done
fi

echo "=== supplementary full-rates DONE $(date) ===" >> "$MASTER_LOG"
touch /tmp/vllm_supp_fullrates.done
