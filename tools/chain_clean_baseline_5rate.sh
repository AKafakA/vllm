#!/bin/bash
# Clean 5-rate × 3-seed baseline on archive-r2 × trim=0,100 (no magic).
# Purpose: get variance envelopes at every rate. If r=16 is noise-dominated,
# its mean across 3 seeds will localise the true systematic gap (or confirm
# it's just Poisson variance at saturation edge).
#
# v3 hook enabled (median), trim DISABLED (TRIM="0,100").
# Single server session; bench seeds {1001, 1002, 1003} per rate.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

MASTER_LOG="/tmp/vllm_clean_baseline.log"
echo "=== clean baseline start $(date) ===" > "$MASTER_LOG"
touch /tmp/vllm_clean_baseline.started

MODEL="Qwen/Qwen3-8B"
PORT=8100
PROFILE="./results/RTX-8000-adaptive-archive-5r/serving-r2.json"
DIR="./results/clean-baseline-5rate-3seed"
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
    > "$DIR/server.log" 2>&1 &
wait_server || { echo "SERVER_FAIL" >> "$MASTER_LOG"; cleanup; exit 1; }

# Warmup.
python3 -m vllm.entrypoints.cli.main bench serve \
    --model "$MODEL" --base-url "http://localhost:${PORT}" \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 --request-rate 4 > /dev/null 2>&1 || true
sleep 3

# 5 rates × 3 seeds. Note: v3 hook uses fixed internal RNG seed 42; bench_serve's
# --seed controls ARRIVAL pattern (which is the main source of r=16 variance).
for RATE in 2 4 8 16 32; do
    for SEED in 1001 1002 1003; do
        echo "  [$(date +%T)] r=${RATE} seed=${SEED}" >> "$MASTER_LOG"
        timeout 1800 python3 -m vllm.entrypoints.cli.main bench serve \
            --model "$MODEL" --base-url "http://localhost:${PORT}" \
            --dataset-name random --random-input-len 256 --random-output-len 128 \
            --num-prompts 2000 --request-rate $RATE --seed $SEED \
            --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
            --save-result --result-dir "$DIR" \
            --result-filename "r${RATE}_s${SEED}.json" > /dev/null 2>&1 \
            && echo "    r=${RATE} s=${SEED} done" >> "$MASTER_LOG" \
            || echo "    r=${RATE} s=${SEED} FAIL" >> "$MASTER_LOG"
        sleep 2
    done
done
cleanup

echo "" >> "$MASTER_LOG"
echo "=== clean baseline DONE $(date) ===" >> "$MASTER_LOG"
touch /tmp/vllm_clean_baseline.done
