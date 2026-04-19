#!/bin/bash
# v3-arrival-delay variant: IPC overhead applied via scheduler hook
# (delays new-request admission, does not touch step latency).
# Oracle has IPC injection REVERTED (clean baseline on that axis).
#
# Waits for TTFT variants chain if still running, then runs full
# 5-rate × 2000p emu-validation on archive-r2 profile + sched_overhead_table.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

MASTER_LOG="/tmp/vllm_ttft_arrival.log"
echo "=== ttft arrival-delay chain start $(date) ===" > "$MASTER_LOG"
touch /tmp/vllm_ttft_arrival.started

echo "  [$(date +%T)] waiting for any prior ttft_variants chain..." >> "$MASTER_LOG"
for i in $(seq 1 60); do
    [ -f /tmp/vllm_ttft_variants.done ] && break
    sleep 30
done

# Preflight: ensure no stale processes on port 8100 or bench serve.
STALE_VLLM=$(pgrep -f "vllm.entrypoints" 2>/dev/null || true)
STALE_BENCH=$(pgrep -f "bench serve" 2>/dev/null || true)
if [ -n "$STALE_VLLM" ] || [ -n "$STALE_BENCH" ]; then
    echo "  [$(date +%T)] PREFLIGHT: killing stale procs ($STALE_VLLM $STALE_BENCH)" >> "$MASTER_LOG"
    pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
    pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
    pkill -9 -f "bench serve" 2>/dev/null || true
    sleep 10
fi

MODEL="Qwen/Qwen3-8B"
PORT=8100
PROFILE="./results/RTX-8000-adaptive-archive-5r/serving-r2.json"
NAME="v3-arrival-delay"
DIR="./results/ttft-variant-${NAME}"
mkdir -p "$DIR"

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

# Copy real baselines from the prior v2 run (same workload, same hardware).
for R in 2 4 8 16 32; do
    cp "./results/ttft-variant-v2-additive-divide/r${R}_real.json" "$DIR/r${R}_real.json" 2>/dev/null || true
done

cleanup
env \
    VLLM_EMULATOR_ENABLE_ORACLE=1 \
    VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
    VLLM_EMULATOR_MODE=realtime \
    VLLM_EMULATOR_EXECUTOR_HOOK=1 \
    VLLM_EMULATOR_SCHEDULER_HOOK=1 \
    VLLM_EMULATOR_PREP_SURROGATE=1 \
    VLLM_EMULATOR_SAMPLE_TRIM="2,98" \
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > "$DIR/server.log" 2>&1 &
wait_server || { echo "SERVER_NEVER_READY" >> "$MASTER_LOG"; cleanup; exit 1; }

# Warmup.
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

echo "" >> "$MASTER_LOG"
echo "=== ttft arrival-delay chain DONE $(date) ===" >> "$MASTER_LOG"
touch /tmp/vllm_ttft_arrival.done
