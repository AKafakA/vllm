#!/bin/bash
# Phase 2b — v5-2d-burst variant: 2D (N_conc, k_burst) overhead lookup.
# Uses sched_overhead_table_v2 (68-cell 2D grid) from Phase 1.
# Gated on Phase 2a (v5-arrival-sample) completion so they don't race.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

MASTER_LOG="/tmp/vllm_apr20_phase2b.log"
echo "=== phase2b v5-2d-burst start $(date) ===" > "$MASTER_LOG"
touch /tmp/vllm_apr20_phase2b.started

# Wait for Phase 2a done.
for i in $(seq 1 180); do
    [ -f /tmp/vllm_apr20_phase2.done ] && break
    sleep 30
done
if [ ! -f /tmp/vllm_apr20_phase2.done ]; then
    echo "Phase 2 not done; aborting 2b" >> "$MASTER_LOG"
    exit 1
fi

MODEL="Qwen/Qwen3-8B"
PORT=8100
PROFILE="./results/RTX-8000-adaptive-archive-5r/serving-r2.json"
NAME="v5-2d-burst"
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

cleanup

for R in 2 4 8 16 32; do
    cp "./results/ttft-variant-v3-arrival-delay/r${R}_real.json" "$DIR/r${R}_real.json" 2>/dev/null || true
done

echo "  [$(date +%T)] starting emu server (agg=2d-burst)" >> "$MASTER_LOG"
env \
    VLLM_EMULATOR_ENABLE_ORACLE=1 \
    VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
    VLLM_EMULATOR_MODE=realtime \
    VLLM_EMULATOR_EXECUTOR_HOOK=1 \
    VLLM_EMULATOR_SCHEDULER_HOOK=1 \
    VLLM_IPC_OVERHEAD_AGG=2d-burst \
    VLLM_EMULATOR_PREP_SURROGATE=1 \
    VLLM_EMULATOR_SAMPLE_TRIM="2,98" \
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > "$DIR/server.log" 2>&1 &
wait_server || { echo "SERVER_FAIL" >> "$MASTER_LOG"; cleanup; exit 1; }

python3 -m vllm.entrypoints.cli.main bench serve \
    --model "$MODEL" --base-url "http://localhost:${PORT}" \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 --request-rate 4 > /dev/null 2>&1 || true
sleep 3

for RATE in 2 4 8 16 32; do
    echo "  [$(date +%T)] ${NAME} r=${RATE}" >> "$MASTER_LOG"
    timeout 1800 python3 -m vllm.entrypoints.cli.main bench serve \
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
echo "=== phase2b DONE $(date) ===" >> "$MASTER_LOG"
touch /tmp/vllm_apr20_phase2b.done
