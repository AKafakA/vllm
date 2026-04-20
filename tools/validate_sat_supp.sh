#!/bin/bash
# Validate the saturation-supplementary profile against archive-r2 baseline.
# Random 256/128 × 5 rates × 2000 prompts. Random-only — sharegpt handled separately.
# Gates on the profile build being done.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

MASTER_LOG="/tmp/vllm_sat_supp_validate.log"
echo "=== sat-supp validation start $(date) ===" > "$MASTER_LOG"
touch /tmp/vllm_sat_supp_validate.started

# Wait for profile build.
for i in $(seq 1 240); do
    [ -f /tmp/vllm_sat_supp.done ] && break
    sleep 30
done
if [ ! -f /tmp/vllm_sat_supp.done ]; then
    echo "profile build did not finish; aborting validation" >> "$MASTER_LOG"
    exit 1
fi

MODEL="Qwen/Qwen3-8B"
PORT=8100
PROFILE="./results/RTX-8000-profile-archive-r2-sat-supp/serving-full.json"
DIR="./results/sat-supp-validate-random"
mkdir -p "$DIR"

# Copy real baselines from v3-arrival-delay chain (same workload, same hw).
for R in 2 4 8 16 32; do
    cp "./results/ttft-variant-v3-arrival-delay/r${R}_real.json" "$DIR/r${R}_real.json" 2>/dev/null || true
done

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

python3 -m vllm.entrypoints.cli.main bench serve \
    --model "$MODEL" --base-url "http://localhost:${PORT}" \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 --request-rate 4 > /dev/null 2>&1 || true
sleep 3

for RATE in 2 4 8 16 32; do
    echo "  [$(date +%T)] sat-supp r=${RATE}" >> "$MASTER_LOG"
    timeout 1800 python3 -m vllm.entrypoints.cli.main bench serve \
        --model "$MODEL" --base-url "http://localhost:${PORT}" \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 2000 --request-rate $RATE \
        --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
        --save-result --result-dir "$DIR" \
        --result-filename "r${RATE}_emu.json" > /dev/null 2>&1 \
        && echo "    r=${RATE} done" >> "$MASTER_LOG" \
        || echo "    r=${RATE} FAIL" >> "$MASTER_LOG"
    sleep 2
done
cleanup
python3 tools/summarize_matrix.py "$DIR" 2>&1 | tee "$DIR/summary.txt" >> "$MASTER_LOG"

echo "" >> "$MASTER_LOG"
echo "=== sat-supp validation DONE $(date) ===" >> "$MASTER_LOG"
touch /tmp/vllm_sat_supp_validate.done
