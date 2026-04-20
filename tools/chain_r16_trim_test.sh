#!/bin/bash
# Test whether VLLM_EMULATOR_SAMPLE_TRIM="2,98" is the mechanism that made
# sat-supp profile regress r=16 from -14% (archive-r2) to -28% (sat-supp).
#
# Hypothesis: sat-supp's dense main-mode samples at (tt=256, c=257) pushed
# p98 lower, so trim=2,98 cuts off the 130-200ms tail samples that were
# load-bearing for oracle predictions at r=16. Removing the trim should
# restore the tail and recover accuracy.
#
# Two comparisons to run:
#   A. sat-supp × random × r=16 × 2000p × trim disabled (TRIM="0,100")
#   B. archive-r2 × random × r=16 × 2000p × trim disabled (TRIM="0,100")
# Baselines (from earlier today): sat-supp trim 2,98 = -28%, archive-r2 trim 2,98 = -14%.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

MASTER_LOG="/tmp/vllm_r16_trim.log"
echo "=== r=16 trim test start $(date) ===" > "$MASTER_LOG"
touch /tmp/vllm_r16_trim.started

# Gate on variance test done.
for i in $(seq 1 60); do
    [ -f /tmp/vllm_r16_variance.done ] && break
    sleep 30
done
if [ ! -f /tmp/vllm_r16_variance.done ]; then
    echo "variance test did not complete in time; proceeding anyway" >> "$MASTER_LOG"
fi

MODEL="Qwen/Qwen3-8B"
PORT=8100
DIR="./results/r16-trim-test"
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

run_one() {
    local LABEL=$1
    local PROFILE=$2
    local TRIM=$3
    echo "  [$(date +%T)] ${LABEL}: profile=${PROFILE##*/} trim=${TRIM}" >> "$MASTER_LOG"
    cleanup
    env \
        VLLM_EMULATOR_ENABLE_ORACLE=1 \
        VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
        VLLM_EMULATOR_MODE=realtime \
        VLLM_EMULATOR_EXECUTOR_HOOK=1 \
        VLLM_EMULATOR_SCHEDULER_HOOK=1 \
        VLLM_IPC_OVERHEAD_AGG=median \
        VLLM_EMULATOR_PREP_SURROGATE=1 \
        VLLM_EMULATOR_SAMPLE_TRIM="$TRIM" \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
        > "$DIR/server_${LABEL}.log" 2>&1 &
    wait_server || { echo "    SERVER_FAIL $LABEL" >> "$MASTER_LOG"; cleanup; return; }

    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "$MODEL" --base-url "http://localhost:${PORT}" \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 100 --request-rate 4 > /dev/null 2>&1 || true
    sleep 2

    timeout 1800 python3 -m vllm.entrypoints.cli.main bench serve \
        --model "$MODEL" --base-url "http://localhost:${PORT}" \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 2000 --request-rate 16 \
        --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
        --save-result --result-dir "$DIR" \
        --result-filename "${LABEL}.json" > /dev/null 2>&1 \
        && echo "    ${LABEL} done" >> "$MASTER_LOG" \
        || echo "    ${LABEL} FAIL" >> "$MASTER_LOG"
    cleanup
}

# A. sat-supp without trim
run_one "sat_notrim" "./results/RTX-8000-profile-archive-r2-sat-supp/serving-full.json" "0,100"
# B. archive-r2 without trim (baseline comparison)
run_one "arc_notrim" "./results/RTX-8000-adaptive-archive-5r/serving-r2.json" "0,100"

echo "=== r=16 trim test DONE $(date) ===" >> "$MASTER_LOG"
touch /tmp/vllm_r16_trim.done
