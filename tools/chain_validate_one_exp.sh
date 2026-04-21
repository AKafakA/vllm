#!/bin/bash
# Parametric validation chain — one experiment cell:
#   - profile pack path (env: PROFILE)
#   - oracle K (env: ORACLE_K, default 1)
#   - output subdir (env: OUT_TAG)
#
# Runs emu hookOn × {r=2,4,8,16,32} × 2000p, fresh server per bench,
# common_warmup, seed=0. Writes hookOn_r{N}.json under results/$OUT_TAG/.
# Touches /tmp/vllm_${OUT_TAG}.done at end.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

source "$(dirname "$0")/_bench_common.sh"

PROFILE="${PROFILE:?set PROFILE=<path to serving-full.json>}"
ORACLE_K="${ORACLE_K:-1}"
OUT_TAG="${OUT_TAG:?set OUT_TAG=<subdir under results/>}"

DIR="./results/$OUT_TAG"
MASTER_LOG="/tmp/vllm_${OUT_TAG}.log"
MARKER="/tmp/vllm_${OUT_TAG}"

mkdir -p "$DIR"
touch "${MARKER}.started"
echo "=== ${OUT_TAG} start $(date) ===" > "$MASTER_LOG"
echo "PROFILE=$PROFILE  ORACLE_K=$ORACLE_K" >> "$MASTER_LOG"

if [ ! -f "$PROFILE" ]; then
    echo "PROFILE NOT FOUND: $PROFILE" >> "$MASTER_LOG"
    exit 1
fi

run_emu_bench() {
    local RATE=$1
    local LABEL="hookOn_r${RATE}"

    common_cleanup
    if ! common_preflight; then
        echo "  preflight FAIL at rate=$RATE" >> "$MASTER_LOG"
        return 1
    fi

    env \
        VLLM_EMULATOR_ENABLE_ORACLE=1 \
        VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
        VLLM_EMULATOR_MODE=realtime \
        VLLM_EMULATOR_EXECUTOR_HOOK=1 \
        VLLM_EMULATOR_SCHEDULER_HOOK=1 \
        VLLM_IPC_OVERHEAD_AGG=median \
        VLLM_EMULATOR_PREP_SURROGATE="${VLLM_EMULATOR_PREP_SURROGATE:-1}" \
        VLLM_EMULATOR_ORACLE_K="$ORACLE_K" \
        VLLM_EMULATOR_ORACLE_MIN_SAMPLES="${VLLM_EMULATOR_ORACLE_MIN_SAMPLES:-30}" \
        VLLM_EMULATOR_IPC_POSITION="${VLLM_EMULATOR_IPC_POSITION:-arrival}" \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$BENCH_MODEL" --max-model-len "$BENCH_MAX_MODEL_LEN" \
        --port "$BENCH_PORT" --trust-remote-code \
        > "$DIR/server_${LABEL}.log" 2>&1 &

    if ! common_wait_server; then
        echo "    SERVER FAIL ${LABEL}" >> "$MASTER_LOG"
        common_cleanup
        return 1
    fi

    common_warmup

    echo "  [$(date +%T)] bench ${LABEL} (ORACLE_K=$ORACLE_K)" >> "$MASTER_LOG"
    timeout 1800 python3 -m vllm.entrypoints.cli.main bench serve \
        --model "$BENCH_MODEL" --base-url "http://localhost:${BENCH_PORT}" \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 2000 --request-rate "$RATE" \
        --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
        --save-result --result-dir "$DIR" \
        --result-filename "${LABEL}.json" > /dev/null 2>&1 \
        && echo "    ${LABEL} done" >> "$MASTER_LOG" \
        || echo "    ${LABEL} FAIL" >> "$MASTER_LOG"

    common_cleanup
}

for RATE in 2 4 8 16 32; do
    run_emu_bench "$RATE"
done

echo "=== ${OUT_TAG} DONE $(date) ===" >> "$MASTER_LOG"
touch "${MARKER}.done"
