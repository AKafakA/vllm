#!/bin/bash
# Generic per-feature A/B harness. Invokes two passes (gate off / gate on)
# against a single profile and writes results under per-pass directories.
#
# Usage:
#   PROFILE=./results/_archive/serving-dense.json \
#   PROFILE_B=./results/_archive/serving-dense.json \
#   FEATURE=f1 \
#   EXTRA_ENV_A="" \
#   EXTRA_ENV_B="VLLM_EMULATOR_OUTLIER_FILTER=iqr_noop_marker" \
#   RATES="2 8 16" \
#   NUM_PROMPTS=1000 \
#   bash tools/validate_feature_ab.sh
#
# PROFILE is used for pass A; PROFILE_B for pass B (may be the same file,
# or a rebuilt variant e.g. for F1's iqr-filtered profile).
# EXTRA_ENV_A and EXTRA_ENV_B are raw `env` fragments injected into each pass.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

MODEL="Qwen/Qwen3-8B"
PORT=8100
FEATURE="${FEATURE:-unknown}"
PROFILE="${PROFILE:?set PROFILE}"
PROFILE_B="${PROFILE_B:-$PROFILE}"
RATES="${RATES:-2 8 16}"
NUM_PROMPTS="${NUM_PROMPTS:-1000}"
MASTER_LOG="/tmp/vllm_${FEATURE}_ab.log"

echo "=== ${FEATURE} A/B start $(date) ===" > "$MASTER_LOG"
echo "  PROFILE_A=$PROFILE" >> "$MASTER_LOG"
echo "  PROFILE_B=$PROFILE_B" >> "$MASTER_LOG"
echo "  RATES=$RATES NUM_PROMPTS=$NUM_PROMPTS" >> "$MASTER_LOG"

if [ ! -f "$PROFILE" ]; then
    echo "FATAL: profile $PROFILE not found" | tee -a "$MASTER_LOG"
    exit 1
fi
if [ ! -f "$PROFILE_B" ]; then
    echo "FATAL: profile_B $PROFILE_B not found" | tee -a "$MASTER_LOG"
    exit 1
fi

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
    local LABEL="$1"
    local PROF="$2"
    local EXTRA_ENV="$3"
    local DIR="./results/RTX-8000-${FEATURE}-${LABEL}-apr18"
    local LOG="$DIR/run.log"

    mkdir -p "$DIR"
    echo "=== PASS ${LABEL} ($FEATURE) start $(date) ===" > "$LOG"
    touch "$DIR/.started"

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
        VLLM_EMULATOR_HOOK_TRACE="$DIR/hook_trace.csv" \
        $EXTRA_ENV \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
        > "$DIR/emu_server.log" 2>&1 &

    if ! wait_server; then
        echo "SERVER_NEVER_READY" >> "$LOG"
        touch "$DIR/.failed"
        cleanup
        return 1
    fi
    touch "$DIR/.server_ready"
    echo "  [$(date +%T)] server ready ($LABEL); env='${EXTRA_ENV}'" >> "$LOG"

    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "$MODEL" --base-url "http://localhost:${PORT}" \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 200 --request-rate 4 > /dev/null 2>&1 || true
    sleep 3
    touch "$DIR/.warmup_done"

    for RATE in $RATES; do
        echo "  [$(date +%T)] r=${RATE}" >> "$LOG"
        touch "$DIR/.r${RATE}_started"
        python3 -m vllm.entrypoints.cli.main bench serve \
            --model "$MODEL" --base-url "http://localhost:${PORT}" \
            --dataset-name random --random-input-len 256 --random-output-len 128 \
            --num-prompts $NUM_PROMPTS --request-rate $RATE \
            --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
            --save-result --result-dir "$DIR" \
            --result-filename "r${RATE}_emu.json" > /dev/null 2>&1 \
            && touch "$DIR/.r${RATE}_done"
    done
    cleanup

    python3 tools/summarize_matrix.py "$DIR" 2>&1 | tee "$DIR/summary.txt" >> "$LOG"
    echo "=== PASS ${LABEL} done $(date) ===" >> "$LOG"
    touch "$DIR/.all_done"
}

run_pass "off" "$PROFILE"   "${EXTRA_ENV_A:-}"
run_pass "on"  "$PROFILE_B" "${EXTRA_ENV_B:-}"

echo "=== ${FEATURE} A/B DONE $(date) ===" >> "$MASTER_LOG"
touch "/tmp/vllm_${FEATURE}_ab.done"
