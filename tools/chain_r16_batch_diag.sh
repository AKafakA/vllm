#!/bin/bash
# r=16 batch-composition diagnostic with FIXED line-buffered trace.
# Two passes: v3 (hook on, median) vs NO hook (baseline emu).
# Captures per-step trace to answer: does arrival-delay change batch composition?
# Re-run of Apr 20 Phase 1.2 with executor_hook.py's trace file now
# line-buffered (survives SIGKILL).
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

MASTER_LOG="/tmp/vllm_r16_batch_diag.log"
echo "=== r16 batch-comp diag start $(date) ===" > "$MASTER_LOG"
touch /tmp/vllm_r16_batch_diag.started

MODEL="Qwen/Qwen3-8B"
PORT=8100
PROFILE="./results/RTX-8000-adaptive-archive-5r/serving-r2.json"
DIAG_DIR="./results/r16-batch-diag-apr20pm"
mkdir -p "$DIAG_DIR"

cleanup() {
    # Graceful then forced — so trace file flushes first.
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

for HOOK_STATE in on off; do
    if [ "$HOOK_STATE" = "on" ]; then
        HOOK_ENV="VLLM_EMULATOR_SCHEDULER_HOOK=1"
        TRACE_PATH="$DIAG_DIR/trace_hook.csv"
    else
        HOOK_ENV="VLLM_EMULATOR_SCHEDULER_HOOK=0"
        TRACE_PATH="$DIAG_DIR/trace_nohook.csv"
    fi
    echo "  [$(date +%T)] r=16 hook=$HOOK_STATE trace=$TRACE_PATH" >> "$MASTER_LOG"

    cleanup
    env \
        VLLM_EMULATOR_ENABLE_ORACLE=1 \
        VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
        VLLM_EMULATOR_MODE=realtime \
        VLLM_EMULATOR_EXECUTOR_HOOK=1 \
        $HOOK_ENV \
        VLLM_IPC_OVERHEAD_AGG=median \
        VLLM_EMULATOR_PREP_SURROGATE=1 \
        VLLM_EMULATOR_SAMPLE_TRIM="2,98" \
        VLLM_EMULATOR_HOOK_TRACE="$TRACE_PATH" \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
        > "$DIAG_DIR/server_${HOOK_STATE}.log" 2>&1 &
    wait_server || { echo "SERVER_FAIL_${HOOK_STATE}" >> "$MASTER_LOG"; cleanup; continue; }

    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "$MODEL" --base-url "http://localhost:${PORT}" \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 50 --request-rate 4 > /dev/null 2>&1 || true
    sleep 2

    timeout 1500 python3 -m vllm.entrypoints.cli.main bench serve \
        --model "$MODEL" --base-url "http://localhost:${PORT}" \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 2000 --request-rate 16 \
        --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
        --save-result --result-dir "$DIAG_DIR" \
        --result-filename "r16_hook${HOOK_STATE}.json" > /dev/null 2>&1 \
        && echo "    hook=$HOOK_STATE done" >> "$MASTER_LOG" \
        || echo "    hook=$HOOK_STATE FAIL" >> "$MASTER_LOG"
    cleanup
done

python3 tools/analyze_batch_trace.py \
    "$DIAG_DIR/trace_hook.csv" "$DIAG_DIR/trace_nohook.csv" \
    > "$DIAG_DIR/analyzer_stdout.log" 2>&1 || true

# Move the analyzer's output to the r=16 diag folder (default output path
# is paper/apr_20/01_batch_composition_r16.md which is fine).
echo "=== r16 batch-comp diag DONE $(date) ===" >> "$MASTER_LOG"
touch /tmp/vllm_r16_batch_diag.done
