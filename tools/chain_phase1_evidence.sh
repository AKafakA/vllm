#!/bin/bash
# Phase 1 — evidence gathering for r=16 root cause.
#   Step 1: extended IPC sweep v2 (N up to 1024, burst_k ∈ {1,2,4,8}).
#   Step 2: batch-composition diagnostic at r=16 (hook on vs off, 2000p each).
# Sequential on single GPU; both write .done markers.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

MASTER_LOG="/tmp/vllm_apr20_phase1.log"
echo "=== phase1 start $(date) ===" > "$MASTER_LOG"
touch /tmp/vllm_apr20_phase1.started

MODEL="Qwen/Qwen3-8B"
PORT=8100
PROFILE="./results/RTX-8000-adaptive-archive-5r/serving-r2.json"
IPC_OUT="./results/RTX-8000-adaptive-archive-5r/ipc_overhead_v2_2d.json"
DIAG_DIR="./results/batch-diag-r16-apr20"
mkdir -p "$DIAG_DIR"

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
        echo "FATAL: vllm procs survived cleanup" >> "$MASTER_LOG"
        exit 1
    fi
}

cleanup
preflight

# === Step 1: Extended IPC sweep v2 (real server, no emu) ===
echo "  [$(date +%T)] step1: starting real server for IPC sweep v2" >> "$MASTER_LOG"
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > "$DIAG_DIR/ipc_v2_server.log" 2>&1 &
wait_server || { echo "IPC_SERVER_FAIL" >> "$MASTER_LOG"; cleanup; exit 1; }

python3 -m vllm.entrypoints.cli.main bench serve \
    --model "$MODEL" --base-url "http://localhost:${PORT}" \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 --request-rate 4 > /dev/null 2>&1 || true
sleep 3

echo "  [$(date +%T)] step1: running profile_ipc_overhead_v2.py" >> "$MASTER_LOG"
python3 tools/profile_ipc_overhead_v2.py \
    $PORT "$MODEL" "$PROFILE" "$IPC_OUT" \
    --burst-k "1 2 4 8" --max-n 1024 --samples-per-cell 5 \
    >> "$MASTER_LOG" 2>&1

cleanup

# Merge v2 table into profile as sched_overhead_table_v2 (preserve original).
python3 - << PYEOF >> "$MASTER_LOG" 2>&1
import json
profile = json.load(open("$PROFILE"))
v2 = json.load(open("$IPC_OUT"))
profile["sched_overhead_table_v2"] = v2
json.dump(profile, open("$PROFILE", "w"), indent=2)
print(f"Merged {len(v2)} 2D IPC overhead entries as sched_overhead_table_v2")
PYEOF

# === Step 2: Batch-composition diagnostic (emu, r=16, 2000p, hook on and off) ===
for HOOK_STATE in on off; do
    if [ "$HOOK_STATE" = "on" ]; then
        HOOK_ENV="VLLM_EMULATOR_SCHEDULER_HOOK=1"
        TRACE_PATH="$DIAG_DIR/trace_hook.csv"
    else
        HOOK_ENV="VLLM_EMULATOR_SCHEDULER_HOOK=0"
        TRACE_PATH="$DIAG_DIR/trace_nohook.csv"
    fi
    echo "  [$(date +%T)] step2: emu r=16 hook=$HOOK_STATE, trace=$TRACE_PATH" >> "$MASTER_LOG"

    cleanup
    env \
        VLLM_EMULATOR_ENABLE_ORACLE=1 \
        VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
        VLLM_EMULATOR_MODE=realtime \
        VLLM_EMULATOR_EXECUTOR_HOOK=1 \
        $HOOK_ENV \
        VLLM_EMULATOR_PREP_SURROGATE=1 \
        VLLM_EMULATOR_SAMPLE_TRIM="2,98" \
        VLLM_EMULATOR_HOOK_TRACE="$TRACE_PATH" \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
        > "$DIAG_DIR/server_${HOOK_STATE}.log" 2>&1 &
    wait_server || { echo "DIAG_SERVER_FAIL_${HOOK_STATE}" >> "$MASTER_LOG"; cleanup; continue; }

    # Warmup small.
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "$MODEL" --base-url "http://localhost:${PORT}" \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 50 --request-rate 4 > /dev/null 2>&1 || true
    sleep 2

    # 2000p at r=16 (standard debug size).
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

# Run analyzer.
python3 tools/analyze_batch_trace.py \
    "$DIAG_DIR/trace_hook.csv" "$DIAG_DIR/trace_nohook.csv" >> "$MASTER_LOG" 2>&1 || true

echo "" >> "$MASTER_LOG"
echo "=== phase1 DONE $(date) ===" >> "$MASTER_LOG"
touch /tmp/vllm_apr20_phase1.done
