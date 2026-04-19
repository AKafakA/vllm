#!/bin/bash
# Re-run IPC sweep (saving raw samples + mean + median) then 5-rate validation
# with MEAN aggregation via VLLM_IPC_OVERHEAD_AGG=mean.
# Expected to close low-rate TTFT residuals (~7ms/request added vs median).
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

MASTER_LOG="/tmp/vllm_ipc_mean.log"
echo "=== ipc-mean chain start $(date) ===" > "$MASTER_LOG"
touch /tmp/vllm_ipc_mean.started

MODEL="Qwen/Qwen3-8B"
PORT=8100
PROFILE="./results/RTX-8000-adaptive-archive-5r/serving-r2.json"
IPC_OUT="./results/RTX-8000-adaptive-archive-5r/ipc_overhead_v2.json"
NAME="v4-arrival-mean"
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
preflight() {
    if pgrep -f "VLLM::EngineCore|bench serve|vllm.entrypoints" > /dev/null; then
        echo "FATAL: vllm procs survived cleanup" >> "$MASTER_LOG"
        exit 1
    fi
}

cleanup
preflight

# === Phase 1: IPC sweep (real server, no emu) ===
echo "  [$(date +%T)] phase1: starting real server" >> "$MASTER_LOG"
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > "$DIR/ipc_server.log" 2>&1 &
wait_server || { echo "PHASE1_SERVER_FAIL" >> "$MASTER_LOG"; cleanup; exit 1; }

python3 -m vllm.entrypoints.cli.main bench serve \
    --model "$MODEL" --base-url "http://localhost:${PORT}" \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 --request-rate 4 > /dev/null 2>&1 || true
sleep 3

echo "  [$(date +%T)] phase1: running profile_ipc_overhead.py" >> "$MASTER_LOG"
python3 tools/profile_ipc_overhead.py \
    $PORT "$MODEL" "$PROFILE" "$IPC_OUT" 256 >> "$MASTER_LOG" 2>&1

cleanup

# Merge into profile (both median and mean present in each entry).
python3 - << PYEOF >> "$MASTER_LOG" 2>&1
import json
profile = json.load(open("$PROFILE"))
overhead = json.load(open("$IPC_OUT"))
profile["sched_overhead_table"] = overhead
json.dump(profile, open("$PROFILE", "w"), indent=2)
print(f"Merged {len(overhead)} ipc-overhead entries (with raw samples)")
PYEOF

# Copy real baselines from the v3 run (same workload, same hardware).
for R in 2 4 8 16 32; do
    cp "./results/ttft-variant-v3-arrival-delay/r${R}_real.json" "$DIR/r${R}_real.json" 2>/dev/null || true
done

# === Phase 2: 5-rate emu validation with MEAN aggregation ===
echo "  [$(date +%T)] phase2: starting emu server (agg=mean)" >> "$MASTER_LOG"
env \
    VLLM_EMULATOR_ENABLE_ORACLE=1 \
    VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
    VLLM_EMULATOR_MODE=realtime \
    VLLM_EMULATOR_EXECUTOR_HOOK=1 \
    VLLM_EMULATOR_SCHEDULER_HOOK=1 \
    VLLM_IPC_OVERHEAD_AGG=mean \
    VLLM_EMULATOR_PREP_SURROGATE=1 \
    VLLM_EMULATOR_SAMPLE_TRIM="2,98" \
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > "$DIR/server.log" 2>&1 &
wait_server || { echo "PHASE2_SERVER_FAIL" >> "$MASTER_LOG"; cleanup; exit 1; }

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
echo "=== ipc-mean chain DONE $(date) ===" >> "$MASTER_LOG"
touch /tmp/vllm_ipc_mean.done
