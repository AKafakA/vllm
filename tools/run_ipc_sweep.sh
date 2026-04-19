#!/bin/bash
# Run IPC overhead sweep against a real vllm server, produce ipc_overhead.json,
# then merge into the archive-r2 profile as sched_overhead_table.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

MODEL="Qwen/Qwen3-8B"
PORT=8100
PROFILE="./results/RTX-8000-adaptive-archive-5r/serving-r2.json"
IPC_OUT="./results/RTX-8000-adaptive-archive-5r/ipc_overhead.json"
SERVER_LOG="./results/RTX-8000-adaptive-archive-5r/ipc_sweep_server.log"
MASTER_LOG="/tmp/vllm_ipc_sweep.log"

echo "=== IPC sweep start $(date) ===" > "$MASTER_LOG"

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
        echo "FATAL: vllm procs survived cleanup. Halting." >> "$MASTER_LOG"
        exit 1
    fi
}

cleanup
preflight

# Start REAL vllm server (no emulator oracle — we want real TTFT).
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > "$SERVER_LOG" 2>&1 &
wait_server || { echo "SERVER_NEVER_READY" >> "$MASTER_LOG"; cleanup; exit 1; }
echo "  [$(date +%T)] real server ready" >> "$MASTER_LOG"

# Warmup so CUDA graphs are captured before we measure.
python3 -m vllm.entrypoints.cli.main bench serve \
    --model "$MODEL" --base-url "http://localhost:${PORT}" \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 --request-rate 4 > /dev/null 2>&1 || true
sleep 3
echo "  [$(date +%T)] warmup done" >> "$MASTER_LOG"

# Run IPC overhead sweep.
python3 tools/profile_ipc_overhead.py \
    $PORT "$MODEL" "$PROFILE" "$IPC_OUT" 256 >> "$MASTER_LOG" 2>&1

cleanup

# Merge into profile JSON under sched_overhead_table.
python3 - << PYEOF >> "$MASTER_LOG" 2>&1
import json
profile = json.load(open("$PROFILE"))
overhead = json.load(open("$IPC_OUT"))
profile["sched_overhead_table"] = overhead
json.dump(profile, open("$PROFILE", "w"), indent=2)
print(f"Merged {len(overhead)} ipc-overhead entries into profile")
PYEOF

echo "=== IPC sweep DONE $(date) ===" >> "$MASTER_LOG"
touch /tmp/vllm_ipc_sweep.done
