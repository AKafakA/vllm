#!/bin/bash
# Recapture real-hardware baselines with v3-matching pre-warmup, so the
# reported metrics don't include cold-graph capture cost. The resulting
# baseline will be SYMMETRIC with v3's profile methodology (both are
# fully warm).
#
# Output: results/RTX-8000-v31-2000p-warm/rN_real.json for rate N in
# {2,4,8,16,32}. These will act as the NEW real baseline for a v3 A/B
# sanity check.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

MODEL="Qwen/Qwen3-8B"
PORT=8100
OUT_DIR="./results/RTX-8000-v31-2000p-warm"
RATES="2 4 8 16 32"
NUM_PROMPTS=2000
LOG="/tmp/recapture_real_warm.log"

mkdir -p "$OUT_DIR"
touch "$OUT_DIR/.started"
echo "=== recapture_real_warm start $(date) ===" > "$LOG"

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

# Single long-lived REAL server (NO emulator env vars). All rates share
# one pre-warm sequence that mirrors v3's profiling script.
cleanup
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > "$OUT_DIR/real_server.log" 2>&1 &
if ! wait_server; then
    echo "SERVER_NEVER_READY" >> "$LOG"
    touch "$OUT_DIR/.failed"
    cleanup
    exit 1
fi
touch "$OUT_DIR/.server_ready"
echo "  [$(date +%T)] server ready" >> "$LOG"

# v3-matching CUDA graph warmup sweep for all padded batch sizes.
echo "  [$(date +%T)] CUDA graph warmup sweep..." >> "$LOG"
for NP in 1 2 4 8 16 24 32 48 64 96 128 160 192 224 256; do
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "$MODEL" --base-url "http://localhost:${PORT}" \
        --dataset-name random --random-input-len 1 --random-output-len 1 \
        --num-prompts $NP --request-rate inf > /dev/null 2>&1 || true
done

# v3-matching high-concurrency burst.
echo "  [$(date +%T)] high-concurrency burst..." >> "$LOG"
python3 -m vllm.entrypoints.cli.main bench serve \
    --model "$MODEL" --base-url "http://localhost:${PORT}" \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 500 --request-rate inf > /dev/null 2>&1 || true

# v3-matching standard warmup.
echo "  [$(date +%T)] standard warmup..." >> "$LOG"
python3 -m vllm.entrypoints.cli.main bench serve \
    --model "$MODEL" --base-url "http://localhost:${PORT}" \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 --request-rate 4 > /dev/null 2>&1 || true

touch "$OUT_DIR/.warmup_done"
echo "  [$(date +%T)] warmup done; graphs should be fully captured." >> "$LOG"

for RATE in $RATES; do
    echo "  [$(date +%T)] measurement r=$RATE" >> "$LOG"
    touch "$OUT_DIR/.r${RATE}_started"
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "$MODEL" --base-url "http://localhost:${PORT}" \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts $NUM_PROMPTS --request-rate $RATE \
        --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
        --save-result --result-dir "$OUT_DIR" \
        --result-filename "r${RATE}_real.json" > /dev/null 2>&1 \
        && touch "$OUT_DIR/.r${RATE}_done"
done

cleanup
echo "=== recapture_real_warm done $(date) ===" >> "$LOG"
touch "$OUT_DIR/.all_done"
touch "/tmp/recapture_real_warm.done"
