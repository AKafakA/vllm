#!/bin/bash
# Wait for archive_5r chain done → run archive_5r_vars reprofile → emu validate.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

MASTER_LOG="/tmp/vllm_archive_5r_vars_chain.log"
echo "=== archive-5r-vars chain start $(date) ===" > "$MASTER_LOG"

echo "  [$(date +%T)] waiting for /tmp/vllm_archive_5r_chain.done" >> "$MASTER_LOG"
for i in $(seq 1 240); do
    [ -f /tmp/vllm_archive_5r_chain.done ] && break
    sleep 30
done
if [ ! -f /tmp/vllm_archive_5r_chain.done ]; then
    echo "  WARN: archive_5r chain not done after 2h wait; proceeding" >> "$MASTER_LOG"
fi

echo "" >> "$MASTER_LOG"
echo "  [$(date +%T)] Phase 1: archive-5r-vars reprofile" >> "$MASTER_LOG"
bash tools/adaptive_profile_archive_5r_vars.sh >> "$MASTER_LOG" 2>&1 || true

PROF="./results/RTX-8000-adaptive-archive-5r-vars/serving-full.json"
if [ ! -f "$PROF" ]; then
    echo "  FATAL: archive_5r_vars profile build failed" >> "$MASTER_LOG"
    touch /tmp/vllm_archive_5r_vars_chain.done
    exit 1
fi

echo "" >> "$MASTER_LOG"
echo "  [$(date +%T)] Phase 2: emu validate" >> "$MASTER_LOG"
DIR="./results/RTX-8000-archive-5r-vars-validate"
mkdir -p "$DIR"
for R in 2 8; do
    cp "./results/RTX-8000-v31-2000p/r${R}_real.json" "$DIR/r${R}_real.json" 2>/dev/null || true
done

pkill -9 -f "VLLM::EngineCore" 2>/dev/null; pkill -9 -f "vllm.entrypoints" 2>/dev/null
fuser 8100/tcp 2>/dev/null | xargs -r kill -9 2>/dev/null; sleep 5

env \
    VLLM_EMULATOR_ENABLE_ORACLE=1 \
    VLLM_EMULATOR_PROFILE_PACK="$PROF" \
    VLLM_EMULATOR_MODE=realtime \
    VLLM_EMULATOR_EXECUTOR_HOOK=1 \
    VLLM_EMULATOR_PREP_SURROGATE=1 \
    VLLM_EMULATOR_SAMPLE_TRIM="2,98" \
python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-8B --max-model-len 4096 --port 8100 --trust-remote-code \
    > "$DIR/emu_server.log" 2>&1 &

for i in $(seq 1 300); do
    curl -s "http://localhost:8100/health" > /dev/null 2>&1 && break
    sleep 1
done

python3 -m vllm.entrypoints.cli.main bench serve \
    --model Qwen/Qwen3-8B --base-url "http://localhost:8100" \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 --request-rate 4 > /dev/null 2>&1 || true
sleep 3

for RATE in 2 8; do
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model Qwen/Qwen3-8B --base-url "http://localhost:8100" \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 500 --request-rate $RATE \
        --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
        --save-result --result-dir "$DIR" \
        --result-filename "r${RATE}_emu.json" > /dev/null 2>&1
done

pkill -9 -f "VLLM::EngineCore" 2>/dev/null; pkill -9 -f "vllm.entrypoints" 2>/dev/null
sleep 3
python3 tools/summarize_matrix.py "$DIR" 2>&1 | tee "$DIR/summary.txt" >> "$MASTER_LOG"

echo "" >> "$MASTER_LOG"
echo "=== archive-5r-vars chain DONE $(date) ===" >> "$MASTER_LOG"
touch /tmp/vllm_archive_5r_vars_chain.done
