#!/bin/bash
# Validate archive-2round profile at r=2, r=8 × 500 prompts.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

MODEL="Qwen/Qwen3-8B"
PORT=8100
PROFILE="./results/RTX-8000-adaptive-archive-5r/serving-r2.json"
DIR="./results/RTX-8000-archive-r2-validate"

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

cleanup
env \
    VLLM_EMULATOR_ENABLE_ORACLE=1 \
    VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
    VLLM_EMULATOR_MODE=realtime \
    VLLM_EMULATOR_EXECUTOR_HOOK=1 \
    VLLM_EMULATOR_PREP_SURROGATE=1 \
    VLLM_EMULATOR_SAMPLE_TRIM="2,98" \
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > "$DIR/emu_server.log" 2>&1 &
wait_server || { echo "SERVER_NEVER_READY"; cleanup; exit 1; }
echo "server ready"

python3 -m vllm.entrypoints.cli.main bench serve \
    --model "$MODEL" --base-url "http://localhost:${PORT}" \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 --request-rate 4 > /dev/null 2>&1 || true
sleep 3

for RATE in 2 8; do
    echo "r=${RATE}"
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "$MODEL" --base-url "http://localhost:${PORT}" \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 500 --request-rate $RATE \
        --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
        --save-result --result-dir "$DIR" \
        --result-filename "r${RATE}_emu.json" > /dev/null 2>&1
done
cleanup
python3 tools/summarize_matrix.py "$DIR" 2>&1 | tee "$DIR/summary.txt"
touch /tmp/vllm_archive_r2.done
