#!/bin/bash
# v6 full validation: 5 rates x 2000 prompts, surrogate ON, against the real
# RTX 8000 baseline. Compared to archive reference numbers from
# paper/apr_18/02_v3_baseline_AB.md.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

MODEL="Qwen/Qwen3-8B"
PORT=8100
PROFILE="./results/RTX-8000-adaptive-v6-5r-single/serving-full.json"
RATES="2 4 8 16 32"
DIR="./results/RTX-8000-v6-validate"
LOG="$DIR/run.log"

mkdir -p "$DIR"
echo "=== v6 validation start $(date) ===" > "$LOG"

if [ ! -f "$PROFILE" ]; then
    echo "FATAL: v6 profile not found at $PROFILE" | tee -a "$LOG"
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

touch "$DIR/.started"
for R in $RATES; do
    cp "./results/RTX-8000-v31-2000p/r${R}_real.json" "$DIR/r${R}_real.json" 2>/dev/null || true
done

cleanup
env \
    VLLM_EMULATOR_ENABLE_ORACLE=1 \
    VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
    VLLM_EMULATOR_MODE=realtime \
    VLLM_EMULATOR_EXECUTOR_HOOK=1 \
    VLLM_EMULATOR_PREP_SURROGATE=1 \
    VLLM_EMULATOR_SAMPLE_TRIM="2,98" \
    VLLM_EMULATOR_HOOK_TRACE="$DIR/hook_trace.csv" \
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > "$DIR/emu_server.log" 2>&1 &
if ! wait_server; then
    echo "SERVER_NEVER_READY" >> "$LOG"; touch "$DIR/.failed"; cleanup; exit 1
fi
touch "$DIR/.server_ready"
echo "  [$(date +%T)] server ready" >> "$LOG"

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
        --num-prompts 2000 --request-rate $RATE \
        --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
        --save-result --result-dir "$DIR" \
        --result-filename "r${RATE}_emu.json" > /dev/null 2>&1 \
        && touch "$DIR/.r${RATE}_done"
done
cleanup

python3 tools/summarize_matrix.py "$DIR" 2>&1 | tee "$DIR/summary.txt" >> "$LOG"
echo "" >> "$LOG"
echo "=== v6 validation DONE $(date) ===" >> "$LOG"
touch "$DIR/.all_done"
touch /tmp/vllm_v6_validate.done
