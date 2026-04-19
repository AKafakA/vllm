#!/bin/bash
# Wait for baseline_exp_chain + archive_r2_full to finish, then:
#  1. Run IPC overhead sweep on real server → ipc_overhead.json
#  2. Merge into archive-r2 profile as sched_overhead_table
#  3. Re-emu-validate archive-r2 × r=2/8 × 500p to measure TTFT improvement
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

MASTER_LOG="/tmp/vllm_ipc_chain.log"
echo "=== IPC chain start $(date) ===" > "$MASTER_LOG"

# Wait up to 4h for both prior chains done.
echo "  [$(date +%T)] waiting for archive_r2_full + baseline_exp done" >> "$MASTER_LOG"
for i in $(seq 1 480); do
    if [ -f /tmp/vllm_archive_r2_full.done ] && [ -f /tmp/vllm_baseline_exp.done ]; then
        break
    fi
    sleep 30
done

echo "  [$(date +%T)] dependencies done, starting IPC sweep" >> "$MASTER_LOG"
bash tools/run_ipc_sweep.sh >> "$MASTER_LOG" 2>&1 || true

# Re-emu-validate archive-r2 with new sched_overhead_table.
echo "" >> "$MASTER_LOG"
echo "  [$(date +%T)] re-emu-validating archive-r2 with IPC overhead" >> "$MASTER_LOG"

DIR="./results/RTX-8000-archive-r2-ipc-validate"
PROFILE="./results/RTX-8000-adaptive-archive-5r/serving-r2.json"
MODEL="Qwen/Qwen3-8B"
PORT=8100
RATES="2 4 8 16 32"

mkdir -p "$DIR"
for R in $RATES; do
    cp "./results/RTX-8000-v31-2000p/r${R}_real.json" "$DIR/r${R}_real.json" 2>/dev/null || true
done

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
wait_server || { echo "SERVER_NEVER_READY" >> "$MASTER_LOG"; cleanup; exit 1; }

python3 -m vllm.entrypoints.cli.main bench serve \
    --model "$MODEL" --base-url "http://localhost:${PORT}" \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 --request-rate 4 > /dev/null 2>&1 || true
sleep 3

for RATE in $RATES; do
    echo "  [$(date +%T)] r=${RATE}" >> "$MASTER_LOG"
    timeout 1500 python3 -m vllm.entrypoints.cli.main bench serve \
        --model "$MODEL" --base-url "http://localhost:${PORT}" \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 2000 --request-rate $RATE \
        --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
        --save-result --result-dir "$DIR" \
        --result-filename "r${RATE}_emu.json" > /dev/null 2>&1 \
        && echo "    r=${RATE} done" >> "$MASTER_LOG" \
        || echo "    r=${RATE} FAILED" >> "$MASTER_LOG"
    # Kill zombie bench at this rate; server stays for next rate.
    pkill -9 -f "bench serve.*request-rate $RATE " 2>/dev/null || true
    sleep 3
done
cleanup

python3 tools/summarize_matrix.py "$DIR" 2>&1 | tee "$DIR/summary.txt" >> "$MASTER_LOG"

echo "" >> "$MASTER_LOG"
echo "=== IPC chain DONE $(date) ===" >> "$MASTER_LOG"
touch /tmp/vllm_ipc_chain.done
