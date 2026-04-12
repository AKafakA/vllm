#!/bin/bash
# Test all oracle modes on dense profile at rate=4.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

LOG=/tmp/vllm_dense_modes_test.log
echo "=== Dense profile all-modes test at $(date) ===" > "$LOG"

MODEL="Qwen/Qwen3-8B"
PORT=8100
NP=200
RATE=4
PROFILE="./results/RTX-8000-dense/profiles/serving-Qwen3-8B-dense.json"
RESULT_DIR="./results/RTX-8000-dense/modes"
mkdir -p "$RESULT_DIR"

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
    echo "ERROR: Server timeout" >> "$LOG"
    return 1
}

test_mode() {
    local MODE=$1
    echo "" >> "$LOG"
    echo "=== Testing mode: $MODE ===" >> "$LOG"

    cleanup
    VLLM_EMULATOR_ENABLE_ORACLE=1 \
    VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
    VLLM_EMULATOR_MODE=realtime \
    VLLM_EMULATOR_EXECUTOR_HOOK=1 \
    VLLM_EMULATOR_PREP_SURROGATE=1 \
    VLLM_EMULATOR_ORACLE_MODE="$MODE" \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
        > "$RESULT_DIR/${MODE}_server.log" 2>&1 &
    wait_server
    grep "ExecutorEmulatorHook" "$RESULT_DIR/${MODE}_server.log" | head -1 >> "$LOG"

    # Warmup
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "$MODEL" --base-url "http://localhost:${PORT}" \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 50 --request-rate 4 > /dev/null 2>&1
    sleep 2

    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "$MODEL" --base-url "http://localhost:${PORT}" \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts $NP --request-rate $RATE \
        --percentile-metrics ttft,tpot,itl --metric-percentiles 50,90,99 \
        --save-result --result-dir "$RESULT_DIR" \
        --result-filename "${MODE}_r${RATE}.json" > /dev/null 2>&1
    cleanup
    echo "  Done $MODE" >> "$LOG"
}

# Test all 4 modes
for mode in step_cycle 2d distribution distribution_global; do
    test_mode "$mode"
done

echo "" >> "$LOG"
echo "=== 4-way comparison (dense profile) ===" >> "$LOG"
python3 -c "
import json

real = json.load(open('./results/RTX-8000-quick/diag/real_r4.json'))
modes = {}
for m in ['step_cycle', '2d', 'distribution', 'distribution_global']:
    try:
        modes[m] = json.load(open('$RESULT_DIR/' + m + '_r${RATE}.json'))
    except FileNotFoundError:
        modes[m] = {}

print(f'{\"Metric\":>16} {\"Real\":>10} {\"step_cyc\":>14} {\"2d\":>14} {\"distrib\":>14} {\"global\":>14}')
print('-' * 84)
for key, label in [
    ('mean_ttft_ms', 'Mean TTFT'),
    ('mean_tpot_ms', 'Mean TPOT'),
    ('median_tpot_ms', 'Med TPOT'),
    ('p99_tpot_ms', 'P99 TPOT'),
    ('p99_itl_ms', 'P99 ITL'),
    ('std_tpot_ms', 'std TPOT'),
    ('output_throughput', 'tok/s'),
    ('max_concurrent_requests', 'max_conc'),
]:
    rv = real.get(key, 0)
    def fmt(d):
        v = d.get(key, 0)
        err = (v-rv)/rv*100 if rv else 0
        return f'{v:>6.2f}({err:+5.1f}%)'
    row = [fmt(modes[m]) for m in ['step_cycle', '2d', 'distribution', 'distribution_global']]
    print(f'{label:>16} {rv:>10.2f} ' + ' '.join(row))
" >> "$LOG" 2>&1

echo "" >> "$LOG"
echo "=== Done at $(date) ===" >> "$LOG"
cat "$LOG"
