#!/bin/bash
# Test the variance-aware distribution oracle.
# 1. Rebuild profile from existing trace (adds p90/p99/std fields)
# 2. Run eval with VLLM_EMULATOR_ORACLE_MODE=distribution at rate=4
# 3. Compare with real to see if variance closes the TPOT gap.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

LOG=/tmp/vllm_variance_test.log
echo "=== Variance oracle test at $(date) ===" > "$LOG"

TRACE="./results/RTX-8000-quick/profiles/step_cycle_Qwen3-8B.jsonl"
PROFILE_OLD="./results/RTX-8000-quick/profiles/serving-Qwen3-8B-step-cycle.json"
PROFILE_NEW="./results/RTX-8000-quick/profiles/serving-Qwen3-8B-variance.json"

# 1. Rebuild profile with variance fields
echo "=== Rebuild profile with variance fields ===" >> "$LOG"
python3 vllm_emulator/profile/build_serving_profile.py \
    "$TRACE" "/dev/null" "$PROFILE_NEW" >> "$LOG" 2>&1

# Verify new fields
python3 -c "
import json
p = json.load(open('$PROFILE_NEW'))
fp = p.get('decode_forward_pass', [])
if fp:
    print('New profile sample (first 3 decode buckets):')
    for s in fp[:3]:
        print(f'  tt={s[\"total_tokens\"]}: p50={s[\"latency_us\"]/1000:.1f}ms, '
              f'p90={s.get(\"p90_us\", 0)/1000:.1f}ms, '
              f'p99={s.get(\"p99_us\", 0)/1000:.1f}ms, '
              f'std={s.get(\"std_us\", 0)/1000:.2f}ms')
" >> "$LOG" 2>&1

# 2. Run eval with distribution oracle
MODEL="Qwen/Qwen3-8B"
PORT=8100
NP=200
RATE=4

cleanup() {
    pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
    pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
    fuser ${PORT}/tcp 2>/dev/null | xargs -r kill -9 2>/dev/null || true
    sleep 5
}

wait_server() {
    for i in $(seq 1 300); do
        curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1 && return 0
        sleep 1
    done
    echo "ERROR: Server timeout" >> "$LOG" && return 1
}

RESULT_DIR="./results/RTX-8000-quick/variance"
mkdir -p "$RESULT_DIR"

echo "" >> "$LOG"
echo "=== Eval with distribution oracle (rate=$RATE, $NP prompts) ===" >> "$LOG"

cleanup
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="$PROFILE_NEW" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
VLLM_EMULATOR_PREP_SURROGATE=1 \
VLLM_EMULATOR_ORACLE_MODE=distribution \
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > "$RESULT_DIR/emu_server.log" 2>&1 &
wait_server
grep "ExecutorEmulatorHook" "$RESULT_DIR/emu_server.log" | head -3 >> "$LOG"

# Warmup
python3 -m vllm.entrypoints.cli.main bench serve \
    --model "$MODEL" --base-url "http://localhost:${PORT}" \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 50 --request-rate 4 > /dev/null 2>&1
sleep 2

echo "Benchmarking emu (distribution) at rate=$RATE..." >> "$LOG"
python3 -m vllm.entrypoints.cli.main bench serve \
    --model "$MODEL" --base-url "http://localhost:${PORT}" \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts $NP --request-rate $RATE \
    --percentile-metrics ttft,tpot,itl --metric-percentiles 50,90,99 \
    --save-result --result-dir "$RESULT_DIR" \
    --result-filename "emu_distribution_r${RATE}.json" > /dev/null 2>&1
cleanup

echo "" >> "$LOG"
echo "=== Comparison: real vs step_cycle vs distribution ===" >> "$LOG"
python3 tools/e2e/compare_results.py \
    "./results/RTX-8000-quick/diag/real_r4.json" \
    "$RESULT_DIR/emu_distribution_r${RATE}.json" >> "$LOG" 2>&1

echo "" >> "$LOG"
echo "=== Distribution mode vs step_cycle mode ===" >> "$LOG"
python3 tools/e2e/analyze_request_latency.py \
    "./results/RTX-8000-quick/diag/real_r4.json" \
    "$RESULT_DIR/emu_distribution_r${RATE}.json" >> "$LOG" 2>&1

echo "" >> "$LOG"
echo "=== Done at $(date) ===" >> "$LOG"
cat "$LOG"
