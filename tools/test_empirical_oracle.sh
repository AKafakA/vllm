#!/bin/bash
# Test Phase 1: empirical distribution sampling (bootstrap from raw samples).
# 1. Rebuild profile from existing trace (stores up to 200 raw samples/bucket)
# 2. Run eval at rate=4 with VLLM_EMULATOR_ORACLE_MODE=distribution
# 3. Compare vs real baseline.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

LOG=/tmp/vllm_empirical_test.log
echo "=== Empirical oracle test at $(date) ===" > "$LOG"

TRACE="./results/RTX-8000-quick/profiles/step_cycle_Qwen3-8B.jsonl"
PROFILE_NEW="./results/RTX-8000-quick/profiles/serving-Qwen3-8B-empirical.json"

# 1. Rebuild profile with raw samples per bucket
echo "=== Rebuild profile with raw samples ===" >> "$LOG"
python3 vllm_emulator/profile/build_serving_profile.py \
    "$TRACE" "/dev/null" "$PROFILE_NEW" >> "$LOG" 2>&1

# Verify new fields present
python3 -c "
import json
p = json.load(open('$PROFILE_NEW'))
fp = p.get('decode_forward_pass', [])
if fp:
    print('First 3 decode buckets:')
    for s in fp[:3]:
        samples = s.get('samples', [])
        print(f'  tt={s[\"total_tokens\"]}: p50={s[\"latency_us\"]/1000:.1f}ms, '
              f'mean={s.get(\"mean_us\",0)/1000:.1f}ms, '
              f'p99={s.get(\"p99_us\",0)/1000:.1f}ms, '
              f'samples={len(samples)}')
# Verify we have samples
n_with_samples = sum(1 for s in fp if s.get('samples'))
print(f'{n_with_samples}/{len(fp)} decode buckets have samples')
" >> "$LOG" 2>&1

# 2. Run eval with empirical distribution oracle
MODEL="Qwen/Qwen3-8B"
PORT=8100
NP=200
RATE=4

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

RESULT_DIR="./results/RTX-8000-quick/empirical"
mkdir -p "$RESULT_DIR"

echo "" >> "$LOG"
echo "=== Emulator eval (distribution oracle with raw samples) ===" >> "$LOG"

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

echo "Benchmarking emu (empirical) at rate=$RATE, $NP prompts..." >> "$LOG"
python3 -m vllm.entrypoints.cli.main bench serve \
    --model "$MODEL" --base-url "http://localhost:${PORT}" \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts $NP --request-rate $RATE \
    --percentile-metrics ttft,tpot,itl --metric-percentiles 50,90,99 \
    --save-result --result-dir "$RESULT_DIR" \
    --result-filename "emu_empirical_r${RATE}.json" > /dev/null 2>&1
cleanup

echo "" >> "$LOG"
echo "=== Comparison: real vs empirical ===" >> "$LOG"
python3 tools/e2e/compare_results.py \
    "./results/RTX-8000-quick/diag/real_r4.json" \
    "$RESULT_DIR/emu_empirical_r${RATE}.json" >> "$LOG" 2>&1

echo "" >> "$LOG"
echo "=== Latency analysis ===" >> "$LOG"
python3 tools/e2e/analyze_request_latency.py \
    "./results/RTX-8000-quick/diag/real_r4.json" \
    "$RESULT_DIR/emu_empirical_r${RATE}.json" >> "$LOG" 2>&1

echo "" >> "$LOG"
echo "=== Three-way comparison (step_cycle vs CDF vs empirical) ===" >> "$LOG"
python3 -c "
import json

real = json.load(open('./results/RTX-8000-quick/diag/real_r4.json'))
sc = json.load(open('./results/RTX-8000-quick/online/Qwen3-8B_r4_emu.json'))
cdf = json.load(open('./results/RTX-8000-quick/variance/emu_distribution_r4.json'))
emp = json.load(open('$RESULT_DIR/emu_empirical_r${RATE}.json'))

print(f'{\"Metric\":>20} {\"Real\":>10} {\"step_cyc\":>12} {\"CDF\":>12} {\"Empirical\":>12}')
print('-' * 72)
for key, label in [
    ('mean_ttft_ms', 'Mean TTFT'),
    ('mean_tpot_ms', 'Mean TPOT'),
    ('p99_tpot_ms', 'P99 TPOT'),
    ('p99_itl_ms', 'P99 ITL'),
    ('output_throughput', 'Output tok/s'),
]:
    rv = real.get(key, 0)
    sv = sc.get(key, 0)
    cv = cdf.get(key, 0)
    ev = emp.get(key, 0)
    def err(v):
        return (v-rv)/rv*100 if rv else 0
    print(f'{label:>20} {rv:>10.2f} {sv:>8.2f}({err(sv):+5.1f}%) {cv:>8.2f}({err(cv):+5.1f}%) {ev:>8.2f}({err(ev):+5.1f}%)')
" >> "$LOG" 2>&1

echo "" >> "$LOG"
echo "=== Done at $(date) ===" >> "$LOG"
cat "$LOG"
