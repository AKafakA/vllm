#!/bin/bash
# Test Phase 2: 2D (tt, concurrency) empirical distribution sampling.
# Compares against 1D empirical (Phase 1) and CDF (previous).
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

LOG=/tmp/vllm_2d_empirical_test.log
echo "=== 2D empirical oracle test at $(date) ===" > "$LOG"

TRACE="./results/RTX-8000-quick/profiles/step_cycle_Qwen3-8B.jsonl"
PROFILE_NEW="./results/RTX-8000-quick/profiles/serving-Qwen3-8B-2d-empirical.json"

# 1. Rebuild profile with 2D distribution samples
echo "=== Rebuild profile with 2D (tt, conc) samples ===" >> "$LOG"
python3 vllm_emulator/profile/build_serving_profile.py \
    "$TRACE" "/dev/null" "$PROFILE_NEW" >> "$LOG" 2>&1

# Verify 2D distribution fields
python3 -c "
import json
p = json.load(open('$PROFILE_NEW'))
d2 = p.get('step_cycle_2d_distribution', [])
print(f'2D distribution: {len(d2)} (tt, conc) cells')
# Check sample counts
n_with_samples = sum(1 for c in d2 if c.get('samples'))
print(f'{n_with_samples}/{len(d2)} cells have samples')
# Show tt coverage
tts = sorted(set(c['tt'] for c in d2))
concs = sorted(set(c['conc'] for c in d2))
print(f'tt range: {tts[0]}-{tts[-1]} ({len(tts)} buckets)')
print(f'conc buckets: {concs}')
# Show sample size for first few cells
for c in d2[:5]:
    s = c.get('samples', [])
    print(f'  tt={c[\"tt\"]}, conc={c[\"conc\"]}: '
          f'p50={c[\"latency_us\"]/1000:.1f}ms, '
          f'mean={c.get(\"mean_us\", 0)/1000:.1f}ms, '
          f'p99={c.get(\"p99_us\", 0)/1000:.1f}ms, '
          f'n_samples={len(s)}')
" >> "$LOG" 2>&1

# 2. Run eval with 2D distribution oracle
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

RESULT_DIR="./results/RTX-8000-quick/empirical-2d"
mkdir -p "$RESULT_DIR"

echo "" >> "$LOG"
echo "=== Emulator eval (2D empirical distribution) ===" >> "$LOG"

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

echo "Benchmarking emu (2D empirical) at rate=$RATE, $NP prompts..." >> "$LOG"
python3 -m vllm.entrypoints.cli.main bench serve \
    --model "$MODEL" --base-url "http://localhost:${PORT}" \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts $NP --request-rate $RATE \
    --percentile-metrics ttft,tpot,itl --metric-percentiles 50,90,99 \
    --save-result --result-dir "$RESULT_DIR" \
    --result-filename "emu_2d_r${RATE}.json" > /dev/null 2>&1
cleanup

echo "" >> "$LOG"
echo "=== Four-way comparison ===" >> "$LOG"
python3 -c "
import json

real = json.load(open('./results/RTX-8000-quick/diag/real_r4.json'))
sc = json.load(open('./results/RTX-8000-quick/online/Qwen3-8B_r4_emu.json'))  # step_cycle (100 prompts)
cdf = json.load(open('./results/RTX-8000-quick/variance/emu_distribution_r4.json'))  # CDF
emp1d = json.load(open('./results/RTX-8000-quick/empirical/emu_empirical_r4.json'))  # 1D empirical
emp2d = json.load(open('$RESULT_DIR/emu_2d_r${RATE}.json'))  # 2D empirical

print(f'{\"Metric\":>18} {\"Real\":>10} {\"step_cyc*\":>12} {\"CDF\":>12} {\"Emp1D\":>12} {\"Emp2D\":>12}')
print('  (*step_cyc is 100-prompt run; others are 200)')
print('-' * 84)
for key, label in [
    ('mean_ttft_ms', 'Mean TTFT'),
    ('median_ttft_ms', 'Med TTFT'),
    ('p99_ttft_ms', 'P99 TTFT'),
    ('mean_tpot_ms', 'Mean TPOT'),
    ('median_tpot_ms', 'Med TPOT'),
    ('p99_tpot_ms', 'P99 TPOT'),
    ('mean_itl_ms', 'Mean ITL'),
    ('p99_itl_ms', 'P99 ITL'),
    ('std_tpot_ms', 'std TPOT'),
    ('output_throughput', 'tok/s'),
    ('max_concurrent_requests', 'max_conc'),
]:
    rv = real.get(key, 0)
    sv = sc.get(key, 0)
    cv = cdf.get(key, 0)
    e1 = emp1d.get(key, 0)
    e2 = emp2d.get(key, 0)
    def err(v):
        return (v-rv)/rv*100 if rv else 0
    if rv == 0 and sv == 0 and cv == 0 and e1 == 0 and e2 == 0:
        continue
    print(f'{label:>18} {rv:>10.2f} {sv:>8.2f}({err(sv):+5.1f}%) {cv:>8.2f}({err(cv):+5.1f}%) {e1:>8.2f}({err(e1):+5.1f}%) {e2:>8.2f}({err(e2):+5.1f}%)')
" >> "$LOG" 2>&1

echo "" >> "$LOG"
echo "=== Done at $(date) ===" >> "$LOG"
cat "$LOG"
