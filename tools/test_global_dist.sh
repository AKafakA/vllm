#!/bin/bash
# Test global distribution oracle + capture emu step_cycle trace.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

LOG=/tmp/vllm_global_test.log
echo "=== Global distribution test at $(date) ===" > "$LOG"

PROFILE="./results/RTX-8000-quick/profiles/serving-Qwen3-8B-2d-empirical.json"

# Quick check: what's the global distribution's stats?
python3 -c "
import json, statistics
p = json.load(open('$PROFILE'))
# Gather all samples from decode_forward_pass
decode_samples = []
for b in p.get('decode_forward_pass', []):
    decode_samples.extend(b.get('samples', []))
prefill_samples = []
for b in p.get('prefill_forward_pass', []):
    prefill_samples.extend(b.get('samples', []))
print(f'Global decode samples: {len(decode_samples)}')
if decode_samples:
    print(f'  median={statistics.median(decode_samples)/1000:.1f}ms')
    print(f'  mean={statistics.mean(decode_samples)/1000:.1f}ms')
    print(f'  p99={sorted(decode_samples)[int(len(decode_samples)*0.99)]/1000:.1f}ms')
print(f'Global prefill samples: {len(prefill_samples)}')
if prefill_samples:
    print(f'  median={statistics.median(prefill_samples)/1000:.1f}ms')
    print(f'  mean={statistics.mean(prefill_samples)/1000:.1f}ms')
" >> "$LOG" 2>&1

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

RESULT_DIR="./results/RTX-8000-quick/global"
mkdir -p "$RESULT_DIR"

echo "" >> "$LOG"
echo "=== Emu eval (distribution_global) with step_cycle trace ===" >> "$LOG"

cleanup
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
VLLM_EMULATOR_PREP_SURROGATE=1 \
VLLM_EMULATOR_ORACLE_MODE=distribution_global \
VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
VLLM_EMULATOR_STEP_TRACE_OUTPUT="$RESULT_DIR/emu_step_cycle.jsonl" \
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

echo "Benchmarking emu (global) at rate=$RATE, $NP prompts..." >> "$LOG"
python3 -m vllm.entrypoints.cli.main bench serve \
    --model "$MODEL" --base-url "http://localhost:${PORT}" \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts $NP --request-rate $RATE \
    --percentile-metrics ttft,tpot,itl --metric-percentiles 50,90,99 \
    --save-result --result-dir "$RESULT_DIR" \
    --result-filename "emu_global_r${RATE}.json" > /dev/null 2>&1
cleanup

echo "" >> "$LOG"
echo "=== Emu step_cycle distribution (vs real) ===" >> "$LOG"
python3 -c "
import json, statistics

def load(path):
    recs = []
    for line in open(path):
        r = json.loads(line)
        if r.get('_header') or 'step_cycle_us' not in r:
            continue
        recs.append(r)
    return recs[200:]  # skip warmup

real = load('./results/RTX-8000-quick/diag/real_step_cycle_r4.jsonl')
emu = load('$RESULT_DIR/emu_step_cycle.jsonl')

print(f'Real: {len(real)} steps, Emu: {len(emu)} steps')

for name, data in [('Real', real), ('Emu (global)', emu)]:
    decode = [r for r in data if r.get('num_new_reqs', 0) == 0]
    if decode:
        lats = sorted([r['step_cycle_us']/1000 for r in decode])
        print(f'{name} decode: median={statistics.median(lats):.1f}ms, '
              f'mean={statistics.mean(lats):.1f}ms, '
              f'p99={lats[int(len(lats)*0.99)]:.1f}ms, '
              f'std={statistics.stdev(lats):.1f}ms')
" >> "$LOG" 2>&1

echo "" >> "$LOG"
echo "=== Five-way comparison ===" >> "$LOG"
python3 -c "
import json

real = json.load(open('./results/RTX-8000-quick/diag/real_r4.json'))
sc = json.load(open('./results/RTX-8000-quick/online/Qwen3-8B_r4_emu.json'))
cdf = json.load(open('./results/RTX-8000-quick/variance/emu_distribution_r4.json'))
emp1d = json.load(open('./results/RTX-8000-quick/empirical/emu_empirical_r4.json'))
emp2d = json.load(open('./results/RTX-8000-quick/empirical-2d/emu_2d_r4.json'))
globl = json.load(open('$RESULT_DIR/emu_global_r4.json'))

print(f'{\"Metric\":>16} {\"Real\":>8} {\"SC*\":>12} {\"CDF\":>12} {\"E1D\":>12} {\"E2D\":>12} {\"Global\":>12}')
print('-' * 88)
for key, label in [
    ('mean_ttft_ms', 'Mean TTFT'),
    ('median_ttft_ms', 'Med TTFT'),
    ('mean_tpot_ms', 'Mean TPOT'),
    ('median_tpot_ms', 'Med TPOT'),
    ('p99_tpot_ms', 'P99 TPOT'),
    ('p99_itl_ms', 'P99 ITL'),
    ('std_tpot_ms', 'std TPOT'),
    ('output_throughput', 'tok/s'),
    ('max_concurrent_requests', 'max_conc'),
]:
    rv = real.get(key, 0)
    row = [sc.get(key, 0), cdf.get(key, 0), emp1d.get(key, 0), emp2d.get(key, 0), globl.get(key, 0)]
    def err(v):
        return (v-rv)/rv*100 if rv else 0
    parts = []
    for v in row:
        parts.append(f'{v:>6.2f}({err(v):+5.1f}%)')
    print(f'{label:>16} {rv:>8.2f} ' + ' '.join(parts))
" >> "$LOG" 2>&1

echo "" >> "$LOG"
echo "=== Done at $(date) ===" >> "$LOG"
cat "$LOG"
