#!/bin/bash
# Detailed tracing: capture real vs emu step_cycle traces at rate=4.
# Then analyze inter-step timing, per-request completion, etc.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

LOG=/tmp/vllm_trace_comparison.log
echo "=== Real vs Emu deep trace at $(date) ===" > "$LOG"

MODEL="Qwen/Qwen3-8B"
PORT=8100
NP=200
RATE=4
PROFILE="./results/RTX-8000-adaptive/profiles/serving-Qwen3-8B-adaptive-filtered.json"
RESULT_DIR="./results/RTX-8000-adaptive/trace-compare"
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
    return 1
}

# --- REAL with full tracing ---
cleanup
rm -f "$RESULT_DIR/real_trace.jsonl"
echo "Starting REAL server with step_cycle trace..." >> "$LOG"
VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
VLLM_EMULATOR_STEP_TRACE_OUTPUT="$RESULT_DIR/real_trace.jsonl" \
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > "$RESULT_DIR/real_server.log" 2>&1 &
wait_server
python3 -m vllm.entrypoints.cli.main bench serve \
    --model "$MODEL" --base-url "http://localhost:${PORT}" \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 50 --request-rate 4 > /dev/null 2>&1
sleep 2
echo "Benchmarking REAL at rate=$RATE..." >> "$LOG"
python3 -m vllm.entrypoints.cli.main bench serve \
    --model "$MODEL" --base-url "http://localhost:${PORT}" \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts $NP --request-rate $RATE \
    --save-result --result-dir "$RESULT_DIR" \
    --result-filename "real.json" > /dev/null 2>&1
cleanup

# --- EMU with full tracing ---
rm -f "$RESULT_DIR/emu_trace.jsonl"
echo "Starting EMU server with step_cycle + hook trace..." >> "$LOG"
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
VLLM_EMULATOR_PREP_SURROGATE=1 \
VLLM_EMULATOR_ORACLE_MODE=distribution \
VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
VLLM_EMULATOR_STEP_TRACE_OUTPUT="$RESULT_DIR/emu_trace.jsonl" \
VLLM_EMULATOR_HOOK_TRACE="$RESULT_DIR/emu_hook.csv" \
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > "$RESULT_DIR/emu_server.log" 2>&1 &
wait_server
python3 -m vllm.entrypoints.cli.main bench serve \
    --model "$MODEL" --base-url "http://localhost:${PORT}" \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 50 --request-rate 4 > /dev/null 2>&1
sleep 2
echo "Benchmarking EMU at rate=$RATE..." >> "$LOG"
python3 -m vllm.entrypoints.cli.main bench serve \
    --model "$MODEL" --base-url "http://localhost:${PORT}" \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts $NP --request-rate $RATE \
    --save-result --result-dir "$RESULT_DIR" \
    --result-filename "emu.json" > /dev/null 2>&1
cleanup

echo "" >> "$LOG"
echo "=== Trace comparison ===" >> "$LOG"
python3 << 'PYEOF' 2>&1 | tee -a "$LOG"
import json
import statistics
from collections import defaultdict

def load(path):
    recs = []
    for line in open(path):
        r = json.loads(line)
        if r.get("_header") or "step_cycle_us" not in r:
            continue
        recs.append(r)
    return recs

real = load('./results/RTX-8000-adaptive/trace-compare/real_trace.jsonl')
emu = load('./results/RTX-8000-adaptive/trace-compare/emu_trace.jsonl')

# Skip warmup
real = real[200:]
emu = emu[200:]
print(f"Real: {len(real)} steps, Emu: {len(emu)} steps")

# Compare distributions
for name, data in [("Real", real), ("Emu", emu)]:
    lats = [r['step_cycle_us']/1000 for r in data]
    decode = [r['step_cycle_us']/1000 for r in data if r.get('num_new_reqs',0)==0]
    prefill = [r['step_cycle_us']/1000 for r in data if r.get('num_new_reqs',0)>0]
    print(f"\n{name} total: mean={statistics.mean(lats):.2f}, median={statistics.median(lats):.2f}, p99={sorted(lats)[int(len(lats)*0.99)]:.2f}ms")
    print(f"  decode (n={len(decode)}): mean={statistics.mean(decode):.2f}ms, median={statistics.median(decode):.2f}, p99={sorted(decode)[int(len(decode)*0.99)]:.2f}ms")
    if prefill:
        print(f"  prefill (n={len(prefill)}): mean={statistics.mean(prefill):.2f}ms, median={statistics.median(prefill):.2f}")

# Concurrency distribution: how often is conc at each level?
print("\n=== Concurrency distribution ===")
for name, data in [("Real", real), ("Emu", emu)]:
    concs = [r.get('num_new_reqs',0)+r.get('num_decode_seqs',0) for r in data]
    print(f"{name}: mean_conc={statistics.mean(concs):.1f}, median={statistics.median(concs)}, p99={sorted(concs)[int(len(concs)*0.99)]}, max={max(concs)}")
    # Histogram
    buckets = [0]*10
    for c in concs:
        if c < 10: buckets[0] += 1
        elif c < 20: buckets[1] += 1
        elif c < 30: buckets[2] += 1
        elif c < 40: buckets[3] += 1
        elif c < 50: buckets[4] += 1
        elif c < 60: buckets[5] += 1
        else: buckets[6] += 1
    total = sum(buckets)
    print(f"  <10: {buckets[0]/total*100:.0f}%, 10-19: {buckets[1]/total*100:.0f}%, 20-29: {buckets[2]/total*100:.0f}%, 30-39: {buckets[3]/total*100:.0f}%, 40-49: {buckets[4]/total*100:.0f}%, 50+: {(buckets[5]+buckets[6])/total*100:.0f}%")

# Rate of step production (steps per second wall-clock)
print("\n=== Step production rate ===")
# use step_count divided by total elapsed time from step_cycle sums
# Actually, step_cycle_us sums give total engine time
for name, data in [("Real", real), ("Emu", emu)]:
    total_us = sum(r['step_cycle_us'] for r in data)
    steps_per_sec = len(data) / (total_us / 1e6)
    print(f"{name}: {len(data)} steps in {total_us/1e6:.2f}s = {steps_per_sec:.1f} steps/s")

# How many tokens per step? = avg concurrency (assuming decode)
# tokens/s = steps/s * avg_conc
for name, data in [("Real", real), ("Emu", emu)]:
    total_us = sum(r['step_cycle_us'] for r in data)
    steps_per_sec = len(data) / (total_us / 1e6)
    avg_conc = statistics.mean([r.get('num_new_reqs',0)+r.get('num_decode_seqs',0) for r in data])
    tokens_per_sec = steps_per_sec * avg_conc
    print(f"{name}: implied tokens/s = {steps_per_sec:.1f} * {avg_conc:.1f} = {tokens_per_sec:.0f}")
PYEOF

echo "=== Done at $(date) ===" >> "$LOG"
