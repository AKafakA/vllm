#!/bin/bash
# Diagnostic: compare real vs emulator per-step traces at rate=4.
# Captures both real step-cycle trace and emulator hook trace for
# side-by-side analysis of where the latency gap comes from.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

MODEL="Qwen/Qwen3-8B"
PORT=8100
RESULT_DIR="./results/RTX-8000-quick/diag"
PROFILE="./results/RTX-8000-quick/profiles/serving-Qwen3-8B-step-cycle.json"
NP=200
RATE=4

mkdir -p "$RESULT_DIR"

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
    echo "ERROR: Server timeout" && return 1
}

warmup() {
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "$MODEL" --base-url "http://localhost:${PORT}" \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 50 --request-rate 4 > /dev/null 2>&1
    sleep 2
}

echo "=== Diagnostic: Rate=$RATE real vs emu per-step traces ==="

# === 1. Real GPU with step-cycle tracing ===
cleanup
echo "Starting real server with step-cycle tracing..."
VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
VLLM_EMULATOR_STEP_TRACE_OUTPUT="$RESULT_DIR/real_step_cycle_r${RATE}.jsonl" \
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > "$RESULT_DIR/real_server.log" 2>&1 &
wait_server
warmup
echo "Benchmarking real at rate=$RATE ($NP prompts)..."
python3 -m vllm.entrypoints.cli.main bench serve \
    --model "$MODEL" --base-url "http://localhost:${PORT}" \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts $NP --request-rate $RATE \
    --save-result --result-dir "$RESULT_DIR" --result-filename "real_r${RATE}.json" \
    > /dev/null 2>&1
cleanup

REAL_RECORDS=$(wc -l < "$RESULT_DIR/real_step_cycle_r${RATE}.jsonl")
echo "Real trace: $REAL_RECORDS records"

# === 2. Emulator with hook trace ===
echo "Starting emulator server with hook tracing..."
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
VLLM_EMULATOR_PREP_SURROGATE=1 \
VLLM_EMULATOR_HOOK_TRACE="$RESULT_DIR/emu_hook_trace_r${RATE}.csv" \
VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
VLLM_EMULATOR_STEP_TRACE_OUTPUT="$RESULT_DIR/emu_step_cycle_r${RATE}.jsonl" \
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > "$RESULT_DIR/emu_server.log" 2>&1 &
wait_server
warmup
echo "Benchmarking emulator at rate=$RATE ($NP prompts)..."
python3 -m vllm.entrypoints.cli.main bench serve \
    --model "$MODEL" --base-url "http://localhost:${PORT}" \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts $NP --request-rate $RATE \
    --save-result --result-dir "$RESULT_DIR" --result-filename "emu_r${RATE}.json" \
    > /dev/null 2>&1
cleanup

EMU_RECORDS=$(wc -l < "$RESULT_DIR/emu_hook_trace_r${RATE}.csv")
echo "Emu hook trace: $EMU_RECORDS records"

# === 3. Compare results ===
echo ""
echo "=== Benchmark comparison ==="
python3 tools/e2e/compare_results.py \
    "$RESULT_DIR/real_r${RATE}.json" \
    "$RESULT_DIR/emu_r${RATE}.json"

# === 4. Analyze per-step traces ===
echo ""
echo "=== Per-step trace analysis ==="
python3 -c "
import json, statistics

# Real step-cycle trace
real_records = []
for line in open('$RESULT_DIR/real_step_cycle_r${RATE}.jsonl'):
    r = json.loads(line)
    if 'step_cycle_us' in r:
        real_records.append(r)

# Group by concurrency
from collections import defaultdict
real_by_conc = defaultdict(list)
for r in real_records:
    n = r.get('num_new_reqs', 0) + r.get('num_decode_seqs', 0)
    real_by_conc[n].append(r['step_cycle_us'])

print(f'Real: {len(real_records)} steps')
print(f'Step cycle by concurrency:')
for n in sorted(real_by_conc.keys()):
    lats = real_by_conc[n]
    if len(lats) >= 3:
        print(f'  N={n:>3}: median={statistics.median(lats)/1000:.1f}ms, '
              f'mean={statistics.mean(lats)/1000:.1f}ms (n={len(lats)})')

# Emu hook trace
import csv
emu_records = []
with open('$RESULT_DIR/emu_hook_trace_r${RATE}.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        emu_records.append(row)

print(f'\nEmu: {len(emu_records)} steps')
# Compare oracle prediction vs real at same total_tokens
emu_by_tt = defaultdict(list)
for r in emu_records:
    tt = int(r['tt'])
    oracle = float(r['oracle_us'])
    emu_by_tt[tt].append(oracle)

print(f'Oracle prediction by total_tokens:')
for tt in sorted(emu_by_tt.keys())[:20]:
    preds = emu_by_tt[tt]
    if len(preds) >= 2:
        print(f'  tt={tt:>4}: oracle_median={statistics.median(preds)/1000:.1f}ms (n={len(preds)})')
"

echo ""
echo "=== Files saved ==="
ls -la "$RESULT_DIR/"
echo ""
echo "Done. Analyze traces in: $RESULT_DIR"
