#!/bin/bash
# Analyze existing traces, then re-run diagnostic with hook trace fix.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

DIAG_DIR="./results/RTX-8000-quick/diag"
PROFILE="./results/RTX-8000-quick/profiles/serving-Qwen3-8B-step-cycle.json"
LOG=/tmp/vllm_analysis.log

echo "=== Analysis at $(date) ===" > "$LOG"

# Phase 1: Analyze existing traces
echo "=== Phase 1: Analyze existing real vs emu traces ===" >> "$LOG"
python3 tools/e2e/analyze_real_vs_emu.py \
    "$DIAG_DIR/real_step_cycle_r4.jsonl" \
    "$DIAG_DIR/emu_step_cycle_r4.jsonl" \
    "$PROFILE" >> "$LOG" 2>&1

# Phase 2: Re-run diagnostic with hook trace fix (flushes every 50 steps)
echo "" >> "$LOG"
echo "=== Phase 2: Re-run diagnostic with hook trace ==" >> "$LOG"

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

# Emulator with hook trace (uses updated code with flush every 50 steps)
cleanup
echo "Starting emulator with hook trace..." >> "$LOG"
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
VLLM_EMULATOR_PREP_SURROGATE=1 \
VLLM_EMULATOR_HOOK_TRACE="$DIAG_DIR/emu_hook_trace_r${RATE}_v2.csv" \
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > "$DIAG_DIR/emu_server_v2.log" 2>&1 &
wait_server

python3 -m vllm.entrypoints.cli.main bench serve \
    --model "$MODEL" --base-url "http://localhost:${PORT}" \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 50 --request-rate 4 > /dev/null 2>&1
sleep 2

echo "Benchmarking emu at rate=$RATE..." >> "$LOG"
python3 -m vllm.entrypoints.cli.main bench serve \
    --model "$MODEL" --base-url "http://localhost:${PORT}" \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts $NP --request-rate $RATE > /dev/null 2>&1
cleanup

HOOK_LINES=$(wc -l < "$DIAG_DIR/emu_hook_trace_r${RATE}_v2.csv" 2>/dev/null || echo 0)
echo "Hook trace v2: $HOOK_LINES lines" >> "$LOG"

# Analyze hook trace
if [ "$HOOK_LINES" -gt 1 ]; then
    echo "" >> "$LOG"
    echo "=== Hook trace analysis ===" >> "$LOG"
    python3 -c "
import csv, statistics
from collections import defaultdict

rows = []
with open('$DIAG_DIR/emu_hook_trace_r${RATE}_v2.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

print(f'Hook trace: {len(rows)} steps')

# Group by n_reqs
by_nreqs = defaultdict(list)
for r in rows:
    n = int(r['n_reqs'])
    by_nreqs[n].append(r)

print(f'Oracle prediction by concurrent requests:')
for n in sorted(by_nreqs.keys()):
    recs = by_nreqs[n]
    if len(recs) < 3:
        continue
    oracle_vals = [float(r['oracle_us']) for r in recs]
    total_lat = [float(r['total_latency_us']) for r in recs]
    sched_comp = [float(r['sched_comp_us']) for r in recs]
    print(f'  N={n:>3}: oracle={statistics.median(oracle_vals)/1000:.1f}ms, '
          f'total={statistics.median(total_lat)/1000:.1f}ms, '
          f'sched_comp={statistics.median(sched_comp)/1000:.1f}ms (n={len(recs)})')
" >> "$LOG" 2>&1
fi

echo "" >> "$LOG"
echo "=== Done at $(date) ===" >> "$LOG"
cat "$LOG"
