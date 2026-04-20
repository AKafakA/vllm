#!/bin/bash
# Saturation-focused supplementary profiling round.
# Targets dense coverage of (tt≥300, conc≥200) buckets where archive-r2's
# profile had only 1,173 samples and they were ~99% concentrated at tt=256.
# Strictly additive: archive-r2's trace stays verbatim; this adds new
# samples and rebuilds the profile.
#
# Pure 256/128 shape (same as archive-r2) — no shape confounding.
# Three saturation rates × 10,000 prompts each, pure decode workload.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

MODEL="Qwen/Qwen3-8B"
PORT=8100
OUT_DIR="./results/RTX-8000-profile-archive-r2-sat-supp"
ARCHIVE_TRACE="./results/RTX-8000-adaptive-archive-5r/step_cycle_trace.jsonl"
SUPP_TRACE="$OUT_DIR/supp_trace.jsonl"
COMBINED_TRACE="$OUT_DIR/combined_trace.jsonl"
PROFILE="$OUT_DIR/serving-full.json"

MASTER_LOG="$OUT_DIR/run.log"
mkdir -p "$OUT_DIR/logs"
rm -f "$SUPP_TRACE"
echo "=== archive-r2 saturation supplementary start $(date) ===" > "$MASTER_LOG"
touch /tmp/vllm_sat_supp.started

if [ ! -f "$ARCHIVE_TRACE" ]; then
    echo "FATAL: archive-r2 trace missing at $ARCHIVE_TRACE" >> "$MASTER_LOG"
    exit 1
fi

cleanup() {
    pkill -TERM -f "vllm.entrypoints" 2>/dev/null
    sleep 3
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
    VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
    VLLM_EMULATOR_STEP_TRACE_OUTPUT="$SUPP_TRACE" \
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > "$OUT_DIR/logs/server.log" 2>&1 &
wait_server || { echo "SERVER_NEVER_READY" >> "$MASTER_LOG"; cleanup; exit 1; }
echo "  [$(date +%T)] server ready" >> "$MASTER_LOG"

# Warmup — compile CUDA graphs, pad engine state (matches archive's pattern).
echo "  [$(date +%T)] warmup rate=4 n=500" >> "$MASTER_LOG"
python3 -m vllm.entrypoints.cli.main bench serve \
    --model "$MODEL" --base-url "http://localhost:${PORT}" \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 500 --request-rate 4 > /dev/null 2>&1 || true
sleep 3

# Saturation rounds. Each wrapped with profiling markers so only the
# saturation-regime samples go into the trace.
echo '{"__marker__": "profiling_start"}' >> "$SUPP_TRACE"

for RATE in 16 32 inf; do
    echo "  [$(date +%T)] saturation rate=${RATE} n=10000" >> "$MASTER_LOG"
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "$MODEL" --base-url "http://localhost:${PORT}" \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 10000 --request-rate $RATE > /dev/null 2>&1 || true
    sleep 3
    echo "    rate=${RATE} done" >> "$MASTER_LOG"
done

echo '{"__marker__": "profiling_stop"}' >> "$SUPP_TRACE"

cleanup

echo "" >> "$MASTER_LOG"
echo "=== Concatenating traces $(date) ===" >> "$MASTER_LOG"
cp "$ARCHIVE_TRACE" "$COMBINED_TRACE"
cat "$SUPP_TRACE" >> "$COMBINED_TRACE"
wc -l "$ARCHIVE_TRACE" "$SUPP_TRACE" "$COMBINED_TRACE" >> "$MASTER_LOG"

echo "" >> "$MASTER_LOG"
echo "=== Building profile $(date) ===" >> "$MASTER_LOG"
python3 vllm_emulator/profile/build_serving_profile_filtered.py \
    "$COMBINED_TRACE" "$PROFILE" \
    --tt-bucket-width 1 --conc-bucket-width 5 >> "$MASTER_LOG" 2>&1

# Carry sched_overhead_table forward (IPC sweep is hardware-specific,
# independent of this saturation sampling).
python3 - << PYEOF >> "$MASTER_LOG" 2>&1
import json
src = json.load(open("./results/RTX-8000-adaptive-archive-5r/serving-r2.json"))
dst = json.load(open("$PROFILE"))
if "sched_overhead_table" in src:
    dst["sched_overhead_table"] = src["sched_overhead_table"]
if "sched_overhead_table_v2" in src:
    dst["sched_overhead_table_v2"] = src["sched_overhead_table_v2"]
json.dump(dst, open("$PROFILE", "w"), indent=2)
print("Carried sched_overhead_table forward from archive-r2")
PYEOF

echo "=== DONE $(date) ===" >> "$MASTER_LOG"
touch /tmp/vllm_sat_supp.done
