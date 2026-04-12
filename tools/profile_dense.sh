#!/bin/bash
# Dense profiling matching previous Qwen2.5 setup: 8 rates × 200 prompts each.
# Replaces the sparse 3-rate quick profile.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

LOG=/tmp/vllm_dense_profile.log
echo "=== Dense profiling at $(date) ===" > "$LOG"

MODEL="Qwen/Qwen3-8B"
PORT=8100
INPUT_LEN=256
OUTPUT_LEN=128
WARMUP_PROMPTS=200  # heavy warmup like previous
PROFILE_RATES="0.5 1 2 3 4 6 8 12"  # 8 rates matching Qwen2.5 full profile
RESULT_DIR="./results/RTX-8000-dense"
PROFILES_DIR="${RESULT_DIR}/profiles"
LOGS_DIR="${RESULT_DIR}/logs"
mkdir -p "$PROFILES_DIR" "$LOGS_DIR"

TRACE_FILE="${PROFILES_DIR}/step_cycle_dense.jsonl"
rm -f "$TRACE_FILE"

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

# Start profiling server with tracing
cleanup
echo "Starting profiling server..." >> "$LOG"
VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
VLLM_EMULATOR_STEP_TRACE_OUTPUT="$TRACE_FILE" \
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > "${LOGS_DIR}/profile_server.log" 2>&1 &
wait_server

# Heavy warmup (200 prompts at rate=4) — reaches thermal steady state
# and captures all CUDA graph sizes
echo "Heavy warmup ($WARMUP_PROMPTS prompts at rate=4)..." >> "$LOG"
python3 -m vllm.entrypoints.cli.main bench serve \
    --model "$MODEL" --base-url "http://localhost:${PORT}" \
    --dataset-name random --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" \
    --num-prompts "$WARMUP_PROMPTS" --request-rate 4 > /dev/null 2>&1
sleep 3

# Profile at multiple rates (low rate → fewer prompts since arrival-limited)
for rate in $PROFILE_RATES; do
    NP=200
    if [[ "$rate" == "0.5" ]]; then NP=50; fi
    echo "  Profiling rate=$rate ($NP prompts)..." >> "$LOG"
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "$MODEL" --base-url "http://localhost:${PORT}" \
        --dataset-name random --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" \
        --num-prompts "$NP" --request-rate "$rate" > /dev/null 2>&1
    sleep 2
done

# Offline: rate=inf for high-concurrency coverage
echo "  Profiling offline (200 prompts, rate=inf)..." >> "$LOG"
python3 -m vllm.entrypoints.cli.main bench serve \
    --model "$MODEL" --base-url "http://localhost:${PORT}" \
    --dataset-name random --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" \
    --num-prompts 200 --request-rate inf > /dev/null 2>&1
cleanup

RECORDS=$(wc -l < "$TRACE_FILE")
echo "Total trace records: $RECORDS" >> "$LOG"

# Build dense profile
PROFILE_NEW="${PROFILES_DIR}/serving-Qwen3-8B-dense.json"
echo "Building dense profile..." >> "$LOG"
python3 vllm_emulator/profile/build_serving_profile.py \
    "$TRACE_FILE" "/dev/null" "$PROFILE_NEW" >> "$LOG" 2>&1

# Verify profile
python3 -c "
import json, statistics
p = json.load(open('$PROFILE_NEW'))
decode = p.get('decode_forward_pass', [])
prefill = p.get('prefill_forward_pass', [])
d2 = p.get('step_cycle_2d_distribution', [])

all_decode_samples = []
for b in decode:
    all_decode_samples.extend(b.get('samples', []))
all_prefill_samples = []
for b in prefill:
    all_prefill_samples.extend(b.get('samples', []))

print(f'Dense profile built:')
print(f'  Decode buckets: {len(decode)}, total samples: {len(all_decode_samples)}')
print(f'  Prefill buckets: {len(prefill)}, total samples: {len(all_prefill_samples)}')
print(f'  2D cells: {len(d2)}')
if all_decode_samples:
    print(f'  Decode global mean: {statistics.mean(all_decode_samples)/1000:.1f}ms')
    print(f'  Decode global p99: {sorted(all_decode_samples)[int(len(all_decode_samples)*0.99)]/1000:.1f}ms')
" >> "$LOG" 2>&1

echo "" >> "$LOG"
echo "=== Done at $(date) ===" >> "$LOG"
cat "$LOG"
