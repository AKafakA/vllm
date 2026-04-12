#!/bin/bash
# Adaptive profiling for Qwen3-8B on RTX 8000.
# Based on tools/e2e/adaptive_profile.sh but paths adapted for dev-gpu.
# Loops over rates until all 2D buckets have MIN_SAMPLES, or MAX_ROUNDS reached.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

MODEL="Qwen/Qwen3-8B"
RESULT_DIR="./results/RTX-8000-adaptive"
PORT=8100
MIN_SAMPLES=10
MAX_ROUNDS=5

mkdir -p "$RESULT_DIR/profiles" "$RESULT_DIR/logs"
TRACE="${RESULT_DIR}/profiles/step_cycle_adaptive.jsonl"
rm -f "$TRACE"

LOG="/tmp/vllm_adaptive_profile.log"
echo "=== Adaptive profiling at $(date) ===" > "$LOG"
echo "=== Min samples: $MIN_SAMPLES, Max rounds: $MAX_ROUNDS ===" >> "$LOG"

cleanup_gpu() {
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

check_coverage() {
    python3 << 'PYEOF'
import json
import sys
from collections import defaultdict

CONC_BOUNDARIES = [1, 3, 5, 10, 20, 50, 100, 200, 300]
TT_BUCKET_WIDTH = 5
MIN_SAMPLES = 10
TRACE_FILE = "./results/RTX-8000-adaptive/profiles/step_cycle_adaptive.jsonl"


def conc_bucket(n):
    for i in range(len(CONC_BOUNDARIES) - 1):
        if CONC_BOUNDARIES[i] <= n < CONC_BOUNDARIES[i + 1]:
            return (CONC_BOUNDARIES[i] + CONC_BOUNDARIES[i + 1]) // 2
    return CONC_BOUNDARIES[-1]


def tt_bucket(tt):
    return (tt // TT_BUCKET_WIDTH) * TT_BUCKET_WIDTH + TT_BUCKET_WIDTH // 2


records = []
for line in open(TRACE_FILE):
    r = json.loads(line)
    if r.get("_header"):
        continue
    if "total_tokens" in r:
        records.append(r)
records = records[200:]

counts = defaultdict(int)
for r in records:
    tt = r["total_tokens"]
    conc = r.get("num_new_reqs", 0) + r.get("num_decode_seqs", 0)
    if conc < 1:
        conc = 1
    counts[(tt_bucket(tt), conc_bucket(conc))] += 1

total = len(counts)
under = [(k, v) for k, v in counts.items() if v < MIN_SAMPLES]
conc_set = sorted(set(cb for (_, cb) in counts.keys()))

print(f"COVERAGE: {total} cells, {len(under)} under {MIN_SAMPLES}, {len(records)} records")
for cb in conc_set:
    u = sum(1 for (_, c), v in counts.items() if c == cb and v < MIN_SAMPLES)
    t = sum(1 for (_, c) in counts.keys() if c == cb)
    status = "OK" if u == 0 else f"{u}/{t} UNDER"
    print(f"    conc={cb:>4}: {t:>3} tt-buckets, {status}")

if len(under) == 0:
    with open("/tmp/coverage_done", "w") as f:
        f.write("done")
    sys.exit(0)
else:
    import os
    try:
        os.remove("/tmp/coverage_done")
    except OSError:
        pass
    sys.exit(1)
PYEOF
}

# Rates covering all concurrency ranges
RATES="1 2 3 4 5 6 8 10 12 16 24 32 0.5 inf"

get_num_prompts() {
    local rate=$1 round=$2
    local base=300
    [[ "$round" -gt 1 ]] && base=500
    [[ "$round" -gt 3 ]] && base=800
    case $rate in
        0.5) echo $((base / 3)) ;;
        *)   echo $base ;;
    esac
}

rm -f /tmp/coverage_done

for ROUND in $(seq 1 $MAX_ROUNDS); do
    echo "" >> "$LOG"
    echo "=== ROUND $ROUND / $MAX_ROUNDS at $(date) ===" >> "$LOG"

    cleanup_gpu
    VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
    VLLM_EMULATOR_STEP_TRACE_OUTPUT="$TRACE" \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
        > "$RESULT_DIR/logs/server_r${ROUND}.log" 2>&1 &
    wait_server

    # CUDA graph warmup sweep for all padded batch sizes
    echo "  CUDA graph warmup sweep..." >> "$LOG"
    for NP in 1 2 4 8 16 24 32 48 64 96 128 160 192 224 256; do
        python3 -m vllm.entrypoints.cli.main bench serve \
            --model "$MODEL" --base-url "http://localhost:${PORT}" \
            --dataset-name random --random-input-len 1 --random-output-len 1 \
            --num-prompts $NP --request-rate inf > /dev/null 2>&1
    done
    # High-concurrency burst
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "$MODEL" --base-url "http://localhost:${PORT}" \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 500 --request-rate inf > /dev/null 2>&1

    # Standard warmup
    echo "  Standard warmup (200 prompts, rate=4)..." >> "$LOG"
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "$MODEL" --base-url "http://localhost:${PORT}" \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 200 --request-rate 4 > /dev/null 2>&1

    # Profile each rate
    for rate in $RATES; do
        NP=$(get_num_prompts $rate $ROUND)
        echo "  rate=$rate ($NP prompts)..." >> "$LOG"
        python3 -m vllm.entrypoints.cli.main bench serve \
            --model "$MODEL" --base-url "http://localhost:${PORT}" \
            --dataset-name random --random-input-len 256 --random-output-len 128 \
            --num-prompts $NP --request-rate $rate > /dev/null 2>&1
    done

    # Variable-length workloads
    echo "  Variable input=64 output=32..." >> "$LOG"
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "$MODEL" --base-url "http://localhost:${PORT}" \
        --dataset-name random --random-input-len 64 --random-output-len 32 \
        --num-prompts 300 --request-rate 8 > /dev/null 2>&1

    echo "  Variable input=512 output=256..." >> "$LOG"
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "$MODEL" --base-url "http://localhost:${PORT}" \
        --dataset-name random --random-input-len 512 --random-output-len 256 \
        --num-prompts 200 --request-rate 4 > /dev/null 2>&1

    echo "  Variable input=128 output=64..." >> "$LOG"
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "$MODEL" --base-url "http://localhost:${PORT}" \
        --dataset-name random --random-input-len 128 --random-output-len 64 \
        --num-prompts 300 --request-rate 16 > /dev/null 2>&1

    cleanup_gpu

    echo "" >> "$LOG"
    TOTAL=$(wc -l < "$TRACE" 2>/dev/null || echo 0)
    echo "=== Coverage check (total records: $TOTAL) ===" >> "$LOG"
    check_coverage >> "$LOG" 2>&1 || true

    if [ -f /tmp/coverage_done ]; then
        echo "=== COVERAGE TARGET MET after round $ROUND ===" >> "$LOG"
        break
    fi
done

if [ ! -f /tmp/coverage_done ]; then
    echo "" >> "$LOG"
    echo "=== WARNING: Coverage target NOT met after $MAX_ROUNDS rounds ===" >> "$LOG"
fi

# Build profile (using filtered builder which is more robust)
echo "" >> "$LOG"
echo "=== Building profile ===" >> "$LOG"
PROFILE="${RESULT_DIR}/profiles/serving-Qwen3-8B-adaptive.json"
python3 vllm_emulator/profile/build_serving_profile_filtered.py \
    "$TRACE" "/dev/null" "$PROFILE" >> "$LOG" 2>&1

echo "" >> "$LOG"
echo "=== Profile: $PROFILE ===" >> "$LOG"
echo "=== DONE $(date) ===" >> "$LOG"
cat "$LOG"
