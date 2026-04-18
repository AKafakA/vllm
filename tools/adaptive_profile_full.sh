#!/bin/bash
# Full adaptive profiling: proven Apr 13 recipe + current uncapped builder.
# 14 rates × up to 5 rounds, variable-shape workloads, CUDA warmup sweep.
# Expected runtime: 3-4 hours. Output: ~300k+ raw records.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

MODEL="Qwen/Qwen3-8B"
PORT=8100
OUT_DIR="./results/RTX-8000-adaptive-v4"
TRACE="$OUT_DIR/step_cycle_trace.jsonl"
PROFILE="$OUT_DIR/serving-full.json"
MIN_SAMPLES=10
MAX_ROUNDS=5

mkdir -p "$OUT_DIR/logs"
rm -f "$TRACE"
touch "$OUT_DIR/.started"

LOG="$OUT_DIR/run.log"
echo "=== Adaptive profile v2 start $(date) ===" > "$LOG"
echo "=== Min samples: $MIN_SAMPLES, Max rounds: $MAX_ROUNDS ===" >> "$LOG"

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

# Coverage-check script (external file, not inline)
cat > "$OUT_DIR/check_coverage.py" << 'PYEOF'
import json
import sys
from collections import defaultdict

CONC_BOUNDARIES = [1, 3, 5, 10, 20, 50, 100, 200, 300]
TT_BUCKET_WIDTH = 5
MIN_SAMPLES = 10
TRACE_FILE = sys.argv[1]
MARKER_FILE = sys.argv[2]


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
    if r.get("_header") or r.get("__marker__"):
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

import os
if len(under) == 0:
    open(MARKER_FILE, "w").write("done")
    sys.exit(0)
else:
    try:
        os.remove(MARKER_FILE)
    except OSError:
        pass
    sys.exit(1)
PYEOF

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

rm -f /tmp/adaptive_coverage_done

for ROUND in $(seq 1 $MAX_ROUNDS); do
    echo "" >> "$LOG"
    echo "=== ROUND $ROUND / $MAX_ROUNDS at $(date) ===" >> "$LOG"
    touch "$OUT_DIR/.round_${ROUND}_started"

    cleanup
    env \
        VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
        VLLM_EMULATOR_STEP_TRACE_OUTPUT="$TRACE" \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
        > "$OUT_DIR/logs/server_r${ROUND}.log" 2>&1 &
    if ! wait_server; then
        echo "SERVER_NEVER_READY round=$ROUND" >> "$LOG"
        touch "$OUT_DIR/.failed"
        cleanup
        exit 1
    fi

    # Standard warmup only. CUDA graph warmup sweep and high-concurrency
    # burst have been REMOVED because they pre-capture every padded batch
    # size before the rate sweep — that shifts the per-bucket distribution
    # SHAPE (tails) even though the average-metric effect is <1%. The
    # oracle's random.choice needs the heavy-tail capture-cost samples
    # for its per-step variance to match real's. By skipping the
    # comprehensive pre-warm, rate-sweep encounters new graphs in-situ
    # and the capture-cost spikes end up recorded in the profile.
    # (Apr 18 v3→v4 fix; diagnostic in paper/apr_18/02_v3_baseline_AB.md.)
    echo "  [$(date +%T)] standard warmup..." >> "$LOG"
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "$MODEL" --base-url "http://localhost:${PORT}" \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 200 --request-rate 4 > /dev/null 2>&1 || true

    # Begin profiling window — standard warmup is EXCLUDED.
    echo '{"__marker__": "profiling_start"}' >> "$TRACE"

    # Profile each rate
    for rate in $RATES; do
        NP=$(get_num_prompts $rate $ROUND)
        echo "  [$(date +%T)] r=$rate n=$NP" >> "$LOG"
        python3 -m vllm.entrypoints.cli.main bench serve \
            --model "$MODEL" --base-url "http://localhost:${PORT}" \
            --dataset-name random --random-input-len 256 --random-output-len 128 \
            --num-prompts $NP --request-rate $rate > /dev/null 2>&1 || true
    done

    # Variable-shape workloads: cover small and large prompt regimes
    echo "  [$(date +%T)] variable 64/32..." >> "$LOG"
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "$MODEL" --base-url "http://localhost:${PORT}" \
        --dataset-name random --random-input-len 64 --random-output-len 32 \
        --num-prompts 300 --request-rate 8 > /dev/null 2>&1 || true

    echo "  [$(date +%T)] variable 512/256..." >> "$LOG"
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "$MODEL" --base-url "http://localhost:${PORT}" \
        --dataset-name random --random-input-len 512 --random-output-len 256 \
        --num-prompts 200 --request-rate 4 > /dev/null 2>&1 || true

    echo "  [$(date +%T)] variable 128/64..." >> "$LOG"
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "$MODEL" --base-url "http://localhost:${PORT}" \
        --dataset-name random --random-input-len 128 --random-output-len 64 \
        --num-prompts 300 --request-rate 16 > /dev/null 2>&1 || true

    # End profiling window — next round's warmup phases excluded.
    echo '{"__marker__": "profiling_stop"}' >> "$TRACE"

    cleanup

    echo "" >> "$LOG"
    TOTAL=$(wc -l < "$TRACE" 2>/dev/null || echo 0)
    echo "=== Coverage check after round $ROUND (total records: $TOTAL) ===" >> "$LOG"
    python3 "$OUT_DIR/check_coverage.py" "$TRACE" /tmp/adaptive_coverage_done >> "$LOG" 2>&1 || true
    touch "$OUT_DIR/.round_${ROUND}_done"

    if [ -f /tmp/adaptive_coverage_done ]; then
        echo "=== COVERAGE TARGET MET after round $ROUND ===" >> "$LOG"
        break
    fi
done

if [ ! -f /tmp/adaptive_coverage_done ]; then
    echo "" >> "$LOG"
    echo "=== WARNING: Coverage target NOT met after $MAX_ROUNDS rounds ===" >> "$LOG"
fi

# Build profile with current (uncapped) builder — 2-arg CLI
echo "" >> "$LOG"
echo "=== Building profile $(date) ===" >> "$LOG"
python3 vllm_emulator/profile/build_serving_profile_filtered.py \
    "$TRACE" "$PROFILE" \
    --tt-bucket-width 1 --conc-bucket-width 5 >> "$LOG" 2>&1

echo "" >> "$LOG"
echo "=== DONE $(date) ===" >> "$LOG"
touch "$OUT_DIR/.all_done"
