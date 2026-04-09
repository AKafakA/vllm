#!/bin/bash
# Adaptive profiling: loops over rates until all 2D table buckets >= MIN_SAMPLES
# Each iteration: warmup → profile all rates → check coverage → repeat if needed
set -e
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=1

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"
PORT=8100
MIN_SAMPLES=10
MAX_ROUNDS=5

TRACE="${RESULT_DIR}/step_cycle_adaptive.jsonl"
rm -f "$TRACE"

cleanup_gpu() {
    pkill -9 -f "python3.*api_server" 2>/dev/null || true
    pkill -9 -f "python3.*bench" 2>/dev/null || true
    fuser /dev/nvidia* 2>/dev/null | tr " " "\n" | sort -u | xargs kill -9 2>/dev/null || true
    sleep 5
}

wait_server() {
    for i in $(seq 1 120); do
        if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then return 0; fi
        sleep 1
    done
    echo "ERROR: Server failed to start"; return 1
}

# Coverage check — exits 0 if all buckets >= MIN_SAMPLES, 1 otherwise
# Also prints which concurrency ranges need more data
check_coverage() {
    python3 << 'PYEOF'
import json, sys
from collections import defaultdict

CONC_BOUNDARIES = [1, 3, 5, 10, 20, 50, 100, 200, 300]
TT_BUCKET_WIDTH = 5
MIN_SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 10
TRACE_FILE = sys.argv[2] if len(sys.argv) > 2 else "/workspace/eval_results/RTX-3060-12GB/step_cycle_adaptive.jsonl"

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
    if "total_tokens" in r:
        records.append(r)

records = records[200:]  # skip warmup

counts = defaultdict(int)
for r in records:
    tt = r["total_tokens"]
    conc = r.get("num_new_reqs", 0) + r.get("num_decode_seqs", 0)
    if conc < 1: conc = 1
    ttb = tt_bucket(tt)
    cb = conc_bucket(conc)
    counts[(ttb, cb)] += 1

total_cells = len(counts)
under_min = [(k, v) for k, v in counts.items() if v < MIN_SAMPLES]
conc_set = sorted(set(cb for (_, cb) in counts.keys()))

print(f"COVERAGE: {total_cells} cells, {len(under_min)} under {MIN_SAMPLES} samples, {len(records)} records")
print(f"  Conc buckets: {conc_set}")

under_by_conc = defaultdict(int)
total_by_conc = defaultdict(int)
for (ttb, cb), cnt in counts.items():
    total_by_conc[cb] += 1
    if cnt < MIN_SAMPLES:
        under_by_conc[cb] += 1

# Print which conc ranges need more data
needs_more = []
for cb in conc_set:
    u = under_by_conc.get(cb, 0)
    t = total_by_conc[cb]
    status = "OK" if u == 0 else f"{u}/{t} UNDER"
    print(f"    conc={cb:>4}: {t:>3} tt-buckets, {status}")
    if u > 0:
        needs_more.append(cb)

if len(under_min) == 0:
    print(f"\nALL BUCKETS >= {MIN_SAMPLES} — DONE")
    # Write "DONE" marker for the outer loop
    with open("/tmp/coverage_done", "w") as f:
        f.write("done")
    sys.exit(0)
else:
    print(f"\n  Still need data at conc buckets: {needs_more}")
    under_min.sort(key=lambda x: x[1])
    print(f"  Worst 10 gaps:")
    for (ttb, cb), cnt in under_min[:10]:
        print(f"    tt={ttb:>4} conc={cb:>4}: {cnt} samples (need {MIN_SAMPLES - cnt} more)")
    # Remove done marker
    import os
    try: os.remove("/tmp/coverage_done")
    except: pass
    sys.exit(1)
PYEOF
}

echo "============================================"
echo "=== Adaptive Profiling ==="
echo "=== Min samples per bucket: $MIN_SAMPLES ==="
echo "=== Max rounds: $MAX_ROUNDS ==="
echo "=== $(date) ==="
echo "============================================"

# Rates to cycle through — dense at mid-range for conc=3-8 coverage
RATES="1 2 3 4 5 6 8 10 12 16 24 32 0.5 inf"

# Prompts per rate — heavier at mid-rates where coverage is sparse
get_num_prompts() {
    local rate=$1
    local round=$2
    # Increase prompts each round to fill remaining gaps
    local base=300
    if [ "$round" -gt 1 ]; then base=500; fi
    if [ "$round" -gt 3 ]; then base=800; fi
    case $rate in
        0.5)   echo $((base / 3)) ;;
        inf)   echo $base ;;
        *)     echo $base ;;
    esac
}

rm -f /tmp/coverage_done

for ROUND in $(seq 1 $MAX_ROUNDS); do
    echo ""
    echo "========================================"
    echo "=== ROUND $ROUND of $MAX_ROUNDS ==="
    echo "=== $(date) ==="
    echo "========================================"

    # Start fresh server for this round
    cleanup_gpu
    VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
    VLLM_EMULATOR_STEP_TRACE_OUTPUT="$TRACE" \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
        > /workspace/adaptive_server_r${ROUND}.log 2>&1 &
    wait_server

    # CUDA graph warmup sweep: compile all padded capture sizes
    # These steps are traced but excluded by the profile builder (skip first N records)
    echo "  CUDA graph warmup sweep (all capture sizes)..."
    for NP in 1 2 4 8 16 24 32 48 64 96 128 160 192 224 256; do
        python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
            --dataset-name random --random-input-len 1 --random-output-len 1 \
            --num-prompts $NP --request-rate inf > /dev/null 2>&1 || true
    done
    # High-concurrency burst to warm large padded sizes
    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 500 --request-rate inf > /dev/null 2>&1 || true

    # Standard warmup
    echo "  Warmup (200 prompts, rate=4)..."
    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 200 --request-rate 4 > /dev/null 2>&1

    # Profile each rate
    for rate in $RATES; do
        NP=$(get_num_prompts $rate $ROUND)
        echo "  rate=$rate ($NP prompts)..."
        python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
            --dataset-name random --random-input-len 256 --random-output-len 128 \
            --num-prompts $NP --request-rate $rate > /dev/null 2>&1
    done

    # Variable-length workloads for broader tt coverage
    echo "  Variable: input=64, output=32 (300p, rate=8)..."
    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 64 --random-output-len 32 \
        --num-prompts 300 --request-rate 8 > /dev/null 2>&1

    echo "  Variable: input=512, output=256 (200p, rate=4)..."
    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 512 --random-output-len 256 \
        --num-prompts 200 --request-rate 4 > /dev/null 2>&1

    echo "  Variable: input=128, output=64 (300p, rate=16)..."
    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 128 --random-output-len 64 \
        --num-prompts 300 --request-rate 16 > /dev/null 2>&1

    cleanup_gpu

    # Check coverage
    echo ""
    echo "=== Coverage check after round $ROUND ==="
    TOTAL=$(wc -l < "$TRACE" 2>/dev/null || echo 0)
    echo "  Total records so far: $TOTAL"
    check_coverage $MIN_SAMPLES "$TRACE" || true

    # Check if done
    if [ -f /tmp/coverage_done ]; then
        echo ""
        echo "=== COVERAGE TARGET MET after round $ROUND ==="
        break
    fi
done

if [ ! -f /tmp/coverage_done ]; then
    echo ""
    echo "=== WARNING: Coverage target NOT met after $MAX_ROUNDS rounds ==="
    echo "=== Proceeding with best available data ==="
fi

# Build profile from adaptive trace
echo ""
echo "=== Building 2D profile from adaptive trace ==="
PROFILE="${RESULT_DIR}/profiles/serving-1.5b-tp1-adaptive.json"
python3 /workspace/vllm-emulator-v18/vllm_emulator/profile/build_serving_profile_filtered.py \
    "$TRACE" \
    "${RESULT_DIR}/profiles/sweep-1.5b-tp1-v14.json" \
    "$PROFILE" \
    "$MODEL" "RTX-3060-12GB"

echo ""
echo "=== Profile saved: $PROFILE ==="
echo "=== DONE $(date) ==="
