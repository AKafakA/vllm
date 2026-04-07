#!/bin/bash
# Profile offline forward pass via LLM() interface (bench throughput)
# Captures step-cycle with CUDA graphs at high batch sizes
# Output: offline_step_cycle.jsonl
set -e
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub
export CUDA_VISIBLE_DEVICES=1

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"
TRACE_FILE="${RESULT_DIR}/offline_step_cycle.jsonl"

pkill -9 -f python3 2>/dev/null || true
fuser /dev/nvidia* 2>/dev/null | tr " " "\n" | sort -u | xargs kill -9 2>/dev/null || true
sleep 5

rm -f "$TRACE_FILE"

echo "=== Offline step-cycle profiling ==="
echo "Traces will be written to: $TRACE_FILE"

# Profile at various prompt counts to get different batch sizes
# Each run: LLM() loads model, processes all prompts, captures step-cycles
# The first run also warms up CUDA graphs

for NP in 10 20 50 100 150 200 300 500; do
    echo ""
    echo "--- Profiling $NP prompts (input=256, output=128) ---"
    VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
    VLLM_EMULATOR_STEP_TRACE_OUTPUT="$TRACE_FILE" \
    python3 -m vllm.entrypoints.cli.main bench throughput \
        --model "$MODEL" --max-model-len 4096 --trust-remote-code \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts $NP 2>&1 | grep "Throughput:"

    pkill -9 -f python3 2>/dev/null || true
    fuser /dev/nvidia* 2>/dev/null | tr " " "\n" | sort -u | xargs kill -9 2>/dev/null || true
    sleep 3
done

# Also profile with short input (1 token) to get pure decode steps at various sizes
for PD_NP in 50 100 200 300; do
    echo ""
    echo "--- Profiling $PD_NP prompts (input=1, output=256) — pure decode ---"
    VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
    VLLM_EMULATOR_STEP_TRACE_OUTPUT="$TRACE_FILE" \
    python3 -m vllm.entrypoints.cli.main bench throughput \
        --model "$MODEL" --max-model-len 4096 --trust-remote-code \
        --dataset-name random --random-input-len 1 --random-output-len 256 \
        --num-prompts $PD_NP 2>&1 | grep "Throughput:"

    pkill -9 -f python3 2>/dev/null || true
    fuser /dev/nvidia* 2>/dev/null | tr " " "\n" | sort -u | xargs kill -9 2>/dev/null || true
    sleep 3
done

RECORDS=$(wc -l < "$TRACE_FILE" 2>/dev/null || echo 0)
echo ""
echo "Total offline trace records: $RECORDS"

echo ""
echo "=== Analyze offline step-cycle data ==="
python3 << 'PYEOF'
import json, statistics
from collections import defaultdict

records = []
for line in open("/workspace/eval_results/RTX-3060-12GB/offline_step_cycle.jsonl"):
    r = json.loads(line)
    if "total_tokens" in r:
        records.append(r)

print(f"Total records: {len(records)}")

# Split by decode-only (num_new_reqs=0) vs mixed
decode_only = [r for r in records if r.get("num_new_reqs", 0) == 0]
mixed = [r for r in records if r.get("num_new_reqs", 0) > 0]
print(f"Decode-only steps: {len(decode_only)}")
print(f"Mixed (prefill+decode) steps: {len(mixed)}")

# Decode-only step-cycle by tt
by_tt = defaultdict(list)
for r in decode_only:
    by_tt[r["total_tokens"]].append(r["step_cycle_us"])

print(f"\nDecode-only step-cycle at key batch sizes:")
for tt in sorted(by_tt.keys()):
    lats = by_tt[tt]
    if len(lats) >= 2:
        med = statistics.median(lats)
        print(f"  tt={tt:>5}: n={len(lats):>4}, median={med/1000:.1f}ms")
    elif tt > 50 or tt % 10 == 0:
        print(f"  tt={tt:>5}: n={len(lats):>4}, value={lats[0]/1000:.1f}ms")

# Summary at key points
print(f"\nKey decode-only batch sizes:")
for target_tt in [10, 50, 100, 150, 200, 250, 300, 400, 500]:
    closest = min(by_tt.keys(), key=lambda k: abs(k - target_tt)) if by_tt else 0
    if closest and abs(closest - target_tt) < 20:
        lats = by_tt[closest]
        med = statistics.median(lats) if len(lats) >= 2 else lats[0]
        print(f"  ~tt={target_tt:>4} (actual={closest}): {med/1000:.1f}ms, n={len(lats)}")
PYEOF

echo ""
echo "DONE $(date)"
