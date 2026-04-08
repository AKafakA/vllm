#!/bin/bash
# Graph-enabled profiling: same config as serving (NO enforce_eager)
# Produces: online step-cycle + graph-enabled sweep + offline step-cycle
set -e
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=1

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"
PORT=8100

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

echo "============================================"
echo "=== Graph-Enabled Profiling ==="
echo "=== Same config as serving (CUDA graphs ON) ==="
echo "=== $(date) ==="
echo "============================================"

ONLINE_TRACE="${RESULT_DIR}/step_cycle_graph.jsonl"
OFFLINE_TRACE="${RESULT_DIR}/offline_step_cycle_graph.jsonl"
SWEEP_TRACE="${RESULT_DIR}/sweep_graph.jsonl"

rm -f "$ONLINE_TRACE" "$OFFLINE_TRACE" "$SWEEP_TRACE"
cleanup_gpu

# =============================================
# PHASE 1: Online step-cycle (dense rates, 500 prompts at high rates)
# =============================================
echo ""
echo "=== PHASE 1: Online step-cycle ==="

VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
VLLM_EMULATOR_STEP_TRACE_OUTPUT="$ONLINE_TRACE" \
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > /workspace/profile_graph_server.log 2>&1 &
wait_server

echo "  Warmup (200 prompts, rate=4)..."
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 --request-rate 4 > /dev/null 2>&1

# Dense rates with enough prompts
for rate in 0.5 1 2 3 4 6 8 10 12 16 24 32 inf; do
    NP=200
    [ "$rate" = "0.5" ] && NP=50
    [ "$rate" = "16" ] || [ "$rate" = "24" ] || [ "$rate" = "32" ] && NP=500
    [ "$rate" = "inf" ] && NP=500
    echo "  rate=$rate ($NP prompts)..."
    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts $NP --request-rate $rate > /dev/null 2>&1
done

# Variable-length prompts for coverage at different tt values
echo "  Variable length: input=64, output=32 (500 prompts, rate=8)..."
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 64 --random-output-len 32 \
    --num-prompts 500 --request-rate 8 > /dev/null 2>&1

echo "  Variable length: input=512, output=256 (200 prompts, rate=4)..."
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 512 --random-output-len 256 \
    --num-prompts 200 --request-rate 4 > /dev/null 2>&1

echo "  Variable length: input=128, output=64 (500 prompts, rate=16)..."
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 128 --random-output-len 64 \
    --num-prompts 500 --request-rate 16 > /dev/null 2>&1

cleanup_gpu
ONLINE_N=$(wc -l < "$ONLINE_TRACE" 2>/dev/null || echo 0)
echo "  Online records: $ONLINE_N"

# =============================================
# PHASE 2: Graph-enabled sweep (NOT enforce_eager)
# Same CUDA graph config as serving
# =============================================
echo ""
echo "=== PHASE 2: Graph-enabled sweep ==="
echo "  (Using bench throughput with various batch sizes — CUDA graphs ON)"

# Warmup to compile CUDA graphs
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "$MODEL" --max-model-len 4096 --trust-remote-code \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 > /dev/null 2>&1

# Sweep at various batch sizes with step-cycle tracing
for NP in 1 2 5 10 20 50 100 150 200 300 500; do
    echo "  sweep batch=$NP..."
    VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
    VLLM_EMULATOR_STEP_TRACE_OUTPUT="$SWEEP_TRACE" \
    python3 -m vllm.entrypoints.cli.main bench throughput \
        --model "$MODEL" --max-model-len 4096 --trust-remote-code \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts $NP 2>&1 | grep "Throughput:" || true
done

# Also pure decode sweep (short input)
for NP in 10 50 100 200 300; do
    echo "  sweep decode batch=$NP..."
    VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
    VLLM_EMULATOR_STEP_TRACE_OUTPUT="$SWEEP_TRACE" \
    python3 -m vllm.entrypoints.cli.main bench throughput \
        --model "$MODEL" --max-model-len 4096 --trust-remote-code \
        --dataset-name random --random-input-len 1 --random-output-len 256 \
        --num-prompts $NP 2>&1 | grep "Throughput:" || true
done

SWEEP_N=$(wc -l < "$SWEEP_TRACE" 2>/dev/null || echo 0)
echo "  Sweep records: $SWEEP_N"

# =============================================
# PHASE 3: Offline step-cycle
# =============================================
echo ""
echo "=== PHASE 3: Offline step-cycle ==="
rm -f "$OFFLINE_TRACE"

python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "$MODEL" --max-model-len 4096 --trust-remote-code \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 > /dev/null 2>&1

for NP in 50 100 200 300; do
    VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
    VLLM_EMULATOR_STEP_TRACE_OUTPUT="$OFFLINE_TRACE" \
    python3 -m vllm.entrypoints.cli.main bench throughput \
        --model "$MODEL" --max-model-len 4096 --trust-remote-code \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts $NP 2>&1 | grep "Throughput:"
done
for NP in 50 100 200; do
    VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
    VLLM_EMULATOR_STEP_TRACE_OUTPUT="$OFFLINE_TRACE" \
    python3 -m vllm.entrypoints.cli.main bench throughput \
        --model "$MODEL" --max-model-len 4096 --trust-remote-code \
        --dataset-name random --random-input-len 1 --random-output-len 256 \
        --num-prompts $NP 2>&1 | grep "Throughput:"
done

OFFLINE_N=$(wc -l < "$OFFLINE_TRACE" 2>/dev/null || echo 0)
echo "  Offline records: $OFFLINE_N"

# =============================================
# PHASE 4: Analysis — compare old vs new data
# =============================================
echo ""
echo "=== PHASE 4: Data quality analysis ==="
python3 << 'PYEOF'
import json, statistics
from collections import defaultdict

# Load new online trace
new_records = []
for line in open("/workspace/eval_results/RTX-3060-12GB/step_cycle_graph.jsonl"):
    r = json.loads(line)
    if "total_tokens" in r:
        new_records.append(r)

# Load old online trace (for comparison)
old_records = []
try:
    for line in open("/workspace/eval_results/RTX-3060-12GB/step_cycle_final.jsonl"):
        r = json.loads(line)
        if "total_tokens" in r:
            old_records.append(r)
except: pass

print(f"Old trace: {len(old_records)} records")
print(f"New trace: {len(new_records)} records")

# Compare coverage at critical tt range (260-290)
for label, records in [("Old", old_records), ("New", new_records)]:
    by_tt = defaultdict(list)
    for r in records[200:]:  # skip warmup
        by_tt[r["total_tokens"]].append(r["step_cycle_us"])

    print(f"\n{label} profile at tt=260-290 (prefill region):")
    for tt in range(260, 291, 2):
        if tt in by_tt:
            lats = by_tt[tt]
            med = statistics.median(lats)
            print(f"  tt={tt}: n={len(lats):>4}, median={med/1000:.1f}ms, "
                  f"min={min(lats)/1000:.1f}ms, max={max(lats)/1000:.1f}ms")

# Load sweep trace
sweep_records = []
try:
    for line in open("/workspace/eval_results/RTX-3060-12GB/sweep_graph.jsonl"):
        r = json.loads(line)
        if "total_tokens" in r:
            sweep_records.append(r)
except: pass

if sweep_records:
    print(f"\nGraph-enabled sweep: {len(sweep_records)} records")
    sweep_by_tt = defaultdict(list)
    for r in sweep_records:
        sweep_by_tt[r["total_tokens"]].append(r["step_cycle_us"])

    print("Sweep at key tt values:")
    for tt in [1, 10, 50, 100, 200, 256, 300, 500]:
        if tt in sweep_by_tt:
            lats = sweep_by_tt[tt]
            med = statistics.median(lats)
            print(f"  tt={tt}: n={len(lats):>4}, median={med/1000:.1f}ms")
        else:
            nearest = min(sweep_by_tt.keys(), key=lambda k: abs(k-tt)) if sweep_by_tt else 0
            if nearest:
                print(f"  tt={tt}: (nearest tt={nearest}, {statistics.median(sweep_by_tt[nearest])/1000:.1f}ms)")

    # Compare old enforce_eager sweep vs new graph-enabled sweep
    try:
        old_sweep = json.load(open("/workspace/eval_results/RTX-3060-12GB/profiles/sweep-1.5b-tp1-v14.json"))
        old_sweep_fp = {e["total_tokens"]: e["latency_us"] for e in old_sweep["forward_pass"]}
        print(f"\nOld sweep (enforce_eager) vs New sweep (graph-enabled):")
        for tt in [1, 10, 50, 100, 200, 256, 300]:
            old_val = old_sweep_fp.get(tt, 0)
            if tt in sweep_by_tt:
                new_val = statistics.median(sweep_by_tt[tt])
                ratio = old_val / new_val if new_val > 0 else 0
                print(f"  tt={tt}: old={old_val/1000:.1f}ms, new={new_val/1000:.1f}ms, ratio={ratio:.1f}x")
    except: pass
PYEOF

# =============================================
# PHASE 5: Outlier analysis at tt=260-290
# =============================================
echo ""
echo "=== PHASE 5: Outlier analysis ==="
python3 << 'PYEOF'
import json, statistics
from collections import defaultdict

records = []
for line in open("/workspace/eval_results/RTX-3060-12GB/step_cycle_graph.jsonl"):
    r = json.loads(line)
    if "total_tokens" in r:
        records.append(r)

# Focus on prefill steps at tt=256-300
prefill = [r for r in records[200:] if r.get("num_new_reqs", 0) > 0
           and 256 <= r["total_tokens"] <= 300]

by_tt = defaultdict(list)
for r in prefill:
    by_tt[r["total_tokens"]].append(r["step_cycle_us"])

print(f"Prefill steps at tt=256-300: {len(prefill)}")
print(f"\nPer-tt breakdown (raw, before any filtering):")
print(f"{'tt':>5} {'n':>5} {'median':>8} {'min':>8} {'max':>8} {'outliers':>10}")

for tt in sorted(by_tt.keys()):
    lats = by_tt[tt]
    if not lats:
        continue
    med = statistics.median(lats)
    mn = min(lats)
    mx = max(lats)
    # Count outliers (>2x median)
    outliers = sum(1 for v in lats if v > med * 2)
    print(f"{tt:>5} {len(lats):>5} {med/1000:>7.1f}ms {mn/1000:>7.1f}ms {mx/1000:>7.1f}ms {outliers:>10}")

# What the profile builder would produce with current 3x threshold
print(f"\nEffect of different outlier thresholds:")
for threshold in [3.0, 2.5, 2.0, 1.5]:
    bad_buckets = 0
    for tt in sorted(by_tt.keys()):
        lats = by_tt[tt]
        if len(lats) < 2:
            continue
        med = statistics.median(lats)
        filtered = [v for v in lats if v > 5000 and v < med * 3]
        if len(filtered) < 2:
            continue
        bucket_med = statistics.median(filtered)
        # Check against neighbors
        neighbors = []
        for tt2 in by_tt:
            if abs(tt2 - tt) <= 10 and tt2 != tt and len(by_tt[tt2]) >= 2:
                n_med = statistics.median([v for v in by_tt[tt2] if v > 5000 and v < statistics.median(by_tt[tt2]) * 3][:] or by_tt[tt2])
                neighbors.append(n_med)
        if neighbors:
            neighbor_med = statistics.median(neighbors)
            if bucket_med > neighbor_med * threshold:
                bad_buckets += 1
    print(f"  threshold={threshold}x: {bad_buckets} outlier buckets detected")

# Suggest: minimum samples per bucket
print(f"\nSample count distribution at tt=256-300:")
for min_n in [1, 2, 5, 10, 20]:
    covered = sum(1 for lats in by_tt.values() if len(lats) >= min_n)
    total = len(by_tt)
    print(f"  min_n={min_n:>3}: {covered}/{total} tt values covered ({covered/total*100:.0f}%)")
PYEOF

cleanup_gpu
echo ""
echo "=== DONE $(date) ==="
