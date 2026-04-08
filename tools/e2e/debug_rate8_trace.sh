#!/bin/bash
# Rate=8 with per-step tracing + analysis
set -e
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=1

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
PROFILE="/workspace/eval_results/RTX-3060-12GB/profiles/serving-1.5b-tp1-extended.json"
PORT=8100
SCRIPT_DIR="/workspace/vllm-emulator-v18/tools/e2e"
ONLINE_DIR="/workspace/eval_results/RTX-3060-12GB/online"
TRACE="/workspace/hook_trace_r8.csv"

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
echo "=== Rate=8 Traced Run ==="
echo "=== $(date) ==="
echo "============================================"

cleanup_gpu
rm -f "$TRACE"

VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
VLLM_EMULATOR_ORACLE_MODE=hybrid \
VLLM_EMULATOR_HOOK_TRACE="$TRACE" \
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > /workspace/trace_r8_server.log 2>&1 &
wait_server

echo "  Warmup..."
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 --request-rate 4 > /dev/null 2>&1
sleep 3

echo "  Bench rate=8 (1000 prompts, traced)..."
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 1000 --request-rate 8 --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,99 \
    --save-result --result-dir "$ONLINE_DIR" --result-filename "trace_rate8_emu.json" > /dev/null 2>&1

cleanup_gpu

echo ""
echo "=== Trace Analysis ==="
TRACE_LINES=$(wc -l < "$TRACE" 2>/dev/null || echo 0)
echo "Trace records: $TRACE_LINES"

python3 << 'PYEOF'
import csv, statistics

rows = []
with open("/workspace/hook_trace_r8.csv") as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append({k: float(v) for k, v in r.items()})

if not rows:
    print("No trace data")
    exit()

# Skip warmup (first ~3000 steps)
warmup_steps = 3000
bench = rows[warmup_steps:] if len(rows) > warmup_steps else rows[len(rows)//2:]
print(f"Total steps: {len(rows)}, benchmark steps: {len(bench)}")

# Inter-step gap (wall-clock time between consecutive steps)
gaps = []
for i in range(1, len(bench)):
    gap_s = bench[i]["wall_s"] - bench[i-1]["wall_s"]
    gaps.append(gap_s * 1000)  # ms

print(f"\n=== Inter-step gap (wall-clock, ms) ===")
print(f"  mean={statistics.mean(gaps):.1f}, median={statistics.median(gaps):.1f}")
print(f"  min={min(gaps):.1f}, max={max(gaps):.1f}")

print(f"\n=== Oracle prediction (ms) ===")
oracle = [r["oracle_us"]/1000 for r in bench]
print(f"  mean={statistics.mean(oracle):.1f}, median={statistics.median(oracle):.1f}")

print(f"\n=== Total latency incl overhead (ms) ===")
total_lat = [r["total_latency_us"]/1000 for r in bench]
print(f"  mean={statistics.mean(total_lat):.1f}, median={statistics.median(total_lat):.1f}")

print(f"\n=== Timer delay after chaining (ms) ===")
timer = [r["timer_delay_us"]/1000 for r in bench]
print(f"  mean={statistics.mean(timer):.1f}, median={statistics.median(timer):.1f}")

print(f"\n=== Concurrency (n_reqs) ===")
nreqs = [r["n_reqs"] for r in bench]
print(f"  mean={statistics.mean(nreqs):.1f}, median={statistics.median(nreqs):.0f}")
print(f"  min={min(nreqs):.0f}, max={max(nreqs):.0f}")

print(f"\n=== Key comparison ===")
gap_mean = statistics.mean(gaps)
timer_mean = statistics.mean(timer)
print(f"  Inter-step gap (wall-clock): {gap_mean:.1f}ms")
print(f"  Timer delay (predicted):     {timer_mean:.1f}ms")
print(f"  Engine overhead per step:    {gap_mean - timer_mean:.1f}ms")
print(f"  Real GPU TPOT:               20.1ms")
print(f"  Emu TPOT (bench):            check result json")

# Breakdown by concurrency bucket
print(f"\n=== Gap vs Timer by concurrency ===")
by_bucket = {}
for i in range(1, len(bench)):
    n = int(bench[i]["n_reqs"])
    bucket = (n // 5) * 5  # bucket by 5s
    if bucket not in by_bucket:
        by_bucket[bucket] = {"gaps": [], "timers": [], "oracles": []}
    by_bucket[bucket]["gaps"].append(gaps[i-1])
    by_bucket[bucket]["timers"].append(bench[i]["timer_delay_us"]/1000)
    by_bucket[bucket]["oracles"].append(bench[i]["oracle_us"]/1000)

for bucket in sorted(by_bucket.keys()):
    d = by_bucket[bucket]
    if len(d["gaps"]) >= 10:
        g = statistics.median(d["gaps"])
        t = statistics.median(d["timers"])
        o = statistics.median(d["oracles"])
        print(f"  reqs={bucket:>3}-{bucket+4}: gap={g:.1f}ms, timer={t:.1f}ms, oracle={o:.1f}ms, overhead={g-t:.1f}ms (n={len(d['gaps'])})")
PYEOF

echo ""
echo "=== Bench result ==="
python3 "$SCRIPT_DIR/compare_results.py" "$ONLINE_DIR/ext_rate8_real.json" "$ONLINE_DIR/trace_rate8_emu.json" 2>/dev/null \
    || python3 "$SCRIPT_DIR/compare_results.py" "$ONLINE_DIR/hybrid_rate8_real.json" "$ONLINE_DIR/trace_rate8_emu.json" 2>/dev/null \
    || echo "No real baseline"

echo ""
echo "=== DONE $(date) ==="
