#!/bin/bash
# pyinstrument + CSV trace at rate=8 with uncapped chain + hybrid linear
set -e
source /workspace/vllm-v18-env/bin/activate
pip install pyinstrument -q 2>/dev/null
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=1

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
PROFILE="/workspace/eval_results/RTX-3060-12GB/profiles/serving-1.5b-tp1-final.json"
PORT=8100
SCRIPT_DIR="/workspace/vllm-emulator-v18/tools/e2e"
ONLINE_DIR="/workspace/eval_results/RTX-3060-12GB/online"

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

echo "=== pyinstrument + trace: uncapped chain + hybrid linear @ rate=8 ==="
echo "=== $(date) ==="

cleanup_gpu

# Run with both pyinstrument and CSV trace
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
VLLM_EMULATOR_ORACLE_MODE=hybrid \
VLLM_EMULATOR_OVERHEAD_SCALING=linear \
VLLM_EMULATOR_TIMER_MODE=chain \
VLLM_EMULATOR_CHAIN_CAP=0 \
VLLM_EMULATOR_HOOK_TRACE="/workspace/pyinst_trace_r8.csv" \
VLLM_EMULATOR_PYINSTRUMENT="/workspace/pyinst_r8.txt" \
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > /workspace/pyinst_server.log 2>&1 &
SERVER_PID=$!
wait_server
grep "ExecutorEmulatorHook\|pyinstrument" /workspace/pyinst_server.log | head -2

echo "  Warmup (short)..."
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 50 --request-rate 4 > /dev/null 2>&1
sleep 2

echo "  Bench rate=8 (100 prompts — short for profiling)..."
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 100 --request-rate 8 2>&1 | tail -5

# Graceful shutdown to trigger pyinstrument save
echo "  Stopping server (saving pyinstrument profile)..."
kill -TERM $SERVER_PID 2>/dev/null || true
sleep 10
kill -9 $SERVER_PID 2>/dev/null || true
sleep 3

echo ""
echo "=== pyinstrument profile (top functions) ==="
if [ -f /workspace/pyinst_r8.txt ]; then
    head -60 /workspace/pyinst_r8.txt
    echo ""
    echo "Full profile: /workspace/pyinst_r8.txt"
    echo "HTML profile: /workspace/pyinst_r8.html"
else
    echo "No pyinstrument output — server may not have shut down cleanly"
    echo "Checking server log:"
    grep "pyinstrument" /workspace/pyinst_server.log | tail -3
fi

echo ""
echo "=== CSV trace analysis ==="
if [ -f /workspace/pyinst_trace_r8.csv ]; then
    python3 << 'PYEOF'
import csv, statistics
rows = []
with open("/workspace/pyinst_trace_r8.csv") as f:
    for r in csv.DictReader(f):
        rows.append({k: float(v) for k, v in r.items()})
bench = rows[len(rows)//2:]
gaps = [(bench[i]["wall_s"]-bench[i-1]["wall_s"])*1000 for i in range(1,len(bench))]
timers = [r["timer_delay_us"]/1000 for r in bench]
oracles = [r["oracle_us"]/1000 for r in bench]
hybrids = [r["hybrid_overhead_us"]/1000 for r in bench]
nreqs = [r["n_reqs"] for r in bench]
print(f"Steps: {len(bench)}")
print(f"Gap: {statistics.mean(gaps):.1f}ms, Timer: {statistics.mean(timers):.1f}ms")
print(f"Oracle: {statistics.mean(oracles):.1f}ms, Hybrid: {statistics.mean(hybrids):.1f}ms")
print(f"Concurrency: mean={statistics.mean(nreqs):.0f}, max={max(nreqs):.0f}")
print(f"Chain inflation: {statistics.mean(timers)-statistics.mean([r['total_latency_us']/1000 for r in bench]):.1f}ms")
PYEOF
else
    echo "No CSV trace"
fi

cleanup_gpu
echo ""
echo "=== DONE $(date) ==="
