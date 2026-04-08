#!/bin/bash
# Debug 1000 prompts at rate=8: trace concurrency growth over time
# Focus: when does concurrency explode? Does it stabilize or grow unbounded?
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
TRACE="/workspace/debug_1k_r8_trace.csv"

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

echo "=== Debug 1000 prompts rate=8: concurrency timeline ==="
echo "=== $(date) ==="

cleanup_gpu
rm -f "$TRACE"

# uncapped chain + hybrid linear (the config that works at 200 but breaks at 1000)
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
VLLM_EMULATOR_ORACLE_MODE=hybrid \
VLLM_EMULATOR_OVERHEAD_SCALING=linear \
VLLM_EMULATOR_TIMER_MODE=chain \
VLLM_EMULATOR_CHAIN_CAP=0 \
VLLM_EMULATOR_HOOK_TRACE="$TRACE" \
VLLM_EMULATOR_PYINSTRUMENT="/workspace/debug_1k_r8_pyinst.txt" \
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > /workspace/debug_1k_server.log 2>&1 &
SERVER_PID=$!
wait_server
grep "ExecutorEmulatorHook\|pyinstrument" /workspace/debug_1k_server.log | head -2

echo "  Warmup..."
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 --request-rate 4 > /dev/null 2>&1
sleep 3

echo "  Bench rate=8 (1000 prompts, traced)..."
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 1000 --request-rate 8 --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,99 \
    --save-result --result-dir "$ONLINE_DIR" --result-filename "debug_1k_r8_emu.json" > /dev/null 2>&1

echo "  Stopping server..."
kill -TERM $SERVER_PID 2>/dev/null || true
sleep 10
kill -9 $SERVER_PID 2>/dev/null || true
sleep 3

echo ""
echo "=== Bench result ==="
python3 "$SCRIPT_DIR/compare_results.py" "$ONLINE_DIR/fc_rate8_real.json" "$ONLINE_DIR/debug_1k_r8_emu.json" 2>/dev/null || echo "No baseline"

echo ""
echo "=== Concurrency timeline ==="
python3 << 'PYEOF'
import csv, statistics

rows = []
with open("/workspace/debug_1k_r8_trace.csv") as f:
    for r in csv.DictReader(f):
        rows.append({k: float(v) for k, v in r.items()})

# Find benchmark start (after warmup — look for step count reset or gap)
# Warmup: 200 prompts at rate=4, then 1000 at rate=8
# Use last 80% of steps as benchmark
total = len(rows)
warmup_end = total // 5  # rough: first 20% is warmup
bench = rows[warmup_end:]
print(f"Total steps: {total}, benchmark steps: {len(bench)}")

# Timeline: concurrency over time (in 10-second windows)
if bench:
    start_time = bench[0]["wall_s"]
    windows = {}
    for r in bench:
        window = int((r["wall_s"] - start_time) / 10) * 10
        if window not in windows:
            windows[window] = {"nreqs": [], "timers": [], "oracles": [], "hybrids": [], "gaps": []}
        windows[window]["nreqs"].append(r["n_reqs"])
        windows[window]["timers"].append(r["timer_delay_us"]/1000)
        windows[window]["oracles"].append(r["oracle_us"]/1000)
        windows[window]["hybrids"].append(r["hybrid_overhead_us"]/1000)

    # Compute gaps
    for i in range(1, len(bench)):
        window = int((bench[i]["wall_s"] - start_time) / 10) * 10
        gap = (bench[i]["wall_s"] - bench[i-1]["wall_s"]) * 1000
        windows[window]["gaps"].append(gap)

    print(f"\nTimeline (10-second windows):")
    print(f"{'Time':>6} {'Steps':>6} {'Conc':>6} {'Oracle':>8} {'Hybrid':>8} {'Timer':>8} {'Gap':>8}")
    for t in sorted(windows.keys()):
        w = windows[t]
        n = len(w["nreqs"])
        conc = statistics.median(w["nreqs"])
        oracle = statistics.median(w["oracles"])
        hybrid = statistics.median(w["hybrids"])
        timer = statistics.median(w["timers"])
        gap = statistics.median(w["gaps"]) if w["gaps"] else 0
        print(f"{t:>5}s {n:>6} {conc:>6.0f} {oracle:>7.1f}ms {hybrid:>7.1f}ms {timer:>7.1f}ms {gap:>7.1f}ms")

# At what point does concurrency exceed 30? 50? 80?
print(f"\nConcurrency milestones:")
for threshold in [20, 30, 40, 50, 60, 80]:
    first = next((i for i, r in enumerate(bench) if r["n_reqs"] >= threshold), None)
    if first is not None:
        time_s = bench[first]["wall_s"] - bench[0]["wall_s"]
        print(f"  First reqs>={threshold}: step {warmup_end+first} at {time_s:.0f}s into benchmark")
    else:
        print(f"  reqs>={threshold}: never reached")

# Feedback loop detection: does TPOT increase cause concurrency increase?
print(f"\nFeedback loop check:")
print(f"  Expected steady-state concurrency at rate=8:")
print(f"    If TPOT=12ms: each req takes 128*12=1536ms, conc=8*1.536=12")
print(f"    If TPOT=20ms: each req takes 128*20=2560ms, conc=8*2.56=20")
print(f"    If TPOT=30ms: each req takes 128*30=3840ms, conc=8*3.84=31")
print(f"    If TPOT=50ms: each req takes 128*50=6400ms, conc=8*6.4=51")
actual_med_conc = statistics.median([r["n_reqs"] for r in bench])
actual_med_timer = statistics.median([r["timer_delay_us"]/1000 for r in bench])
implied_tpot = actual_med_timer  # timer ≈ TPOT at per-step level
implied_req_time = 128 * implied_tpot / 1000  # seconds
implied_conc = 8 * implied_req_time
print(f"  Actual median concurrency: {actual_med_conc:.0f}")
print(f"  Actual median timer: {actual_med_timer:.1f}ms")
print(f"  Implied steady-state conc from timer: {implied_conc:.0f}")
if implied_conc > actual_med_conc * 1.5:
    print(f"  → FEEDBACK LOOP: timer overestimation drives concurrency up")
elif actual_med_conc > implied_conc * 1.5:
    print(f"  → CONCURRENCY BUILDUP: more requests than timer explains")
else:
    print(f"  → STABLE: concurrency matches timer prediction")
PYEOF

echo ""
echo "=== pyinstrument (top 40 lines) ==="
head -40 /workspace/debug_1k_r8_pyinst.txt 2>/dev/null || echo "No pyinstrument output"

cleanup_gpu
echo ""
echo "=== DONE $(date) ==="
