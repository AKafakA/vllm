#!/bin/bash
# Side-by-side: real GPU vs emulator at rate=8, 1000 prompts
# Both with step-cycle tracing to compare concurrency timelines
set -e
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=1

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
PROFILE="/workspace/eval_results/RTX-3060-12GB/profiles/serving-1.5b-tp1-final.json"
PORT=8100
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"

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

echo "=== Real vs Emu: rate=8, 1000 prompts, traced ==="
echo "=== $(date) ==="

# =============================================
# RUN 1: Real GPU with step-cycle tracing
# =============================================
echo ""
echo "=== REAL GPU (traced) ==="
cleanup_gpu
REAL_TRACE="/workspace/real_r8_1k_trace.jsonl"
rm -f "$REAL_TRACE"

VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
VLLM_EMULATOR_STEP_TRACE_OUTPUT="$REAL_TRACE" \
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > /workspace/real_r8_1k_server.log 2>&1 &
wait_server

echo "  Warmup..."
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 --request-rate 4 > /dev/null 2>&1
sleep 3

echo "  Bench rate=8 (1000 prompts)..."
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 1000 --request-rate 8 --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,99 \
    2>&1 | tail -5

cleanup_gpu
REAL_RECORDS=$(wc -l < "$REAL_TRACE" 2>/dev/null || echo 0)
echo "  Real trace records: $REAL_RECORDS"

# =============================================
# RUN 2: Emulator (uncapped chain + step_cycle, no hybrid)
# =============================================
echo ""
echo "=== EMULATOR (traced, step_cycle, uncapped chain) ==="
cleanup_gpu
EMU_TRACE="/workspace/emu_r8_1k_hook_trace.csv"
rm -f "$EMU_TRACE"

VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
VLLM_EMULATOR_ORACLE_MODE=step_cycle \
VLLM_EMULATOR_TIMER_MODE=chain \
VLLM_EMULATOR_CHAIN_CAP=0 \
VLLM_EMULATOR_HOOK_TRACE="$EMU_TRACE" \
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > /workspace/emu_r8_1k_server.log 2>&1 &
wait_server

echo "  Warmup..."
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 --request-rate 4 > /dev/null 2>&1
sleep 3

echo "  Bench rate=8 (1000 prompts)..."
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 1000 --request-rate 8 --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,99 \
    2>&1 | tail -5

cleanup_gpu
EMU_RECORDS=$(wc -l < "$EMU_TRACE" 2>/dev/null || echo 0)
echo "  Emu trace records: $EMU_RECORDS"

# =============================================
# COMPARE: concurrency timelines
# =============================================
echo ""
echo "=== COMPARISON ==="
python3 << 'PYEOF'
import json, csv, statistics
from collections import defaultdict

# Load real GPU trace
real_records = []
for line in open("/workspace/real_r8_1k_trace.jsonl"):
    r = json.loads(line)
    if "total_tokens" in r:
        real_records.append(r)

# Load emu trace
emu_records = []
with open("/workspace/emu_r8_1k_hook_trace.csv") as f:
    for r in csv.DictReader(f):
        emu_records.append({k: float(v) for k, v in r.items()})

print(f"Real steps: {len(real_records)}, Emu steps: {len(emu_records)}")

# Skip warmup (first 20% of each)
real_bench = real_records[len(real_records)//5:]
emu_bench = emu_records[len(emu_records)//5:]

# Real GPU concurrency: num_new_reqs + num_decode_seqs
real_conc = [r.get("num_new_reqs",0) + r.get("num_decode_seqs",0) for r in real_bench]
# Emu concurrency: from n_reqs column
emu_conc = [int(r["n_reqs"]) for r in emu_bench]

print(f"\n=== Concurrency stats ===")
print(f"  Real: mean={statistics.mean(real_conc):.1f}, median={statistics.median(real_conc):.0f}, max={max(real_conc)}")
print(f"  Emu:  mean={statistics.mean(emu_conc):.1f}, median={statistics.median(emu_conc):.0f}, max={max(emu_conc)}")

# Real step-cycle stats
real_cycles = [r["step_cycle_us"] for r in real_bench]
print(f"\n=== Step cycle ===")
print(f"  Real: mean={statistics.mean(real_cycles)/1000:.1f}ms, median={statistics.median(real_cycles)/1000:.1f}ms")

# Emu timer stats
emu_timers = [r["total_latency_us"]/1000 for r in emu_bench]
print(f"  Emu timer: mean={statistics.mean(emu_timers):.1f}ms, median={statistics.median(emu_timers):.1f}ms")

# Timeline comparison (by step index, normalized to percentage through benchmark)
print(f"\n=== Concurrency timeline (by % through benchmark) ===")
print(f"{'%':>5} {'Real_conc':>10} {'Real_cycle':>12} {'Emu_conc':>10} {'Emu_timer':>12}")
for pct in range(0, 100, 10):
    ri = int(pct/100 * len(real_bench))
    ei = int(pct/100 * len(emu_bench))
    if ri < len(real_bench) and ei < len(emu_bench):
        rc = real_conc[ri]
        rcyc = real_bench[ri]["step_cycle_us"]/1000
        ec = emu_conc[ei]
        etim = emu_bench[ei]["total_latency_us"]/1000
        print(f"{pct:>4}% {rc:>10} {rcyc:>11.1f}ms {ec:>10} {etim:>11.1f}ms")

# Concurrency distribution comparison
print(f"\n=== Concurrency distribution ===")
real_buckets = defaultdict(int)
emu_buckets = defaultdict(int)
for c in real_conc:
    real_buckets[(c//5)*5] += 1
for c in emu_conc:
    emu_buckets[(c//5)*5] += 1

all_buckets = sorted(set(list(real_buckets.keys()) + list(emu_buckets.keys())))
print(f"{'Bucket':>8} {'Real_steps':>12} {'Real_%':>8} {'Emu_steps':>12} {'Emu_%':>8}")
for b in all_buckets:
    rn = real_buckets.get(b, 0)
    en = emu_buckets.get(b, 0)
    rp = rn/len(real_bench)*100 if real_bench else 0
    ep = en/len(emu_bench)*100 if emu_bench else 0
    if rn > 10 or en > 10:
        print(f"{b:>3}-{b+4:>3} {rn:>12} {rp:>7.1f}% {en:>12} {ep:>7.1f}%")

# Key question: where does emu concurrency diverge from real?
print(f"\n=== KEY QUESTION ===")
if statistics.mean(emu_conc) > statistics.mean(real_conc) * 1.3:
    print(f"  Emu has {statistics.mean(emu_conc)/statistics.mean(real_conc):.1f}x more concurrency than real!")
    print(f"  → Emu is processing requests SLOWER, causing queue buildup")
elif statistics.mean(emu_conc) < statistics.mean(real_conc) * 0.7:
    print(f"  Emu has LESS concurrency than real")
    print(f"  → Emu is processing requests FASTER")
else:
    print(f"  Concurrency is similar (within 30%)")
    print(f"  → The per-step timing may be close but something else causes TPOT gap")
PYEOF

echo ""
echo "=== DONE $(date) ==="
