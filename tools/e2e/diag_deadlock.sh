#!/bin/bash
# Fine-grained timing: schedule vs execute_model vs sample_tokens
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1
MODEL="Qwen/Qwen2.5-1.5B-Instruct"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"
PORT=8100

cleanup_gpu() {
    pkill -9 -f "python3.*api_server" 2>/dev/null || true
    fuser /dev/nvidia* 2>/dev/null | tr " " "\n" | sort -u | xargs kill -9 2>/dev/null || true
    sleep 10
}
wait_server() {
    for i in $(seq 1 120); do curl -s http://localhost:$PORT/health > /dev/null 2>&1 && return 0; sleep 1; done
    echo "ERROR"; return 1
}

for MODE in real emu; do
    TRACE="${RESULT_DIR}/diag_fine_${MODE}.jsonl"
    echo "=== $MODE R=8 ==="
    cleanup_gpu
    rm -f "$TRACE"
    if [ "$MODE" = "emu" ]; then
        VLLM_EMULATOR_ENABLE_ORACLE=1 \
        VLLM_EMULATOR_PROFILE_PACK="${RESULT_DIR}/profiles/serving-1.5b-tp1-fresh.json" \
        VLLM_EMULATOR_MODE=realtime VLLM_EMULATOR_EXECUTOR_HOOK=1 \
        VLLM_EMULATOR_ORACLE_MODE=2d VLLM_EMULATOR_TIMER_MODE=chain \
        VLLM_EMULATOR_CHAIN_CAP=0 VLLM_EMULATOR_OUTPUT_OVERHEAD=0 \
        VLLM_EMULATOR_TRACE_STEP_CYCLE=1 VLLM_EMULATOR_STEP_TRACE_OUTPUT="$TRACE" \
        python3 -m vllm.entrypoints.openai.api_server \
            --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
            > /workspace/diag_fine_${MODE}_srv.log 2>&1 &
    else
        VLLM_EMULATOR_TRACE_STEP_CYCLE=1 VLLM_EMULATOR_STEP_TRACE_OUTPUT="$TRACE" \
        python3 -m vllm.entrypoints.openai.api_server \
            --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
            > /workspace/diag_fine_${MODE}_srv.log 2>&1 &
    fi
    wait_server || continue
    # Sweep warmup (real only)
    if [ "$MODE" = "real" ]; then
        for SWEEP_NP in 1 2 4 8 16 24 32 48 64 96 128 160 192 224 256; do
            python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
                --dataset-name random --random-input-len 1 --random-output-len 1 \
                --num-prompts $SWEEP_NP --request-rate inf > /dev/null 2>&1 || true
        done
    fi
    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 200 --request-rate 4 > /dev/null 2>&1
    echo '{"__marker__": "benchmark_start"}' >> "$TRACE"
    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 1000 --request-rate 8 > /dev/null 2>&1
    cleanup_gpu
done

echo ""
echo "=== Fine-Grained Breakdown ==="
python3 << 'PYEOF'
import json, statistics
from collections import defaultdict

for label, fname in [("Real", "/workspace/eval_results/RTX-3060-12GB/diag_fine_real.jsonl"),
                     ("Emu", "/workspace/eval_results/RTX-3060-12GB/diag_fine_emu.jsonl")]:
    recs = []
    in_bench = False
    for line in open(fname):
        r = json.loads(line)
        if r.get("__marker__") == "benchmark_start":
            in_bench = True
            continue
        if in_bench and "step_cycle_us" in r and "exec_ms" in r:
            recs.append(r)

    if not recs:
        print(f"\n{label}: no detailed records")
        continue

    print(f"\n=== {label} ({len(recs)} steps with detail) ===")
    for field in ["sched_ms", "exec_ms", "sample_ms", "wait_ms", "update_ms"]:
        vals = [r.get(field, 0) for r in recs if field in r]
        if vals:
            print(f"  {field:>12}: median={statistics.median(vals):.2f}ms mean={statistics.mean(vals):.2f}ms "
                  f"p90={sorted(vals)[int(len(vals)*0.9)]:.2f}ms")

    # exec_ms by total_tokens
    print(f"\n  exec_ms by total_tokens:")
    by_tt = defaultdict(list)
    for r in recs:
        tt_bucket = (r.get("total_tokens", 0) // 10) * 10
        by_tt[tt_bucket].append(r.get("exec_ms", 0))
    for tt in sorted(by_tt):
        if tt in [0, 10, 20, 50, 100, 200, 260]:
            vals = by_tt[tt]
            print(f"    tt={tt:>4}: n={len(vals):>4} median={statistics.median(vals):.2f}ms")

    # exec_ms by concurrency
    print(f"\n  exec_ms by concurrency:")
    by_conc = defaultdict(list)
    for r in recs:
        conc = r.get("num_decode_seqs", 0) + r.get("num_new_reqs", 0)
        by_conc[conc].append(r.get("exec_ms", 0))
    for c in [1, 5, 10, 15, 20, 25, 30]:
        nearby = []
        for cc in range(max(1, c-2), c+3):
            if cc in by_conc:
                nearby.extend(by_conc[cc])
        if nearby:
            print(f"    conc~{c:>3}: n={len(nearby):>4} median={statistics.median(nearby):.2f}ms")
PYEOF
echo ""
echo "=== DONE $(date) ==="
