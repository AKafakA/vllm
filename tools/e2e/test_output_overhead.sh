#!/bin/bash
# Quick test: add output overhead table to existing profile and test
# No re-profiling — uses existing data
set -e
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=1

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"
PORT=8100
SCRIPT_DIR="/workspace/vllm-emulator-v18/tools/e2e"
ONLINE_DIR="${RESULT_DIR}/online"
REAL_TAG="split2d"

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

echo "=== Output Overhead Quick Test ==="
echo "=== $(date) ==="

# Step 1: Inject output overhead table into existing profile
echo ""
echo "=== Step 1: Compute and inject output overhead ==="
PROFILE_IN="${RESULT_DIR}/profiles/serving-1.5b-tp1-split2d.json"
PROFILE_OUT="${RESULT_DIR}/profiles/serving-1.5b-tp1-with-overhead.json"

python3 << PYEOF
import json, statistics
from collections import defaultdict

# Load existing profile
profile = json.load(open("$PROFILE_IN"))

# Load step_cycle trace
records = []
for line in open("${RESULT_DIR}/step_cycle_adaptive.jsonl"):
    r = json.loads(line)
    if "total_tokens" in r:
        records.append(r)

# Decode step_cycle by concurrency
decode_by_conc = defaultdict(list)
for r in records[5000:]:
    if r.get("num_new_reqs", 0) == 0 and r.get("num_decode_seqs", 0) > 0:
        decode_by_conc[r["num_decode_seqs"]].append(r["step_cycle_us"])

# Compute overhead from real baselines
overhead_table = []
for rate in [1, 4, 8]:
    f = "${ONLINE_DIR}/${REAL_TAG}_r{}_real.json".format(rate)
    try:
        d = json.load(open(f))
        client_tpot_us = d["mean_tpot_ms"] * 1000
        mean_e2e_s = d.get("mean_e2el_ms", 0) / 1000
        avg_conc = int(rate * mean_e2e_s)

        # Step_cycle at that concurrency
        nearby = []
        for c in range(max(1, avg_conc - 3), avg_conc + 4):
            if c in decode_by_conc:
                nearby.extend(decode_by_conc[c])
        if nearby:
            step_cycle_us = statistics.median(nearby)
            overhead_us = client_tpot_us - step_cycle_us
            overhead_table.append({
                "num_requests": avg_conc,
                "overhead_us": round(max(0, overhead_us), 1),
            })
            print(f"  conc={avg_conc}: step_cycle={step_cycle_us/1000:.1f}ms, "
                  f"client_TPOT={client_tpot_us/1000:.1f}ms, "
                  f"overhead={overhead_us/1000:.1f}ms")
    except Exception as e:
        print(f"  R={rate}: {e}")

profile["output_overhead_table"] = sorted(overhead_table, key=lambda e: e["num_requests"])
json.dump(profile, open("$PROFILE_OUT", "w"), indent=2)
print(f"\nSaved to $PROFILE_OUT")
print(f"Overhead table: {json.dumps(overhead_table)}")
PYEOF

# Step 2: Test Option A + chain with output overhead
echo ""
echo "=== Step 2: Test with output overhead ==="
for RATE in 1 4 8; do
    TAG="overhead_optA_chain"
    echo "  $TAG rate=$RATE..."
    cleanup_gpu
    VLLM_EMULATOR_ENABLE_ORACLE=1 \
    VLLM_EMULATOR_PROFILE_PACK="$PROFILE_OUT" \
    VLLM_EMULATOR_MODE=realtime \
    VLLM_EMULATOR_EXECUTOR_HOOK=1 \
    VLLM_EMULATOR_ORACLE_MODE=2d \
    VLLM_EMULATOR_TIMER_MODE=chain \
    VLLM_EMULATOR_CHAIN_CAP=0 \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
        > /workspace/${TAG}_r${RATE}_server.log 2>&1 &
    wait_server

    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 200 --request-rate 4 > /dev/null 2>&1
    sleep 3

    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 1000 --request-rate $RATE --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,99 \
        --save-result --result-dir "$ONLINE_DIR" --result-filename "${TAG}_r${RATE}_emu.json" > /dev/null 2>&1
    cleanup_gpu
done

echo ""
echo "=== SUMMARY ==="
echo "Option A + chain + output overhead:"
for RATE in 1 4 8; do
    python3 -c "
import json
e=json.load(open('$ONLINE_DIR/overhead_optA_chain_r${RATE}_emu.json'))
r=json.load(open('$ONLINE_DIR/${REAL_TAG}_r${RATE}_real.json'))
tpot=(e['mean_tpot_ms']-r['mean_tpot_ms'])/r['mean_tpot_ms']*100
e2e=(e.get('mean_e2el_ms',0)-r.get('mean_e2el_ms',0))/r.get('mean_e2el_ms',1)*100
ttft=(e['mean_ttft_ms']-r['mean_ttft_ms'])/r['mean_ttft_ms']*100
print(f'  R=${RATE}: TPOT={tpot:+.1f}% E2E={e2e:+.1f}% TTFT={ttft:+.1f}%')
" 2>/dev/null || echo "  R=$RATE: error"
done
echo ""
echo "Without overhead (Option A + chain, no hybrid):"
echo "  R=1: TPOT=-4.0% E2E=-4.2% TTFT=-11.2%"
echo "  R=4: TPOT=-4.7% E2E=-5.1% TTFT=-13.7%"
echo "  R=8: TPOT=-13.2% E2E=-13.4% TTFT=-17.7%"
echo ""
echo "=== DONE $(date) ==="
