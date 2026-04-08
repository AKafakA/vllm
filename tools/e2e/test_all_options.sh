#!/bin/bash
# Round-robin test of all timer architecture options (D, E, F)
# Plus baseline (A=chain, B=pool) for comparison
# All use 2D oracle + hybrid overhead, 1000 prompts, rates 1,4,8
set -e
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=1

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"
PROFILE="${RESULT_DIR}/profiles/serving-1.5b-tp1-2d.json"
PORT=8100
SCRIPT_DIR="/workspace/vllm-emulator-v18/tools/e2e"
ONLINE_DIR="${RESULT_DIR}/online"

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

run_emu_test() {
    local TAG=$1
    local RATE=$2
    local TIMER_MODE=$3
    local CHAIN_CAP=${4:-0}

    echo ""
    echo "=== ${TAG} Rate=$RATE (timer=$TIMER_MODE cap=$CHAIN_CAP) ==="
    cleanup_gpu
    VLLM_EMULATOR_ENABLE_ORACLE=1 \
    VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
    VLLM_EMULATOR_MODE=realtime \
    VLLM_EMULATOR_EXECUTOR_HOOK=1 \
    VLLM_EMULATOR_ORACLE_MODE=2d \
    VLLM_EMULATOR_TIMER_MODE=$TIMER_MODE \
    VLLM_EMULATOR_CHAIN_CAP=$CHAIN_CAP \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
        > /workspace/${TAG}_r${RATE}_server.log 2>&1 &
    wait_server

    # Warmup
    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 200 --request-rate 4 > /dev/null 2>&1
    sleep 3

    # Benchmark
    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 1000 --request-rate $RATE --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,99 \
        --save-result --result-dir "$ONLINE_DIR" --result-filename "${TAG}_r${RATE}_emu.json" > /dev/null 2>&1

    cleanup_gpu
    python3 "$SCRIPT_DIR/compare_results.py" \
        "$ONLINE_DIR/fc_rate${RATE}_real.json" "$ONLINE_DIR/${TAG}_r${RATE}_emu.json" 2>/dev/null || echo "  compare failed"
}

echo "============================================"
echo "=== Round-Robin Timer Options Test ==="
echo "=== $(date) ==="
echo "============================================"

# Check profile exists
if [ ! -f "$PROFILE" ]; then
    echo "ERROR: Profile not found at $PROFILE"
    echo "Run profile_graph_enabled.sh + build_serving_profile.py first"
    exit 1
fi

# Check real baselines exist
for RATE in 1 4 8; do
    if [ ! -f "$ONLINE_DIR/fc_rate${RATE}_real.json" ]; then
        echo "ERROR: Real baseline not found for rate=$RATE"
        exit 1
    fi
done

echo ""
echo "Profile: $PROFILE"
echo "Baselines: fc_rate{1,4,8}_real.json"
echo ""

# =============================================
# Test each option at rates 1, 4, 8
# Order: E (split), D (dcap), F (block), A (chain), B (pool)
# =============================================

# Option B: Pool (baseline)
for RATE in 1 4 8; do
    run_emu_test "optB_pool" $RATE "pool"
done

# Option A: Uncapped chain (baseline)
for RATE in 1 4 8; do
    run_emu_test "optA_chain" $RATE "chain" 0
done

# Option E: Split (chain prefill, pool decode)
for RATE in 1 4 8; do
    run_emu_test "optE_split" $RATE "split" 0
done

# Option D: Chain + profiled drift correction
for RATE in 1 4 8; do
    run_emu_test "optD_dcap" $RATE "dcap" 0
done

# Option F: Full engine-thread blocking
for RATE in 1 4 8; do
    run_emu_test "optF_block" $RATE "block"
done

# =============================================
# SUMMARY
# =============================================
echo ""
echo "============================================"
echo "=== FINAL SUMMARY ==="
echo "============================================"
echo ""

for OPT in optB_pool optA_chain optE_split optD_dcap optF_block; do
    echo "--- $OPT ---"
    for RATE in 1 4 8; do
        REAL="$ONLINE_DIR/fc_rate${RATE}_real.json"
        EMU="$ONLINE_DIR/${OPT}_r${RATE}_emu.json"
        if [ -f "$EMU" ] && [ -f "$REAL" ]; then
            python3 -c "
import json
e=json.load(open('$EMU'))
r=json.load(open('$REAL'))
tpot=(e['mean_tpot_ms']-r['mean_tpot_ms'])/r['mean_tpot_ms']*100
e2e=(e.get('mean_e2el_ms',0)-r.get('mean_e2el_ms',0))/r.get('mean_e2el_ms',1)*100
ttft=(e['mean_ttft_ms']-r['mean_ttft_ms'])/r['mean_ttft_ms']*100
tp_err=0
if r.get('output_throughput',0)>0 and e.get('output_throughput',0)>0:
    tp_err=(e['output_throughput']-r['output_throughput'])/r['output_throughput']*100
print(f'  R=$RATE: TPOT={tpot:+.1f}% E2E={e2e:+.1f}% TTFT={ttft:+.1f}% Thru={tp_err:+.1f}%')
" 2>/dev/null || echo "  R=$RATE: error"
        else
            echo "  R=$RATE: missing results"
        fi
    done
    echo ""
done

echo "=== Targets: TPOT<5% E2E<5% TTFT<15% Thru<1% ==="
echo "=== DONE $(date) ==="
