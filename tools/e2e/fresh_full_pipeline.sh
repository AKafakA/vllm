#!/bin/bash
# Fresh full pipeline: adaptive profiling + output overhead + baselines + emu
# All-in-one: produces a complete profile with output overhead table
# then tests at R=1,4,8 with Option A + chain
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
PROFILE_BUILDER="/workspace/vllm-emulator-v18/vllm_emulator/profile/build_serving_profile_filtered.py"
MAX_ROUNDS=5
TRACE="${RESULT_DIR}/step_cycle_fresh.jsonl"
TPOT_DIR="${RESULT_DIR}/profiling_tpot"
PROFILE="${RESULT_DIR}/profiles/serving-1.5b-tp1-fresh.json"

mkdir -p "$ONLINE_DIR" "$TPOT_DIR"
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

cuda_graph_warmup_sweep() {
    echo "    CUDA graph warmup sweep..."
    for SWEEP_NP in 1 2 4 8 16 24 32 48 64 96 128 160 192 224 256; do
        python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
            --dataset-name random --random-input-len 1 --random-output-len 1 \
            --num-prompts $SWEEP_NP --request-rate inf > /dev/null 2>&1 || true
    done
    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 500 --request-rate inf > /dev/null 2>&1 || true
}

echo "============================================"
echo "=== Fresh Full Pipeline ==="
echo "=== $(date) ==="
echo "============================================"

# =============================================
# STEP 1: Adaptive profiling (5 rounds, with CUDA graph sweep + TPOT capture)
# =============================================
echo ""
echo "=== STEP 1: Adaptive profiling ==="

RATES="1 2 3 4 5 6 8 10 12 16 24 32 0.5 inf"

get_num_prompts() {
    local round=$1
    local base=300
    [ "$round" -gt 1 ] && base=500
    [ "$round" -gt 3 ] && base=800
    echo $base
}

for ROUND in $(seq 1 $MAX_ROUNDS); do
    NP=$(get_num_prompts $ROUND)
    echo ""
    echo "  === Round $ROUND/$MAX_ROUNDS (${NP}p/rate) ==="
    cleanup_gpu

    VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
    VLLM_EMULATOR_STEP_TRACE_OUTPUT="$TRACE" \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
        > /workspace/fresh_profile_server.log 2>&1 &
    wait_server

    # CUDA graph warmup sweep
    cuda_graph_warmup_sweep

    # Standard warmup
    echo "    Warmup (200 prompts, rate=4)..."
    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 200 --request-rate 4 > /dev/null 2>&1

    # Write marker to trace: everything after this is clean profiling data
    echo "{\"__marker__\": \"profiling_start\", \"round\": $ROUND}" >> "$TRACE"

    # Profile each rate + save bench serve results for TPOT
    for rate in $RATES; do
        RP=$NP
        [ "$rate" = "0.5" ] && RP=$((NP / 3))
        echo "    rate=$rate ($RP prompts)..."
        python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
            --dataset-name random --random-input-len 256 --random-output-len 128 \
            --num-prompts $RP --request-rate $rate \
            --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,99 \
            --save-result --result-dir "$TPOT_DIR" --result-filename "round${ROUND}_rate${rate}.json" \
            > /dev/null 2>&1
    done

    # Variable length
    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 128 --random-output-len 64 \
        --num-prompts 300 --request-rate 8 > /dev/null 2>&1

    cleanup_gpu
    TOTAL=$(wc -l < "$TRACE" 2>/dev/null || echo 0)
    echo "    Total records: $TOTAL"
done

# =============================================
# STEP 2: Build profile with output overhead
# =============================================
echo ""
echo "=== STEP 2: Build profile ==="

# First build the base profile (2D tables, 1D sections)
python3 "$PROFILE_BUILDER" \
    "$TRACE" \
    "${RESULT_DIR}/profiles/sweep-1.5b-tp1-v14.json" \
    "$PROFILE" \
    "$MODEL" "RTX-3060-12GB"

# Output overhead table removed — 2D table + surrogate handles timing

# =============================================
# STEP 3: Real baselines (with sweep warmup)
# =============================================
echo ""
echo "=== STEP 3: Real baselines ==="
for RATE in 1 4 8; do
    echo "  Real rate=$RATE..."
    cleanup_gpu
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
        > /workspace/fresh_real_server.log 2>&1 &
    wait_server
    cuda_graph_warmup_sweep
    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 200 --request-rate 4 > /dev/null 2>&1
    sleep 3
    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 1000 --request-rate $RATE --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,99 \
        --save-result --result-dir "$ONLINE_DIR" --result-filename "fresh_r${RATE}_real.json" > /dev/null 2>&1
    cleanup_gpu
done

# =============================================
# STEP 4: Emulator (Option A + chain)
# =============================================
echo ""
echo "=== STEP 4: Emulator ==="
for RATE in 1 4 8; do
    echo "  Emu rate=$RATE..."
    cleanup_gpu
    VLLM_EMULATOR_ENABLE_ORACLE=1 \
    VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
    VLLM_EMULATOR_MODE=realtime \
    VLLM_EMULATOR_EXECUTOR_HOOK=1 \
    VLLM_EMULATOR_ORACLE_MODE=2d \
    VLLM_EMULATOR_TIMER_MODE=chain \
    VLLM_EMULATOR_CHAIN_CAP=0 \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
        > /workspace/fresh_emu_server.log 2>&1 &
    wait_server
    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 200 --request-rate 4 > /dev/null 2>&1
    sleep 3
    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 1000 --request-rate $RATE --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,99 \
        --save-result --result-dir "$ONLINE_DIR" --result-filename "fresh_r${RATE}_emu.json" > /dev/null 2>&1
    cleanup_gpu
    python3 "$SCRIPT_DIR/compare_results.py" \
        "$ONLINE_DIR/fresh_r${RATE}_real.json" "$ONLINE_DIR/fresh_r${RATE}_emu.json" 2>/dev/null || echo "  compare failed"
done

# =============================================
# SUMMARY
# =============================================
echo ""
echo "============================================"
echo "=== FINAL RESULTS ==="
echo "============================================"
echo ""
echo "Fresh pipeline (Option A + chain + output overhead):"
for RATE in 1 4 8; do
    python3 -c "
import json
e=json.load(open('$ONLINE_DIR/fresh_r${RATE}_emu.json'))
r=json.load(open('$ONLINE_DIR/fresh_r${RATE}_real.json'))
tpot=(e['mean_tpot_ms']-r['mean_tpot_ms'])/r['mean_tpot_ms']*100
e2e=(e.get('mean_e2el_ms',0)-r.get('mean_e2el_ms',0))/r.get('mean_e2el_ms',1)*100
ttft=(e['mean_ttft_ms']-r['mean_ttft_ms'])/r['mean_ttft_ms']*100
tp_err=0
if r.get('output_throughput',0)>0:
    tp_err=(e['output_throughput']-r['output_throughput'])/r['output_throughput']*100
print(f'  R=${RATE}: TPOT={tpot:+.1f}% E2E={e2e:+.1f}% TTFT={ttft:+.1f}% Thru={tp_err:+.1f}%')
" 2>/dev/null || echo "  R=$RATE: error"
done
echo ""
echo "Targets: TPOT<5% E2E<5% TTFT<15% Thru<1%"
echo ""
echo "=== DONE $(date) ==="
