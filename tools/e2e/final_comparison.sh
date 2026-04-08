#!/bin/bash
# FINAL COMPARISON: fresh profile + all oracle modes × all rates
# Modes: step_cycle, hybrid (linear), hybrid (sqrt), 2d
# Rates: 1, 4, 8 with 1000 prompts + offline
# All ThreadPool mechanism, fresh real baselines
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
ONLINE_TRACE="${RESULT_DIR}/step_cycle_final.jsonl"
OFFLINE_TRACE="${RESULT_DIR}/offline_step_cycle.jsonl"

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

run_emu() {
    local RATE=$1 TAG=$2 ORACLE_MODE=$3 SCALING=$4
    cleanup_gpu
    VLLM_EMULATOR_ENABLE_ORACLE=1 \
    VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
    VLLM_EMULATOR_MODE=realtime \
    VLLM_EMULATOR_EXECUTOR_HOOK=1 \
    VLLM_EMULATOR_ORACLE_MODE=$ORACLE_MODE \
    VLLM_EMULATOR_OVERHEAD_SCALING=$SCALING \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
        > /workspace/fc_emu_${TAG}.log 2>&1 &
    wait_server
    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 200 --request-rate 4 > /dev/null 2>&1
    sleep 3
    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 1000 --request-rate $RATE --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,99 \
        --save-result --result-dir "$ONLINE_DIR" --result-filename "fc_${TAG}_emu.json" > /dev/null 2>&1
}

echo "============================================"
echo "=== FINAL COMPARISON ==="
echo "=== $(date) ==="
echo "============================================"

rm -f "$ONLINE_TRACE" "$OFFLINE_TRACE"
rm -f "$ONLINE_DIR"/fc_*.json

# =============================================
# PHASE 1: Fresh profiling (online + offline)
# =============================================
echo ""
echo "=== PHASE 1: Online profiling ==="
cleanup_gpu
VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
VLLM_EMULATOR_STEP_TRACE_OUTPUT="$ONLINE_TRACE" \
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > /workspace/fc_profile_server.log 2>&1 &
wait_server

echo "  Warmup..."
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 --request-rate 4 > /dev/null 2>&1

for rate in 0.5 1 2 4 8 12 16 32 inf; do
    NP=200
    [ "$rate" = "0.5" ] && NP=50
    [ "$rate" = "16" ] || [ "$rate" = "32" ] || [ "$rate" = "inf" ] && NP=500
    echo "  Profile rate=$rate ($NP)..."
    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts $NP --request-rate $rate > /dev/null 2>&1
done
cleanup_gpu
echo "  Online records: $(wc -l < $ONLINE_TRACE)"

echo ""
echo "=== Offline profiling ==="
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
echo "  Offline records: $(wc -l < $OFFLINE_TRACE 2>/dev/null || echo 0)"

# =============================================
# PHASE 2: Build profile
# =============================================
echo ""
echo "=== PHASE 2: Build profile ==="
PROFILE="${RESULT_DIR}/profiles/serving-1.5b-tp1-final.json"
python3 /workspace/vllm-emulator-v18/vllm_emulator/profile/build_serving_profile.py \
    "$ONLINE_TRACE" \
    "${RESULT_DIR}/profiles/sweep-1.5b-tp1-v14.json" \
    "$PROFILE" \
    "$MODEL" "RTX-3060-12GB"

# =============================================
# PHASE 3: Real baselines (1000 prompts)
# =============================================
echo ""
echo "=== PHASE 3: Real baselines ==="
for RATE in 1 4 8; do
    cleanup_gpu
    echo "  Real rate=$RATE..."
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
        > /workspace/fc_real_${RATE}.log 2>&1 &
    wait_server
    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 200 --request-rate 4 > /dev/null 2>&1
    sleep 3
    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 1000 --request-rate $RATE --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,99 \
        --save-result --result-dir "$ONLINE_DIR" --result-filename "fc_rate${RATE}_real.json" > /dev/null 2>&1
done
cleanup_gpu

# =============================================
# PHASE 4: All oracle modes
# =============================================
echo ""
echo "=== PHASE 4: Oracle mode comparison ==="

# Mode 1: step_cycle (no overhead)
echo ""
echo "--- step_cycle ---"
for RATE in 1 4 8; do
    echo "  rate=$RATE..."
    run_emu $RATE "sc_r${RATE}" "step_cycle" "linear"
done

# Mode 2: hybrid linear (original)
echo ""
echo "--- hybrid linear ---"
for RATE in 1 4 8; do
    echo "  rate=$RATE..."
    run_emu $RATE "hl_r${RATE}" "hybrid" "linear"
done

# Mode 3: hybrid sqrt (sublinear)
echo ""
echo "--- hybrid sqrt ---"
for RATE in 1 4 8; do
    echo "  rate=$RATE..."
    run_emu $RATE "hs_r${RATE}" "hybrid" "sqrt"
done

# Mode 4: 2d (regression)
echo ""
echo "--- 2d ---"
for RATE in 1 4 8; do
    echo "  rate=$RATE..."
    run_emu $RATE "2d_r${RATE}" "2d" "linear"
done

cleanup_gpu

# =============================================
# PHASE 5: Offline
# =============================================
echo ""
echo "=== PHASE 5: Offline ==="
cleanup_gpu
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "$MODEL" --max-model-len 4096 --trust-remote-code \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 > /dev/null 2>&1
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "$MODEL" --max-model-len 4096 --trust-remote-code \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 2>&1 | tee /workspace/fc_offline_real.txt | grep "Throughput:"
cleanup_gpu

VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_PROFILE_USAGE=offline \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "$MODEL" --max-model-len 4096 --trust-remote-code \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 > /dev/null 2>&1
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_PROFILE_USAGE=offline \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "$MODEL" --max-model-len 4096 --trust-remote-code \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 2>&1 | tee /workspace/fc_offline_emu.txt | grep "Throughput:"
cleanup_gpu

# =============================================
# RESULTS
# =============================================
echo ""
echo "============================================"
echo "=== FINAL RESULTS ==="
echo "============================================"
echo ""
printf "%-16s %8s %8s %8s %8s %8s %8s %8s %8s %8s\n" \
    "Mode" "R1_TPOT" "R1_E2E" "R1_TTFT" "R4_TPOT" "R4_E2E" "R4_TTFT" "R8_TPOT" "R8_E2E" "R8_TTFT"

for MODE_TAG in "sc:step_cycle" "hl:hybrid_lin" "hs:hybrid_sqrt" "2d:2d_regr"; do
    PREFIX=$(echo $MODE_TAG | cut -d: -f1)
    LABEL=$(echo $MODE_TAG | cut -d: -f2)
    VALS=""
    for RATE in 1 4 8; do
        F="$ONLINE_DIR/fc_${PREFIX}_r${RATE}_emu.json"
        R="$ONLINE_DIR/fc_rate${RATE}_real.json"
        if [ -f "$F" ] && [ -f "$R" ]; then
            V=$(python3 -c "
import json
e=json.load(open('$F')); r=json.load(open('$R'))
tpot=(e['mean_tpot_ms']-r['mean_tpot_ms'])/r['mean_tpot_ms']*100
e2e=(e.get('mean_e2el_ms',0)-r.get('mean_e2el_ms',0))/r.get('mean_e2el_ms',1)*100 if r.get('mean_e2el_ms') else 0
ttft=(e['mean_ttft_ms']-r['mean_ttft_ms'])/r['mean_ttft_ms']*100
print(f'{tpot:+6.1f}% {e2e:+6.1f}% {ttft:+6.1f}%')
" 2>/dev/null || echo "   ERR    ERR    ERR")
            VALS="$VALS $V"
        else
            VALS="$VALS    N/A    N/A    N/A"
        fi
    done
    printf "%-16s%s\n" "$LABEL" "$VALS"
done

echo ""
echo "--- Offline ---"
echo "  Real: $(grep 'Throughput:' /workspace/fc_offline_real.txt 2>/dev/null || echo MISSING)"
echo "  Emu:  $(grep 'Throughput:' /workspace/fc_offline_emu.txt 2>/dev/null || echo MISSING)"

echo ""
echo "=== DONE $(date) ==="
