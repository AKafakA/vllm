#!/bin/bash
# Chain comparison using same baselines/profile from final_comparison
# Tests: uncapped chain × {step_cycle, hybrid_sqrt}
#        capped chain × {step_cycle, hybrid_sqrt}
# Reuses fc_rate*_real.json baselines and serving-1.5b-tp1-final.json profile
set -e
source /workspace/vllm-v18-env/bin/activate
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

run_emu() {
    local RATE=$1 TAG=$2 ORACLE_MODE=$3 SCALING=$4 CHAIN_CAP=$5
    cleanup_gpu
    VLLM_EMULATOR_ENABLE_ORACLE=1 \
    VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
    VLLM_EMULATOR_MODE=realtime \
    VLLM_EMULATOR_EXECUTOR_HOOK=1 \
    VLLM_EMULATOR_ORACLE_MODE=$ORACLE_MODE \
    VLLM_EMULATOR_OVERHEAD_SCALING=$SCALING \
    VLLM_EMULATOR_CHAIN_CAP=$CHAIN_CAP \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
        > /workspace/fcc_emu_${TAG}.log 2>&1 &
    wait_server
    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 200 --request-rate 4 > /dev/null 2>&1
    sleep 3
    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 1000 --request-rate $RATE --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,99 \
        --save-result --result-dir "$ONLINE_DIR" --result-filename "fcc_${TAG}_emu.json" > /dev/null 2>&1
}

# Verify real baselines exist
if [ ! -f "$ONLINE_DIR/fc_rate1_real.json" ]; then
    echo "ERROR: Real baselines not found. Run final_comparison.sh first."
    exit 1
fi

echo "============================================"
echo "=== CHAIN COMPARISON ==="
echo "=== Uses same baselines from final_comparison ==="
echo "=== $(date) ==="
echo "============================================"

# Uncapped chain + step_cycle
echo ""
echo "--- uncapped chain + step_cycle ---"
for RATE in 1 4 8; do
    echo "  rate=$RATE..."
    run_emu $RATE "uc_sc_r${RATE}" "step_cycle" "linear" "0"
done

# Uncapped chain + hybrid sqrt
echo ""
echo "--- uncapped chain + hybrid sqrt ---"
for RATE in 1 4 8; do
    echo "  rate=$RATE..."
    run_emu $RATE "uc_hs_r${RATE}" "hybrid" "sqrt" "0"
done

# Capped chain + step_cycle
echo ""
echo "--- capped chain + step_cycle ---"
for RATE in 1 4 8; do
    echo "  rate=$RATE..."
    run_emu $RATE "cc_sc_r${RATE}" "step_cycle" "linear" "1"
done

# Capped chain + hybrid sqrt
echo ""
echo "--- capped chain + hybrid sqrt ---"
for RATE in 1 4 8; do
    echo "  rate=$RATE..."
    run_emu $RATE "cc_hs_r${RATE}" "hybrid" "sqrt" "1"
done

cleanup_gpu

# =============================================
# RESULTS
# =============================================
echo ""
echo "============================================"
echo "=== CHAIN + THREADPOOL COMPARISON ==="
echo "============================================"
echo ""
printf "%-24s %8s %8s %8s %8s %8s %8s %8s %8s %8s\n" \
    "Mode" "R1_TPOT" "R1_E2E" "R1_TTFT" "R4_TPOT" "R4_E2E" "R4_TTFT" "R8_TPOT" "R8_E2E" "R8_TTFT"

# ThreadPool results (from final_comparison)
for MODE_TAG in "fc_sc:pool+sc" "fc_hl:pool+hybrid_lin" "fc_hs:pool+hybrid_sqrt"; do
    PREFIX=$(echo $MODE_TAG | cut -d: -f1)
    LABEL=$(echo $MODE_TAG | cut -d: -f2)
    VALS=""
    for RATE in 1 4 8; do
        F="$ONLINE_DIR/${PREFIX}_r${RATE}_emu.json"
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
    printf "%-24s%s\n" "$LABEL" "$VALS"
done

# Chain results
for MODE_TAG in "uc_sc:unchain+sc" "uc_hs:unchain+hybrid_sqrt" "cc_sc:cap_chain+sc" "cc_hs:cap_chain+hybrid_sqrt"; do
    PREFIX=$(echo $MODE_TAG | cut -d: -f1)
    LABEL=$(echo $MODE_TAG | cut -d: -f2)
    VALS=""
    for RATE in 1 4 8; do
        F="$ONLINE_DIR/fcc_${PREFIX}_r${RATE}_emu.json"
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
    printf "%-24s%s\n" "$LABEL" "$VALS"
done

echo ""
echo "=== DONE $(date) ==="
