#!/bin/bash
# Test corrected oracle mode (bucketed correction table)
# Rebuilds profile with correction table, tests rates 1, 4, 8
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

echo "=== Corrected Oracle Test ==="
echo "=== $(date) ==="

# Rebuild profile with correction table
echo ""
echo "=== Rebuild profile ==="
PROFILE="${RESULT_DIR}/profiles/serving-1.5b-tp1-corrected.json"
python3 /workspace/vllm-emulator-v18/vllm_emulator/profile/build_serving_profile.py \
    "${RESULT_DIR}/step_cycle_final.jsonl" \
    "${RESULT_DIR}/profiles/sweep-1.5b-tp1-v14.json" \
    "$PROFILE" \
    "$MODEL" "RTX-3060-12GB"

# Show correction table
echo ""
echo "=== Correction table ==="
python3 -c "
import json
p = json.load(open('$PROFILE'))
for e in p.get('correction_table', []):
    print(f'  N={e[\"num_requests\"]:>3}: correction={e[\"correction_us\"]/1000:+.1f}ms (n={e[\"num_samples\"]})')
"

# Test corrected mode at rates 1, 4, 8 (1000 prompts)
# Use uncapped chain (best TTFT) + corrected oracle
for RATE in 1 4 8; do
    TAG="corr_r${RATE}"
    echo ""
    echo "=== Rate=$RATE (corrected, uncapped chain) ==="
    cleanup_gpu
    VLLM_EMULATOR_ENABLE_ORACLE=1 \
    VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
    VLLM_EMULATOR_MODE=realtime \
    VLLM_EMULATOR_EXECUTOR_HOOK=1 \
    VLLM_EMULATOR_ORACLE_MODE=corrected \
    VLLM_EMULATOR_TIMER_MODE=chain \
    VLLM_EMULATOR_CHAIN_CAP=0 \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
        > /workspace/corr_emu_${RATE}.log 2>&1 &
    wait_server
    if [ "$RATE" = "1" ]; then
        grep "ExecutorEmulatorHook" /workspace/corr_emu_${RATE}.log | head -1
    fi

    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 200 --request-rate 4 > /dev/null 2>&1
    sleep 3

    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 1000 --request-rate $RATE --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,99 \
        --save-result --result-dir "$ONLINE_DIR" --result-filename "${TAG}_emu.json" > /dev/null 2>&1

    cleanup_gpu
    python3 "$SCRIPT_DIR/compare_results.py" \
        "$ONLINE_DIR/fc_rate${RATE}_real.json" "$ONLINE_DIR/${TAG}_emu.json" 2>/dev/null || echo "  No baseline"
done

echo ""
echo "=== SUMMARY: corrected vs others ==="
for RATE in 1 4 8; do
    echo ""
    echo "--- Rate=$RATE ---"
    REAL="$ONLINE_DIR/fc_rate${RATE}_real.json"
    for TAG_LABEL in "fc_sc:step_cycle" "fc_hl:hybrid_lin" "fc_hs:hybrid_sqrt" "corr:corrected"; do
        PREFIX=$(echo $TAG_LABEL | cut -d: -f1)
        LABEL=$(echo $TAG_LABEL | cut -d: -f2)
        F="$ONLINE_DIR/${PREFIX}_r${RATE}_emu.json"
        if [ -f "$F" ] && [ -f "$REAL" ]; then
            python3 -c "
import json
e=json.load(open('$F')); r=json.load(open('$REAL'))
tpot=(e['mean_tpot_ms']-r['mean_tpot_ms'])/r['mean_tpot_ms']*100
e2e=(e.get('mean_e2el_ms',0)-r.get('mean_e2el_ms',0))/r.get('mean_e2el_ms',1)*100 if r.get('mean_e2el_ms') else 0
ttft=(e['mean_ttft_ms']-r['mean_ttft_ms'])/r['mean_ttft_ms']*100
print(f'  {\"$LABEL\":>14}: TPOT={tpot:+6.1f}% E2E={e2e:+6.1f}% TTFT={ttft:+6.1f}%')
" 2>/dev/null || echo "  ${LABEL}: error"
        fi
    done
done

echo ""
echo "=== DONE $(date) ==="
