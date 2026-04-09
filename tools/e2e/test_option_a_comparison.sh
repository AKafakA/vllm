#!/bin/bash
# Compare Option A (1D prefill + 2D decode) vs current (split 2D for both)
# Uses the same profile and baselines from the pipeline run
set -e
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=1

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"
PROFILE="${RESULT_DIR}/profiles/serving-1.5b-tp1-split2d.json"
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

run_emu_chain() {
    local TAG=$1
    local RATE=$2
    local REAL_TAG=$3
    echo "  $TAG rate=$RATE..."
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
    python3 "$SCRIPT_DIR/compare_results.py" \
        "$ONLINE_DIR/${REAL_TAG}_r${RATE}_real.json" "$ONLINE_DIR/${TAG}_r${RATE}_emu.json" 2>/dev/null || echo "  compare failed"
}

echo "============================================"
echo "=== Option A Comparison Test ==="
echo "=== $(date) ==="
echo "============================================"

if [ ! -f "$PROFILE" ]; then
    echo "ERROR: Profile not found. Run pipeline first."
    exit 1
fi

# Determine which real baselines to use
REAL_TAG="split2d"
if [ ! -f "$ONLINE_DIR/split2d_r1_real.json" ]; then
    # Try warmswept baselines
    REAL_TAG="warmswept"
    if [ ! -f "$ONLINE_DIR/warmswept_r1_real.json" ]; then
        REAL_TAG="fc_rate"
        echo "Using original baselines (fc_rate*_real.json)"
    fi
fi
echo "Using baselines: ${REAL_TAG}_r{1,4,8}_real.json"

# The pipeline already tested with the OLD oracle (split 2D for both).
# Those results are in split2d_r{1,4,8}_emu.json.
# Now test with Option A oracle (1D prefill + 2D decode).
echo ""
echo "=== Option A: 1D prefill + 2D decode ==="
for RATE in 1 4 8; do
    run_emu_chain "optA_1d2d" $RATE "$REAL_TAG"
done

echo ""
echo "============================================"
echo "=== SUMMARY ==="
echo "============================================"
echo ""
echo "Option A (1D prefill + 2D decode):"
for RATE in 1 4 8; do
    REAL="$ONLINE_DIR/${REAL_TAG}_r${RATE}_real.json"
    EMU="$ONLINE_DIR/optA_1d2d_r${RATE}_emu.json"
    if [ -f "$EMU" ] && [ -f "$REAL" ]; then
        python3 -c "
import json
e=json.load(open('$EMU'))
r=json.load(open('$REAL'))
tpot=(e['mean_tpot_ms']-r['mean_tpot_ms'])/r['mean_tpot_ms']*100
e2e=(e.get('mean_e2el_ms',0)-r.get('mean_e2el_ms',0))/r.get('mean_e2el_ms',1)*100
ttft=(e['mean_ttft_ms']-r['mean_ttft_ms'])/r['mean_ttft_ms']*100
print(f'  R=$RATE: TPOT={tpot:+.1f}% E2E={e2e:+.1f}% TTFT={ttft:+.1f}%')
" 2>/dev/null || echo "  R=$RATE: error"
    fi
done

echo ""
echo "Pipeline results (split 2D for both):"
for RATE in 1 4 8; do
    REAL="$ONLINE_DIR/${REAL_TAG}_r${RATE}_real.json"
    EMU="$ONLINE_DIR/split2d_r${RATE}_emu.json"
    if [ -f "$EMU" ] && [ -f "$REAL" ]; then
        python3 -c "
import json
e=json.load(open('$EMU'))
r=json.load(open('$REAL'))
tpot=(e['mean_tpot_ms']-r['mean_tpot_ms'])/r['mean_tpot_ms']*100
e2e=(e.get('mean_e2el_ms',0)-r.get('mean_e2el_ms',0))/r.get('mean_e2el_ms',1)*100
ttft=(e['mean_ttft_ms']-r['mean_ttft_ms'])/r['mean_ttft_ms']*100
print(f'  R=$RATE: TPOT={tpot:+.1f}% E2E={e2e:+.1f}% TTFT={ttft:+.1f}%')
" 2>/dev/null || echo "  R=$RATE: error"
    fi
done

echo ""
echo "Previous best (filtered adaptive, no sweep warmup):"
echo "  R=1: TPOT=-0.8% E2E=-1.3% TTFT=-13.0%"
echo "  R=4: TPOT=+3.4% E2E=+2.3% TTFT=-22.0%"
echo "  R=8: TPOT=+4.6% E2E=+3.7% TTFT=-19.1%"
echo ""
echo "=== DONE $(date) ==="
