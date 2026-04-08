#!/bin/bash
# Test 2D table + hybrid overhead (combines correct GPU base with host-side costs)
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

echo "=== 2D Table + Hybrid Overhead Test ==="
echo "=== $(date) ==="

# Test rates 1, 4, 8 with 2D + auto-calibrated hybrid overhead (ThreadPool)
for RATE in 1 4 8; do
    TAG="2dh_r${RATE}"
    echo ""
    echo "=== Rate=$RATE (2D + hybrid, ThreadPool) ==="
    cleanup_gpu
    VLLM_EMULATOR_ENABLE_ORACLE=1 \
    VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
    VLLM_EMULATOR_MODE=realtime \
    VLLM_EMULATOR_EXECUTOR_HOOK=1 \
    VLLM_EMULATOR_ORACLE_MODE=2d \
    VLLM_EMULATOR_TIMER_MODE=pool \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
        > /workspace/2dh_emu_${RATE}.log 2>&1 &
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
        --save-result --result-dir "$ONLINE_DIR" --result-filename "${TAG}_emu.json" > /dev/null 2>&1

    cleanup_gpu
    python3 "$SCRIPT_DIR/compare_results.py" \
        "$ONLINE_DIR/fc_rate${RATE}_real.json" "$ONLINE_DIR/${TAG}_emu.json" 2>/dev/null || echo "  No baseline"
done

# Also test with chain (uncapped) for rate=8
echo ""
echo "=== Rate=8 (2D + hybrid + uncapped chain) ==="
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
    > /workspace/2dh_chain_emu_8.log 2>&1 &
wait_server
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 --request-rate 4 > /dev/null 2>&1
sleep 3
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 1000 --request-rate 8 --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,99 \
    --save-result --result-dir "$ONLINE_DIR" --result-filename "2dh_chain_r8_emu.json" > /dev/null 2>&1
cleanup_gpu
python3 "$SCRIPT_DIR/compare_results.py" \
    "$ONLINE_DIR/fc_rate8_real.json" "$ONLINE_DIR/2dh_chain_r8_emu.json" 2>/dev/null || echo "  No baseline"

# Summary
echo ""
echo "=== SUMMARY ==="
echo "2D + hybrid (ThreadPool):"
for RATE in 1 4 8; do
    python3 -c "
import json
e=json.load(open('$ONLINE_DIR/2dh_r${RATE}_emu.json'))
r=json.load(open('$ONLINE_DIR/fc_rate${RATE}_real.json'))
tpot=(e['mean_tpot_ms']-r['mean_tpot_ms'])/r['mean_tpot_ms']*100
e2e=(e.get('mean_e2el_ms',0)-r.get('mean_e2el_ms',0))/r.get('mean_e2el_ms',1)*100
ttft=(e['mean_ttft_ms']-r['mean_ttft_ms'])/r['mean_ttft_ms']*100
print(f'  R=${RATE}: TPOT={tpot:+.1f}% E2E={e2e:+.1f}% TTFT={ttft:+.1f}%')
" 2>/dev/null || echo "  R=$RATE: error"
done
echo ""
echo "2D + hybrid (chain uncapped) R=8:"
python3 -c "
import json
e=json.load(open('$ONLINE_DIR/2dh_chain_r8_emu.json'))
r=json.load(open('$ONLINE_DIR/fc_rate8_real.json'))
tpot=(e['mean_tpot_ms']-r['mean_tpot_ms'])/r['mean_tpot_ms']*100
e2e=(e.get('mean_e2el_ms',0)-r.get('mean_e2el_ms',0))/r.get('mean_e2el_ms',1)*100
ttft=(e['mean_ttft_ms']-r['mean_ttft_ms'])/r['mean_ttft_ms']*100
print(f'  R=8: TPOT={tpot:+.1f}% E2E={e2e:+.1f}% TTFT={ttft:+.1f}%')
" 2>/dev/null || echo "  R=8: error"

echo ""
echo "Comparison:"
echo "  2D only:       R1=-4.5%  R4=-10.0%  R8=-25.7%"
echo "  1D step_cycle: R1=-2.0%  R4=-9.1%   R8=+13.0%"
echo "  1D hybrid_lin: R1=+2.2%  R4=+3.6%   R8=+61.7%"
echo ""
echo "=== DONE $(date) ==="
