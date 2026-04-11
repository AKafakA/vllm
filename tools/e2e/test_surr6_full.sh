#!/bin/bash
# Full test: surrogate v6 at R=1,4,8 + offline
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1
MODEL="Qwen/Qwen2.5-1.5B-Instruct"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"
PROFILE="${RESULT_DIR}/profiles/serving-1.5b-tp1-fresh.json"
PORT=8100
ONLINE_DIR="${RESULT_DIR}/online"
REAL_TAG="fresh"

cleanup_gpu() {
    pkill -9 -f "python3.*api_server" 2>/dev/null || true
    pkill -9 -f "python3.*bench" 2>/dev/null || true
    fuser /dev/nvidia* 2>/dev/null | tr " " "\n" | sort -u | xargs kill -9 2>/dev/null || true
    sleep 10
}
wait_server() {
    for i in $(seq 1 120); do curl -s http://localhost:$PORT/health > /dev/null 2>&1 && return 0; sleep 1; done
    echo "ERROR: Server failed to start"; return 1
}

echo "=== Surrogate v6 Full Test ==="
echo "=== $(date) ==="

# Online: R=4, R=8, R=1
for RATE in 4 8 1; do
    TAG="surr6_r${RATE}"
    echo ""
    echo "=== Online R=$RATE ==="
    cleanup_gpu
    VLLM_EMULATOR_ENABLE_ORACLE=1 \
    VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
    VLLM_EMULATOR_MODE=realtime \
    VLLM_EMULATOR_EXECUTOR_HOOK=1 \
    VLLM_EMULATOR_ORACLE_MODE=2d \
    VLLM_EMULATOR_TIMER_MODE=chain \
    VLLM_EMULATOR_CHAIN_CAP=0 \
    VLLM_EMULATOR_OUTPUT_OVERHEAD=0 \
    VLLM_EMULATOR_PREP_SURROGATE=1 \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
        > /workspace/${TAG}_server.log 2>&1 &
    wait_server || continue
    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 200 --request-rate 4 > /dev/null 2>&1
    sleep 3
    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 1000 --request-rate $RATE --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,99 \
        --save-result --result-dir "$ONLINE_DIR" --result-filename "${TAG}_emu.json" > /dev/null 2>&1
    cleanup_gpu
    python3 -c "
import json
e=json.load(open('$ONLINE_DIR/${TAG}_emu.json'))
r=json.load(open('$ONLINE_DIR/${REAL_TAG}_r${RATE}_real.json'))
tpot=(e['mean_tpot_ms']-r['mean_tpot_ms'])/r['mean_tpot_ms']*100
e2e=(e.get('mean_e2el_ms',0)-r.get('mean_e2el_ms',0))/r.get('mean_e2el_ms',1)*100
ttft=(e['mean_ttft_ms']-r['mean_ttft_ms'])/r['mean_ttft_ms']*100
print(f'R=${RATE}: TPOT={tpot:+.1f}% E2E={e2e:+.1f}% TTFT={ttft:+.1f}%')
" 2>/dev/null || echo "R=$RATE: compare error"
done

# Offline
echo ""
echo "=== Offline ==="
cleanup_gpu
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
VLLM_EMULATOR_ORACLE_MODE=2d \
VLLM_EMULATOR_PREP_SURROGATE=1 \
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "$MODEL" --max-model-len 4096 --trust-remote-code \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 500 2>&1 | tee /workspace/surr6_offline.txt
EMU_TPUT=$(grep "Throughput:" /workspace/surr6_offline.txt | grep -oP '[\d.]+(?= requests)')
echo "Offline: Real=18.52 req/s Emu=${EMU_TPUT} req/s"

echo ""
echo "=== SUMMARY ==="
echo "v6 (per-layer × total_tokens scaling):"
for RATE in 1 4 8; do
    python3 -c "
import json
e=json.load(open('$ONLINE_DIR/surr6_r${RATE}_emu.json'))
r=json.load(open('$ONLINE_DIR/${REAL_TAG}_r${RATE}_real.json'))
tpot=(e['mean_tpot_ms']-r['mean_tpot_ms'])/r['mean_tpot_ms']*100
e2e=(e.get('mean_e2el_ms',0)-r.get('mean_e2el_ms',0))/r.get('mean_e2el_ms',1)*100
ttft=(e['mean_ttft_ms']-r['mean_ttft_ms'])/r['mean_ttft_ms']*100
print(f'  R=${RATE}: TPOT={tpot:+.1f}% E2E={e2e:+.1f}% TTFT={ttft:+.1f}%')
" 2>/dev/null || echo "  R=$RATE: error"
done
echo ""
echo "v5 was:"
echo "  R=1: TPOT=+3.4% E2E=+3.4% TTFT=+3.0%"
echo "  R=4: TPOT=+4.0% E2E=+3.6% TTFT=-6.8%"
echo "  R=8: TPOT=-2.4% E2E=-2.7% TTFT=-11.7%"
echo "  Offline: +2.8%"
echo ""
echo "=== DONE $(date) ==="
