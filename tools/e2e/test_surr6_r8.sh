#!/bin/bash
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

echo "=== Surrogate v6 R=8 ==="
echo "=== $(date) ==="
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
    > /workspace/surr6_r8_srv.log 2>&1 &
wait_server || exit 1

python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 --request-rate 4 > /dev/null 2>&1
sleep 3

python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 1000 --request-rate 8 --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,99 \
    --save-result --result-dir "$ONLINE_DIR" --result-filename "surr6_r8_emu.json" > /dev/null 2>&1
cleanup_gpu

python3 -c "
import json
e=json.load(open('$ONLINE_DIR/surr6_r8_emu.json'))
r=json.load(open('$ONLINE_DIR/${REAL_TAG}_r8_real.json'))
tpot=(e['mean_tpot_ms']-r['mean_tpot_ms'])/r['mean_tpot_ms']*100
e2e=(e.get('mean_e2el_ms',0)-r.get('mean_e2el_ms',0))/r.get('mean_e2el_ms',1)*100
ttft=(e['mean_ttft_ms']-r['mean_ttft_ms'])/r['mean_ttft_ms']*100
print(f'R=8: TPOT={tpot:+.1f}% E2E={e2e:+.1f}% TTFT={ttft:+.1f}%')
"

echo ""
echo "v5 was: R=8: TPOT=-2.4% E2E=-2.7% TTFT=-11.7%"
echo "=== DONE $(date) ==="
