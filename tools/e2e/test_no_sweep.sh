#!/bin/bash
# Test with sweep removed from online profile
# Uses step_cycle oracle + uncapped chain
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

echo "=== No-Sweep Profile Test ==="
echo "=== $(date) ==="

# Rebuild profile WITHOUT sweep merge
echo "=== Rebuild profile (no sweep merge) ==="
PROFILE="${RESULT_DIR}/profiles/serving-1.5b-tp1-nosweep.json"
python3 /workspace/vllm-emulator-v18/vllm_emulator/profile/build_serving_profile.py \
    "${RESULT_DIR}/step_cycle_final.jsonl" \
    "${RESULT_DIR}/profiles/sweep-1.5b-tp1-v14.json" \
    "$PROFILE" \
    "$MODEL" "RTX-3060-12GB"

# Check what the oracle now predicts at tt=278
echo ""
echo "=== Oracle at critical tt values ==="
python3 -c "
import sys; sys.path.insert(0, '/workspace/vllm-emulator-v18')
import json
from vllm_emulator.oracle import create_oracle_from_profile_pack
p = json.load(open('$PROFILE'))
oracle = create_oracle_from_profile_pack(p)
for tt in [256, 270, 272, 278, 285, 300, 500]:
    lat_pf = oracle.estimate_step_latency_us(tt, has_prefill=True, oracle_mode='step_cycle')
    lat_dc = oracle.estimate_step_latency_us(tt, has_prefill=False, oracle_mode='step_cycle')
    print(f'  tt={tt:>4}: prefill={lat_pf/1000:.1f}ms  decode={lat_dc/1000:.1f}ms')
print('  (Previously tt=278 prefill was 88ms from sweep, now should be ~18ms)')
"

# Test rates 1, 4, 8 with step_cycle + uncapped chain
for RATE in 1 4 8; do
    TAG="nosweep_r${RATE}"
    echo ""
    echo "=== Rate=$RATE (step_cycle, no sweep, uncapped chain) ==="
    cleanup_gpu
    VLLM_EMULATOR_ENABLE_ORACLE=1 \
    VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
    VLLM_EMULATOR_MODE=realtime \
    VLLM_EMULATOR_EXECUTOR_HOOK=1 \
    VLLM_EMULATOR_ORACLE_MODE=step_cycle \
    VLLM_EMULATOR_TIMER_MODE=chain \
    VLLM_EMULATOR_CHAIN_CAP=0 \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
        > /workspace/nosweep_emu_${RATE}.log 2>&1 &
    wait_server

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

# Also test hybrid_lin with no-sweep profile (should fix rate=8 too)
echo ""
echo "=== Rate=8 hybrid_lin + no-sweep ==="
cleanup_gpu
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
VLLM_EMULATOR_ORACLE_MODE=hybrid \
VLLM_EMULATOR_OVERHEAD_SCALING=linear \
VLLM_EMULATOR_TIMER_MODE=chain \
VLLM_EMULATOR_CHAIN_CAP=0 \
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > /workspace/nosweep_hl_emu_8.log 2>&1 &
wait_server
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 --request-rate 4 > /dev/null 2>&1
sleep 3
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 1000 --request-rate 8 --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,99 \
    --save-result --result-dir "$ONLINE_DIR" --result-filename "nosweep_hl_r8_emu.json" > /dev/null 2>&1
cleanup_gpu
python3 "$SCRIPT_DIR/compare_results.py" \
    "$ONLINE_DIR/fc_rate8_real.json" "$ONLINE_DIR/nosweep_hl_r8_emu.json" 2>/dev/null || echo "  No baseline"

echo ""
echo "=== SUMMARY ==="
echo "No-sweep step_cycle:"
for RATE in 1 4 8; do
    python3 -c "
import json
e=json.load(open('$ONLINE_DIR/nosweep_r${RATE}_emu.json'))
r=json.load(open('$ONLINE_DIR/fc_rate${RATE}_real.json'))
tpot=(e['mean_tpot_ms']-r['mean_tpot_ms'])/r['mean_tpot_ms']*100
e2e=(e.get('mean_e2el_ms',0)-r.get('mean_e2el_ms',0))/r.get('mean_e2el_ms',1)*100
ttft=(e['mean_ttft_ms']-r['mean_ttft_ms'])/r['mean_ttft_ms']*100
print(f'  R={\"$RATE\"}: TPOT={tpot:+.1f}% E2E={e2e:+.1f}% TTFT={ttft:+.1f}%')
" 2>/dev/null || echo "  R=$RATE: error"
done
echo "No-sweep hybrid_lin R=8:"
python3 -c "
import json
e=json.load(open('$ONLINE_DIR/nosweep_hl_r8_emu.json'))
r=json.load(open('$ONLINE_DIR/fc_rate8_real.json'))
tpot=(e['mean_tpot_ms']-r['mean_tpot_ms'])/r['mean_tpot_ms']*100
e2e=(e.get('mean_e2el_ms',0)-r.get('mean_e2el_ms',0))/r.get('mean_e2el_ms',1)*100
ttft=(e['mean_ttft_ms']-r['mean_ttft_ms'])/r['mean_ttft_ms']*100
print(f'  R=8: TPOT={tpot:+.1f}% E2E={e2e:+.1f}% TTFT={ttft:+.1f}%')
" 2>/dev/null || echo "  R=8: error"

echo "Previous (with sweep) for comparison:"
echo "  step_cycle: R1=-2.0% R4=-9.1% R8=+13.0%"
echo "  hybrid_lin: R1=+2.2% R4=+3.6% R8=+61.7%"

echo ""
echo "=== DONE $(date) ==="
