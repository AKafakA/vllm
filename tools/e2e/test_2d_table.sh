#!/bin/bash
# Test proper 2D oracle: (tt, concurrency) -> latency_us
# Uses the new step_cycle_2d_table from graph-enabled profiling
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

echo "=== 2D Table Oracle Test ==="
echo "=== $(date) ==="

# Step 1: Rebuild profile with 2D table from graph-enabled trace
echo ""
echo "=== Step 1: Rebuild profile with 2D table ==="
PROFILE="${RESULT_DIR}/profiles/serving-1.5b-tp1-2d.json"
python3 /workspace/vllm-emulator-v18/vllm_emulator/profile/build_serving_profile.py \
    "${RESULT_DIR}/step_cycle_graph.jsonl" \
    "${RESULT_DIR}/profiles/sweep-1.5b-tp1-v14.json" \
    "$PROFILE" \
    "$MODEL" "RTX-3060-12GB"

# Step 2: Verify oracle predictions at key (tt, conc) combos
echo ""
echo "=== Step 2: Oracle verification ==="
python3 -c "
import sys; sys.path.insert(0, '/workspace/vllm-emulator-v18')
import json
from vllm_emulator.oracle import create_oracle_from_profile_pack
p = json.load(open('$PROFILE'))
oracle = create_oracle_from_profile_pack(p)

print('2D table entries:', len(p.get('step_cycle_2d_table', [])))
print()
print('Oracle predictions (2D mode):')
print(f\"{'tt':>6} {'conc':>6} {'2d_lat':>10} {'1d_lat':>10} {'ratio':>8}\")
for tt in [5, 50, 100, 256, 270, 300, 500]:
    for conc in [1, 5, 10, 20, 50, 100, 200]:
        lat_2d = oracle.estimate_step_latency_us(tt, num_requests=conc, oracle_mode='2d')
        lat_1d = oracle.estimate_step_latency_us(tt, has_prefill=False, oracle_mode='step_cycle')
        ratio = lat_2d / lat_1d if lat_1d > 0 else 0
        if conc in [1, 10, 50, 200]:
            print(f'{tt:>6} {conc:>6} {lat_2d/1000:>9.1f}ms {lat_1d/1000:>9.1f}ms {ratio:>7.1f}x')
    print()
"

# Step 3: Run real baselines (if not already available)
for RATE in 1 4 8; do
    if [ ! -f "$ONLINE_DIR/fc_rate${RATE}_real.json" ]; then
        echo "=== Real baseline rate=$RATE ==="
        cleanup_gpu
        python3 -m vllm.entrypoints.openai.api_server \
            --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
            > /workspace/real_r${RATE}_server.log 2>&1 &
        wait_server
        # Warmup
        python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
            --dataset-name random --random-input-len 256 --random-output-len 128 \
            --num-prompts 200 --request-rate 4 > /dev/null 2>&1
        sleep 3
        python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
            --dataset-name random --random-input-len 256 --random-output-len 128 \
            --num-prompts 1000 --request-rate $RATE --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,99 \
            --save-result --result-dir "$ONLINE_DIR" --result-filename "fc_rate${RATE}_real.json" > /dev/null 2>&1
        cleanup_gpu
    fi
done

# Step 4: Test 2D oracle at rates 1, 4, 8
for RATE in 1 4 8; do
    TAG="2d_r${RATE}"
    echo ""
    echo "=== Rate=$RATE (2D table oracle, ThreadPool) ==="
    cleanup_gpu
    VLLM_EMULATOR_ENABLE_ORACLE=1 \
    VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
    VLLM_EMULATOR_MODE=realtime \
    VLLM_EMULATOR_EXECUTOR_HOOK=1 \
    VLLM_EMULATOR_ORACLE_MODE=2d \
    VLLM_EMULATOR_TIMER_MODE=pool \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
        > /workspace/2d_emu_${RATE}.log 2>&1 &
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

# Also test with chain (uncapped) for comparison
echo ""
echo "=== Rate=8 (2D table + uncapped chain) ==="
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
    > /workspace/2d_chain_emu_8.log 2>&1 &
wait_server
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 --request-rate 4 > /dev/null 2>&1
sleep 3
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 1000 --request-rate 8 --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,99 \
    --save-result --result-dir "$ONLINE_DIR" --result-filename "2d_chain_r8_emu.json" > /dev/null 2>&1
cleanup_gpu
python3 "$SCRIPT_DIR/compare_results.py" \
    "$ONLINE_DIR/fc_rate8_real.json" "$ONLINE_DIR/2d_chain_r8_emu.json" 2>/dev/null || echo "  No baseline"

# Summary
echo ""
echo "=== SUMMARY ==="
echo "2D table (ThreadPool):"
for RATE in 1 4 8; do
    python3 -c "
import json
e=json.load(open('$ONLINE_DIR/2d_r${RATE}_emu.json'))
r=json.load(open('$ONLINE_DIR/fc_rate${RATE}_real.json'))
tpot=(e['mean_tpot_ms']-r['mean_tpot_ms'])/r['mean_tpot_ms']*100
e2e=(e.get('mean_e2el_ms',0)-r.get('mean_e2el_ms',0))/r.get('mean_e2el_ms',1)*100
ttft=(e['mean_ttft_ms']-r['mean_ttft_ms'])/r['mean_ttft_ms']*100
print(f'  R=${RATE}: TPOT={tpot:+.1f}% E2E={e2e:+.1f}% TTFT={ttft:+.1f}%')
" 2>/dev/null || echo "  R=$RATE: error"
done
echo ""
echo "2D table (chain uncapped) R=8:"
python3 -c "
import json
e=json.load(open('$ONLINE_DIR/2d_chain_r8_emu.json'))
r=json.load(open('$ONLINE_DIR/fc_rate8_real.json'))
tpot=(e['mean_tpot_ms']-r['mean_tpot_ms'])/r['mean_tpot_ms']*100
e2e=(e.get('mean_e2el_ms',0)-r.get('mean_e2el_ms',0))/r.get('mean_e2el_ms',1)*100
ttft=(e['mean_ttft_ms']-r['mean_ttft_ms'])/r['mean_ttft_ms']*100
print(f'  R=8: TPOT={tpot:+.1f}% E2E={e2e:+.1f}% TTFT={ttft:+.1f}%')
" 2>/dev/null || echo "  R=8: error"

echo ""
echo "Previous best (step_cycle, ThreadPool):"
echo "  R=1: TPOT=-2.0% R=4: TPOT=-9.1% R=8: TPOT=+13.0%"
echo ""
echo "=== DONE $(date) ==="
