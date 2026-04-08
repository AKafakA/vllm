#!/bin/bash
# Full comparison: fresh profile + all rates + all oracle modes + offline
# 1000 prompts per rate, fresh real baselines shared across modes
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

echo "============================================"
echo "=== FULL ORACLE COMPARISON ==="
echo "=== $(date) ==="
echo "============================================"

rm -f "$ONLINE_TRACE" "$OFFLINE_TRACE"
rm -f "$ONLINE_DIR"/final_*.json

# =============================================
# PHASE 1: Fresh online profiling
# =============================================
echo ""
echo "=== PHASE 1: Online profiling ==="
cleanup_gpu

VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
VLLM_EMULATOR_STEP_TRACE_OUTPUT="$ONLINE_TRACE" \
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > /workspace/final_profile_server.log 2>&1 &
wait_server

echo "  Warmup (200 prompts, rate=4)..."
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 --request-rate 4 > /dev/null 2>&1

for rate in 0.5 1 2 4 8 12 16 32 inf; do
    NP=200
    if [ "$rate" = "0.5" ]; then NP=50; fi
    if [ "$rate" = "16" ] || [ "$rate" = "32" ] || [ "$rate" = "inf" ]; then NP=500; fi
    echo "  Profile rate=$rate ($NP prompts)..."
    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts $NP --request-rate $rate > /dev/null 2>&1
done
cleanup_gpu
echo "  Online records: $(wc -l < $ONLINE_TRACE 2>/dev/null || echo 0)"

# =============================================
# PHASE 2: Offline profiling
# =============================================
echo ""
echo "=== PHASE 2: Offline profiling ==="
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
# PHASE 3: Build profile
# =============================================
echo ""
echo "=== PHASE 3: Build profile ==="
PROFILE="${RESULT_DIR}/profiles/serving-1.5b-tp1-final.json"
python3 /workspace/vllm-emulator-v18/paper/isca-2026-mlarchsys/scripts/temp/build_serving_profile_2d.py \
    "$ONLINE_TRACE" \
    "${RESULT_DIR}/profiles/sweep-1.5b-tp1-v14.json" \
    "$PROFILE" \
    "$MODEL" "RTX-3060-12GB"

# =============================================
# PHASE 4: Real baselines (shared across all oracle modes)
# =============================================
echo ""
echo "=== PHASE 4: Real baselines (1000 prompts) ==="

for RATE in 1 2 4 8; do
    cleanup_gpu
    echo "  Real rate=$RATE..."
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
        > /workspace/final_real_${RATE}.log 2>&1 &
    wait_server
    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 200 --request-rate 4 > /dev/null 2>&1
    sleep 3
    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 1000 --request-rate $RATE --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,99 \
        --save-result --result-dir "$ONLINE_DIR" --result-filename "final_rate${RATE}_real.json" > /dev/null 2>&1
done
cleanup_gpu

# =============================================
# PHASE 5: Emu with all oracle modes
# =============================================
echo ""
echo "=== PHASE 5: Emu — all oracle modes ==="

for MODE in step_cycle hybrid 2d; do
    echo ""
    echo "--- Oracle: $MODE ---"
    for RATE in 1 2 4 8; do
        TAG="final_${MODE}_rate${RATE}"
        cleanup_gpu
        echo "  Emu rate=$RATE ($MODE)..."
        VLLM_EMULATOR_ENABLE_ORACLE=1 \
        VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
        VLLM_EMULATOR_MODE=realtime \
        VLLM_EMULATOR_EXECUTOR_HOOK=1 \
        VLLM_EMULATOR_ORACLE_MODE=$MODE \
        python3 -m vllm.entrypoints.openai.api_server \
            --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
            > /workspace/final_emu_${MODE}_${RATE}.log 2>&1 &
        wait_server
        if [ "$RATE" = "1" ]; then
            grep "ExecutorEmulatorHook" /workspace/final_emu_${MODE}_${RATE}.log | head -1
        fi
        python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
            --dataset-name random --random-input-len 256 --random-output-len 128 \
            --num-prompts 200 --request-rate 4 > /dev/null 2>&1
        sleep 3
        python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
            --dataset-name random --random-input-len 256 --random-output-len 128 \
            --num-prompts 1000 --request-rate $RATE --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,99 \
            --save-result --result-dir "$ONLINE_DIR" --result-filename "${TAG}_emu.json" > /dev/null 2>&1
    done
done
cleanup_gpu

# =============================================
# PHASE 6: Offline
# =============================================
echo ""
echo "=== PHASE 6: Offline (200 prompts) ==="

cleanup_gpu
echo "  Real warmup..."
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "$MODEL" --max-model-len 4096 --trust-remote-code \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 > /dev/null 2>&1
echo "  Real..."
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "$MODEL" --max-model-len 4096 --trust-remote-code \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 2>&1 | tee /workspace/final_offline_real.txt | grep "Throughput:"

cleanup_gpu
echo "  Emu warmup..."
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_PROFILE_USAGE=offline \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
VLLM_EMULATOR_ORACLE_MODE=step_cycle \
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "$MODEL" --max-model-len 4096 --trust-remote-code \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 > /dev/null 2>&1
echo "  Emu..."
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_PROFILE_USAGE=offline \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
VLLM_EMULATOR_ORACLE_MODE=step_cycle \
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "$MODEL" --max-model-len 4096 --trust-remote-code \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 2>&1 | tee /workspace/final_offline_emu.txt | grep "Throughput:"
cleanup_gpu

# =============================================
# FULL RESULTS
# =============================================
echo ""
echo "============================================"
echo "=== FULL RESULTS ==="
echo "============================================"

for MODE in step_cycle hybrid 2d; do
    echo ""
    echo "--- Oracle: $MODE ---"
    for RATE in 1 2 4 8; do
        TAG="final_${MODE}_rate${RATE}"
        python3 -c "
import json
try:
    e=json.load(open('$ONLINE_DIR/${TAG}_emu.json'))
    r=json.load(open('$ONLINE_DIR/final_rate${RATE}_real.json'))
    tpot=(e['mean_tpot_ms']-r['mean_tpot_ms'])/r['mean_tpot_ms']*100
    e2e=(e.get('mean_e2el_ms',0)-r.get('mean_e2el_ms',0))/r.get('mean_e2el_ms',1)*100 if r.get('mean_e2el_ms') else 0
    ttft=(e['mean_ttft_ms']-r['mean_ttft_ms'])/r['mean_ttft_ms']*100
    print(f'  R={\"$RATE\":>1}: TPOT={tpot:+6.1f}% E2E={e2e:+6.1f}% TTFT={ttft:+6.1f}%')
except Exception as ex:
    print(f'  R={\"$RATE\":>1}: ERROR {ex}')
" 2>/dev/null
    done
done

echo ""
echo "--- Offline ---"
echo "  Real: $(grep 'Throughput:' /workspace/final_offline_real.txt 2>/dev/null || echo MISSING)"
echo "  Emu:  $(grep 'Throughput:' /workspace/final_offline_emu.txt 2>/dev/null || echo MISSING)"

echo ""
echo "=== DONE $(date) ==="
