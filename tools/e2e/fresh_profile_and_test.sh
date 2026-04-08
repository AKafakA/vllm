#!/bin/bash
# Fresh profiling (online + offline) + focused test (rate 1, 4, offline)
# 200-prompt warmup before each phase, 1000 prompts for benchmark
set -e
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub
export CUDA_VISIBLE_DEVICES=1

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"
PORT=8100
SCRIPT_DIR="/workspace/vllm-emulator-v18/tools/e2e"
ONLINE_DIR="${RESULT_DIR}/online"
ONLINE_TRACE="${RESULT_DIR}/step_cycle_fresh.jsonl"
OFFLINE_TRACE="${RESULT_DIR}/offline_step_cycle.jsonl"
PROFILE="${RESULT_DIR}/profiles/serving-1.5b-tp1-fresh.json"

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
echo "=== Fresh Profile + Focused Test ==="
echo "=== $(date) ==="
echo "============================================"

cleanup_gpu
rm -f "$ONLINE_TRACE" "$OFFLINE_TRACE"
rm -f "$ONLINE_DIR"/fresh_*.json

# =============================================
# PHASE 1: Online profiling (step-cycle trace from API server)
# =============================================
echo ""
echo "=== PHASE 1: Online profiling ==="

VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
VLLM_EMULATOR_STEP_TRACE_OUTPUT="$ONLINE_TRACE" \
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > /workspace/fresh_profile_server.log 2>&1 &
wait_server

# Warmup (thermal equilibrium)
echo "  Warmup (200 prompts, rate=4)..."
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 --request-rate 4 > /dev/null 2>&1

# Profile at multiple rates
for rate in 0.5 1 2 4 8 12; do
    NP=200
    if [ "$rate" = "0.5" ]; then NP=50; fi
    echo "  Profile rate=$rate ($NP prompts)..."
    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts $NP --request-rate $rate > /dev/null 2>&1
done
# Offline (high concurrency coverage)
echo "  Profile offline (200 prompts, rate=inf)..."
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 --request-rate inf > /dev/null 2>&1

cleanup_gpu
ONLINE_RECORDS=$(wc -l < "$ONLINE_TRACE" 2>/dev/null || echo 0)
echo "  Online trace records: $ONLINE_RECORDS"

# =============================================
# PHASE 2: Offline profiling (step-cycle trace from bench throughput)
# =============================================
echo ""
echo "=== PHASE 2: Offline profiling ==="

# Warmup
echo "  Warmup (200 prompts)..."
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "$MODEL" --max-model-len 4096 --trust-remote-code \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 > /dev/null 2>&1

for NP in 50 100 200 300; do
    echo "  Profile $NP prompts (256+128)..."
    VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
    VLLM_EMULATOR_STEP_TRACE_OUTPUT="$OFFLINE_TRACE" \
    python3 -m vllm.entrypoints.cli.main bench throughput \
        --model "$MODEL" --max-model-len 4096 --trust-remote-code \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts $NP 2>&1 | grep "Throughput:"
done

for NP in 50 100 200; do
    echo "  Profile pure decode $NP prompts (1+256)..."
    VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
    VLLM_EMULATOR_STEP_TRACE_OUTPUT="$OFFLINE_TRACE" \
    python3 -m vllm.entrypoints.cli.main bench throughput \
        --model "$MODEL" --max-model-len 4096 --trust-remote-code \
        --dataset-name random --random-input-len 1 --random-output-len 256 \
        --num-prompts $NP 2>&1 | grep "Throughput:"
done

OFFLINE_RECORDS=$(wc -l < "$OFFLINE_TRACE" 2>/dev/null || echo 0)
echo "  Offline trace records: $OFFLINE_RECORDS"

# =============================================
# PHASE 3: Build profile
# =============================================
echo ""
echo "=== PHASE 3: Build profile ==="
python3 /workspace/vllm-emulator-v18/vllm_emulator/profile/build_serving_profile.py \
    "$ONLINE_TRACE" \
    "${RESULT_DIR}/profiles/sweep-1.5b-tp1-v14.json" \
    "$PROFILE" \
    "$MODEL" "RTX-3060-12GB"

# =============================================
# PHASE 4: Rate=1 (1000 prompts)
# =============================================
echo ""
echo "=== PHASE 4: Rate=1 (1000 prompts) ==="

cleanup_gpu
echo "  Real server..."
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > /workspace/fresh_real_1.log 2>&1 &
wait_server
echo "  Warmup..."
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 --request-rate 4 > /dev/null 2>&1
sleep 3
echo "  Bench real..."
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 1000 --request-rate 1 --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,99 \
    --save-result --result-dir "$ONLINE_DIR" --result-filename "fresh_rate1_real.json" > /dev/null 2>&1

cleanup_gpu
echo "  Emu server..."
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > /workspace/fresh_emu_1.log 2>&1 &
wait_server
echo "  Warmup..."
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 --request-rate 4 > /dev/null 2>&1
sleep 3
echo "  Bench emu..."
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 1000 --request-rate 1 --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,99 \
    --save-result --result-dir "$ONLINE_DIR" --result-filename "fresh_rate1_emu.json" > /dev/null 2>&1

cleanup_gpu
echo "--- Rate=1 ---"
python3 "$SCRIPT_DIR/compare_results.py" "$ONLINE_DIR/fresh_rate1_real.json" "$ONLINE_DIR/fresh_rate1_emu.json"

# =============================================
# PHASE 5: Rate=4 (1000 prompts)
# =============================================
echo ""
echo "=== PHASE 5: Rate=4 (1000 prompts) ==="

cleanup_gpu
echo "  Real server..."
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > /workspace/fresh_real_4.log 2>&1 &
wait_server
echo "  Warmup..."
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 --request-rate 4 > /dev/null 2>&1
sleep 3
echo "  Bench real..."
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 1000 --request-rate 4 --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,99 \
    --save-result --result-dir "$ONLINE_DIR" --result-filename "fresh_rate4_real.json" > /dev/null 2>&1

cleanup_gpu
echo "  Emu server..."
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > /workspace/fresh_emu_4.log 2>&1 &
wait_server
echo "  Warmup..."
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 --request-rate 4 > /dev/null 2>&1
sleep 3
echo "  Bench emu..."
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 1000 --request-rate 4 --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,99 \
    --save-result --result-dir "$ONLINE_DIR" --result-filename "fresh_rate4_emu.json" > /dev/null 2>&1

cleanup_gpu
echo "--- Rate=4 ---"
python3 "$SCRIPT_DIR/compare_results.py" "$ONLINE_DIR/fresh_rate4_real.json" "$ONLINE_DIR/fresh_rate4_emu.json"

# =============================================
# PHASE 6: Offline throughput (200 prompts)
# =============================================
echo ""
echo "=== PHASE 6: Offline throughput ==="

cleanup_gpu
echo "  Real warmup..."
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "$MODEL" --max-model-len 4096 --trust-remote-code \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 > /dev/null 2>&1
echo "  Real benchmark..."
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "$MODEL" --max-model-len 4096 --trust-remote-code \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 2>&1 | tee /workspace/fresh_offline_real.txt | grep "Throughput:"

cleanup_gpu
echo "  Emu warmup..."
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_PROFILE_USAGE=offline \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "$MODEL" --max-model-len 4096 --trust-remote-code \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 > /dev/null 2>&1
echo "  Emu benchmark..."
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_PROFILE_USAGE=offline \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "$MODEL" --max-model-len 4096 --trust-remote-code \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 2>&1 | tee /workspace/fresh_offline_emu.txt | grep "Throughput:"

cleanup_gpu

# =============================================
# SUMMARY
# =============================================
echo ""
echo "============================================"
echo "=== SUMMARY ==="
echo "============================================"
echo ""
echo "--- Rate=1 (1000 prompts) ---"
python3 "$SCRIPT_DIR/compare_results.py" "$ONLINE_DIR/fresh_rate1_real.json" "$ONLINE_DIR/fresh_rate1_emu.json" 2>/dev/null || echo "  MISSING"
echo ""
echo "--- Rate=4 (1000 prompts) ---"
python3 "$SCRIPT_DIR/compare_results.py" "$ONLINE_DIR/fresh_rate4_real.json" "$ONLINE_DIR/fresh_rate4_emu.json" 2>/dev/null || echo "  MISSING"
echo ""
echo "--- Offline (200 prompts) ---"
echo "  Real: $(grep 'Throughput:' /workspace/fresh_offline_real.txt 2>/dev/null || echo MISSING)"
echo "  Emu:  $(grep 'Throughput:' /workspace/fresh_offline_emu.txt 2>/dev/null || echo MISSING)"
echo ""
echo "--- Sched compensation ---"
grep "sched_comp" /workspace/fresh_emu_1.log /workspace/fresh_emu_4.log 2>/dev/null | head -2

echo ""
echo "=== DONE $(date) ==="
