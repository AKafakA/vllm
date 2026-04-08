#!/bin/bash
# Test v8: warm GPU equally for profiling and benchmarking
# Warmup: 200 prompts via bench throughput (throwaway)
# Then: actual benchmark on warm GPU
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
TRACE_FILE="${RESULT_DIR}/offline_step_cycle_v2.jsonl"

echo "============================================"
echo "=== PHASE 1: Re-profile offline (warm GPU) ==="
echo "============================================"

pkill -9 -f python3 2>/dev/null || true
fuser /dev/nvidia* 2>/dev/null | tr " " "\n" | sort -u | xargs kill -9 2>/dev/null || true
sleep 5
rm -f "$TRACE_FILE"

# Warmup: 200 prompts (throwaway, same as benchmark)
echo "  Warmup (200 prompts, throwaway)..."
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "$MODEL" --max-model-len 4096 --trust-remote-code \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 2>&1 | grep "Throughput:"

# Profile: 200 prompts with tracing (GPU is now warm)
echo "  Profiling (200 prompts with tracing)..."
VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
VLLM_EMULATOR_STEP_TRACE_OUTPUT="$TRACE_FILE" \
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "$MODEL" --max-model-len 4096 --trust-remote-code \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 2>&1 | grep "Throughput:"

# Also profile pure decode (short input)
echo "  Profiling pure decode (200 prompts, input=1)..."
VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
VLLM_EMULATOR_STEP_TRACE_OUTPUT="$TRACE_FILE" \
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "$MODEL" --max-model-len 4096 --trust-remote-code \
    --dataset-name random --random-input-len 1 --random-output-len 256 \
    --num-prompts 200 2>&1 | grep "Throughput:"

# Profile at different batch sizes for coverage
for NP in 50 100 300; do
    echo "  Profiling $NP prompts..."
    VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
    VLLM_EMULATOR_STEP_TRACE_OUTPUT="$TRACE_FILE" \
    python3 -m vllm.entrypoints.cli.main bench throughput \
        --model "$MODEL" --max-model-len 4096 --trust-remote-code \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts $NP 2>&1 | grep "Throughput:"
done

RECORDS=$(wc -l < "$TRACE_FILE" 2>/dev/null || echo 0)
echo "  Total offline trace records: $RECORDS"

# Build profile
echo ""
echo "=== Build profile v8 ==="
# Copy v2 trace to the expected location for profile builder
cp "$TRACE_FILE" "${RESULT_DIR}/offline_step_cycle.jsonl"

python3 /workspace/vllm-emulator-v18/vllm_emulator/profile/build_serving_profile.py \
    "${RESULT_DIR}/step_cycle_full.jsonl" \
    "${RESULT_DIR}/profiles/sweep-1.5b-tp1-v14.json" \
    "${RESULT_DIR}/profiles/serving-1.5b-tp1-v8.json" \
    "$MODEL" "RTX-3060-12GB"

PROFILE="${RESULT_DIR}/profiles/serving-1.5b-tp1-v8.json"

echo ""
echo "============================================"
echo "=== PHASE 2: Benchmark offline (warm GPU) ==="
echo "============================================"

pkill -9 -f python3 2>/dev/null || true
fuser /dev/nvidia* 2>/dev/null | tr " " "\n" | sort -u | xargs kill -9 2>/dev/null || true
sleep 5

# Real: warmup then benchmark
echo "  Real warmup..."
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "$MODEL" --max-model-len 4096 --trust-remote-code \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 > /dev/null 2>&1

echo "  Real benchmark (200 prompts)..."
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "$MODEL" --max-model-len 4096 --trust-remote-code \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 2>&1 | tee /workspace/v8_offline_real.txt | grep "Throughput:"

pkill -9 -f python3 2>/dev/null || true
fuser /dev/nvidia* 2>/dev/null | tr " " "\n" | sort -u | xargs kill -9 2>/dev/null || true
sleep 5

# Emu: warmup then benchmark
echo "  Emu warmup..."
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
VLLM_EMULATOR_MODE=offline \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "$MODEL" --max-model-len 4096 --trust-remote-code \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 > /dev/null 2>&1

echo "  Emu benchmark (200 prompts)..."
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
VLLM_EMULATOR_MODE=offline \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "$MODEL" --max-model-len 4096 --trust-remote-code \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 2>&1 | tee /workspace/v8_offline_emu.txt | grep "Throughput:"

pkill -9 -f python3 2>/dev/null || true
fuser /dev/nvidia* 2>/dev/null | tr " " "\n" | sort -u | xargs kill -9 2>/dev/null || true
sleep 5

echo ""
echo "============================================"
echo "=== PHASE 3: Online rate=1 sanity ==="
echo "============================================"
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > /workspace/v8_online.log 2>&1 &
for i in $(seq 1 120); do if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then break; fi; sleep 1; done

python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 --request-rate 4 > /dev/null 2>&1
sleep 3

python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 --request-rate 1 --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
    --save-result --result-dir "$ONLINE_DIR" --result-filename "v8_rate1_emu.json" > /dev/null 2>&1

pkill -9 -f python3 2>/dev/null || true
fuser /dev/nvidia* 2>/dev/null | tr " " "\n" | sort -u | xargs kill -9 2>/dev/null || true
sleep 3

echo "--- Online rate=1 ---"
python3 "$SCRIPT_DIR/compare_results.py" "$ONLINE_DIR/fullv2_rate1_real.json" "$ONLINE_DIR/v8_rate1_emu.json"

echo ""
echo "============================================"
echo "=== SUMMARY ==="
echo "============================================"
echo "Offline 200 (warm GPU):"
echo "  Real: $(grep 'Throughput:' /workspace/v8_offline_real.txt)"
echo "  Emu:  $(grep 'Throughput:' /workspace/v8_offline_emu.txt)"
echo ""
echo "Online rate=1:"
python3 "$SCRIPT_DIR/compare_results.py" "$ONLINE_DIR/fullv2_rate1_real.json" "$ONLINE_DIR/v8_rate1_emu.json" 2>/dev/null || echo "  MISSING"

echo ""
echo "DONE $(date)"
