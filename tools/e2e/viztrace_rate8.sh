#!/bin/bash
# Viztracer profiling of rate=8 emulation
# Captures call tree with timestamps to identify where time is spent
set -e
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=1

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
PROFILE="/workspace/eval_results/RTX-3060-12GB/profiles/serving-1.5b-tp1-fresh.json"
PORT=8100

echo "=== Viztracer Rate=8 Debug ==="
echo "=== $(date) ==="

pkill -9 -f "python3.*api_server" 2>/dev/null || true
fuser /dev/nvidia* 2>/dev/null | tr " " "\n" | sort -u | xargs kill -9 2>/dev/null || true
sleep 5

# Start server with viztracer (short run: 100 prompts to keep trace manageable)
echo "Starting server with viztracer..."
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
VLLM_EMULATOR_ORACLE_MODE=hybrid \
python3 -m viztracer --tracer_entries 5000000 \
    --ignore_c_function --ignore_frozen \
    --min_duration 100us \
    --output_file /workspace/viztrace_r8_emu.json \
    -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > /workspace/viztrace_server.log 2>&1 &
SERVER_PID=$!

for i in $(seq 1 180); do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "  Server UP after ${i}s"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "  DIED after ${i}s"
        tail -20 /workspace/viztrace_server.log
        exit 1
    fi
    sleep 1
done

# Short warmup
echo "  Warmup..."
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 50 --request-rate 4 > /dev/null 2>&1
sleep 2

# Short rate=8 benchmark (100 prompts to keep trace manageable)
echo "  Bench rate=8 (100 prompts)..."
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 100 --request-rate 8 2>&1 | tail -5

# Give viztracer time to flush
echo "  Stopping server (viztracer flushing)..."
kill $SERVER_PID 2>/dev/null || true
sleep 10

echo ""
echo "=== Viztracer output ==="
ls -la /workspace/viztrace_r8_emu.json 2>/dev/null || echo "No trace file"

# Also do a real GPU trace for comparison
echo ""
echo "=== Real GPU rate=8 (100 prompts, no viztracer) ==="
pkill -9 -f python3 2>/dev/null || true
fuser /dev/nvidia* 2>/dev/null | tr " " "\n" | sort -u | xargs kill -9 2>/dev/null || true
sleep 5

python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > /workspace/viztrace_real_server.log 2>&1 &
for i in $(seq 1 120); do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then break; fi; sleep 1
done
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 50 --request-rate 4 > /dev/null 2>&1
sleep 2
echo "  Bench real..."
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 100 --request-rate 8 2>&1 | tail -5

pkill -9 -f python3 2>/dev/null || true
fuser /dev/nvidia* 2>/dev/null | tr " " "\n" | sort -u | xargs kill -9 2>/dev/null || true

echo ""
echo "=== DONE $(date) ==="
