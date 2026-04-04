#!/bin/bash
# Centralized emulator test: run emulator on CPU-only host with serving profile
# Compare TTFT/TPOT against real GPU baseline (from Vast)
source ~/vllm-emulator/venv/bin/activate

MODEL="Qwen/Qwen2.5-0.5B-Instruct"
PROFILE_DIR=~/vllm-emulator/profiles
PORT=8100

echo "============================================================"
echo "Centralized Emulator Test (CPU-only, single instance)"
echo "  Host: $(hostname)"
echo "  Model: ${MODEL}"
echo "  Profile: ${PROFILE_DIR}/serving-0.5b-tp1.json"
echo "============================================================"

# Kill any existing
pkill -f api_server 2>/dev/null
sleep 3

# Start emulator server
echo ""
echo "Starting emulator server..."
LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:${HOME}/vllm-emulator/venv/lib/libiomp5.so" \
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="${PROFILE_DIR}/serving-0.5b-tp1.json" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 2048 \
    --port ${PORT} --trust-remote-code \
    --load-format dummy --enforce-eager \
    > ~/vllm-setup/centralized_emu_server.log 2>&1 &
SERVER_PID=$!

for i in $(seq 1 120); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
        echo "  Server ready after ${i}s"
        break
    fi
    if ! kill -0 ${SERVER_PID} 2>/dev/null; then
        echo "  Server died!"
        tail -20 ~/vllm-setup/centralized_emu_server.log
        exit 1
    fi
    sleep 1
done

if ! curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
    echo "  TIMEOUT"
    kill ${SERVER_PID} 2>/dev/null
    exit 1
fi

# Check if executor hook is active
grep ExecutorEmulatorHook ~/vllm-setup/centralized_emu_server.log 2>/dev/null | head -1

# Benchmark at multiple rates
echo ""
echo "Running benchmarks..."
mkdir -p ~/vllm-emulator/results

for rate in 1 2 4; do
    echo "  Rate=${rate}..."
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "${MODEL}" --base-url http://localhost:${PORT} \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 30 --request-rate ${rate} \
        --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
        --save-result --result-dir ~/vllm-emulator/results \
        --result-filename "cpu_emu_rate${rate}.json" 2>&1 | grep -E "TTFT|TPOT"
done

kill ${SERVER_PID} 2>/dev/null
wait ${SERVER_PID} 2>/dev/null

echo ""
echo "============================================================"
echo "RESULTS (CPU Emulator)"
echo "============================================================"
python3 -c "
import json, os
rd = os.path.expanduser('~/vllm-emulator/results')
for rate in [1, 2, 4]:
    f = f'{rd}/cpu_emu_rate{rate}.json'
    if os.path.exists(f):
        d = json.load(open(f))
        print(f'CPU Emu rate={rate}: TTFT={d[\"mean_ttft_ms\"]:.1f}ms  TPOT={d[\"mean_tpot_ms\"]:.1f}ms')
"

echo ""
echo "Compare these against the Vast real baseline results."
echo "DONE"
