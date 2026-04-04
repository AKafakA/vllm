#!/bin/bash
# Test GPU vLLM + CUDA mock on CPU-only host
export PATH="${HOME}/.local/bin:${PATH}"
source ~/vllm-emulator-gpu/venv/bin/activate

MODEL="Qwen/Qwen2.5-0.5B-Instruct"
PROFILE_DIR=~/vllm-emulator/profiles
PORT=8100

pkill -f api_server 2>/dev/null || true
sleep 3

echo "=== GPU vLLM + CUDA Mock on CPU Host ==="

# Key: VLLM_EMULATOR_MOCK_CUDA must be imported BEFORE vllm
# We use a wrapper that imports the mock first
# Use CUDA stubs + mock for GPU vLLM on CPU
STUB_DIR="${HOME}/vllm-emulator/cuda_stubs"
LD_LIBRARY_PATH="${STUB_DIR}:${LD_LIBRARY_PATH:-}" \
VLLM_EMULATOR_MOCK_CUDA=1 \
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="${PROFILE_DIR}/serving-0.5b-tp1.json" \
VLLM_EMULATOR_MODE=realtime \
python3 -c "
import vllm_emulator.cuda_mock  # Must be first!
import runpy
runpy.run_module('vllm.entrypoints.openai.api_server', run_name='__main__')
" --model "${MODEL}" --max-model-len 2048 \
    --port ${PORT} --trust-remote-code \
    --load-format dummy --enforce-eager \
    > ~/vllm-setup/cuda_mock_server.log 2>&1 &
SERVER_PID=$!

echo "  Server PID: ${SERVER_PID}"
echo "  Waiting..."

for i in $(seq 1 120); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
        echo "  Server ready after ${i}s!"
        break
    fi
    if ! kill -0 ${SERVER_PID} 2>/dev/null; then
        echo "  Server died after ${i}s"
        break
    fi
    sleep 1
done

if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
    echo "  Testing inference..."
    curl -s -X POST http://localhost:${PORT}/v1/completions \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"${MODEL}\", \"prompt\": \"Hello\", \"max_tokens\": 5}" \
        | python3 -m json.tool 2>/dev/null | head -10

    echo ""
    echo "  Running benchmark at rate=1..."
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "${MODEL}" --base-url http://localhost:${PORT} \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 20 --request-rate 1 \
        --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
        --save-result --result-dir ~/vllm-emulator/results \
        --result-filename "cuda_mock_rate1.json" 2>&1 | grep -E "TTFT|TPOT"

    echo ""
    echo "  GPU real baseline (from Vast):"
    python3 -c "
import json, os
f = os.path.expanduser('~/vllm-emulator/gpu_baseline/cluster_real_rate1.json')
if os.path.exists(f):
    d = json.load(open(f))
    print(f'  TTFT={d[\"mean_ttft_ms\"]:.1f}ms  TPOT={d[\"mean_tpot_ms\"]:.1f}ms')
"
else
    echo "  Server failed to start."
    echo "  Last 20 lines of log:"
    tail -20 ~/vllm-setup/cuda_mock_server.log
fi

kill ${SERVER_PID} 2>/dev/null
wait ${SERVER_PID} 2>/dev/null
echo "DONE"
