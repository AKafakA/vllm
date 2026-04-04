#!/bin/bash
# Test GPU vLLM build with emulator skipping GPU init
# Uses the gpu-vllm venv (precompiled) + CUDA mock + stub libraries
export PATH="${HOME}/.local/bin:${PATH}"
source ~/vllm-emulator-gpu/venv/bin/activate

MODEL="Qwen/Qwen2.5-0.5B-Instruct"
PROFILE_DIR=~/vllm-emulator/profiles
STUB_DIR="${HOME}/vllm-emulator/cuda_stubs"
PORT=8100

pkill -f api_server 2>/dev/null || true
sleep 3

echo "=== GPU vLLM + CUDA Mock + Skip GPU Init ==="

# CUDA stubs + mock + emulator with GPU init skip
LD_LIBRARY_PATH="${STUB_DIR}:${LD_LIBRARY_PATH:-}" \
VLLM_EMULATOR_MOCK_CUDA=1 \
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="${PROFILE_DIR}/serving-0.5b-tp1.json" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
VLLM_EMULATOR_MEMORY=$((12*1024*1024*1024)) \
python3 -c "
import vllm_emulator.cuda_mock  # Must be first!
import runpy
runpy.run_module('vllm.entrypoints.openai.api_server', run_name='__main__')
" --model "${MODEL}" --max-model-len 2048 \
    --port ${PORT} --trust-remote-code \
    --load-format dummy --enforce-eager \
    > ~/vllm-setup/gpu_vllm_skip_server.log 2>&1 &
SERVER_PID=$!

echo "  PID: ${SERVER_PID}"
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
    grep -i 'emulator\|mock\|skip\|hook' ~/vllm-setup/gpu_vllm_skip_server.log 2>/dev/null | head -5

    echo ""
    echo "  Benchmark rate=1..."
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "${MODEL}" --base-url http://localhost:${PORT} \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 20 --request-rate 1 \
        --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
        --save-result --result-dir ~/vllm-emulator/results \
        --result-filename "gpu_vllm_skip_rate1.json" 2>&1 | grep -E "TTFT|TPOT"

    echo ""
    echo "  Comparison:"
    python3 -c "
import json, os
gpu_dir = os.path.expanduser('~/vllm-emulator/gpu_baseline')
cpu_dir = os.path.expanduser('~/vllm-emulator/results')
gf = f'{gpu_dir}/cluster_real_rate1.json'
cf = f'{cpu_dir}/gpu_vllm_skip_rate1.json'
if os.path.exists(gf) and os.path.exists(cf):
    g, c = json.load(open(gf)), json.load(open(cf))
    te = (c['mean_ttft_ms']-g['mean_ttft_ms'])/g['mean_ttft_ms']*100
    pe = (c['mean_tpot_ms']-g['mean_tpot_ms'])/g['mean_tpot_ms']*100
    print(f'GPU real: TTFT={g[\"mean_ttft_ms\"]:.1f}ms TPOT={g[\"mean_tpot_ms\"]:.1f}ms')
    print(f'GPU-vLLM-on-CPU: TTFT={c[\"mean_ttft_ms\"]:.1f}ms TPOT={c[\"mean_tpot_ms\"]:.1f}ms')
    print(f'Error: TTFT {te:+.1f}%  TPOT {pe:+.1f}%')
"
else
    echo "  FAILED"
    tail -20 ~/vllm-setup/gpu_vllm_skip_server.log
fi

kill ${SERVER_PID} 2>/dev/null
echo "DONE"
