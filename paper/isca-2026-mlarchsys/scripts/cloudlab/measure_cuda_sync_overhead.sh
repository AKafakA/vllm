#!/bin/bash
# Measure CUDA sync overhead: GPU emu (Vast) vs GPU-vLLM-on-CPU (CloudLab)
# The delta = CUDA sync overhead (constant per GPU type)
export PATH="${HOME}/.local/bin:${PATH}"
source ~/vllm-emulator-gpu/venv/bin/activate

MODEL="Qwen/Qwen2.5-0.5B-Instruct"
PROFILE_DIR=~/vllm-emulator/profiles
STUB_DIR="${HOME}/vllm-emulator/cuda_stubs"
PORT=8100

pkill -f api_server 2>/dev/null || true
sleep 3

echo "=== Measuring CUDA Sync Overhead ==="

# Start GPU-vLLM-on-CPU emulator
LD_LIBRARY_PATH="${STUB_DIR}:${LD_LIBRARY_PATH:-}" \
VLLM_EMULATOR_MOCK_CUDA=1 \
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="${PROFILE_DIR}/serving-0.5b-tp1.json" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
VLLM_EMULATOR_MEMORY=$((12*1024*1024*1024)) \
python3 -c "
import vllm_emulator.cuda_mock
import runpy
runpy.run_module('vllm.entrypoints.openai.api_server', run_name='__main__')
" --model "${MODEL}" --max-model-len 2048 \
    --port ${PORT} --trust-remote-code \
    --load-format dummy --enforce-eager \
    > ~/vllm-setup/sync_overhead_server.log 2>&1 &

for i in $(seq 1 60); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
        echo "  Server ready after ${i}s"
        break
    fi
    sleep 1
done

if ! curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
    echo "  FAILED"
    exit 1
fi

mkdir -p ~/vllm-emulator/results

for rate in 1 2 4; do
    echo "  Rate=${rate}..."
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "${MODEL}" --base-url http://localhost:${PORT} \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 30 --request-rate ${rate} \
        --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
        --save-result --result-dir ~/vllm-emulator/results \
        --result-filename "gpu_on_cpu_rate${rate}.json" 2>&1 | grep -E "TTFT|TPOT"
done

kill $(pgrep -f api_server) 2>/dev/null

echo ""
echo "=== CUDA Sync Overhead Analysis ==="
python3 -c "
import json, os
gpu_dir = os.path.expanduser('~/vllm-emulator/gpu_baseline')
cpu_dir = os.path.expanduser('~/vllm-emulator/results')

print(f'{\"Rate\":<6} {\"GPU_real_TTFT\":>13} {\"CPU_emu_TTFT\":>13} {\"TTFT_delta\":>11} {\"GPU_real_TPOT\":>13} {\"CPU_emu_TPOT\":>13} {\"TPOT_delta\":>11}')
print('-' * 85)

for rate in [1, 2, 4]:
    gf = f'{gpu_dir}/cluster_real_rate{rate}.json'
    cf = f'{cpu_dir}/gpu_on_cpu_rate{rate}.json'
    if os.path.exists(gf) and os.path.exists(cf):
        g, c = json.load(open(gf)), json.load(open(cf))
        ttft_delta = g['mean_ttft_ms'] - c['mean_ttft_ms']
        tpot_delta = g['mean_tpot_ms'] - c['mean_tpot_ms']
        print(f'{rate:<6} {g[\"mean_ttft_ms\"]:>13.1f} {c[\"mean_ttft_ms\"]:>13.1f} {ttft_delta:>+11.1f} {g[\"mean_tpot_ms\"]:>13.1f} {c[\"mean_tpot_ms\"]:>13.1f} {tpot_delta:>+11.1f}')

print()
print('If TTFT_delta is constant across rates → CUDA sync is a fixed overhead.')
print('If it varies → overhead depends on scheduling/batch dynamics.')
"
echo "DONE"
