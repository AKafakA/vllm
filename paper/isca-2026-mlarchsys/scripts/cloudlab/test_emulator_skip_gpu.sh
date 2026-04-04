#!/bin/bash
# Test emulator with GPU init skipped on CPU host
source ~/vllm-emulator/venv/bin/activate

MODEL="Qwen/Qwen2.5-0.5B-Instruct"
PROFILE_DIR=~/vllm-emulator/profiles
PORT=8100

pkill -f api_server 2>/dev/null || true
sleep 3

echo "=== Emulator (skip GPU init, CPU-vLLM) ==="

LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:${HOME}/vllm-emulator/venv/lib/libiomp5.so" \
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="${PROFILE_DIR}/serving-0.5b-tp1.json" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_MEMORY=$((12*1024*1024*1024)) \
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 2048 \
    --port ${PORT} --trust-remote-code \
    --load-format dummy --enforce-eager \
    > ~/vllm-setup/skip_gpu_server.log 2>&1 &
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
    # Check which hook is active
    grep -i 'emulator\|hook\|skip' ~/vllm-setup/skip_gpu_server.log 2>/dev/null | head -5

    echo ""
    echo "  Running benchmark..."
    for rate in 1 2; do
        echo "  Rate=${rate}:"
        python3 -m vllm.entrypoints.cli.main bench serve \
            --model "${MODEL}" --base-url http://localhost:${PORT} \
            --dataset-name random --random-input-len 256 --random-output-len 128 \
            --num-prompts 20 --request-rate ${rate} \
            --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
            --save-result --result-dir ~/vllm-emulator/results \
            --result-filename "skip_gpu_rate${rate}.json" 2>&1 | grep -E "TTFT|TPOT"
    done

    echo ""
    echo "  Comparison with GPU real (from Vast):"
    python3 -c "
import json, os
gpu_dir = os.path.expanduser('~/vllm-emulator/gpu_baseline')
cpu_dir = os.path.expanduser('~/vllm-emulator/results')
for rate in [1, 2]:
    gf = f'{gpu_dir}/cluster_real_rate{rate}.json'
    cf = f'{cpu_dir}/skip_gpu_rate{rate}.json'
    if os.path.exists(gf) and os.path.exists(cf):
        g, c = json.load(open(gf)), json.load(open(cf))
        te = (c['mean_ttft_ms']-g['mean_ttft_ms'])/g['mean_ttft_ms']*100
        pe = (c['mean_tpot_ms']-g['mean_tpot_ms'])/g['mean_tpot_ms']*100
        tok = '✓' if abs(te)<5 else '✗'
        pok = '✓' if abs(pe)<5 else '✗'
        print(f'rate={rate}: GPU_real TTFT={g[\"mean_ttft_ms\"]:.1f} TPOT={g[\"mean_tpot_ms\"]:.1f}')
        print(f'         CPU_emu  TTFT={c[\"mean_ttft_ms\"]:.1f} TPOT={c[\"mean_tpot_ms\"]:.1f}')
        print(f'         Error:   TTFT {te:+.1f}% {tok}  TPOT {pe:+.1f}% {pok}')
"
else
    echo "  FAILED"
    tail -20 ~/vllm-setup/skip_gpu_server.log
fi

kill ${SERVER_PID} 2>/dev/null
wait ${SERVER_PID} 2>/dev/null
echo ""
echo "DONE"
