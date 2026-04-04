#!/bin/bash
# Test worker hook on CPU-only (no executor hook)
source ~/vllm-emulator/venv/bin/activate

MODEL="Qwen/Qwen2.5-0.5B-Instruct"
PROFILE_DIR=~/vllm-emulator/profiles
PORT=8100

pkill -f api_server 2>/dev/null || true
sleep 3

echo "=== Worker Hook on CPU (no executor hook) ==="

LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:${HOME}/vllm-emulator/venv/lib/libiomp5.so" \
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="${PROFILE_DIR}/serving-0.5b-tp1.json" \
VLLM_EMULATOR_MODE=realtime \
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 2048 \
    --port ${PORT} --trust-remote-code \
    --load-format dummy --enforce-eager \
    > ~/vllm-setup/worker_hook_cpu_server.log 2>&1 &

for i in $(seq 1 120); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
        echo "  Server ready after ${i}s"
        break
    fi
    sleep 1
done

if ! curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
    echo "  FAILED"
    tail -20 ~/vllm-setup/worker_hook_cpu_server.log
    exit 1
fi

# Check which hook is active
grep -i 'ExecutorEmulatorHook\|GpuWorkerHook\|emulat' ~/vllm-setup/worker_hook_cpu_server.log 2>/dev/null | head -3

mkdir -p ~/vllm-emulator/results

for rate in 1 2 4; do
    echo "  Rate=${rate}..."
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "${MODEL}" --base-url http://localhost:${PORT} \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 30 --request-rate ${rate} \
        --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
        --save-result --result-dir ~/vllm-emulator/results \
        --result-filename "cpu_worker_rate${rate}.json" 2>&1 | grep -E "TTFT|TPOT"
done

kill $(pgrep -f "api_server.*0.5B") 2>/dev/null

echo ""
echo "=== COMPARISON ==="
python3 -c "
import json, os
cpu_dir = os.path.expanduser('~/vllm-emulator/results')
gpu_dir = os.path.expanduser('~/vllm-emulator/gpu_baseline')

print(f'{\"Config\":<30} {\"TTFT\":>8} {\"TPOT\":>8}')
print('-' * 50)
for rate in [1, 2, 4]:
    gf = f'{gpu_dir}/cluster_real_rate{rate}.json'
    cf = f'{cpu_dir}/cpu_worker_rate{rate}.json'
    if os.path.exists(gf):
        g = json.load(open(gf))
        print(f'{\"GPU real rate=\"+str(rate):<30} {g[\"mean_ttft_ms\"]:>8.1f} {g[\"mean_tpot_ms\"]:>8.1f}')
    if os.path.exists(cf):
        c = json.load(open(cf))
        print(f'{\"CPU worker rate=\"+str(rate):<30} {c[\"mean_ttft_ms\"]:>8.1f} {c[\"mean_tpot_ms\"]:>8.1f}')
    print()

print('Error:')
for rate in [1, 2, 4]:
    gf = f'{gpu_dir}/cluster_real_rate{rate}.json'
    cf = f'{cpu_dir}/cpu_worker_rate{rate}.json'
    if os.path.exists(gf) and os.path.exists(cf):
        g, c = json.load(open(gf)), json.load(open(cf))
        te = (c['mean_ttft_ms']-g['mean_ttft_ms'])/g['mean_ttft_ms']*100
        pe = (c['mean_tpot_ms']-g['mean_tpot_ms'])/g['mean_tpot_ms']*100
        tok = '✓' if abs(te)<10 else '✗'
        pok = '✓' if abs(pe)<10 else '✗'
        print(f'  rate={rate}: TTFT {te:+.1f}% {tok}  TPOT {pe:+.1f}% {pok}')
"
echo "DONE"
