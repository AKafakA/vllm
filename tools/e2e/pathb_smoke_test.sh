#!/bin/bash
# Path B smoke test: emulator on CPU-only CloudLab host (no real GPU)
# Simulates RTX 3060 12GB using calibrated profile from Vast
#
# Path B = MOCK_CUDA mode on CPU-only host, using calibrated profile
# No real GPU baseline available here — compare results against
# Path A real results from Vast after the run.
#
# Usage: bash pathb_smoke_test.sh
# Run on: CloudLab hp140 (CPU-only)
set -e

# === Environment ===
source ~/vllm-emulator-gpu/venv/bin/activate
CODE_DIR="${HOME}/vllm-emulator/code"
cd "$CODE_DIR"

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
PROFILE="${HOME}/eval_results/pathb/profiles/serving-1.5b-tp1-calibrated.json"
PORT=8100
NUM_PROMPTS=50
RESULT_DIR="${HOME}/eval_results/pathb"
ONLINE_DIR="${RESULT_DIR}/online"
TIMESTAMP=$(date +%Y%m%d_%H%M)
LOG_DIR="${RESULT_DIR}/logs"

mkdir -p "$ONLINE_DIR" "$LOG_DIR"

# === Verify profile exists ===
if [ ! -f "$PROFILE" ]; then
    echo "ERROR: Calibrated profile not found at $PROFILE"
    echo "Copy it from Vast first."
    exit 1
fi

# === Note: model weights are NOT needed for Path B (MockModelRunner.load_model
# is a no-op). However, vLLM downloads the full HF repo for tokenizer + config.
# For 1.5B this is ~3GB — acceptable. For larger models, consider pre-caching
# only tokenizer/config files to save bandwidth. ===

echo "============================================"
echo "=== Path B Smoke Test ($TIMESTAMP) ==="
echo "=== Host: $(hostname) ==="
echo "=== Model: $MODEL ==="
echo "=== Profile: $PROFILE ==="
echo "=== Mock CUDA: yes (CPU-only host) ==="
echo "============================================"

# === Helper functions ===
wait_for_server() {
    for i in $(seq 1 180); do
        if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
            echo "  Server ready ($i s)"
            return 0
        fi
        sleep 1
    done
    echo "  ERROR: Server not ready after 180s"
    return 1
}

heavy_warmup() {
    local rate=${1:-1}
    for i in $(seq 1 5); do
        curl -s --max-time 30 http://localhost:$PORT/v1/completions \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"$MODEL\",\"prompt\":\"warmup test\",\"max_tokens\":3,\"temperature\":0}" > /dev/null
        sleep 0.2
    done
    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 30 --request-rate $rate > /dev/null 2>&1
    sleep 2
    echo "  Warmup done (rate=$rate)"
}

kill_servers() {
    pkill -f api_server 2>/dev/null || true
    sleep 2
    pkill -9 -f EngineCore 2>/dev/null || true
    sleep 3
}

run_bench() {
    local rate=$1
    local num_prompts=$2
    local result_file=$3
    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts $num_prompts --request-rate $rate \
        --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
        --save-result --result-dir "$ONLINE_DIR" --result-filename "$result_file" 2>&1 | tee -a "${LOG_DIR}/bench_rate${rate}.log"
}

# === Run emulator at rates 1, 2, 4 ===
for RATE in 1 2 4; do
    echo ""
    echo "============================================"
    echo "=== Path B: 1.5B emulator rate=$RATE ==="
    echo "============================================"

    kill_servers

    echo "  Starting emulator server (MOCK_CUDA)..."
    VLLM_EMULATOR_MOCK_CUDA=1 \
    VLLM_EMULATOR_ENABLE_ORACLE=1 \
    VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
    VLLM_EMULATOR_MODE=realtime \
    VLLM_EMULATOR_EXECUTOR_HOOK=1 \
    VLLM_EMULATOR_MEMORY=12884901888 \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
        > "${LOG_DIR}/server_emu_rate${RATE}.log" 2>&1 &
    SERVER_PID=$!

    wait_for_server

    # Verify hooks are active
    echo "  Checking hooks..."
    grep -m1 "ExecutorEmulatorHook" "${LOG_DIR}/server_emu_rate${RATE}.log" && echo "  [OK] ExecutorEmulatorHook active" || echo "  [WARN] ExecutorEmulatorHook not found in log"
    grep -m1 "MOCK_CUDA" "${LOG_DIR}/server_emu_rate${RATE}.log" && echo "  [OK] MOCK_CUDA active" || echo "  [WARN] MOCK_CUDA not found in log"
    grep -m1 "GpuCostOracle" "${LOG_DIR}/server_emu_rate${RATE}.log" && echo "  [OK] Oracle active" || echo "  [WARN] Oracle not found in log"

    heavy_warmup $RATE
    echo "  Benchmarking emulator rate=$RATE..."
    run_bench $RATE $NUM_PROMPTS "pathb_emu_1.5b_rate${RATE}.json"

    echo "  Results saved to: ${ONLINE_DIR}/pathb_emu_1.5b_rate${RATE}.json"
done

kill_servers

# === Summary ===
echo ""
echo "============================================"
echo "=== Path B SUMMARY ($TIMESTAMP) ==="
echo "============================================"
echo "Results saved in: $ONLINE_DIR"
echo ""
for RATE in 1 2 4; do
    RESULT_FILE="${ONLINE_DIR}/pathb_emu_1.5b_rate${RATE}.json"
    if [ -f "$RESULT_FILE" ]; then
        echo "--- Rate=$RATE ---"
        python3 -c "
import json
with open('$RESULT_FILE') as f:
    d = json.load(f)
print(f'  TTFT mean: {d.get(\"mean_ttft_ms\", \"N/A\"):.2f} ms')
print(f'  TPOT mean: {d.get(\"mean_tpot_ms\", \"N/A\"):.2f} ms')
print(f'  Throughput: {d.get(\"output_throughput\", \"N/A\"):.2f} tok/s')
" 2>/dev/null || echo "  (could not parse results)"
    else
        echo "--- Rate=$RATE: MISSING ---"
    fi
done

echo ""
echo "To compare with Path A real results from Vast, copy Vast results here and run:"
echo "  python3 ${CODE_DIR}/tools/e2e/compare_results.py <vast_real.json> <pathb_emu.json>"
echo ""
echo "Path B smoke test DONE ($TIMESTAMP)"
