#!/bin/bash
# Diagnose TTFT: run real + emu with TTFT tracing to compare timing breakdown
set -e
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"
PROFILE="${RESULT_DIR}/profiles/serving-1.5b-tp1-calibrated.json"
PORT=8100
NUM_PROMPTS=50

pkill -9 -f EngineCore 2>/dev/null || true
pkill -9 -f api_server 2>/dev/null || true
sleep 5

warmup() {
    for i in $(seq 1 10); do
        curl -s --max-time 10 http://localhost:$PORT/v1/completions \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"$MODEL\",\"prompt\":\"warmup $i\",\"max_tokens\":5,\"temperature\":0}" > /dev/null
        sleep 0.2
    done
    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 30 --request-rate 1 > /dev/null 2>&1
    sleep 3
    echo "  Warmup done"
}

# === REAL with TTFT trace ===
echo "=== Real GPU with TTFT trace ==="
VLLM_EMULATOR_TRACE_TTFT=1 \
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > /workspace/ttft_trace_real.log 2>&1 &
for i in $(seq 1 120); do if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then break; fi; sleep 1; done
warmup

echo "  Benchmarking rate=1 ($NUM_PROMPTS prompts)..."
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts $NUM_PROMPTS --request-rate 1 > /dev/null 2>&1
sleep 5

echo ""
echo "=== Real TTFT breakdown ==="
grep "TTFT-TRACE" /workspace/ttft_trace_real.log | tail -20

pkill -f api_server 2>/dev/null || true; sleep 3
pkill -9 -f EngineCore 2>/dev/null || true; sleep 5

# === EMULATOR with TTFT trace ===
echo ""
echo "=== Emulator with TTFT trace ==="
VLLM_EMULATOR_TRACE_TTFT=1 \
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > /workspace/ttft_trace_emu.log 2>&1 &
for i in $(seq 1 120); do if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then break; fi; sleep 1; done
grep -m1 ExecutorEmulatorHook /workspace/ttft_trace_emu.log || true
warmup

echo "  Benchmarking rate=1 ($NUM_PROMPTS prompts)..."
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts $NUM_PROMPTS --request-rate 1 > /dev/null 2>&1
sleep 5

echo ""
echo "=== Emulator TTFT breakdown ==="
grep "TTFT-TRACE" /workspace/ttft_trace_emu.log | tail -20

pkill -f api_server 2>/dev/null || true; sleep 3
pkill -9 -f EngineCore 2>/dev/null || true

echo ""
echo "=== COMPARISON ==="
echo "Real GPU TTFT (EngineCore-side):"
grep "Summary" /workspace/ttft_trace_real.log | tail -3
echo ""
echo "Emulator TTFT (EngineCore-side):"
grep "Summary" /workspace/ttft_trace_emu.log | tail -3
echo "DONE"
