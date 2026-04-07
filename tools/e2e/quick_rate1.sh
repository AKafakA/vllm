#!/bin/bash
# Quick rate=1 A2A test with verbose error checking
set -e
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"
PROFILE="${RESULT_DIR}/profiles/serving-1.5b-tp1-calibrated.json"
PORT=8100

pkill -9 -f EngineCore 2>/dev/null || true
pkill -9 -f api_server 2>/dev/null || true
sleep 5

# === REAL ===
echo "=== Real rate=1 ==="
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > /workspace/q1_real.log 2>&1 &
for i in $(seq 1 120); do if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then echo "  Ready ($i s)"; break; fi; sleep 1; done

# Light warmup only (5 curl requests + 10 bench prompts at rate=1)
echo "  Light warmup..."
for i in $(seq 1 5); do
    curl -s --max-time 10 http://localhost:$PORT/v1/completions \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$MODEL\",\"prompt\":\"warmup test\",\"max_tokens\":3,\"temperature\":0}" > /dev/null
    sleep 0.2
done
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 10 --request-rate 1 > /dev/null 2>&1
echo "  Warmup done, sleeping 5s..."
sleep 5

echo "  Benchmarking..."
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 50 --request-rate 1 --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
    --save-result --result-dir "${RESULT_DIR}/online" --result-filename "q1r_real.json" 2>&1 | tail -20

# Check for failures
echo "  Result check:"
python3 -c "
import json
d = json.load(open('${RESULT_DIR}/online/q1r_real.json'))
print(f'  completed={d[\"completed\"]}, failed={d[\"failed\"]}, duration={d[\"duration\"]:.1f}s')
print(f'  mean_ttft={d[\"mean_ttft_ms\"]:.1f}, mean_tpot={d[\"mean_tpot_ms\"]:.1f}')
"

pkill -f api_server 2>/dev/null || true; sleep 2; pkill -9 -f EngineCore 2>/dev/null || true; sleep 5

# === EMU ===
echo ""
echo "=== Emu rate=1 ==="
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > /workspace/q1_emu.log 2>&1 &
for i in $(seq 1 120); do if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then echo "  Ready ($i s)"; break; fi; sleep 1; done
grep -m1 ExecutorEmulatorHook /workspace/q1_emu.log

# Same light warmup
echo "  Light warmup..."
for i in $(seq 1 5); do
    curl -s --max-time 10 http://localhost:$PORT/v1/completions \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$MODEL\",\"prompt\":\"warmup test\",\"max_tokens\":3,\"temperature\":0}" > /dev/null
    sleep 0.2
done
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 10 --request-rate 1 > /dev/null 2>&1
echo "  Warmup done, sleeping 5s..."
sleep 5

echo "  Benchmarking..."
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 50 --request-rate 1 --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
    --save-result --result-dir "${RESULT_DIR}/online" --result-filename "q1r_emu.json" 2>&1 | tail -20

echo "  Result check:"
python3 -c "
import json
d = json.load(open('${RESULT_DIR}/online/q1r_emu.json'))
print(f'  completed={d[\"completed\"]}, failed={d[\"failed\"]}, duration={d[\"duration\"]:.1f}s')
print(f'  mean_ttft={d[\"mean_ttft_ms\"]:.1f}, mean_tpot={d[\"mean_tpot_ms\"]:.1f}')
"

pkill -f api_server 2>/dev/null || true; sleep 2; pkill -9 -f EngineCore 2>/dev/null || true

echo ""
echo "=== COMPARE ==="
python3 /workspace/vllm-emulator-v18/tools/e2e/compare_results.py \
    "${RESULT_DIR}/online/q1r_real.json" "${RESULT_DIR}/online/q1r_emu.json"
echo "DONE"
