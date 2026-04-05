#!/bin/bash
# Rebuild profile with emulator calibration params, then test rate=1
set -e
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"
PORT=8100
ONLINE_DIR="${RESULT_DIR}/online"

echo "=== Rebuild profile with calibration params ==="
python3 /workspace/vllm-emulator-v18/paper/isca-2026-mlarchsys/scripts/temp/build_serving_profile_2d.py \
    "${RESULT_DIR}/step_cycle_1.5b_full.jsonl" \
    "${RESULT_DIR}/profiles/sweep-1.5b-tp1-v14.json" \
    "${RESULT_DIR}/profiles/serving-1.5b-tp1-calibrated.json" \
    "${MODEL}" "RTX-3060-12GB"

PROFILE="${RESULT_DIR}/profiles/serving-1.5b-tp1-calibrated.json"

# Show calibration params
python3 -c "
import json
p = json.load(open('${PROFILE}'))
print('Calibration params:')
for k in ['cuda_graph_warmup_us', 'sched_factor', 'decode_sched_ratio']:
    print(f'  {k}: {p.get(k, \"N/A\")}')
"

heavy_warmup() {
    for i in $(seq 1 5); do
        curl -s --max-time 10 http://localhost:$PORT/v1/completions \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"$MODEL\",\"prompt\":\"warmup test\",\"max_tokens\":3,\"temperature\":0}" > /dev/null
        sleep 0.2
    done
    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 30 --request-rate 1 > /dev/null 2>&1
    sleep 2
    echo "  Warmup done"
}

pkill -9 -f EngineCore 2>/dev/null || true; pkill -9 -f api_server 2>/dev/null || true; sleep 5

# === REAL ===
echo ""
echo "=== Real rate=1 ==="
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > /workspace/rebuild_real.log 2>&1 &
for i in $(seq 1 120); do if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then break; fi; sleep 1; done
heavy_warmup

python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 50 --request-rate 1 --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
    --save-result --result-dir "$ONLINE_DIR" --result-filename "cal_real_rate1.json" > /dev/null 2>&1

pkill -f api_server 2>/dev/null || true; sleep 2; pkill -9 -f EngineCore 2>/dev/null || true; sleep 5

# === EMULATOR (auto params from profile, no env var overrides) ===
echo ""
echo "=== Emu rate=1 (auto-calibrated from profile) ==="
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="${PROFILE}" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > /workspace/rebuild_emu.log 2>&1 &
for i in $(seq 1 120); do if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then break; fi; sleep 1; done

# Print what the hook auto-computed
grep ExecutorEmulatorHook /workspace/rebuild_emu.log | head -1

heavy_warmup

python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 50 --request-rate 1 --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
    --save-result --result-dir "$ONLINE_DIR" --result-filename "cal_emu_rate1.json" > /dev/null 2>&1

pkill -f api_server 2>/dev/null || true; sleep 2; pkill -9 -f EngineCore 2>/dev/null || true

echo ""
echo "=== RESULTS ==="
python3 /workspace/vllm-emulator-v18/tools/e2e/compare_results.py \
    "$ONLINE_DIR/cal_real_rate1.json" "$ONLINE_DIR/cal_emu_rate1.json"
echo "DONE"
