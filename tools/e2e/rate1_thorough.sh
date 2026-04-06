#!/bin/bash
# Thorough rate=1 test: more prompts + extensive warmup to reduce variance
set -e
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"
PROFILE="${RESULT_DIR}/profiles/serving-1.5b-tp1-calibrated.json"
PORT=8100
ONLINE_DIR="${RESULT_DIR}/online"
NUM_PROMPTS=200  # 4x more than usual for stable statistics

# Rebuild profile
python3 /workspace/vllm-emulator-v18/paper/isca-2026-mlarchsys/scripts/temp/build_serving_profile_2d.py \
    "${RESULT_DIR}/step_cycle_1.5b_full.jsonl" \
    "${RESULT_DIR}/profiles/sweep-1.5b-tp1-v14.json" \
    "$PROFILE" "$MODEL" "RTX-3060-12GB" > /dev/null 2>&1

thorough_warmup() {
    echo "  Phase 1: curl warmup (10 requests)..."
    for i in $(seq 1 10); do
        curl -s --max-time 10 http://localhost:$PORT/v1/completions \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"$MODEL\",\"prompt\":\"warmup test prompt number $i\",\"max_tokens\":5,\"temperature\":0}" > /dev/null
        sleep 0.2
    done

    echo "  Phase 2: bench warmup at rate=1 (50 prompts)..."
    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 50 --request-rate 1 > /dev/null 2>&1

    echo "  Phase 3: bench warmup at rate=2 (30 prompts, exercise batch shapes)..."
    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 30 --request-rate 2 > /dev/null 2>&1

    sleep 3
    echo "  Warmup complete"
}

pkill -9 -f EngineCore 2>/dev/null || true; pkill -9 -f api_server 2>/dev/null || true; sleep 8

# === REAL ===
echo "=== Real rate=1 (${NUM_PROMPTS} prompts, thorough warmup) ==="
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > /workspace/r1t_real.log 2>&1 &
for i in $(seq 1 120); do if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then break; fi; sleep 1; done
thorough_warmup

echo "  Benchmarking ${NUM_PROMPTS} prompts at rate=1..."
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts $NUM_PROMPTS --request-rate 1 --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
    --save-result --result-dir "$ONLINE_DIR" --result-filename "r1t_real.json" > /dev/null 2>&1

echo "  Real result:"
python3 -c "
import json
d = json.load(open('${ONLINE_DIR}/r1t_real.json'))
print(f'  completed={d[\"completed\"]}, failed={d[\"failed\"]}')
print(f'  mean_ttft={d[\"mean_ttft_ms\"]:.1f}, median_ttft={d[\"median_ttft_ms\"]:.1f}')
print(f'  mean_tpot={d[\"mean_tpot_ms\"]:.1f}, throughput={d[\"output_throughput\"]:.1f}')
"

pkill -f api_server 2>/dev/null || true; sleep 3; pkill -9 -f EngineCore 2>/dev/null || true; sleep 8

# === EMU ===
echo ""
echo "=== Emu rate=1 (${NUM_PROMPTS} prompts, thorough warmup) ==="
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > /workspace/r1t_emu.log 2>&1 &
for i in $(seq 1 120); do if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then break; fi; sleep 1; done
grep -m1 ExecutorEmulatorHook /workspace/r1t_emu.log
thorough_warmup

echo "  Benchmarking ${NUM_PROMPTS} prompts at rate=1..."
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts $NUM_PROMPTS --request-rate 1 --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
    --save-result --result-dir "$ONLINE_DIR" --result-filename "r1t_emu.json" > /dev/null 2>&1

echo "  Emu result:"
python3 -c "
import json
d = json.load(open('${ONLINE_DIR}/r1t_emu.json'))
print(f'  completed={d[\"completed\"]}, failed={d[\"failed\"]}')
print(f'  mean_ttft={d[\"mean_ttft_ms\"]:.1f}, median_ttft={d[\"median_ttft_ms\"]:.1f}')
print(f'  mean_tpot={d[\"mean_tpot_ms\"]:.1f}, throughput={d[\"output_throughput\"]:.1f}')
"

pkill -f api_server 2>/dev/null || true; sleep 3; pkill -9 -f EngineCore 2>/dev/null || true

echo ""
echo "=== COMPARE ==="
python3 /workspace/vllm-emulator-v18/tools/e2e/compare_results.py \
    "$ONLINE_DIR/r1t_real.json" "$ONLINE_DIR/r1t_emu.json"
echo "DONE"
