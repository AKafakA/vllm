#!/bin/bash
# Extend profile with high-concurrency data, rebuild, test rate=8
set -e
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub
export CUDA_VISIBLE_DEVICES=1

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"
PORT=8100
SCRIPT_DIR="/workspace/vllm-emulator-v18/tools/e2e"
ONLINE_DIR="${RESULT_DIR}/online"
TRACE="${RESULT_DIR}/step_cycle_fresh.jsonl"

cleanup_gpu() {
    pkill -9 -f "python3.*api_server" 2>/dev/null || true
    pkill -9 -f "python3.*bench" 2>/dev/null || true
    fuser /dev/nvidia* 2>/dev/null | tr " " "\n" | sort -u | xargs kill -9 2>/dev/null || true
    sleep 5
}

wait_server() {
    for i in $(seq 1 120); do
        if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then return 0; fi
        sleep 1
    done
    echo "ERROR: Server failed to start"; return 1
}

echo "============================================"
echo "=== Extend Profile + Test Rate=8 ==="
echo "=== $(date) ==="
echo "============================================"

# Check current profile coverage
echo "=== Current trace coverage ==="
python3 -c "
import json, statistics
from collections import Counter
records = [json.loads(l) for l in open('${TRACE}') if 'total_tokens' in json.loads(l)]
tts = [r['total_tokens'] for r in records]
print(f'Records: {len(records)}, max_tt: {max(tts)}, tt>20: {sum(1 for t in tts if t > 20)}, tt>50: {sum(1 for t in tts if t > 50)}')
# Show decode-only at high tt
dec = [r for r in records if r.get('num_new_reqs',0)==0 and r['total_tokens']>10]
print(f'Decode-only tt>10: {len(dec)}')
tt_counts = Counter(r['total_tokens'] for r in dec)
for tt in sorted(tt_counts.keys()):
    if tt >= 10:
        lats = [r['step_cycle_us'] for r in dec if r['total_tokens']==tt]
        print(f'  tt={tt:>3}: n={len(lats):>3}, median={statistics.median(lats)/1000:.1f}ms')
"

# Phase 1: Add high-rate profiling to extend trace
echo ""
echo "=== Phase 1: High-rate profiling (rate=16, 32, inf) ==="

cleanup_gpu
VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
VLLM_EMULATOR_STEP_TRACE_OUTPUT="$TRACE" \
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > /workspace/extend_profile_server.log 2>&1 &
wait_server

# Warmup
echo "  Warmup..."
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 --request-rate 4 > /dev/null 2>&1

for rate in 16 32; do
    echo "  Profile rate=$rate (500 prompts)..."
    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 500 --request-rate $rate > /dev/null 2>&1
done

echo "  Profile rate=inf (500 prompts)..."
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 500 --request-rate inf > /dev/null 2>&1

cleanup_gpu

echo ""
echo "=== Extended trace coverage ==="
python3 -c "
import json, statistics
from collections import Counter
records = [json.loads(l) for l in open('${TRACE}') if 'total_tokens' in json.loads(l)]
tts = [r['total_tokens'] for r in records]
print(f'Records: {len(records)}, max_tt: {max(tts)}, tt>20: {sum(1 for t in tts if t > 20)}, tt>50: {sum(1 for t in tts if t > 50)}, tt>100: {sum(1 for t in tts if t > 100)}')
dec = [r for r in records if r.get('num_new_reqs',0)==0 and r['total_tokens']>10]
print(f'Decode-only tt>10: {len(dec)}')
for tt_range in [(10,20), (20,50), (50,100), (100,200)]:
    lo, hi = tt_range
    recs = [r for r in dec if lo <= r['total_tokens'] < hi]
    if recs:
        med = statistics.median(r['step_cycle_us'] for r in recs)
        print(f'  tt={lo}-{hi}: n={len(recs)}, median={med/1000:.1f}ms')
"

# Phase 2: Rebuild profile
echo ""
echo "=== Phase 2: Rebuild profile ==="
python3 /workspace/vllm-emulator-v18/vllm_emulator/profile/build_serving_profile.py \
    "$TRACE" \
    "${RESULT_DIR}/profiles/sweep-1.5b-tp1-v14.json" \
    "${RESULT_DIR}/profiles/serving-1.5b-tp1-extended.json" \
    "$MODEL" "RTX-3060-12GB"

PROFILE="${RESULT_DIR}/profiles/serving-1.5b-tp1-extended.json"

# Phase 3: Test rate=8 with extended profile
echo ""
echo "=== Phase 3: Rate=8 with extended profile (hybrid) ==="

cleanup_gpu
echo "  Real server..."
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > /workspace/ext_real_8.log 2>&1 &
wait_server
echo "  Warmup..."
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 --request-rate 4 > /dev/null 2>&1
sleep 3
echo "  Bench real (1000 prompts)..."
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 1000 --request-rate 8 --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,99 \
    --save-result --result-dir "$ONLINE_DIR" --result-filename "ext_rate8_real.json" > /dev/null 2>&1

cleanup_gpu
echo "  Emu server (hybrid, extended profile)..."
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
VLLM_EMULATOR_ORACLE_MODE=hybrid \
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > /workspace/ext_emu_8.log 2>&1 &
wait_server
grep "ExecutorEmulatorHook" /workspace/ext_emu_8.log | head -1
echo "  Warmup..."
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 --request-rate 4 > /dev/null 2>&1
sleep 3
echo "  Bench emu (1000 prompts)..."
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 1000 --request-rate 8 --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,99 \
    --save-result --result-dir "$ONLINE_DIR" --result-filename "ext_rate8_emu.json" > /dev/null 2>&1

cleanup_gpu
echo ""
echo "=== Rate=8 Results (extended profile) ==="
python3 "$SCRIPT_DIR/compare_results.py" "$ONLINE_DIR/ext_rate8_real.json" "$ONLINE_DIR/ext_rate8_emu.json"

# Also quick check rate=4 didn't regress
echo ""
echo "=== Rate=4 sanity (extended profile) ==="
cleanup_gpu
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
VLLM_EMULATOR_ORACLE_MODE=hybrid \
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > /workspace/ext_emu_4.log 2>&1 &
wait_server
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 --request-rate 4 > /dev/null 2>&1
sleep 3
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 1000 --request-rate 4 --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,99 \
    --save-result --result-dir "$ONLINE_DIR" --result-filename "ext_rate4_emu.json" > /dev/null 2>&1
cleanup_gpu
python3 "$SCRIPT_DIR/compare_results.py" "$ONLINE_DIR/fresh_rate4_real.json" "$ONLINE_DIR/ext_rate4_emu.json"

echo ""
echo "=== DONE $(date) ==="
