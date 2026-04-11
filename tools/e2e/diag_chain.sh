#!/bin/bash
# Chain diagnostics: compare surrogate vs no-surrogate at R=1
# Short runs (50 prompts) to capture chain dynamics
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1
MODEL="Qwen/Qwen2.5-1.5B-Instruct"
PORT=8500
PROFILE="/workspace/eval_results/RTX-3060-12GB/profiles/sweep-1.5b-tp1-v14.json"

cleanup_gpu() {
    pkill -9 -f "python3.*api_server" 2>/dev/null || true
    fuser /dev/nvidia* 2>/dev/null | tr " " "\n" | sort -u | xargs kill -9 2>/dev/null || true
    sleep 10
}
wait_server() {
    for i in $(seq 1 120); do curl -s http://localhost:$PORT/health > /dev/null 2>&1 && return 0; sleep 1; done
    echo "ERROR: Server failed to start"; return 1
}

echo "=== Chain Diagnostics ==="
echo "=== $(date) ==="

# Test 1: Surrogate mode at R=1 (50 prompts — quick)
echo ""
echo "=== Test 1: Surrogate (default) R=1 ==="
cleanup_gpu
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
VLLM_EMULATOR_ORACLE_MODE=2d \
VLLM_EMULATOR_PREP_SURROGATE=1 \
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > /workspace/diag_chain_surr.log 2>&1 &
wait_server || exit 1

# Quick warmup (no full CUDA sweep — just basic)
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 20 --request-rate 1 > /dev/null 2>&1

# Actual test
python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 50 --request-rate 1 \
    --percentile-metrics ttft,tpot,e2el 2>&1 | grep -E "Mean TTFT|Mean TPOT|Mean E2EL"

echo ""
echo "Chain diag entries (surrogate):"
grep "ChainDiag" /workspace/diag_chain_surr.log 2>/dev/null | head -20
echo "..."
grep "ChainDiag" /workspace/diag_chain_surr.log 2>/dev/null | tail -5
cleanup_gpu

# Test 2: No surrogate at R=1 (to see baseline chain without prep)
echo ""
echo "=== Test 2: No surrogate R=1 ==="
cleanup_gpu
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
VLLM_EMULATOR_ORACLE_MODE=2d \
VLLM_EMULATOR_PREP_SURROGATE=0 \
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > /workspace/diag_chain_nosurr.log 2>&1 &
wait_server || exit 1

python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 20 --request-rate 1 > /dev/null 2>&1

python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 50 --request-rate 1 \
    --percentile-metrics ttft,tpot,e2el 2>&1 | grep -E "Mean TTFT|Mean TPOT|Mean E2EL"

echo ""
echo "Chain diag entries (no surrogate):"
grep "ChainDiag" /workspace/diag_chain_nosurr.log 2>/dev/null | head -20
echo "..."
grep "ChainDiag" /workspace/diag_chain_nosurr.log 2>/dev/null | tail -5
cleanup_gpu

# Test 3: Surrogate at R=8 (50 prompts)
echo ""
echo "=== Test 3: Surrogate R=8 ==="
cleanup_gpu
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
VLLM_EMULATOR_ORACLE_MODE=2d \
VLLM_EMULATOR_PREP_SURROGATE=1 \
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > /workspace/diag_chain_surr_r8.log 2>&1 &
wait_server || exit 1

python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 20 --request-rate 4 > /dev/null 2>&1

python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 100 --request-rate 8 \
    --percentile-metrics ttft,tpot,e2el 2>&1 | grep -E "Mean TTFT|Mean TPOT|Mean E2EL"

echo ""
echo "Chain diag entries (surrogate R=8):"
grep "ChainDiag" /workspace/diag_chain_surr_r8.log 2>/dev/null | head -20
echo "..."
grep "ChainDiag" /workspace/diag_chain_surr_r8.log 2>/dev/null | tail -5
cleanup_gpu

# Test 4: No surrogate at R=8
echo ""
echo "=== Test 4: No surrogate R=8 ==="
cleanup_gpu
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
VLLM_EMULATOR_ORACLE_MODE=2d \
VLLM_EMULATOR_PREP_SURROGATE=0 \
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > /workspace/diag_chain_nosurr_r8.log 2>&1 &
wait_server || exit 1

python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 20 --request-rate 4 > /dev/null 2>&1

python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 100 --request-rate 8 \
    --percentile-metrics ttft,tpot,e2el 2>&1 | grep -E "Mean TTFT|Mean TPOT|Mean E2EL"

echo ""
echo "Chain diag entries (no surrogate R=8):"
grep "ChainDiag" /workspace/diag_chain_nosurr_r8.log 2>/dev/null | head -20
echo "..."
grep "ChainDiag" /workspace/diag_chain_nosurr_r8.log 2>/dev/null | tail -5
cleanup_gpu

echo ""
echo "=== DONE $(date) ==="
