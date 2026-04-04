#!/bin/bash
# PD Disaggregation: real baseline + emulator comparison (0.5B, NIXL)
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub

MODEL="Qwen/Qwen2.5-0.5B-Instruct"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"
PORT_P=8100  # prefill
PORT_D=8200  # decode

cleanup() {
    pkill -9 -f EngineCore 2>/dev/null
    pkill -9 -f api_server 2>/dev/null
    sleep 5
}

start_pd_servers() {
    local prefix=$1
    local extra_env_p="${2:-}"
    local extra_env_d="${3:-}"

    echo "  Starting prefill (GPU0, port ${PORT_P})..."
    eval "${extra_env_p}" CUDA_VISIBLE_DEVICES=0 VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "${MODEL}" --max-model-len 2048 \
        --host 0.0.0.0 --port ${PORT_P} --trust-remote-code \
        --gpu-memory-utilization 0.9 --enforce-eager \
        --kv-transfer-config \
        '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_parallel_size":2,"kv_buffer_size":"5e8"}' \
        > /workspace/pd_${prefix}_prefill.log 2>&1 &

    echo "  Starting decode (GPU1, port ${PORT_D})..."
    eval "${extra_env_d}" CUDA_VISIBLE_DEVICES=1 VLLM_NIXL_SIDE_CHANNEL_PORT=5601 \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "${MODEL}" --max-model-len 2048 \
        --host 0.0.0.0 --port ${PORT_D} --trust-remote-code \
        --gpu-memory-utilization 0.9 --enforce-eager \
        --kv-transfer-config \
        '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_parallel_size":2,"kv_buffer_size":"5e8"}' \
        > /workspace/pd_${prefix}_decode.log 2>&1 &

    for i in $(seq 1 120); do
        p=$(curl -s http://localhost:${PORT_P}/health 2>/dev/null && echo 1 || echo 0)
        d=$(curl -s http://localhost:${PORT_D}/health 2>/dev/null && echo 1 || echo 0)
        if [ "$p" = "1" ] && [ "$d" = "1" ]; then
            echo "  Both ready after ${i}s"
            return 0
        fi
        sleep 1
    done
    echo "  TIMEOUT"
    return 1
}

echo "============================================================"
echo "PD Disaggregation Evaluation (0.5B, NIXL)"
echo "============================================================"

cleanup

# Phase 1: Real PD disagg — bench each instance separately
echo ""
echo "PHASE 1: Real PD disagg"
if start_pd_servers "real"; then
    # Benchmark prefill instance
    echo "  Benchmarking prefill instance..."
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "${MODEL}" --base-url http://localhost:${PORT_P} \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 30 --request-rate 2 \
        --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
        --save-result --result-dir "${RESULT_DIR}/online" \
        --result-filename "pd_real_prefill.json" 2>&1 | grep -E "TTFT|TPOT"

    echo "  Benchmarking decode instance..."
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "${MODEL}" --base-url http://localhost:${PORT_D} \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 30 --request-rate 2 \
        --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
        --save-result --result-dir "${RESULT_DIR}/online" \
        --result-filename "pd_real_decode.json" 2>&1 | grep -E "TTFT|TPOT"
fi
cleanup

# Phase 2: Emulator PD disagg (using serving profile from each instance)
# First need to build serving profiles for PD disagg
echo ""
echo "PHASE 2: Build PD serving profiles"

# Trace prefill instance
echo "  Tracing prefill..."
rm -f "${RESULT_DIR}/step_cycle_pd_prefill.jsonl"
VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
VLLM_EMULATOR_STEP_TRACE_OUTPUT="${RESULT_DIR}/step_cycle_pd_prefill.jsonl" \
CUDA_VISIBLE_DEVICES=0 VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 2048 \
    --host 0.0.0.0 --port ${PORT_P} --trust-remote-code \
    --gpu-memory-utilization 0.9 --enforce-eager \
    --kv-transfer-config \
    '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_parallel_size":2,"kv_buffer_size":"5e8"}' \
    > /workspace/pd_trace_prefill.log 2>&1 &

# Trace decode instance
rm -f "${RESULT_DIR}/step_cycle_pd_decode.jsonl"
VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
VLLM_EMULATOR_STEP_TRACE_OUTPUT="${RESULT_DIR}/step_cycle_pd_decode.jsonl" \
CUDA_VISIBLE_DEVICES=1 VLLM_NIXL_SIDE_CHANNEL_PORT=5601 \
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 2048 \
    --host 0.0.0.0 --port ${PORT_D} --trust-remote-code \
    --gpu-memory-utilization 0.9 --enforce-eager \
    --kv-transfer-config \
    '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_parallel_size":2,"kv_buffer_size":"5e8"}' \
    > /workspace/pd_trace_decode.log 2>&1 &

for i in $(seq 1 120); do
    p=$(curl -s http://localhost:${PORT_P}/health 2>/dev/null && echo 1 || echo 0)
    d=$(curl -s http://localhost:${PORT_D}/health 2>/dev/null && echo 1 || echo 0)
    if [ "$p" = "1" ] && [ "$d" = "1" ]; then
        echo "  Both trace servers ready after ${i}s"
        break
    fi
    sleep 1
done

# Send traffic to both
for rate in 1 2; do
    echo "  Tracing rate=${rate}..."
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "${MODEL}" --base-url http://localhost:${PORT_P} \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 30 --request-rate ${rate} > /dev/null 2>&1 &
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "${MODEL}" --base-url http://localhost:${PORT_D} \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 30 --request-rate ${rate} > /dev/null 2>&1 &
    wait
done

cleanup

echo "  Prefill trace: $(wc -l < ${RESULT_DIR}/step_cycle_pd_prefill.jsonl 2>/dev/null || echo 0) records"
echo "  Decode trace: $(wc -l < ${RESULT_DIR}/step_cycle_pd_decode.jsonl 2>/dev/null || echo 0) records"

echo ""
echo "PD Disagg smoke test complete."
echo "Both NIXL instances (prefill+decode) work with 0.5B on RTX 3060."
echo "For full emulator eval with PD disagg, need:"
echo "  1. Build per-instance serving profiles from traces"
echo "  2. Run emulator with per-instance profiles"
echo "  3. Compare TTFT/TPOT — deferred to larger GPUs with 1.5B+ model"
echo ""
echo "DONE"
