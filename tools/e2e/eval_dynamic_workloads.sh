#!/bin/bash
# Evaluate emulator accuracy with dynamic (variable-length) workloads.
# Tests ShareGPT (conversation-style) and BurstGPT (bursty, variable-length).
#
# These workloads test if the surrogate approach generalizes beyond
# fixed-length random prompts — critical for paper claims.
#
# Usage:
#   ./eval_dynamic_workloads.sh --model Qwen/Qwen3-8B \
#       --profile-pack ./results/RTX-8000/profiles/serving-Qwen3-8B-step-cycle.json \
#       --result-dir ./results/RTX-8000/dynamic
#
# Prerequisites:
#   - Profile pack already generated (from profile_and_eval.sh)
#   - ShareGPT dataset: auto-downloaded via HuggingFace
#   - BurstGPT dataset: needs --burstgpt-path (CSV file)
set -euo pipefail

MODEL=""
PROFILE_PACK=""
RESULT_DIR=""
PORT=8100
MAX_MODEL_LEN=4096
WARMUP_PROMPTS=50
EVAL_PROMPTS=500
SHAREGPT_OUTPUT_LEN=256
BURSTGPT_PATH=""
RATES="1 4 8 inf"
SKIP_REAL=0
SKIP_SHAREGPT=0
SKIP_BURSTGPT=0

usage() {
    echo "Usage: $0 --model MODEL --profile-pack PATH --result-dir DIR [OPTIONS]"
    echo ""
    echo "Required:"
    echo "  --model MODEL              HuggingFace model name"
    echo "  --profile-pack PATH        Profile pack JSON from profiling step"
    echo "  --result-dir DIR           Output directory"
    echo ""
    echo "Optional:"
    echo "  --port PORT                Server port (default: 8100)"
    echo "  --eval-prompts N           Number of prompts (default: 500)"
    echo "  --rates 'R1 R2 ...'        Request rates (default: '1 4 8 inf')"
    echo "  --sharegpt-output-len N    Max output tokens for ShareGPT (default: 256)"
    echo "  --burstgpt-path PATH       Path to BurstGPT CSV (skip if not provided)"
    echo "  --skip-real                Skip real GPU runs (emu only)"
    echo "  --skip-sharegpt            Skip ShareGPT evaluation"
    echo "  --skip-burstgpt            Skip BurstGPT evaluation"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2;;
        --profile-pack) PROFILE_PACK="$2"; shift 2;;
        --result-dir) RESULT_DIR="$2"; shift 2;;
        --port) PORT="$2"; shift 2;;
        --eval-prompts) EVAL_PROMPTS="$2"; shift 2;;
        --rates) RATES="$2"; shift 2;;
        --sharegpt-output-len) SHAREGPT_OUTPUT_LEN="$2"; shift 2;;
        --burstgpt-path) BURSTGPT_PATH="$2"; shift 2;;
        --skip-real) SKIP_REAL=1; shift;;
        --skip-sharegpt) SKIP_SHAREGPT=1; shift;;
        --skip-burstgpt) SKIP_BURSTGPT=1; shift;;
        --help|-h) usage;;
        *) echo "Unknown: $1"; usage;;
    esac
done

[[ -z "$MODEL" ]] && { echo "ERROR: --model required"; usage; }
[[ -z "$PROFILE_PACK" ]] && { echo "ERROR: --profile-pack required"; usage; }
[[ -z "$RESULT_DIR" ]] && { echo "ERROR: --result-dir required"; usage; }

MODEL_SHORT="${MODEL##*/}"
LOGS_DIR="${RESULT_DIR}/logs"
mkdir -p "$RESULT_DIR/sharegpt" "$RESULT_DIR/burstgpt" "$LOGS_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cleanup_gpu() {
    fuser "${PORT}/tcp" 2>/dev/null | xargs -r kill -9 2>/dev/null || true
    sleep 3
}

wait_for_server() {
    for i in $(seq 1 180); do
        if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then return 0; fi
        sleep 1
    done
    echo "ERROR: Server timeout"; return 1
}

warmup() {
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "$MODEL" --base-url "http://localhost:${PORT}" \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts "$WARMUP_PROMPTS" --request-rate 4 > /dev/null 2>&1
    sleep 2
}

# Download ShareGPT dataset if not present
SHAREGPT_PATH="${RESULT_DIR}/ShareGPT_V3_unfiltered_cleaned_split.json"
download_sharegpt() {
    if [[ ! -f "$SHAREGPT_PATH" ]]; then
        echo "Downloading ShareGPT dataset..."
        python3 -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id='anon8231489123/ShareGPT_Vicuna_unfiltered',
                       filename='ShareGPT_V3_unfiltered_cleaned_split.json',
                       repo_type='dataset',
                       local_dir='${RESULT_DIR}')
print(f'Downloaded to {path}')
"
    fi
}

run_eval() {
    local dataset_name="$1" dataset_args="$2" tag_prefix="$3" out_dir="$4"

    for RATE in $RATES; do
        TAG="${tag_prefix}_r${RATE}"
        echo ""
        echo "--- ${dataset_name} Rate=$RATE ---"

        # Real
        if [[ "$SKIP_REAL" -eq 0 ]]; then
            cleanup_gpu
            python3 -m vllm.entrypoints.openai.api_server \
                --model "$MODEL" --max-model-len "$MAX_MODEL_LEN" \
                --port "$PORT" --trust-remote-code \
                > "${LOGS_DIR}/${TAG}_real_server.log" 2>&1 &
            wait_for_server; warmup
            echo "  Real benchmark..."
            python3 -m vllm.entrypoints.cli.main bench serve \
                --model "$MODEL" --base-url "http://localhost:${PORT}" \
                --dataset-name "$dataset_name" $dataset_args \
                --num-prompts "$EVAL_PROMPTS" --request-rate "$RATE" \
                --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
                --save-result --result-dir "$out_dir" \
                --result-filename "${TAG}_real.json" > /dev/null 2>&1
        fi

        # Emulator
        cleanup_gpu
        VLLM_EMULATOR_ENABLE_ORACLE=1 \
        VLLM_EMULATOR_PROFILE_PACK="$PROFILE_PACK" \
        VLLM_EMULATOR_MODE=realtime \
        VLLM_EMULATOR_EXECUTOR_HOOK=1 \
        VLLM_EMULATOR_PREP_SURROGATE=1 \
        python3 -m vllm.entrypoints.openai.api_server \
            --model "$MODEL" --max-model-len "$MAX_MODEL_LEN" \
            --port "$PORT" --trust-remote-code \
            > "${LOGS_DIR}/${TAG}_emu_server.log" 2>&1 &
        wait_for_server; warmup
        echo "  Emu benchmark..."
        python3 -m vllm.entrypoints.cli.main bench serve \
            --model "$MODEL" --base-url "http://localhost:${PORT}" \
            --dataset-name "$dataset_name" $dataset_args \
            --num-prompts "$EVAL_PROMPTS" --request-rate "$RATE" \
            --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
            --save-result --result-dir "$out_dir" \
            --result-filename "${TAG}_emu.json" > /dev/null 2>&1

        cleanup_gpu

        if [[ "$SKIP_REAL" -eq 0 ]] && [[ -f "${out_dir}/${TAG}_real.json" ]]; then
            python3 "$SCRIPT_DIR/compare_results.py" \
                "${out_dir}/${TAG}_real.json" "${out_dir}/${TAG}_emu.json" || true
        fi
    done
}

echo "============================================"
echo "=== Dynamic Workload Evaluation ==="
echo "============================================"
echo "Model: $MODEL | Profile: $PROFILE_PACK"
echo ""

# === ShareGPT ===
if [[ "$SKIP_SHAREGPT" -eq 0 ]]; then
    echo "========== ShareGPT (variable-length conversations) =========="
    download_sharegpt
    SGPT_ARGS="--dataset-path $SHAREGPT_PATH --sharegpt-output-len $SHAREGPT_OUTPUT_LEN"
    run_eval "sharegpt" "$SGPT_ARGS" "sharegpt_${MODEL_SHORT}" "$RESULT_DIR/sharegpt"
fi

# === BurstGPT ===
if [[ "$SKIP_BURSTGPT" -eq 0 ]]; then
    if [[ -n "$BURSTGPT_PATH" ]] && [[ -f "$BURSTGPT_PATH" ]]; then
        echo ""
        echo "========== BurstGPT (bursty arrival, variable-length) =========="
        BGPT_ARGS="--dataset-path $BURSTGPT_PATH"
        run_eval "burstgpt" "$BGPT_ARGS" "burstgpt_${MODEL_SHORT}" "$RESULT_DIR/burstgpt"
    else
        echo "Skipping BurstGPT (no --burstgpt-path provided or file not found)"
    fi
fi

echo ""
echo "============================================"
echo "=== SUMMARY ==="
echo "============================================"
for ds in sharegpt burstgpt; do
    if [[ -d "$RESULT_DIR/$ds" ]]; then
        echo "--- $ds ---"
        for RATE in $RATES; do
            TAG="${ds}_${MODEL_SHORT}_r${RATE}"
            if [[ -f "${RESULT_DIR}/${ds}/${TAG}_real.json" ]]; then
                echo "Rate=$RATE:"
                python3 "$SCRIPT_DIR/compare_results.py" \
                    "${RESULT_DIR}/${ds}/${TAG}_real.json" \
                    "${RESULT_DIR}/${ds}/${TAG}_emu.json" 2>/dev/null || echo "  MISSING"
            fi
        done
        echo ""
    fi
done
echo "ALL DONE. Results in: $RESULT_DIR"
