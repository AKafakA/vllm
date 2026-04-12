#!/bin/bash
# Ablation study: evaluate emulator accuracy under different vLLM configurations.
#
# Each ablation changes one server flag, profiles from scratch, and compares
# real vs emulator at multiple rates. This tests whether the surrogate
# approach works across all vLLM operating modes.
#
# Ablations:
#   1. chunked_prefill: on (default) vs off
#   2. prefix_caching: on (default) vs off
#   3. async_scheduling: on vs off
#
# Usage:
#   ./eval_ablations.sh --model Qwen/Qwen3-8B --result-dir ./results/RTX-8000/ablations
#
# Each ablation produces its own profile pack and results.
set -euo pipefail

MODEL=""
RESULT_DIR=""
PORT=8100
MAX_MODEL_LEN=4096
INPUT_LEN=256
OUTPUT_LEN=128
WARMUP_PROMPTS=50
EVAL_PROMPTS=500
RATES="1 4 8"
PROFILE_RATES="1 4 8"
SKIP_REAL=0
# Which ablations to run (space-separated, or "all")
ABLATIONS="all"

usage() {
    echo "Usage: $0 --model MODEL --result-dir DIR [OPTIONS]"
    echo ""
    echo "Required:"
    echo "  --model MODEL          HuggingFace model name"
    echo "  --result-dir DIR       Output directory"
    echo ""
    echo "Optional:"
    echo "  --port PORT            Server port (default: 8100)"
    echo "  --eval-prompts N       Eval prompts per rate (default: 500)"
    echo "  --rates 'R1 R2 ...'    Eval rates (default: '1 4 8')"
    echo "  --ablations 'A1 A2..'  Which ablations (default: all)"
    echo "                         Options: chunked_prefill prefix_caching async_scheduling"
    echo "  --skip-real            Skip real GPU runs"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2;;
        --result-dir) RESULT_DIR="$2"; shift 2;;
        --port) PORT="$2"; shift 2;;
        --eval-prompts) EVAL_PROMPTS="$2"; shift 2;;
        --rates) RATES="$2"; shift 2;;
        --ablations) ABLATIONS="$2"; shift 2;;
        --skip-real) SKIP_REAL=1; shift;;
        --help|-h) usage;;
        *) echo "Unknown: $1"; usage;;
    esac
done

[[ -z "$MODEL" ]] && { echo "ERROR: --model required"; usage; }
[[ -z "$RESULT_DIR" ]] && { echo "ERROR: --result-dir required"; usage; }

MODEL_SHORT="${MODEL##*/}"
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
        --dataset-name random --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" \
        --num-prompts "$WARMUP_PROMPTS" --request-rate 4 > /dev/null 2>&1
    sleep 2
}

# Run one ablation: profile + eval with given server flags
# Args: $1=ablation_name $2=server_extra_flags
run_ablation() {
    local NAME="$1"
    local SERVER_FLAGS="$2"
    local ABL_DIR="${RESULT_DIR}/${NAME}"
    local PROFILES_DIR="${ABL_DIR}/profiles"
    local ONLINE_DIR="${ABL_DIR}/online"
    local LOGS_DIR="${ABL_DIR}/logs"
    mkdir -p "$PROFILES_DIR" "$ONLINE_DIR" "$LOGS_DIR"

    echo ""
    echo "========================================================"
    echo "=== Ablation: ${NAME} ==="
    echo "=== Server flags: ${SERVER_FLAGS} ==="
    echo "========================================================"

    # Step 1: Profile with this configuration
    local TRACE_FILE="${PROFILES_DIR}/step_cycle.jsonl"
    rm -f "$TRACE_FILE"

    cleanup_gpu
    echo "  Profiling..."
    VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
    VLLM_EMULATOR_STEP_TRACE_OUTPUT="$TRACE_FILE" \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --max-model-len "$MAX_MODEL_LEN" \
        --port "$PORT" --trust-remote-code $SERVER_FLAGS \
        > "${LOGS_DIR}/profile_server.log" 2>&1 &
    wait_for_server; warmup

    for rate in $PROFILE_RATES; do
        NP=200
        python3 -m vllm.entrypoints.cli.main bench serve \
            --model "$MODEL" --base-url "http://localhost:${PORT}" \
            --dataset-name random --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" \
            --num-prompts "$NP" --request-rate "$rate" > /dev/null 2>&1
    done
    # High concurrency
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "$MODEL" --base-url "http://localhost:${PORT}" \
        --dataset-name random --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" \
        --num-prompts 200 --request-rate inf > /dev/null 2>&1
    cleanup_gpu

    local PROFILE="${PROFILES_DIR}/serving-${NAME}.json"
    python3 "${SCRIPT_DIR}/../../vllm_emulator/profile/build_serving_profile.py" \
        "$TRACE_FILE" "/dev/null" "$PROFILE"
    echo "  Profile: $PROFILE ($(wc -l < "$TRACE_FILE") records)"

    # Step 2: Evaluate at each rate
    for RATE in $RATES; do
        TAG="${NAME}_r${RATE}"
        echo "  --- Rate=$RATE ---"

        # Real
        if [[ "$SKIP_REAL" -eq 0 ]]; then
            cleanup_gpu
            python3 -m vllm.entrypoints.openai.api_server \
                --model "$MODEL" --max-model-len "$MAX_MODEL_LEN" \
                --port "$PORT" --trust-remote-code $SERVER_FLAGS \
                > "${LOGS_DIR}/${TAG}_real.log" 2>&1 &
            wait_for_server; warmup
            python3 -m vllm.entrypoints.cli.main bench serve \
                --model "$MODEL" --base-url "http://localhost:${PORT}" \
                --dataset-name random --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" \
                --num-prompts "$EVAL_PROMPTS" --request-rate "$RATE" \
                --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
                --save-result --result-dir "$ONLINE_DIR" \
                --result-filename "${TAG}_real.json" > /dev/null 2>&1
        fi

        # Emulator (same server flags)
        cleanup_gpu
        VLLM_EMULATOR_ENABLE_ORACLE=1 \
        VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
        VLLM_EMULATOR_MODE=realtime \
        VLLM_EMULATOR_EXECUTOR_HOOK=1 \
        VLLM_EMULATOR_PREP_SURROGATE=1 \
        python3 -m vllm.entrypoints.openai.api_server \
            --model "$MODEL" --max-model-len "$MAX_MODEL_LEN" \
            --port "$PORT" --trust-remote-code $SERVER_FLAGS \
            > "${LOGS_DIR}/${TAG}_emu.log" 2>&1 &
        wait_for_server; warmup
        python3 -m vllm.entrypoints.cli.main bench serve \
            --model "$MODEL" --base-url "http://localhost:${PORT}" \
            --dataset-name random --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" \
            --num-prompts "$EVAL_PROMPTS" --request-rate "$RATE" \
            --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
            --save-result --result-dir "$ONLINE_DIR" \
            --result-filename "${TAG}_emu.json" > /dev/null 2>&1
        cleanup_gpu

        if [[ "$SKIP_REAL" -eq 0 ]] && [[ -f "${ONLINE_DIR}/${TAG}_real.json" ]]; then
            python3 "$SCRIPT_DIR/compare_results.py" \
                "${ONLINE_DIR}/${TAG}_real.json" "${ONLINE_DIR}/${TAG}_emu.json" || true
        fi
    done
}

# ============================================================
# Define ablation configurations
# ============================================================

# Resolve which ablations to run
if [[ "$ABLATIONS" == "all" ]]; then
    ABLATIONS="chunked_prefill_off prefix_caching_off async_scheduling_off async_scheduling_on"
fi

echo "============================================"
echo "=== Ablation Study ==="
echo "============================================"
echo "Model: $MODEL"
echo "Ablations: $ABLATIONS"
echo ""

for ABL in $ABLATIONS; do
    case "$ABL" in
        chunked_prefill_off)
            run_ablation "no_chunked_prefill" "--no-enable-chunked-prefill"
            ;;
        prefix_caching_off)
            run_ablation "no_prefix_caching" "--no-enable-prefix-caching"
            ;;
        async_scheduling_off)
            run_ablation "no_async_scheduling" "--no-async-scheduling"
            ;;
        async_scheduling_on)
            run_ablation "async_scheduling" "--async-scheduling"
            ;;
        *)
            echo "Unknown ablation: $ABL (skipping)"
            ;;
    esac
done

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================"
echo "=== ABLATION SUMMARY ==="
echo "============================================"
for ABL in $ABLATIONS; do
    case "$ABL" in
        chunked_prefill_off) NAME="no_chunked_prefill";;
        prefix_caching_off) NAME="no_prefix_caching";;
        async_scheduling_off) NAME="no_async_scheduling";;
        async_scheduling_on) NAME="async_scheduling";;
        *) continue;;
    esac
    echo "--- $NAME ---"
    for RATE in $RATES; do
        TAG="${NAME}_r${RATE}"
        R="${RESULT_DIR}/${NAME}/online/${TAG}_real.json"
        E="${RESULT_DIR}/${NAME}/online/${TAG}_emu.json"
        if [[ -f "$R" ]] && [[ -f "$E" ]]; then
            echo "  Rate=$RATE:"
            python3 "$SCRIPT_DIR/compare_results.py" "$R" "$E" 2>/dev/null || echo "    MISSING"
        fi
    done
    echo ""
done
echo "ALL DONE. Results in: $RESULT_DIR"
