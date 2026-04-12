#!/bin/bash
# Portable profile + evaluation script for the vLLM emulator.
# Works across hosts/models without hardcoded paths.
#
# Usage:
#   ./profile_and_eval.sh --model Qwen/Qwen3-8B --result-dir ./results
#
# Required: venv activated, vllm-emulator installed (pip install -e .)
# The script auto-detects GPU name, collects metadata during profiling,
# and produces a self-contained profile pack.
set -euo pipefail

# ============================================================
# Default configuration (override via CLI flags)
# ============================================================
MODEL=""
RESULT_DIR=""
PORT=8100
MAX_MODEL_LEN=4096
INPUT_LEN=256
OUTPUT_LEN=128
WARMUP_PROMPTS=200
EVAL_PROMPTS=1000
RATES="1 2 4 8 16"
PROFILE_RATES="0.5 1 2 3 4 6 8 12"
SKIP_PROFILE=0
SKIP_REAL=0
PROFILE_PACK=""  # Set automatically after profiling, or provide to skip

# ============================================================
# Parse CLI arguments
# ============================================================
usage() {
    echo "Usage: $0 --model MODEL --result-dir DIR [OPTIONS]"
    echo ""
    echo "Required:"
    echo "  --model MODEL          HuggingFace model name (e.g. Qwen/Qwen3-8B)"
    echo "  --result-dir DIR       Output directory for results"
    echo ""
    echo "Optional:"
    echo "  --port PORT            Server port (default: 8100)"
    echo "  --max-model-len LEN    Max sequence length (default: 4096)"
    echo "  --input-len LEN        Random input length (default: 256)"
    echo "  --output-len LEN       Random output length (default: 128)"
    echo "  --warmup-prompts N     Warmup prompts (default: 200)"
    echo "  --eval-prompts N       Eval prompts (default: 1000)"
    echo "  --rates 'R1 R2 ...'    Space-separated rates (default: '1 2 4 8 16')"
    echo "  --profile-rates 'R..'  Rates for profiling (default: '0.5 1 2 3 4 6 8 12')"
    echo "  --profile-pack PATH    Skip profiling, use existing profile"
    echo "  --skip-real            Skip real GPU benchmark (emu only)"
    echo "  --skip-profile         Skip profiling (requires --profile-pack)"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2;;
        --result-dir) RESULT_DIR="$2"; shift 2;;
        --port) PORT="$2"; shift 2;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift 2;;
        --input-len) INPUT_LEN="$2"; shift 2;;
        --output-len) OUTPUT_LEN="$2"; shift 2;;
        --warmup-prompts) WARMUP_PROMPTS="$2"; shift 2;;
        --eval-prompts) EVAL_PROMPTS="$2"; shift 2;;
        --rates) RATES="$2"; shift 2;;
        --profile-rates) PROFILE_RATES="$2"; shift 2;;
        --profile-pack) PROFILE_PACK="$2"; SKIP_PROFILE=1; shift 2;;
        --skip-real) SKIP_REAL=1; shift;;
        --skip-profile) SKIP_PROFILE=1; shift;;
        --help|-h) usage;;
        *) echo "Unknown option: $1"; usage;;
    esac
done

[[ -z "$MODEL" ]] && { echo "ERROR: --model is required"; usage; }
[[ -z "$RESULT_DIR" ]] && { echo "ERROR: --result-dir is required"; usage; }

# Derived paths
PROFILES_DIR="${RESULT_DIR}/profiles"
ONLINE_DIR="${RESULT_DIR}/online"
OFFLINE_DIR="${RESULT_DIR}/offline"
LOGS_DIR="${RESULT_DIR}/logs"
mkdir -p "$PROFILES_DIR" "$ONLINE_DIR" "$OFFLINE_DIR" "$LOGS_DIR"

# Model short name for filenames (e.g. Qwen/Qwen3-8B -> Qwen3-8B)
MODEL_SHORT="${MODEL##*/}"

# Detect GPU name
GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0).replace(' ', '-'))" 2>/dev/null || echo "unknown-gpu")
echo "GPU: $GPU_NAME"
echo "Model: $MODEL ($MODEL_SHORT)"
echo "Results: $RESULT_DIR"
echo ""

# ============================================================
# Helper functions
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cleanup_gpu() {
    # Kill any vllm servers on our port and any leftover GPU processes
    fuser "${PORT}/tcp" 2>/dev/null | xargs -r kill -9 2>/dev/null || true
    pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
    pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
    sleep 5
}

wait_for_server() {
    local max_wait=${1:-180}
    for i in $(seq 1 "$max_wait"); do
        if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    echo "ERROR: Server did not start within ${max_wait}s"
    return 1
}

warmup() {
    echo "  Warmup (${WARMUP_PROMPTS} prompts at rate=4)..."
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "$MODEL" --base-url "http://localhost:${PORT}" \
        --dataset-name random --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" \
        --num-prompts "$WARMUP_PROMPTS" --request-rate 4 > /dev/null 2>&1
    sleep 2
}

run_bench_serve() {
    local rate="$1" output_json="$2"
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "$MODEL" --base-url "http://localhost:${PORT}" \
        --dataset-name random --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" \
        --num-prompts "$EVAL_PROMPTS" --request-rate "$rate" \
        --percentile-metrics ttft,tpot --metric-percentiles 50,99 \
        --save-result --result-dir "$(dirname "$output_json")" \
        --result-filename "$(basename "$output_json")" > /dev/null 2>&1
}

# ============================================================
# STEP 1: Profile (auto-collects GPU + model metadata)
# ============================================================
if [[ "$SKIP_PROFILE" -eq 0 ]]; then
    echo "============================================"
    echo "=== STEP 1: Profiling (auto-metadata) ==="
    echo "============================================"

    TRACE_FILE="${PROFILES_DIR}/step_cycle_${MODEL_SHORT}.jsonl"
    rm -f "$TRACE_FILE"

    cleanup_gpu
    echo "Starting profiling server..."
    VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
    VLLM_EMULATOR_STEP_TRACE_OUTPUT="$TRACE_FILE" \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --max-model-len "$MAX_MODEL_LEN" \
        --port "$PORT" --trust-remote-code \
        > "${LOGS_DIR}/profile_server.log" 2>&1 &
    wait_for_server

    warmup

    # Profile at multiple rates for diverse concurrency coverage
    for rate in $PROFILE_RATES; do
        NP=200
        [[ "$rate" == "0.5" ]] && NP=50
        echo "  Profiling rate=$rate ($NP prompts)..."
        python3 -m vllm.entrypoints.cli.main bench serve \
            --model "$MODEL" --base-url "http://localhost:${PORT}" \
            --dataset-name random --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" \
            --num-prompts "$NP" --request-rate "$rate" > /dev/null 2>&1
    done

    # Offline (rate=inf) for high-concurrency coverage
    echo "  Profiling offline (200 prompts, rate=inf)..."
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "$MODEL" --base-url "http://localhost:${PORT}" \
        --dataset-name random --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" \
        --num-prompts 200 --request-rate inf > /dev/null 2>&1

    cleanup_gpu

    RECORDS=$(wc -l < "$TRACE_FILE")
    echo "Trace records: $RECORDS"

    # Build profile pack (reads _header for GPU/model metadata automatically)
    PROFILE_PACK="${PROFILES_DIR}/serving-${MODEL_SHORT}-step-cycle.json"
    echo "Building profile..."
    python3 "${SCRIPT_DIR}/../../vllm_emulator/profile/build_serving_profile.py" \
        "$TRACE_FILE" \
        "/dev/null" \
        "$PROFILE_PACK"
    echo "Profile saved to: $PROFILE_PACK"
else
    [[ -z "$PROFILE_PACK" ]] && { echo "ERROR: --profile-pack required with --skip-profile"; exit 1; }
    echo "Using existing profile: $PROFILE_PACK"
fi

echo ""

# ============================================================
# STEP 2: Evaluate at each rate (independent server per rate)
# ============================================================
echo "============================================"
echo "=== STEP 2: Online serving evaluation ==="
echo "============================================"

for RATE in $RATES; do
    TAG="${MODEL_SHORT}_r${RATE}"
    echo ""
    echo "--- Rate=$RATE (${EVAL_PROMPTS} prompts) ---"

    # --- Real GPU ---
    if [[ "$SKIP_REAL" -eq 0 ]]; then
        cleanup_gpu
        echo "  Starting real server..."
        python3 -m vllm.entrypoints.openai.api_server \
            --model "$MODEL" --max-model-len "$MAX_MODEL_LEN" \
            --port "$PORT" --trust-remote-code \
            > "${LOGS_DIR}/real_${RATE}.log" 2>&1 &
        wait_for_server
        warmup
        echo "  Benchmarking real..."
        run_bench_serve "$RATE" "${ONLINE_DIR}/${TAG}_real.json"
    fi

    # --- Emulator ---
    cleanup_gpu
    echo "  Starting emulator server..."
    VLLM_EMULATOR_ENABLE_ORACLE=1 \
    VLLM_EMULATOR_PROFILE_PACK="$PROFILE_PACK" \
    VLLM_EMULATOR_MODE=realtime \
    VLLM_EMULATOR_EXECUTOR_HOOK=1 \
    VLLM_EMULATOR_PREP_SURROGATE=1 \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --max-model-len "$MAX_MODEL_LEN" \
        --port "$PORT" --trust-remote-code \
        > "${LOGS_DIR}/emu_${RATE}.log" 2>&1 &
    wait_for_server
    grep -m1 "ExecutorEmulatorHook" "${LOGS_DIR}/emu_${RATE}.log" | strings || true
    warmup
    echo "  Benchmarking emulator..."
    run_bench_serve "$RATE" "${ONLINE_DIR}/${TAG}_emu.json"

    cleanup_gpu

    # Compare
    if [[ "$SKIP_REAL" -eq 0 ]] && [[ -f "${ONLINE_DIR}/${TAG}_real.json" ]]; then
        echo "  Results:"
        python3 "$SCRIPT_DIR/compare_results.py" \
            "${ONLINE_DIR}/${TAG}_real.json" \
            "${ONLINE_DIR}/${TAG}_emu.json" || true
    fi
done

# ============================================================
# STEP 3: Offline throughput evaluation
# ============================================================
echo ""
echo "============================================"
echo "=== STEP 3: Offline throughput evaluation ==="
echo "============================================"

if [[ "$SKIP_REAL" -eq 0 ]]; then
    cleanup_gpu
    echo "  Benchmarking real offline..."
    python3 -m vllm.entrypoints.cli.main bench throughput \
        --model "$MODEL" --max-model-len "$MAX_MODEL_LEN" --trust-remote-code \
        --dataset-name random --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" \
        --num-prompts "$EVAL_PROMPTS" \
        --output-json "${OFFLINE_DIR}/${MODEL_SHORT}_offline_real.json" \
        > "${LOGS_DIR}/offline_real.log" 2>&1
fi

cleanup_gpu
echo "  Benchmarking emulator offline..."
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="$PROFILE_PACK" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
VLLM_EMULATOR_PREP_SURROGATE=1 \
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model "$MODEL" --max-model-len "$MAX_MODEL_LEN" --trust-remote-code \
    --dataset-name random --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" \
    --num-prompts "$EVAL_PROMPTS" \
    --output-json "${OFFLINE_DIR}/${MODEL_SHORT}_offline_emu.json" \
    > "${LOGS_DIR}/offline_emu.log" 2>&1

cleanup_gpu

# ============================================================
# SUMMARY
# ============================================================
echo ""
echo "============================================"
echo "=== SUMMARY ==="
echo "============================================"
echo "GPU: $GPU_NAME | Model: $MODEL | Prompts: $EVAL_PROMPTS"
echo ""
for RATE in $RATES; do
    TAG="${MODEL_SHORT}_r${RATE}"
    if [[ -f "${ONLINE_DIR}/${TAG}_real.json" ]] && [[ -f "${ONLINE_DIR}/${TAG}_emu.json" ]]; then
        echo "--- Rate=$RATE ---"
        python3 "$SCRIPT_DIR/compare_results.py" \
            "${ONLINE_DIR}/${TAG}_real.json" \
            "${ONLINE_DIR}/${TAG}_emu.json" 2>/dev/null || echo "  MISSING"
        echo ""
    fi
done

if [[ -f "${OFFLINE_DIR}/${MODEL_SHORT}_offline_real.json" ]]; then
    echo "--- Offline ---"
    R_TP=$(python3 -c "import json; d=json.load(open('${OFFLINE_DIR}/${MODEL_SHORT}_offline_real.json')); print(f\"real={d.get('tokens_per_second', d.get('request_throughput', 0)):.2f}\")" 2>/dev/null || echo "?")
    E_TP=$(python3 -c "import json; d=json.load(open('${OFFLINE_DIR}/${MODEL_SHORT}_offline_emu.json')); print(f\"emu={d.get('tokens_per_second', d.get('request_throughput', 0)):.2f}\")" 2>/dev/null || echo "?")
    echo "  Throughput: $R_TP  $E_TP"
fi

echo ""
echo "ALL DONE. Results in: $RESULT_DIR"
