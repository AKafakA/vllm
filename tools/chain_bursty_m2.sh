#!/bin/bash
# Burstiness ablation: repeat M2 (Qwen3-8B / RTX 8000 / ShareGPT)
# with vLLM bench's --burstiness=0.25 (very bursty arrivals) for
# both real and emu sides. Reuses the existing M2 dense profile
# (captured under Poisson arrivals); tests whether the oracle
# generalises to an arrival pattern it was not profiled on.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

source "$(dirname "$0")/_bench_common.sh"

export BENCH_MODEL="Qwen/Qwen3-8B"
export BENCH_PORT=8100
export BENCH_MAX_MODEL_LEN=4096
export EXTRA_SERVER_ARGS=""

export VLLM_EMULATOR_SCHEDULER_HOOK=0
export VLLM_EMULATOR_IPC_POSITION=disabled
export VLLM_EMULATOR_PREP_SURROGATE=0
export VLLM_EMULATOR_ORACLE_AGG=sample

TAG="apr24-bursty025-m2"
RATES="${RATES:-2 4 8 16 32}"
BURSTINESS="0.25"
SHAREGPT="./results/sharegpt_filtered_256_128.json"
PROFILE="./results/RTX-8000-adaptive-apr23-sharegpt-dense/serving-full.json"
REAL_DIR="./results/bursty025-m2-5rate-real"
EMU_DIR="./results/validate-${TAG}"

MASTER_MARKER="/tmp/vllm_overnight_${TAG}"
MASTER_LOG="/tmp/vllm_overnight_${TAG}.log"
mkdir -p "$REAL_DIR" "$EMU_DIR"
touch "${MASTER_MARKER}.started"
echo "=== ${TAG} start $(date -u) ===" > "$MASTER_LOG"
echo "RATES=$RATES  BURSTINESS=$BURSTINESS  PROFILE=$PROFILE" >> "$MASTER_LOG"

if [ ! -f "$PROFILE" ]; then
    echo "PROFILE missing at $PROFILE" >> "$MASTER_LOG"
    touch "${MASTER_MARKER}.done"; exit 1
fi

# Stage 1: real baseline at bursty arrivals.
for RATE in $RATES; do
    echo "" >> "$MASTER_LOG"
    echo "[$(date +%T)] REAL r=$RATE burstiness=$BURSTINESS" >> "$MASTER_LOG"
    common_cleanup
    if ! common_preflight; then echo "  preflight FAIL" >> "$MASTER_LOG"; continue; fi

    python3 -m vllm.entrypoints.openai.api_server \
        --model "$BENCH_MODEL" --max-model-len "$BENCH_MAX_MODEL_LEN" \
        --port "$BENCH_PORT" --trust-remote-code $EXTRA_SERVER_ARGS \
        > "$REAL_DIR/server_real_r${RATE}.log" 2>&1 &
    if ! common_wait_server; then
        echo "  server FAIL" >> "$MASTER_LOG"; common_cleanup; continue
    fi
    common_warmup

    timeout 2400 python3 -m vllm.entrypoints.cli.main bench serve \
        --model "$BENCH_MODEL" --base-url "http://localhost:${BENCH_PORT}" \
        --dataset-name sharegpt --dataset-path "$SHAREGPT" \
        --num-prompts 2000 --request-rate "$RATE" \
        --burstiness "$BURSTINESS" --seed 0 \
        --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
        --save-result --result-dir "$REAL_DIR" \
        --result-filename "real_r${RATE}.json" > /dev/null 2>&1 || true
    common_cleanup
done

# Stage 2: emu validation at bursty arrivals, same profile.
for RATE in $RATES; do
    echo "" >> "$MASTER_LOG"
    echo "[$(date +%T)] EMU r=$RATE burstiness=$BURSTINESS" >> "$MASTER_LOG"
    common_cleanup
    if ! common_preflight; then echo "  preflight FAIL" >> "$MASTER_LOG"; continue; fi

    env \
        VLLM_EMULATOR_ENABLE_ORACLE=1 \
        VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
        VLLM_EMULATOR_MODE=realtime \
        VLLM_EMULATOR_EXECUTOR_HOOK=1 \
        VLLM_EMULATOR_SCHEDULER_HOOK=0 \
        VLLM_EMULATOR_IPC_POSITION=disabled \
        VLLM_EMULATOR_PREP_SURROGATE=0 \
        VLLM_EMULATOR_ORACLE_AGG=sample \
        VLLM_EMULATOR_ORACLE_K=auto \
        VLLM_EMULATOR_ORACLE_MIN_SAMPLES=30 \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$BENCH_MODEL" --max-model-len "$BENCH_MAX_MODEL_LEN" \
        --port "$BENCH_PORT" --trust-remote-code $EXTRA_SERVER_ARGS \
        > "$EMU_DIR/server_hookOn_r${RATE}.log" 2>&1 &
    if ! common_wait_server; then
        echo "  server FAIL" >> "$MASTER_LOG"; common_cleanup; continue
    fi
    common_warmup

    timeout 2400 python3 -m vllm.entrypoints.cli.main bench serve \
        --model "$BENCH_MODEL" --base-url "http://localhost:${BENCH_PORT}" \
        --dataset-name sharegpt --dataset-path "$SHAREGPT" \
        --num-prompts 2000 --request-rate "$RATE" \
        --burstiness "$BURSTINESS" --seed 0 \
        --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
        --save-result --result-dir "$EMU_DIR" \
        --result-filename "hookOn_r${RATE}.json" > /dev/null 2>&1 || true
    common_cleanup
done

echo "" >> "$MASTER_LOG"
echo "=== ${TAG} DONE $(date -u) ===" >> "$MASTER_LOG"
touch "${MASTER_MARKER}.done"
