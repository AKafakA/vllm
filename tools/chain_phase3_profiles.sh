#!/bin/bash
# Phase 3 — Profile-shape investigation.
#   3.1: Build 3 profiles (fixedmix, shareptsampled, archiver2ext).
#   3.2: Capture ShareGPT real baseline (5 rates × 2000p).
#   3.3: 8-cell emu matrix (4 profiles × 2 workloads, 3 rates each).
#
# Gates on Phase 2 completion.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

MASTER_LOG="/tmp/vllm_apr20_phase3.log"
echo "=== phase3 profile-shape start $(date) ===" > "$MASTER_LOG"
touch /tmp/vllm_apr20_phase3.started

# Gate on Phase 2 done.
for i in $(seq 1 180); do
    [ -f /tmp/vllm_apr20_phase2.done ] && break
    sleep 30
done
if [ ! -f /tmp/vllm_apr20_phase2.done ]; then
    echo "Phase 2 not done in time; aborting" >> "$MASTER_LOG"
    exit 1
fi

MODEL="Qwen/Qwen3-8B"
PORT=8100
SHAREGPT="./results/sharegpt_filtered_256_128.json"
ARCHIVE_PROFILE="./results/RTX-8000-adaptive-archive-5r/serving-r2.json"
V5_PASSED_MARKER="/tmp/vllm_v5_passed"

cleanup() {
    pkill -9 -f "VLLM::EngineCore" 2>/dev/null
    pkill -9 -f "vllm.entrypoints" 2>/dev/null
    pkill -9 -f "bench serve" 2>/dev/null
    fuser ${PORT}/tcp 2>/dev/null | xargs -r kill -9 2>/dev/null
    sleep 5
}
wait_server() {
    for i in $(seq 1 300); do
        curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1 && return 0
        sleep 1
    done
    return 1
}

# === Phase 3.1: Build 3 candidate profiles sequentially ===
echo "  [$(date +%T)] 3.1: building profile fixedmix-2r" >> "$MASTER_LOG"
bash tools/adaptive_profile_fixedmix_2r.sh 2>&1 | tail -5 >> "$MASTER_LOG"

echo "  [$(date +%T)] 3.1: building profile shareptsampled-2r" >> "$MASTER_LOG"
bash tools/adaptive_profile_shareptsampled_2r.sh 2>&1 | tail -5 >> "$MASTER_LOG"

echo "  [$(date +%T)] 3.1: building profile archiver2ext-2r" >> "$MASTER_LOG"
bash tools/adaptive_profile_archiver2ext_2r.sh 2>&1 | tail -5 >> "$MASTER_LOG"

# === Phase 3.2: Capture ShareGPT real baseline (5 rates × 2000p) ===
REAL_SHAREGPT_DIR="./results/workload-real-sharegpt"
mkdir -p "$REAL_SHAREGPT_DIR"
cleanup
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > "$REAL_SHAREGPT_DIR/server.log" 2>&1 &
wait_server || { echo "REAL_SERVER_FAIL" >> "$MASTER_LOG"; cleanup; }

python3 -m vllm.entrypoints.cli.main bench serve \
    --model "$MODEL" --base-url "http://localhost:${PORT}" \
    --dataset-name sharegpt --dataset-path "$SHAREGPT" \
    --num-prompts 200 --request-rate 4 > /dev/null 2>&1 || true
sleep 3

for RATE in 2 4 8 16 32; do
    echo "  [$(date +%T)] real sharegpt r=$RATE" >> "$MASTER_LOG"
    timeout 1500 python3 -m vllm.entrypoints.cli.main bench serve \
        --model "$MODEL" --base-url "http://localhost:${PORT}" \
        --dataset-name sharegpt --dataset-path "$SHAREGPT" \
        --num-prompts 2000 --request-rate $RATE \
        --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
        --save-result --result-dir "$REAL_SHAREGPT_DIR" \
        --result-filename "r${RATE}_real.json" > /dev/null 2>&1 \
        && echo "    r=${RATE} done" >> "$MASTER_LOG" \
        || echo "    r=${RATE} FAIL" >> "$MASTER_LOG"
    sleep 2
done
cleanup

# === Phase 3.3: 8-cell emu matrix (3 rates per cell: 2, 8, 32) ===
# Determine hook config: v5-sample if Phase 2 passed, else v3 (median).
if [ -f "$V5_PASSED_MARKER" ]; then
    HOOK_AGG=sample
    HOOK_NAME=v5-sample
else
    HOOK_AGG=median
    HOOK_NAME=v3-median
fi
echo "  [$(date +%T)] 3.3: emu matrix with hook=$HOOK_NAME" >> "$MASTER_LOG"

declare -A PROFILES
PROFILES[archive-r2]="./results/RTX-8000-adaptive-archive-5r/serving-r2.json"
PROFILES[fixedmix]="./results/RTX-8000-profile-fixedmix-2r/serving-full.json"
PROFILES[shareptsampled]="./results/RTX-8000-profile-shareptsampled-2r/serving-full.json"
PROFILES[archiver2ext]="./results/RTX-8000-profile-archiver2ext-2r/serving-full.json"

for PROF_KEY in archive-r2 fixedmix shareptsampled archiver2ext; do
    PROF_PATH="${PROFILES[$PROF_KEY]}"
    if [ ! -f "$PROF_PATH" ]; then
        echo "    SKIP $PROF_KEY (missing profile)" >> "$MASTER_LOG"
        continue
    fi
    for WORKLOAD in random sharegpt; do
        CELL_DIR="./results/workload-emu-${PROF_KEY}-${WORKLOAD}"
        mkdir -p "$CELL_DIR"
        # Copy real baselines.
        if [ "$WORKLOAD" = "random" ]; then
            for R in 2 8 32; do
                cp "./results/ttft-variant-v3-arrival-delay/r${R}_real.json" \
                    "$CELL_DIR/r${R}_real.json" 2>/dev/null || true
            done
        else
            for R in 2 8 32; do
                cp "$REAL_SHAREGPT_DIR/r${R}_real.json" \
                    "$CELL_DIR/r${R}_real.json" 2>/dev/null || true
            done
        fi

        echo "  [$(date +%T)] cell: profile=$PROF_KEY workload=$WORKLOAD" >> "$MASTER_LOG"
        cleanup
        env \
            VLLM_EMULATOR_ENABLE_ORACLE=1 \
            VLLM_EMULATOR_PROFILE_PACK="$PROF_PATH" \
            VLLM_EMULATOR_MODE=realtime \
            VLLM_EMULATOR_EXECUTOR_HOOK=1 \
            VLLM_EMULATOR_SCHEDULER_HOOK=1 \
            VLLM_IPC_OVERHEAD_AGG="$HOOK_AGG" \
            VLLM_EMULATOR_PREP_SURROGATE=1 \
            VLLM_EMULATOR_SAMPLE_TRIM="2,98" \
        python3 -m vllm.entrypoints.openai.api_server \
            --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
            > "$CELL_DIR/server.log" 2>&1 &
        wait_server || { echo "    server fail $PROF_KEY $WORKLOAD" >> "$MASTER_LOG"; cleanup; continue; }

        # Warmup.
        if [ "$WORKLOAD" = "random" ]; then
            python3 -m vllm.entrypoints.cli.main bench serve \
                --model "$MODEL" --base-url "http://localhost:${PORT}" \
                --dataset-name random --random-input-len 256 --random-output-len 128 \
                --num-prompts 100 --request-rate 4 > /dev/null 2>&1 || true
        else
            python3 -m vllm.entrypoints.cli.main bench serve \
                --model "$MODEL" --base-url "http://localhost:${PORT}" \
                --dataset-name sharegpt --dataset-path "$SHAREGPT" \
                --num-prompts 100 --request-rate 4 > /dev/null 2>&1 || true
        fi
        sleep 2

        for RATE in 2 8 32; do
            if [ "$WORKLOAD" = "random" ]; then
                DS_ARGS="--dataset-name random --random-input-len 256 --random-output-len 128"
            else
                DS_ARGS="--dataset-name sharegpt --dataset-path $SHAREGPT"
            fi
            timeout 1500 python3 -m vllm.entrypoints.cli.main bench serve \
                --model "$MODEL" --base-url "http://localhost:${PORT}" \
                $DS_ARGS \
                --num-prompts 2000 --request-rate $RATE \
                --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
                --save-result --result-dir "$CELL_DIR" \
                --result-filename "r${RATE}_emu.json" > /dev/null 2>&1 \
                && echo "    r=${RATE} done" >> "$MASTER_LOG" \
                || echo "    r=${RATE} FAIL" >> "$MASTER_LOG"
            sleep 2
        done
        cleanup
    done
done

echo "" >> "$MASTER_LOG"
echo "=== phase3 DONE $(date) ===" >> "$MASTER_LOG"
touch /tmp/vllm_apr20_phase3.done
touch /tmp/vllm_apr20.done
