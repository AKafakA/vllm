#!/bin/bash
# Self-chaining baseline experiments: burstiness + sharegpt + combined.
# Waits for archive-r2-full to finish, then runs each experiment (real + emu)
# sequentially. Total ~3h.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

MODEL="Qwen/Qwen3-8B"
PORT=8100
PROFILE="./results/RTX-8000-adaptive-archive-5r/serving-r2.json"
MASTER_LOG="/tmp/vllm_baseline_exp.log"
echo "=== baseline-exp chain start $(date) ===" > "$MASTER_LOG"
touch /tmp/vllm_baseline_exp.started

cleanup() {
    pkill -9 -f "VLLM::EngineCore" 2>/dev/null
    pkill -9 -f "vllm.entrypoints" 2>/dev/null
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

preflight() {
    if pgrep -f "VLLM::EngineCore|bench serve|vllm.entrypoints" > /dev/null; then
        echo "FATAL: vllm/bench processes survived cleanup. Another script active?" >> "$MASTER_LOG"
        ps aux | grep -E "vllm|bench serve" | grep -v grep >> "$MASTER_LOG"
        exit 1
    fi
}

start_real() {
    cleanup
    preflight
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
        > "$1/server.log" 2>&1 &
    wait_server || return 1
}
start_emu() {
    cleanup
    preflight
    env \
        VLLM_EMULATOR_ENABLE_ORACLE=1 \
        VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
        VLLM_EMULATOR_MODE=realtime \
        VLLM_EMULATOR_EXECUTOR_HOOK=1 \
        VLLM_EMULATOR_PREP_SURROGATE=1 \
        VLLM_EMULATOR_SAMPLE_TRIM="2,98" \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
        > "$1/server.log" 2>&1 &
    wait_server || return 1
}

# Wait for archive-r2-full
echo "  [$(date +%T)] waiting for /tmp/vllm_archive_r2_full.done" >> "$MASTER_LOG"
for i in $(seq 1 120); do
    [ -f /tmp/vllm_archive_r2_full.done ] && break
    sleep 30
done
echo "  [$(date +%T)] archive-r2-full done, proceeding" >> "$MASTER_LOG"

# ===== Exp 0: Download + filter sharegpt =====
echo "" >> "$MASTER_LOG"
echo "=== Exp 0: sharegpt download+filter $(date) ===" >> "$MASTER_LOG"
if [ ! -f "./results/sharegpt_filtered_256_128.json" ]; then
    python3 tools/download_filter_sharegpt.py >> "$MASTER_LOG" 2>&1 || {
        echo "  sharegpt prep failed; Exp 2/3 will be skipped" >> "$MASTER_LOG"
    }
fi
SHAREGPT_PATH="./results/sharegpt_filtered_256_128.json"

# ===== Exp 1: Burstiness sweep at rate=4, 256/128, burstiness={0.3, 1.0, 3.0} =====
echo "" >> "$MASTER_LOG"
echo "=== Exp 1: burstiness sweep $(date) ===" >> "$MASTER_LOG"
for BURST in 0.3 1.0 3.0; do
    for MODE in real emu; do
        DIR="./results/exp1-burst${BURST}-${MODE}"
        mkdir -p "$DIR"
        echo "  [$(date +%T)] exp1 burst=$BURST mode=$MODE" >> "$MASTER_LOG"
        if [ "$MODE" = "real" ]; then start_real "$DIR"; else start_emu "$DIR"; fi
        # warmup
        python3 -m vllm.entrypoints.cli.main bench serve \
            --model "$MODEL" --base-url "http://localhost:${PORT}" \
            --dataset-name random --random-input-len 256 --random-output-len 128 \
            --num-prompts 200 --request-rate 4 --burstiness $BURST > /dev/null 2>&1 || true
        sleep 3
        # main bench
        python3 -m vllm.entrypoints.cli.main bench serve \
            --model "$MODEL" --base-url "http://localhost:${PORT}" \
            --dataset-name random --random-input-len 256 --random-output-len 128 \
            --num-prompts 2000 --request-rate 4 --burstiness $BURST \
            --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
            --save-result --result-dir "$DIR" \
            --result-filename "r4_${MODE}.json" > /dev/null 2>&1 || true
        cleanup
    done
done
touch /tmp/vllm_exp1_burst.done
echo "  [$(date +%T)] Exp 1 done" >> "$MASTER_LOG"

# ===== Exp 2: Sharegpt filtered (≤256/≤128) at rates 2/8/32 =====
if [ -f "$SHAREGPT_PATH" ]; then
    echo "" >> "$MASTER_LOG"
    echo "=== Exp 2: sharegpt filtered $(date) ===" >> "$MASTER_LOG"
    for MODE in real emu; do
        DIR="./results/exp2-sharegpt-${MODE}"
        mkdir -p "$DIR"
        if [ "$MODE" = "real" ]; then start_real "$DIR"; else start_emu "$DIR"; fi
        # warmup (random)
        python3 -m vllm.entrypoints.cli.main bench serve \
            --model "$MODEL" --base-url "http://localhost:${PORT}" \
            --dataset-name random --random-input-len 256 --random-output-len 128 \
            --num-prompts 200 --request-rate 4 > /dev/null 2>&1 || true
        sleep 3
        for RATE in 2 8 32; do
            echo "  [$(date +%T)] exp2 sharegpt mode=$MODE rate=$RATE" >> "$MASTER_LOG"
            python3 -m vllm.entrypoints.cli.main bench serve \
                --model "$MODEL" --base-url "http://localhost:${PORT}" \
                --dataset-name sharegpt --dataset-path "$SHAREGPT_PATH" \
                --num-prompts 2000 --request-rate $RATE \
                --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
                --save-result --result-dir "$DIR" \
                --result-filename "r${RATE}_${MODE}.json" > /dev/null 2>&1 || true
        done
        cleanup
    done
    touch /tmp/vllm_exp2_sharegpt.done
    echo "  [$(date +%T)] Exp 2 done" >> "$MASTER_LOG"
else
    echo "  Exp 2 SKIPPED (no sharegpt file)" >> "$MASTER_LOG"
fi

# ===== Exp 3: Combined — sharegpt + burstiness=0.3 at rate=4 =====
if [ -f "$SHAREGPT_PATH" ]; then
    echo "" >> "$MASTER_LOG"
    echo "=== Exp 3: sharegpt + burstiness=0.3 $(date) ===" >> "$MASTER_LOG"
    for MODE in real emu; do
        DIR="./results/exp3-combined-${MODE}"
        mkdir -p "$DIR"
        if [ "$MODE" = "real" ]; then start_real "$DIR"; else start_emu "$DIR"; fi
        python3 -m vllm.entrypoints.cli.main bench serve \
            --model "$MODEL" --base-url "http://localhost:${PORT}" \
            --dataset-name random --random-input-len 256 --random-output-len 128 \
            --num-prompts 200 --request-rate 4 > /dev/null 2>&1 || true
        sleep 3
        echo "  [$(date +%T)] exp3 mode=$MODE" >> "$MASTER_LOG"
        python3 -m vllm.entrypoints.cli.main bench serve \
            --model "$MODEL" --base-url "http://localhost:${PORT}" \
            --dataset-name sharegpt --dataset-path "$SHAREGPT_PATH" \
            --num-prompts 2000 --request-rate 4 --burstiness 0.3 \
            --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
            --save-result --result-dir "$DIR" \
            --result-filename "r4_${MODE}.json" > /dev/null 2>&1 || true
        cleanup
    done
    touch /tmp/vllm_exp3_combined.done
    echo "  [$(date +%T)] Exp 3 done" >> "$MASTER_LOG"
fi

echo "" >> "$MASTER_LOG"
echo "=== baseline-exp chain DONE $(date) ===" >> "$MASTER_LOG"
touch /tmp/vllm_baseline_exp.done
