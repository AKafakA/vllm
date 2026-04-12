#!/bin/bash
# Rate sweep with FILTERED adaptive profile + 2d oracle + DISABLE_DEFER_ADD.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

LOG=/tmp/vllm_filtered_sweep.log
echo "=== Filtered profile rate sweep at $(date) ===" > "$LOG"

MODEL="Qwen/Qwen3-8B"
PORT=8100
NP=200
PROFILE="./results/RTX-8000-adaptive/profiles/serving-Qwen3-8B-adaptive-filtered.json"
RESULT_DIR="./results/RTX-8000-adaptive/filtered-sweep"
mkdir -p "$RESULT_DIR"

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

run_rate() {
    local RATE=$1
    local TAG="r${RATE}"
    echo "" >> "$LOG"
    echo "=== Rate=$RATE ===" >> "$LOG"

    # REAL
    cleanup
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
        > "$RESULT_DIR/${TAG}_real.log" 2>&1 &
    wait_server
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "$MODEL" --base-url "http://localhost:${PORT}" \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 50 --request-rate 4 > /dev/null 2>&1
    sleep 2
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "$MODEL" --base-url "http://localhost:${PORT}" \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts $NP --request-rate $RATE \
        --percentile-metrics ttft,tpot,itl --metric-percentiles 50,90,99 \
        --save-result --result-dir "$RESULT_DIR" \
        --result-filename "${TAG}_real.json" > /dev/null 2>&1
    cleanup

    # EMU — proven setup: filtered profile + 2d oracle + DISABLE_DEFER_ADD
    VLLM_EMULATOR_ENABLE_ORACLE=1 \
    VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
    VLLM_EMULATOR_MODE=realtime \
    VLLM_EMULATOR_EXECUTOR_HOOK=1 \
    VLLM_EMULATOR_PREP_SURROGATE=1 \
    VLLM_EMULATOR_ORACLE_MODE=2d \
    VLLM_EMULATOR_DISABLE_DEFER_ADD=1 \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
        > "$RESULT_DIR/${TAG}_emu.log" 2>&1 &
    wait_server
    grep "ExecutorEmulatorHook" "$RESULT_DIR/${TAG}_emu.log" | head -1 >> "$LOG"
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "$MODEL" --base-url "http://localhost:${PORT}" \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 50 --request-rate 4 > /dev/null 2>&1
    sleep 2
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "$MODEL" --base-url "http://localhost:${PORT}" \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts $NP --request-rate $RATE \
        --percentile-metrics ttft,tpot,itl --metric-percentiles 50,90,99 \
        --save-result --result-dir "$RESULT_DIR" \
        --result-filename "${TAG}_emu.json" > /dev/null 2>&1
    cleanup

    python3 -c "
import json
r = json.load(open('$RESULT_DIR/${TAG}_real.json'))
e = json.load(open('$RESULT_DIR/${TAG}_emu.json'))
def err(k): rv,ev = r.get(k,0),e.get(k,0); return (ev-rv)/rv*100 if rv else 0
print(f'  TPOT: real={r.get(\"mean_tpot_ms\",0):.1f}ms, emu={e.get(\"mean_tpot_ms\",0):.1f}ms, err={err(\"mean_tpot_ms\"):+.1f}%')
print(f'  TTFT: real={r.get(\"mean_ttft_ms\",0):.1f}ms, emu={e.get(\"mean_ttft_ms\",0):.1f}ms, err={err(\"mean_ttft_ms\"):+.1f}%')
print(f'  P99_TPOT: real={r.get(\"p99_tpot_ms\",0):.1f}ms, emu={e.get(\"p99_tpot_ms\",0):.1f}ms, err={err(\"p99_tpot_ms\"):+.1f}%')
print(f'  tok/s: real={r.get(\"output_throughput\",0):.1f}, emu={e.get(\"output_throughput\",0):.1f}, err={err(\"output_throughput\"):+.1f}%')
print(f'  max_conc: real={r.get(\"max_concurrent_requests\",0)}, emu={e.get(\"max_concurrent_requests\",0)}')
" >> "$LOG" 2>&1
}

for rate in 1 2 4 8; do
    run_rate $rate
done

echo "" >> "$LOG"
echo "=== SUMMARY: Filtered profile + 2d oracle ===" >> "$LOG"
python3 -c "
import json
print(f'{\"Rate\":>5} {\"RealTPOT\":>10} {\"EmuTPOT\":>10} {\"Gap%\":>8} {\"Status\":>8}')
for rate in [1, 2, 4, 8]:
    try:
        r = json.load(open(f'$RESULT_DIR/r{rate}_real.json'))
        e = json.load(open(f'$RESULT_DIR/r{rate}_emu.json'))
        rt = r.get('mean_tpot_ms', 0); et = e.get('mean_tpot_ms', 0)
        gp = (et-rt)/rt*100 if rt else 0
        status = 'PASS' if abs(gp) < 5 else 'FAIL'
        print(f'{rate:>5} {rt:>10.1f} {et:>10.1f} {gp:>+7.1f}% {status:>8}')
    except FileNotFoundError:
        pass
" >> "$LOG" 2>&1

echo "=== Done at $(date) ===" >> "$LOG"
