#!/bin/bash
# Quick check r=4 + r=32 (only those two rates, ~30 min) for profile-reuse
# decision. Mirrors run_one_full_sharegpt_cell.sh but only at the two rates.
#
# Pass criteria: TPOT/ITL/E2E mean ≤ 6% at BOTH rates → REUSE profile.
# Otherwise reprofile.
#
# Required env (from caller):
#   CELL_TAG, BENCH_MODEL, REUSE_PROFILE
# Optional:
#   EXTRA_SERVER_ARGS, RATES_OVERRIDE (default "4 32")

set -uo pipefail
ulimit -n 65536 2>/dev/null || true
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator
source tools/_bench_common.sh

: "${CELL_TAG:?required}"
: "${BENCH_MODEL:?required}"
: "${REUSE_PROFILE:?required}"
EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS:-}"
RATES_OVERRIDE="${RATES_OVERRIDE:-4 32}"
STUB_DIR="${STUB_DIR:-$HOME/cuda_stubs}"

if echo "$EXTRA_SERVER_ARGS" | grep -qE '\-\-max-num-seqs'; then
    echo "FATAL: --max-num-seqs in EXTRA_SERVER_ARGS forbidden" >&2
    exit 1
fi

OUT="./results/${CELL_TAG}"
mkdir -p "$OUT"
LOG="$OUT/run.log"
echo "=== quickcheck $CELL_TAG start $(date -u) ===" > "$LOG"
echo "REUSE_PROFILE=$REUSE_PROFILE" >> "$LOG"
echo "RATES=$RATES_OVERRIDE" >> "$LOG"

DELTA_CSV="$OUT/per_rate_deltas.csv"
echo "rate,ttft_mean_pct,tpot_mean_pct,itl_mean_pct,e2e_mean_pct,tput_pct" > "$DELTA_CSV"

run_real() {
    local R="$1"
    common_cleanup; sleep 2
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$BENCH_MODEL" --max-model-len 4096 --port "$BENCH_PORT" \
        --trust-remote-code $EXTRA_SERVER_ARGS \
        > "$OUT/server_real_r${R}.log" 2>&1 &
    common_wait_server || { echo "real r=$R server FAIL" >> "$LOG"; return 1; }
    common_warmup
    echo "[$(date -u +%T)] real r=$R start" >> "$LOG"
    timeout 1800 python3 -m vllm.entrypoints.cli.main bench serve \
        --model "$BENCH_MODEL" --base-url "http://localhost:$BENCH_PORT" \
        --dataset-name sharegpt --dataset-path ./results/sharegpt_full_4096.json \
        --num-prompts 2000 --request-rate "$R" --seed 0 \
        --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
        --save-result --result-dir "$OUT" --result-filename "real_r${R}.json" \
        > "$OUT/bench_real_r${R}.log" 2>&1 || true
    common_cleanup
}

run_emu_v4() {
    local R="$1"
    common_cleanup; sleep 2
    env LD_LIBRARY_PATH="$STUB_DIR:${LD_LIBRARY_PATH:-}" \
        CUDA_VISIBLE_DEVICES="" \
        VLLM_EMULATOR_ENABLE_ORACLE=1 \
        VLLM_EMULATOR_PROFILE_PACK="$REUSE_PROFILE" \
        VLLM_EMULATOR_MODE=realtime \
        VLLM_EMULATOR_EXECUTOR_HOOK=1 \
        VLLM_EMULATOR_SCHEDULER_HOOK=0 \
        VLLM_EMULATOR_IPC_POSITION=disabled \
        VLLM_EMULATOR_PREP_SURROGATE=0 \
        VLLM_EMULATOR_ORACLE_AGG=sample \
        VLLM_EMULATOR_ORACLE_K=auto \
        VLLM_EMULATOR_ORACLE_MIN_SAMPLES=30 \
        VLLM_EMULATOR_BW_SLOPE_SOURCE=disabled \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$BENCH_MODEL" --max-model-len 4096 --port "$BENCH_PORT" \
        --trust-remote-code $EXTRA_SERVER_ARGS \
        > "$OUT/server_emu_r${R}.log" 2>&1 &
    common_wait_server || { echo "emu r=$R server FAIL" >> "$LOG"; return 1; }
    common_warmup
    echo "[$(date -u +%T)] emu r=$R start" >> "$LOG"
    timeout 1800 python3 -m vllm.entrypoints.cli.main bench serve \
        --model "$BENCH_MODEL" --base-url "http://localhost:$BENCH_PORT" \
        --dataset-name sharegpt --dataset-path ./results/sharegpt_full_4096.json \
        --num-prompts 2000 --request-rate "$R" --seed 0 \
        --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
        --save-result --result-dir "$OUT" --result-filename "emu_r${R}.json" \
        > "$OUT/bench_emu_r${R}.log" 2>&1 || true
    common_cleanup
}

for R in $RATES_OVERRIDE; do
    run_real "$R"
    run_emu_v4 "$R"
done

# Compute deltas
for R in $RATES_OVERRIDE; do
    python3 - "$R" "$OUT" "$DELTA_CSV" <<'PY' >> "$LOG" 2>&1
import json, sys
R = int(sys.argv[1]); out = sys.argv[2]; delta_csv = sys.argv[3]
try:
    r = json.load(open(f"{out}/real_r{R}.json"))
    e = json.load(open(f"{out}/emu_r{R}.json"))
    pct = lambda a, b: (b/a-1)*100 if a > 0 else 0.0
    d_ttft = pct(r["mean_ttft_ms"], e["mean_ttft_ms"])
    d_tpot = pct(r["mean_tpot_ms"], e["mean_tpot_ms"])
    d_itl  = pct(r["mean_itl_ms"],  e["mean_itl_ms"])
    d_e2e  = pct(r.get("mean_e2el_ms", 0), e.get("mean_e2el_ms", 0))
    d_tput = pct(r["output_throughput"], e["output_throughput"])
    print(f"r={R:>2} | TTFT_mean {d_ttft:+7.2f}% | TPOT_mean {d_tpot:+6.2f}% | "
          f"ITL_mean {d_itl:+6.2f}% | E2E_mean {d_e2e:+6.2f}% | tput {d_tput:+5.2f}%")
    with open(delta_csv, "a") as f:
        f.write(f"{R},{d_ttft:.4f},{d_tpot:.4f},{d_itl:.4f},{d_e2e:.4f},{d_tput:.4f}\n")
except Exception as ex:
    print(f"r={R}: ERROR {ex}")
PY
done

echo "[$(date -u +%T)] quickcheck DONE" >> "$LOG"
