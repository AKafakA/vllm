#!/bin/bash
# Phase 5 driver — CloudLab cpu_host (no GPU) emu-only validation.
#
# Pairs against the RTX 8000 M2-Main real bench (must exist before this
# launches). The cpu_host runs ONLY the emu side — there's no GPU here,
# so no real bench can run. Pass criteria: TPOT/ITL/E2E mean ≤ 6% vs
# RTX 8000 M2 real (delta computed on the openclaw side or via copy of
# real_r*.json).
#
# Doubles as the e2e install.sh test — must boot, no manual intervention.
#
# Required:
#   - install.sh and tools/setup_cuda_stubs.sh have been run
#   - ~/cuda_stubs/ contains libcuda.so.1 + libcudart.so.12 (set by stubs)
#   - RTX 8000 M2 profile pack copied to ./results/RTX-8000-adaptive-apr26-m2-main/
#     (or apr25-m2-main-full as fallback)
#
# Usage:
#   nohup bash tools/orchestrator_cpu_host_apr26.sh \
#         > /tmp/vllm_orchestrator_cpu_host_apr26.log 2>&1 &

set -uo pipefail
ulimit -n 65536 2>/dev/null || true
export PATH="$HOME/.local/bin:$PATH"

# Best-effort venv detection — cpu_host repos may live at different paths.
for venv in ~/emu-venv ~/vllm-emulator-release/.venv ~/Code/llm/vllm-emulator/.venv; do
    if [ -f "$venv/bin/activate" ]; then
        # shellcheck disable=SC1090
        source "$venv/bin/activate"
        VENV_USED="$venv"
        break
    fi
done
: "${VENV_USED:?no venv found at ~/emu-venv ~/vllm-emulator-release/.venv ~/Code/llm/vllm-emulator/.venv}"

# Repo root — cpu_host has the release tree at one of two paths.
for repo in ~/vllm-emulator-release ~/Code/llm/vllm-emulator; do
    if [ -f "$repo/install.sh" ] || [ -d "$repo/vllm_emulator" ]; then
        REPO_ROOT="$repo"
        break
    fi
done
: "${REPO_ROOT:?no vllm-emulator repo found}"
cd "$REPO_ROOT"

LOG="/tmp/vllm_orchestrator_cpu_host_apr26.log"
echo "=== orchestrator_cpu_host_apr26 start $(date -u) ===" > "$LOG"
echo "REPO_ROOT=$REPO_ROOT  VENV=$VENV_USED" >> "$LOG"

# Verify cuda stubs (required for Mode B).
STUB_DIR="${STUB_DIR:-$HOME/cuda_stubs}"
if [ ! -e "$STUB_DIR/libcuda.so.1" ]; then
    echo "FATAL: $STUB_DIR/libcuda.so.1 missing. Run: bash tools/setup_cuda_stubs.sh" >> "$LOG"
    exit 1
fi
echo "STUB_DIR=$STUB_DIR ($(ls "$STUB_DIR" | wc -l) entries)" >> "$LOG"

# Profile pack (RTX 8000 M2). Try apr26 first (fresh from Phase 3 Cell 1),
# fall back to apr25 if Phase 3 hasn't progressed yet.
PROFILE=""
for cand in \
    "./results/RTX-8000-adaptive-apr26-m2-main/serving-full.json" \
    "./results/RTX-8000-adaptive-apr25-m2-main-full/serving-full.json"; do
    if [ -f "$cand" ]; then
        PROFILE="$cand"; break
    fi
done
: "${PROFILE:?no RTX 8000 M2 profile pack found locally}"
echo "PROFILE=$PROFILE" >> "$LOG"

OUT="./results/apr26-cpu-host-m2"
mkdir -p "$OUT"
DELTA_CSV="$OUT/per_rate_deltas.csv"
echo "rate,ttft_mean_pct,tpot_mean_pct,itl_mean_pct,e2e_mean_pct,tput_pct" > "$DELTA_CSV"

RATES=(2 4 8 16 32)
PROMPTS=2000
SHAREGPT="./results/sharegpt_full_4096.json"

run_emu_v4() {
    local R="$1"
    local SRV="$OUT/server_emu_r${R}.log"
    pkill -9 -f vllm.entrypoints 2>/dev/null; sleep 3
    env LD_LIBRARY_PATH="$STUB_DIR:${LD_LIBRARY_PATH:-}" \
        CUDA_VISIBLE_DEVICES="" \
        VLLM_EMULATOR_ENABLE_ORACLE=1 \
        VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
        VLLM_EMULATOR_MOCK_CUDA=1 \
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
        --model Qwen/Qwen3-8B --max-model-len 4096 --port 8100 \
        --trust-remote-code \
        > "$SRV" 2>&1 &

    # Wait for server.
    for i in $(seq 1 600); do
        curl -s http://localhost:8100/health >/dev/null 2>&1 && break
        sleep 1
    done
    if ! curl -s http://localhost:8100/health >/dev/null; then
        echo "  r=$R server FAIL" >> "$LOG"
        return 1
    fi

    echo "[$(date -u +%T)] emu r=$R start" >> "$LOG"
    timeout 1800 python3 -m vllm.entrypoints.cli.main bench serve \
        --model Qwen/Qwen3-8B --base-url http://localhost:8100 \
        --dataset-name sharegpt --dataset-path "$SHAREGPT" \
        --num-prompts "$PROMPTS" --request-rate "$R" --seed 0 \
        --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
        --save-result --result-dir "$OUT" --result-filename "emu_r${R}.json" \
        > "$OUT/bench_emu_r${R}.log" 2>&1 || true
    pkill -9 -f vllm.entrypoints 2>/dev/null; sleep 5
    echo "[$(date -u +%T)] emu r=$R done" >> "$LOG"
}

for R in "${RATES[@]}"; do
    run_emu_v4 "$R"
done

# Witness assertion: confirm v4 path fired (no kernels dispatched).
HOOK_COUNT=$(grep -c '\[HookDebug\]' "$OUT"/server_emu_r*.log 2>/dev/null | awk -F: '{s+=$2}END{print s+0}')
ORACLE_COUNT=$(grep -c '\[OracleDebug\]' "$OUT"/server_emu_r*.log 2>/dev/null | awk -F: '{s+=$2}END{print s+0}')
MOCK_LINE=$(grep -m1 'EmulatorCudaMock' "$OUT"/server_emu_r*.log 2>/dev/null | head -1)
echo "" >> "$LOG"
echo "WITNESS: HookDebug=$HOOK_COUNT OracleDebug=$ORACLE_COUNT" >> "$LOG"
echo "MOCK_LINE: $MOCK_LINE" >> "$LOG"

# Deltas vs RTX 8000 M2 real (paired by rate).
echo "" >> "$LOG"
echo "=== deltas vs RTX 8000 M2 real ===" >> "$LOG"
REAL_DIR=""
for cand in "./results/apr26-m2-main" "./results/apr25-m2-main-full"; do
    if [ -f "$cand/real_r4.json" ]; then
        REAL_DIR="$cand"; break
    fi
done
if [ -z "$REAL_DIR" ]; then
    echo "WARN: no RTX 8000 M2 real baseline found locally — copy real_r*.json before computing deltas" >> "$LOG"
else
    for R in "${RATES[@]}"; do
        python3 - "$R" "$REAL_DIR" "$OUT" "$DELTA_CSV" <<'PY' >> "$LOG" 2>&1
import json, sys
R = int(sys.argv[1]); real_dir = sys.argv[2]; out = sys.argv[3]; delta_csv = sys.argv[4]
try:
    r = json.load(open(f"{real_dir}/real_r{R}.json"))
    e = json.load(open(f"{out}/emu_r{R}.json"))
    pct = lambda a, b: (b/a-1)*100 if a > 0 else 0.0
    d_ttft = pct(r["mean_ttft_ms"], e["mean_ttft_ms"])
    d_tpot = pct(r["mean_tpot_ms"], e["mean_tpot_ms"])
    d_itl  = pct(r["mean_itl_ms"],  e["mean_itl_ms"])
    d_e2e  = pct(r.get("mean_e2el_ms", 0), e.get("mean_e2el_ms", 0))
    d_tput = pct(r["output_throughput"], e["output_throughput"])
    print(f"r={R:>2} | TTFT_mean {d_ttft:+7.2f}% | TPOT_mean {d_tpot:+6.2f}% | ITL_mean {d_itl:+6.2f}% | E2E_mean {d_e2e:+6.2f}% | tput {d_tput:+5.2f}%")
    with open(delta_csv, "a") as f:
        f.write(f"{R},{d_ttft:.4f},{d_tpot:.4f},{d_itl:.4f},{d_e2e:.4f},{d_tput:.4f}\n")
except Exception as ex:
    print(f"r={R}: ERROR {ex}")
PY
    done
fi

echo "=== orchestrator_cpu_host_apr26 DONE $(date -u) ===" >> "$LOG"
touch /tmp/vllm_orchestrator_cpu_host_apr26.done
