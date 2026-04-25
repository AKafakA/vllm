#!/bin/bash
# Phase 3 driver — RTX 8000 cells, all at DEFAULT max-num-seqs, v4 emu.
#
# Cell pipeline (serial):
#   1. M2-Main         : REUSE existing profile (already DEFAULT) — re-bench Stages 1+3
#   2. R3 prefix-off   : QUICK CHECK r=4+r=32 → full re-bench OR reprofile+rebench
#   3. TRITON          : QUICK CHECK r=4+r=32 → full re-bench OR reprofile+rebench
#   4. Burstiness      : REUSE M2 profile — re-bench Stages 1+3
#   5. Qwen3-4B        : NEW — full 3 stages (clean Qwen3-8B from HF cache first)
#   6. Qwen3-14B       : NEW — full 3 stages
#   7. Llama-3.1-8B    : NEW — preflight HF token, then full 3 stages
#
# Each cell calls run_v4_cell.sh which itself calls run_one_full_sharegpt_cell.sh.
# parity_audit + per-rate MEAN deltas are computed inline by the cell runner.
# After each cell, this orchestrator runs correctness_cron.sh once to update
# parity_log.md + rerun_queue.md.
#
# Halt marker: each cell-runner polls /tmp/correctness_halt_<cell_tag>.flag
# (touched by correctness_cron.sh on parity violation) — currently advisory.
# Polling will be added in a future enhancement; for now violations land in
# rerun_queue.md and the orchestrator continues.
#
# Usage:
#   nohup bash tools/orchestrator_rtx8000_apr26.sh \
#         > /tmp/vllm_orchestrator_apr26.log 2>&1 &

set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

LOG="/tmp/vllm_orchestrator_apr26.log"
echo "=== orchestrator_apr26 start $(date -u) ===" > "$LOG"

# Cell-tag prefix for this batch.
TAG_PREFIX="apr26"

run_cell() {
    local cell_tag="$1" model="$2" extra="$3"
    local reuse="${4:-}"

    echo "" >> "$LOG"
    echo "[$(date -u +%T)] --- CELL: $cell_tag ($model) ---" >> "$LOG"
    echo "    EXTRA=$extra  REUSE=$reuse" >> "$LOG"

    if [ -n "$reuse" ]; then
        REUSE_PROFILE="$reuse" \
            bash tools/run_v4_cell.sh "$cell_tag" "$model" "$extra"
    else
        bash tools/run_v4_cell.sh "$cell_tag" "$model" "$extra"
    fi
    rc=$?
    echo "[$(date -u +%T)] CELL $cell_tag done (rc=$rc)" >> "$LOG"

    # Run correctness sweep once per cell.
    bash tools/correctness_cron.sh "./results/$cell_tag" >> "$LOG" 2>&1 || true
}

# Run a quick check (only r=4 + r=32, ~30 min) before deciding whether to
# reuse a profile. If quick check passes (<6% TPOT/ITL/E2E mean), reuse;
# otherwise reprofile.
quick_check_r4_r32() {
    local cell_tag="$1" model="$2" extra="$3" reuse="$4"
    local quick_tag="${cell_tag}-quickcheck"

    echo "" >> "$LOG"
    echo "[$(date -u +%T)] QUICK CHECK $cell_tag (r=4 + r=32 against $reuse)" >> "$LOG"

    # Generate quick-check script inline — only r=4 and r=32, 2000 prompts.
    REUSE_PROFILE="$reuse" CELL_TAG="$quick_tag" BENCH_MODEL="$model" \
    EXTRA_SERVER_ARGS="$extra" RATES_OVERRIDE="4 32" \
        bash tools/run_quickcheck_two_rates.sh
    rc=$?
    echo "[$(date -u +%T)] QUICK CHECK $cell_tag done (rc=$rc)" >> "$LOG"

    # Read deltas and decide.
    python3 - "$quick_tag" <<'PY'
import sys, csv
from pathlib import Path
tag = sys.argv[1]
csv_path = Path(f"./results/{tag}/per_rate_deltas.csv")
if not csv_path.exists():
    print("DECISION: reprofile (no delta CSV — quick check failed to complete)")
    sys.exit(1)
worst = 0.0
with open(csv_path) as f:
    for row in csv.DictReader(f):
        for k in ("tpot_mean_pct", "itl_mean_pct", "e2e_mean_pct"):
            v = abs(float(row[k]))
            if v > worst:
                worst = v
if worst <= 6.0:
    print(f"DECISION: REUSE (worst mean delta {worst:.2f}% <= 6%)")
    sys.exit(0)
else:
    print(f"DECISION: REPROFILE (worst mean delta {worst:.2f}% > 6%)")
    sys.exit(2)
PY
    return $?
}

# ---------------- 1. M2-Main (reuse existing profile) ----------------
M2_PROFILE="./results/RTX-8000-adaptive-apr25-m2-main-full/serving-full.json"
if [ -f "$M2_PROFILE" ]; then
    run_cell "${TAG_PREFIX}-m2-main" Qwen/Qwen3-8B "" "$M2_PROFILE"
else
    echo "WARN: M2 profile missing at $M2_PROFILE — running full 3-stage" >> "$LOG"
    run_cell "${TAG_PREFIX}-m2-main" Qwen/Qwen3-8B ""
fi
NEW_M2_PROFILE="./results/RTX-8000-adaptive-${TAG_PREFIX}-m2-main/serving-full.json"
[ -f "$NEW_M2_PROFILE" ] && M2_PROFILE="$NEW_M2_PROFILE"

# ---------------- 2. R3 prefix-cache OFF ----------------
R3_PROFILE="./results/RTX-8000-adaptive-apr25-r3-prefix-off/serving-full.json"
R3_REUSE=""
if [ -f "$R3_PROFILE" ]; then
    if quick_check_r4_r32 "${TAG_PREFIX}-r3-prefix-off" Qwen/Qwen3-8B "--no-prefix-caching" "$R3_PROFILE"; then
        R3_REUSE="$R3_PROFILE"
    fi
fi
run_cell "${TAG_PREFIX}-r3-prefix-off" Qwen/Qwen3-8B "--no-prefix-caching" "$R3_REUSE"

# ---------------- 3. TRITON ----------------
TRITON_PROFILE="./results/RTX-8000-adaptive-apr25-triton/serving-full.json"
TRITON_REUSE=""
if [ -f "$TRITON_PROFILE" ]; then
    if quick_check_r4_r32 "${TAG_PREFIX}-triton" Qwen/Qwen3-8B "--attention-backend TRITON_ATTN" "$TRITON_PROFILE"; then
        TRITON_REUSE="$TRITON_PROFILE"
    fi
fi
run_cell "${TAG_PREFIX}-triton" Qwen/Qwen3-8B "--attention-backend TRITON_ATTN" "$TRITON_REUSE"

# ---------------- 4. Burstiness γ=0.25 (reuse M2 profile) ----------------
# Burstiness is set via bench --burstiness flag, not server args. We pass it
# through the bench loop; for now, reuse M2 profile + standard bench.
# TODO: extend run_v4_cell.sh to accept --burstiness.
run_cell "${TAG_PREFIX}-burstiness-g25" Qwen/Qwen3-8B "" "$M2_PROFILE"

# ---------------- 5. Qwen3-4B (NEW; clean HF cache first) ----------------
echo "" >> "$LOG"
echo "[$(date -u +%T)] cleaning HF cache of Qwen3-8B before Qwen3-4B" >> "$LOG"
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen3-8B 2>/dev/null
run_cell "${TAG_PREFIX}-m1b-qwen3-4b" Qwen/Qwen3-4B ""

# ---------------- 6. Qwen3-14B (NEW) ----------------
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen3-4B 2>/dev/null
run_cell "${TAG_PREFIX}-m5-qwen3-14b" Qwen/Qwen3-14B ""

# ---------------- 7. Llama-3.1-8B (NEW; preflight) ----------------
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen3-14B 2>/dev/null
if bash tools/preflight_hf_token.sh meta-llama/Llama-3.1-8B 2>&1 | tee -a "$LOG" | grep -q "PREFLIGHT OK"; then
    run_cell "${TAG_PREFIX}-m3-llama31-8b" meta-llama/Llama-3.1-8B ""
else
    echo "[$(date -u +%T)] SKIPPING Llama cell — HF preflight failed" >> "$LOG"
fi

echo "" >> "$LOG"
echo "=== orchestrator_apr26 DONE $(date -u) ===" >> "$LOG"
touch /tmp/vllm_orchestrator_apr26.done
