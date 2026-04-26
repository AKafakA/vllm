#!/bin/bash
# Phase 3 driver — RTX 8000 cells, all at DEFAULT max-num-seqs, v4 emu,
# fresh dense profile per cell (r=1/2/4/8 lifted to 2000 prompts each).
#
# Pipeline (serial): each cell does its OWN 3-stage capture (real bench +
# dense profile + emu validate). No profile reuse — Apr 25 night decision.
#
# Cells:
#   1. M2-Main         — Qwen3-8B, no special flags
#   2. R3 prefix-off   — Qwen3-8B, --no-prefix-caching
#   3. TRITON          — Qwen3-8B, --attention-backend TRITON_ATTN
#   4. Burstiness      — DEFERRED (needs --burstiness flag wiring)
#   5. Qwen3-4B        — clean Qwen3-8B from HF cache first
#   6. Qwen3-14B       — clean Qwen3-4B from HF cache first
#   7. Llama-3.1-8B    — preflight HF token first
#
# Per cell: ~7.8 h (real ~1.5h + profile ~4.7h + emu ~1.5h + cleanup overhead)
# Total: ~55 h ≈ 2.3 days. Done ~Apr 28 07:30 UTC. Fits May 1 with ~3.7 days buffer.
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

TAG_PREFIX="apr26"

run_cell() {
    local cell_tag="$1" model="$2" extra="$3"
    echo "" >> "$LOG"
    echo "[$(date -u +%T)] --- CELL: $cell_tag ($model) ---" >> "$LOG"
    echo "    EXTRA=$extra  fresh dense profile" >> "$LOG"
    bash tools/run_v4_cell.sh "$cell_tag" "$model" "$extra"
    rc=$?
    echo "[$(date -u +%T)] CELL $cell_tag done (rc=$rc)" >> "$LOG"
    bash tools/correctness_cron.sh "./results/$cell_tag" >> "$LOG" 2>&1 || true
}

# 1. M2-Main (fresh dense profile, no reuse)
run_cell "${TAG_PREFIX}-m2-main" Qwen/Qwen3-8B ""

# 2. R3 prefix-cache OFF (vllm 0.18 uses --no-enable-prefix-caching, NOT --no-prefix-caching)
run_cell "${TAG_PREFIX}-r3-prefix-off" Qwen/Qwen3-8B "--no-enable-prefix-caching"

# 3. TRITON
run_cell "${TAG_PREFIX}-triton" Qwen/Qwen3-8B "--attention-backend TRITON_ATTN"

# 4. Burstiness γ=0.25 (DEFERRED — needs --burstiness flag wiring)
echo "[$(date -u +%T)] CELL burstiness-g25 DEFERRED" >> "$LOG"

# Qwen3-4B moved to vast A10 orchestrator (Phase 4) — runs there in parallel.

# 5. Llama-3.1-8B (cross-family — high paper priority)
echo "[$(date -u +%T)] cleaning HF cache of Qwen3-8B before Llama" >> "$LOG"
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen3-8B 2>/dev/null
if bash tools/preflight_hf_token.sh meta-llama/Llama-3.1-8B 2>&1 | tee -a "$LOG" | grep -q "PREFLIGHT OK"; then
    run_cell "${TAG_PREFIX}-m3-llama31-8b" meta-llama/Llama-3.1-8B ""
else
    echo "[$(date -u +%T)] SKIPPING Llama cell — HF preflight failed" >> "$LOG"
fi

# 6. Qwen3-14B (model-scale upper bound)
rm -rf ~/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B 2>/dev/null
run_cell "${TAG_PREFIX}-m5-qwen3-14b" Qwen/Qwen3-14B ""

echo "" >> "$LOG"
echo "=== orchestrator_apr26 DONE $(date -u) ===" >> "$LOG"
touch /tmp/vllm_orchestrator_apr26.done
