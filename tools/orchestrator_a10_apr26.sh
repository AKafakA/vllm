#!/bin/bash
# Phase 4 driver — A10 (vast) single cell at DEFAULT max-num-seqs, v4 emu.
# Re-profiles Qwen3-8B on A10 + 5-rate validate. Used to refresh the
# A10 hardware-axis row in paper Table 1 with parity-clean numbers.
#
# Why a separate orchestrator from the RTX 8000 one: A10 only runs the
# M4 (Qwen3-8B / A10 / DEFAULT) cell — there's no model-scale or
# attention-backend axis on A10 for the paper. Single cell, ~6 h serial.
#
# Usage:
#   nohup bash tools/orchestrator_a10_apr26.sh \
#         > /tmp/vllm_orchestrator_a10_apr26.log 2>&1 &

set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
# vast: HARD-pin HF cache to /dev/shm (31G tmpfs RAM-disk; today's witness
# already cached Qwen3-8B there). /workspace is on the 20G overlay which
# fills up at 8.6G partial download. Override any inherited HF_HOME from
# .bashrc — must NOT use ${HF_HOME:-default} fallback because vast .bashrc
# sets HF_HOME=/workspace/.hf_home which would win.
export HF_HOME="/dev/shm/hf_home"
export HUGGINGFACE_HUB_CACHE="/dev/shm/hf_home/hub"
unset HF_HUB_CACHE TRANSFORMERS_CACHE 2>/dev/null
mkdir -p "$HF_HOME/hub"
source /workspace/vllm-emulator/.venv/bin/activate
cd /workspace/vllm-emulator

LOG="/tmp/vllm_orchestrator_a10_apr26.log"
echo "=== orchestrator_a10_apr26 start $(date -u) ===" > "$LOG"

# Quick sanity: confirm we're on A10 (script aborts if HW prefix can't be set).
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo none)"
case "$GPU_NAME" in
    *A10*) echo "GPU=$GPU_NAME OK" >> "$LOG" ;;
    *) echo "FATAL: not an A10 host (got $GPU_NAME)" >> "$LOG"; exit 1 ;;
esac

run_a10_cell() {
    local cell_tag="$1" model="$2"
    echo "" >> "$LOG"
    echo "[$(date -u +%T)] --- CELL: $cell_tag ($model) ---" >> "$LOG"
    bash tools/run_v4_cell.sh "$cell_tag" "$model" "" 2>&1 | tee -a "$LOG"
    local rc="${PIPESTATUS[0]}"
    echo "[$(date -u +%T)] CELL $cell_tag done (rc=$rc)" >> "$LOG"
    bash tools/correctness_cron.sh "./results/$cell_tag" >> "$LOG" 2>&1 || true
}

# Cell 1: M4 Qwen3-8B (the headline A10 row in paper Table 1)
run_a10_cell apr26-m4-a10-qwen3-8b Qwen/Qwen3-8B

# Cell 2: Qwen3-4B model-scale (moved from RTX 8000 to free up RTX 8000 time)
echo "[$(date -u +%T)] cleaning HF cache of Qwen3-8B before Qwen3-4B" >> "$LOG"
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen3-8B 2>/dev/null
run_a10_cell apr26-m1b-qwen3-4b-a10 Qwen/Qwen3-4B

echo "" >> "$LOG"
echo "=== orchestrator_a10_apr26 DONE $(date -u) ===" >> "$LOG"
touch /tmp/vllm_orchestrator_a10_apr26.done
