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

# Single cell — full 3 stages at DEFAULT max-num-seqs.
echo "" >> "$LOG"
echo "[$(date -u +%T)] --- CELL: apr26-m4-a10-qwen3-8b ---" >> "$LOG"
bash tools/run_v4_cell.sh apr26-m4-a10-qwen3-8b Qwen/Qwen3-8B "" 2>&1 | tee -a "$LOG"
RC="${PIPESTATUS[0]}"
echo "[$(date -u +%T)] CELL apr26-m4-a10-qwen3-8b done (rc=$RC)" >> "$LOG"

# Run correctness sweep on this cell.
bash tools/correctness_cron.sh "./results/apr26-m4-a10-qwen3-8b" >> "$LOG" 2>&1 || true

echo "" >> "$LOG"
echo "=== orchestrator_a10_apr26 DONE $(date -u) ===" >> "$LOG"
touch /tmp/vllm_orchestrator_a10_apr26.done
