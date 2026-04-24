#!/bin/bash
# Master batch for P2 headline cells: M1b + M2 on canonical ShareGPT.
# Each sub-chain writes its own .done marker; failures don't block
# the next. Runs AFTER M3 completes (uses same RTX 8000 GPU).
set -uo pipefail
cd ~/Code/llm/vllm-emulator

MASTER_MARKER="/tmp/vllm_overnight_apr24-canonical-headline-batch"
MASTER_LOG="/tmp/vllm_overnight_apr24-canonical-headline-batch.log"
touch "${MASTER_MARKER}.started"
echo "=== canonical-headline batch start $(date -u) ===" > "$MASTER_LOG"

for CELL_SCRIPT in \
    tools/chain_redo_dense_m1b_canonical.sh \
    tools/chain_redo_dense_m2_canonical.sh; do
    CELL_NAME=$(basename "$CELL_SCRIPT" .sh)
    echo "" >> "$MASTER_LOG"
    echo "[$(date +%T)] --> $CELL_NAME" >> "$MASTER_LOG"
    bash "$CELL_SCRIPT" >> "$MASTER_LOG" 2>&1 || \
        echo "  $CELL_NAME returned non-zero (continuing)" >> "$MASTER_LOG"
done

echo "" >> "$MASTER_LOG"
echo "=== canonical-headline batch DONE $(date -u) ===" >> "$MASTER_LOG"
touch "${MASTER_MARKER}.done"
