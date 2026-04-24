#!/bin/bash
# Extensions to the paper's Table 1 after the main canonical batch:
# M5 14B (model-scale extension) + burstiness=0.25 (traffic
# ablation). Runs serially on RTX 8000, each writes its own .done.
set -uo pipefail
cd ~/Code/llm/vllm-emulator

MASTER_MARKER="/tmp/vllm_overnight_apr24-extensions-batch"
MASTER_LOG="/tmp/vllm_overnight_apr24-extensions-batch.log"
touch "${MASTER_MARKER}.started"
echo "=== extensions batch start $(date -u) ===" > "$MASTER_LOG"

for CELL_SCRIPT in \
    tools/chain_config_triton_m2.sh \
    tools/chain_bursty_m2.sh \
    tools/chain_redo_dense_m5.sh; do
    CELL_NAME=$(basename "$CELL_SCRIPT" .sh)
    echo "" >> "$MASTER_LOG"
    echo "[$(date +%T)] --> $CELL_NAME" >> "$MASTER_LOG"
    bash "$CELL_SCRIPT" >> "$MASTER_LOG" 2>&1 || \
        echo "  $CELL_NAME returned non-zero (continuing)" >> "$MASTER_LOG"
done

echo "" >> "$MASTER_LOG"
echo "=== extensions batch DONE $(date -u) ===" >> "$MASTER_LOG"
touch "${MASTER_MARKER}.done"
