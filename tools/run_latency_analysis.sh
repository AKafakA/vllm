#!/bin/bash
# Analyze per-request latency gap from quick test results.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

LOG=/tmp/vllm_latency_analysis.log
echo "=== Per-request latency analysis at $(date) ===" > "$LOG"

for RATE in 1 4 8; do
    REAL="./results/RTX-8000-quick/online/Qwen3-8B_r${RATE}_real.json"
    EMU="./results/RTX-8000-quick/online/Qwen3-8B_r${RATE}_emu.json"
    if [ -f "$REAL" ] && [ -f "$EMU" ]; then
        echo "" >> "$LOG"
        echo "========== Rate=$RATE ==========" >> "$LOG"
        python3 tools/e2e/analyze_request_latency.py "$REAL" "$EMU" >> "$LOG" 2>&1
    fi
done

# Also analyze the diag rate=4 data (200 prompts, more reliable)
REAL_DIAG="./results/RTX-8000-quick/diag/real_r4.json"
EMU_DIAG="./results/RTX-8000-quick/diag/emu_r4.json"
if [ -f "$REAL_DIAG" ] && [ -f "$EMU_DIAG" ]; then
    echo "" >> "$LOG"
    echo "========== Rate=4 (diag, 200 prompts) ==========" >> "$LOG"
    python3 tools/e2e/analyze_request_latency.py "$REAL_DIAG" "$EMU_DIAG" >> "$LOG" 2>&1
fi

echo "" >> "$LOG"
echo "=== Done at $(date) ===" >> "$LOG"
cat "$LOG"
