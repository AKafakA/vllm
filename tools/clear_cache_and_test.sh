#!/bin/bash
# Clear stale FlashInfer JIT cache and run quick test.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

LOG=/tmp/vllm_fix_and_test.log
echo "=== Clear cache and test at $(date) ===" > "$LOG"

# Clear stale FlashInfer JIT cache (has hardcoded paths from old venv)
rm -rf ~/.cache/flashinfer
echo "Cleared FlashInfer JIT cache" >> "$LOG"

# Verify install
python3 -c "import vllm._C; print('vllm._C OK')" >> "$LOG" 2>&1
python3 -c "from vllm.version import __version__; print(f'vllm={__version__}')" >> "$LOG" 2>&1
echo "=== Install verified ===" >> "$LOG"

# Run quick smoke test
echo "Starting quick test at $(date)" >> "$LOG"
rm -rf results/RTX-8000-quick
bash tools/e2e/profile_and_eval.sh \
    --model Qwen/Qwen3-8B \
    --result-dir ./results/RTX-8000-quick \
    --rates '1 4 8' \
    --warmup-prompts 50 \
    --eval-prompts 100 \
    --profile-rates '1 4 8' \
    >> /tmp/vllm_quick_test.log 2>&1

echo "=== Quick test done at $(date) ===" >> "$LOG"
