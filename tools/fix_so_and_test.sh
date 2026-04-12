#!/bin/bash
# Fix the .so with cached version and run quick test.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

LOG=/tmp/vllm_fix_and_test.log
echo "=== Fix .so and test at $(date) ===" > "$LOG"

# Use the cached .so that we already verified works
CACHED="$HOME/.cache/uv/archive-v0/u1_2p-3xG1cqE7VZwuZto/vllm/_C.abi3.so"
if [ -f "$CACHED" ]; then
    cp "$CACHED" vllm/_C.abi3.so
    echo "Copied cached .so ($(ls -la vllm/_C.abi3.so | awk '{print $5}') bytes)" >> "$LOG"
else
    echo "ERROR: cached .so not found at $CACHED" >> "$LOG"
    exit 1
fi

# Verify
python3 -c "import vllm._C; print('vllm._C OK')" >> "$LOG" 2>&1
python3 -c "from vllm.version import __version__; print(f'vllm={__version__}')" >> "$LOG" 2>&1
python3 -c "from vllm_emulator.profiler.trace_profiler import StepCycleTracer; print('emulator OK')" >> "$LOG" 2>&1
echo "=== Install verified ===" >> "$LOG"

# Install test deps
uv pip install pytest pytest-asyncio >> "$LOG" 2>&1

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
