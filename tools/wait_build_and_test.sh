#!/bin/bash
# Wait for vllm source build to finish, then run quick smoke test.
# Run via: nohup bash tools/wait_build_and_test.sh > /dev/null 2>&1 &
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

LOG=/tmp/vllm_build_from_source.log

echo "Watcher started at $(date)" >> "$LOG"

# Wait for ninja (the actual CUDA compiler) to finish
while pgrep -x ninja > /dev/null 2>&1; do
    sleep 60
done
sleep 10

# Wait for uv to finish
while pgrep -f "uv pip install" > /dev/null 2>&1; do
    sleep 10
done
sleep 5

echo "Build processes done at $(date)" >> "$LOG"

# Verify vllm._C import
if python3 -c "import vllm._C; print('vllm._C OK')" >> "$LOG" 2>&1; then
    echo "Build SUCCEEDED" >> "$LOG"

    # Clean up failed attempt
    rm -rf ./results/RTX-8000-quick

    # Run quick smoke test
    echo "Starting quick test at $(date)" >> "$LOG"
    bash tools/e2e/profile_and_eval.sh \
        --model Qwen/Qwen3-8B \
        --result-dir ./results/RTX-8000-quick \
        --rates '1 4 8' \
        --warmup-prompts 50 \
        --eval-prompts 100 \
        --profile-rates '1 4 8' \
        > /tmp/vllm_quick_test.log 2>&1
    echo "Quick test done at $(date)" >> "$LOG"
else
    echo "BUILD FAILED - vllm._C import error" >> "$LOG"
fi
