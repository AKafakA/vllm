#!/bin/bash
# Quick smoke test wrapper — activates venv, cleans up, runs test.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

# Clean up any leftover processes (ignore errors)
fuser 8100/tcp 2>/dev/null | xargs -r kill -9 2>/dev/null
sleep 3

# Now enable strict mode
set -e

rm -rf results/RTX-8000-quick 2>/dev/null || true
bash tools/e2e/profile_and_eval.sh \
    --model Qwen/Qwen3-8B \
    --result-dir ./results/RTX-8000-quick \
    --rates '1 4 8' \
    --warmup-prompts 50 \
    --eval-prompts 100 \
    --profile-rates '1 4 8' \
    >> /tmp/vllm_quick_test.log 2>&1
