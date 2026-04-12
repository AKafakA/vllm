#!/bin/bash
# Run evaluation only (profile already built). Uses updated cleanup.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

# Kill ALL leftover GPU processes
pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
sleep 5

PROFILE="./results/RTX-8000-quick/profiles/serving-Qwen3-8B-step-cycle.json"

bash tools/e2e/profile_and_eval.sh \
    --model Qwen/Qwen3-8B \
    --result-dir ./results/RTX-8000-quick \
    --rates '1 4 8' \
    --warmup-prompts 50 \
    --eval-prompts 100 \
    --skip-profile \
    --profile-pack "$PROFILE"
