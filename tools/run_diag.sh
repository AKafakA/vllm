#!/bin/bash
# Wrapper to run diagnostics with venv
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
sleep 5

bash tools/e2e/diag_rate4.sh
