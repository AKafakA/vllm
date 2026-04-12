#!/bin/bash
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

python3 paper/temp/analysis_apr12/compare_profile_vs_real_rate4.py \
    ./results/RTX-8000-quick/diag/real_step_cycle_r4.jsonl \
    ./results/RTX-8000-dense/profiles/serving-Qwen3-8B-dense.json
