#!/bin/bash
# Fresh end-to-end: profile + b2b eval for 1.5B
set -e
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub

bash /workspace/vllm-emulator-v18/paper/isca-2026-mlarchsys/scripts/temp/build_serving_profile.sh \
    "Qwen/Qwen2.5-1.5B-Instruct" 1 "1.5b-tp1-fresh"

bash /workspace/vllm-emulator-v18/paper/isca-2026-mlarchsys/scripts/temp/eval_online_b2b.sh \
    "Qwen/Qwen2.5-1.5B-Instruct" 1 "1.5b-tp1-fresh" \
    "/workspace/eval_results/RTX-3060-12GB/profiles/serving-1.5b-tp1-fresh-step-cycle.json"

echo "FRESH E2E DONE"
