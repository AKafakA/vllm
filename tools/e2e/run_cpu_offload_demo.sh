#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_cpu_offload_demo.sh <MODEL_NAME> <PROFILE_PACK>
# Example:
#   ./run_cpu_offload_demo.sh Qwen/Qwen2.5-7B-Instruct examples/profiles/a100-sxm-80gb.json

MODEL_NAME=${1:-}
PROFILE_PACK=${2:-}

if [[ -z "$MODEL_NAME" || -z "$PROFILE_PACK" ]]; then
  echo "Usage: $0 <MODEL_NAME> <PROFILE_PACK>"
  exit 1
fi

export VLLM_EMULATOR_ENABLE_ORACLE=1
export VLLM_EMULATOR_PROFILE_PACK="$PROFILE_PACK"

# Enable offload oracle simulation
export VLLM_EMULATOR_ENABLE_OFFLOAD_ORACLE=1
export VLLM_EMULATOR_OFFLOAD_PROFILE_PACK="$PROFILE_PACK"
export VLLM_EMULATOR_OFFLOAD_BLOCKING_MODE=online

python3 examples/others/lmcache/cpu_offload_lmcache.py \
  --model "$MODEL_NAME"
