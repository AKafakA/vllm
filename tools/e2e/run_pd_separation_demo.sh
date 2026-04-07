#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_pd_separation_demo.sh <PROFILE_PACK>
# Example:
#   ./run_pd_separation_demo.sh examples/profiles/a100-sxm-80gb.json

PROFILE_PACK=${1:-}

if [[ -z "$PROFILE_PACK" ]]; then
  echo "Usage: $0 <PROFILE_PACK>"
  exit 1
fi

export VLLM_EMULATOR_ENABLE_ORACLE=1
export VLLM_EMULATOR_PROFILE_PACK="$PROFILE_PACK"

python3 examples/emulator/pd_separation_example.py
