#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_offline_online_emulator.sh <MODEL_NAME> <PROFILE_PACK> <TRACE_JSONL>
# Example:
#   ./run_offline_online_emulator.sh Qwen/Qwen2.5-7B-Instruct examples/profiles/a100-sxm-80gb.json /path/to/burstgpt.jsonl

MODEL_NAME=${1:-}
PROFILE_PACK=${2:-}
TRACE_PATH=${3:-}

if [[ -z "$MODEL_NAME" || -z "$PROFILE_PACK" || -z "$TRACE_PATH" ]]; then
  echo "Usage: $0 <MODEL_NAME> <PROFILE_PACK> <TRACE_JSONL>"
  exit 1
fi

export VLLM_EMULATOR_ENABLE_ORACLE=1
export VLLM_EMULATOR_PROFILE_PACK="$PROFILE_PACK"

echo "[1/2] Offline mode (no blocking)"
export VLLM_EMULATOR_BLOCKING_MODE=offline
python3 examples/offline_inference/batch_llm_inference.py \
  --model "$MODEL_NAME" \
  --input-jsonl "$TRACE_PATH" \
  --max-batch-size 16 \
  --max-num-seqs 16

echo "[2/2] Online mode (blocking)"
export VLLM_EMULATOR_BLOCKING_MODE=online
python3 examples/offline_inference/batch_llm_inference.py \
  --model "$MODEL_NAME" \
  --input-jsonl "$TRACE_PATH" \
  --max-batch-size 16 \
  --max-num-seqs 16
