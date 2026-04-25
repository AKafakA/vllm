#!/bin/bash
# Source this script (don't execute) to set up the v4 (CUDA-invisible)
# emulator environment in the current shell.
#
# Usage:
#   source tools/activate_emulator.sh
#   python3 -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-8B \
#       --max-model-len 4096 --port 8100 --trust-remote-code
#
# This sets:
#   - venv (sources .venv/bin/activate)
#   - STUB_DIR  (default: $HOME/cuda_stubs)
#   - LD_LIBRARY_PATH prepended with STUB_DIR
#   - CUDA_VISIBLE_DEVICES=""    (hides any real GPU)
#   - All v4 emulator env vars (oracle on, executor hook on, etc.)
#
# Run on any host: a real GPU box (where the GPU is hidden) OR a no-GPU
# CPU-only host (where stubs satisfy torch's libcuda dlopen).

# Resolve script dir even when sourced.
_EMU_SCRIPT_DIR="${BASH_SOURCE[0]:-$0}"
_EMU_REPO_ROOT="$(cd "$(dirname "$_EMU_SCRIPT_DIR")/.." && pwd)"

# Activate the project venv if present.
if [ -f "$_EMU_REPO_ROOT/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$_EMU_REPO_ROOT/.venv/bin/activate"
fi

export STUB_DIR="${STUB_DIR:-$HOME/cuda_stubs}"

# Prepend STUB_DIR to LD_LIBRARY_PATH so torch's dlopen finds our stubs
# before any system CUDA. Harmless if STUB_DIR is empty or doesn't exist.
if [ -d "$STUB_DIR" ]; then
    export LD_LIBRARY_PATH="$STUB_DIR:${LD_LIBRARY_PATH:-}"
fi

# v4 invariants — DO NOT modify without re-running m2_quick_default_v4.
export CUDA_VISIBLE_DEVICES=""
export VLLM_EMULATOR_ENABLE_ORACLE=1
export VLLM_EMULATOR_MODE=realtime
export VLLM_EMULATOR_EXECUTOR_HOOK=1
export VLLM_EMULATOR_SCHEDULER_HOOK=0
export VLLM_EMULATOR_IPC_POSITION=disabled
export VLLM_EMULATOR_PREP_SURROGATE=0
export VLLM_EMULATOR_ORACLE_AGG=sample
export VLLM_EMULATOR_ORACLE_K=auto
export VLLM_EMULATOR_ORACLE_MIN_SAMPLES=30
export VLLM_EMULATOR_BW_SLOPE_SOURCE=disabled

# File-descriptor ulimit for bench client (1024 default truncates at r>=8;
# silently drops requests, biases mean_ttft_ms / mean_e2el_ms). See
# MEMORY.md / feedback_bench_ulimit.md.
ulimit -n 65536 2>/dev/null || true

# Reminder: VLLM_EMULATOR_PROFILE_PACK must be set per-call, e.g.
#   export VLLM_EMULATOR_PROFILE_PACK=./results/<tag>/serving-full.json
# (this script doesn't set it because the path is workload-specific.)

echo "[activate_emulator] v4 env active. STUB_DIR=$STUB_DIR"
echo "[activate_emulator] Set VLLM_EMULATOR_PROFILE_PACK=<path> before launching api_server."
