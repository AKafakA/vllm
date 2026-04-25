#!/bin/bash
# vLLM-emulator one-shot installer.
#
# Steps:
#   1. Create venv (.venv) if missing
#   2. Install vllm (precompiled wheel) + emulator package in editable mode
#   3. Generate cuda_stubs/ for the v4 (CUDA-invisible) path
#   4. Verify imports
#   5. Print activation hint
#
# Modes (auto-detected, override with --mode):
#   - real-gpu host (libcuda.so.1 in ldconfig): stubs are skipped
#     (v4 mode hides the GPU via CUDA_VISIBLE_DEVICES="")
#   - no-gpu host (Mode B): stubs are linked from torch's cu128 wheel
#
# Usage:
#   bash install.sh                 # defaults
#   bash install.sh --vllm 0.18.1   # pin a specific vllm version
#
# After install:
#   source tools/activate_emulator.sh
#   export VLLM_EMULATOR_PROFILE_PACK=./results/<tag>/serving-full.json
#   python3 -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-8B \
#       --max-model-len 4096 --port 8100 --trust-remote-code

set -euo pipefail

VLLM_VERSION="0.18.1"
SKIP_VENV=""
SKIP_STUBS=""
MODE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --vllm) VLLM_VERSION="$2"; shift 2 ;;
        --skip-venv) SKIP_VENV=1; shift ;;
        --skip-stubs) SKIP_STUBS=1; shift ;;
        --mode) MODE="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--vllm <version>] [--skip-venv] [--skip-stubs] [--mode real-gpu|no-gpu]"
            exit 0
            ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

echo "=== vllm-emulator installer ==="
echo "REPO_ROOT=$REPO_ROOT"
echo "VLLM_VERSION=$VLLM_VERSION"

# -------- Mode detection --------
if [ -z "$MODE" ]; then
    if ldconfig -p 2>/dev/null | grep -q 'libcuda\.so\.1'; then
        MODE="real-gpu"
    else
        MODE="no-gpu"
    fi
fi
echo "MODE=$MODE  (use --mode to override)"

# -------- venv --------
if [ -z "$SKIP_VENV" ]; then
    if [ ! -d .venv ]; then
        if command -v uv >/dev/null 2>&1; then
            echo "[1/4] uv venv .venv --python 3.12"
            uv venv .venv --python 3.12
        else
            echo "[1/4] python3 -m venv .venv"
            python3 -m venv .venv
        fi
    else
        echo "[1/4] .venv exists, reusing"
    fi
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# -------- vllm + emulator install --------
echo "[2/4] pip install vllm==$VLLM_VERSION + emulator (editable)"
PIP="python3 -m pip"
$PIP install --upgrade pip wheel >/dev/null

# Use precompiled wheel — much faster + no nvcc needed on no-gpu hosts.
export VLLM_USE_PRECOMPILED=1
export SETUPTOOLS_SCM_PRETEND_VERSION="$VLLM_VERSION"

# Editable install picks up vllm + emulator + entry-point plugin
$PIP install --no-build-isolation -e .

# -------- CUDA stubs (v4 mode) --------
if [ -z "$SKIP_STUBS" ] && [ "$MODE" = "no-gpu" ]; then
    echo "[3/4] Generating CUDA stub libs in ~/cuda_stubs/"
    bash "$REPO_ROOT/tools/setup_cuda_stubs.sh"
elif [ "$MODE" = "real-gpu" ]; then
    echo "[3/4] Real CUDA detected — stubs not needed (v4 hides GPU via CUDA_VISIBLE_DEVICES=\"\")"
else
    echo "[3/4] Skipping CUDA stubs (--skip-stubs)"
fi

# -------- Verify --------
echo "[4/4] Verifying imports"
python3 - <<'PY'
import vllm
from vllm_emulator.hooks import ExecutorEmulatorHook, get_executor_hook
from vllm_emulator.oracle import BaseGpuCostOracle, ProfileGpuCostOracle, create_oracle_from_profile_pack
from vllm_emulator.platform import emulator_platform_plugin
print(f"vllm {getattr(vllm, '__version__', '?')} + vllm-emulator imports OK")
PY

# -------- Compat check (warn-only, doesn't block install) --------
if ! bash "$REPO_ROOT/tools/check_vllm_compat.sh"; then
    echo "WARN: vllm compat check flagged drift. v4 hooks may not work."
    echo "      Migrate the patches in vllm/ and vllm_emulator/cuda_mock.py."
fi

cat <<EOF

=== install complete ===

Activate the v4 emulator environment:
  source tools/activate_emulator.sh
  export VLLM_EMULATOR_PROFILE_PACK=./results/<tag>/serving-full.json

Then launch any vllm command — it will run on the emulator path.
EOF
