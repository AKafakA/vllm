#!/bin/bash
# Create a CUDA stub directory for the v4 (CUDA-invisible) emulator path.
#
# The v4 path runs vllm with CUDA_VISIBLE_DEVICES="" + LD_LIBRARY_PATH
# pointing at this stub directory. Torch (installed from the cu128 wheel
# even on no-GPU hosts) needs to dlopen libcuda.so.1 + libcudart.so.* at
# import time — our stubs satisfy that, then cuda_mock.py patches the
# Python-level torch.cuda.* APIs so no real CUDA call is ever made.
#
# This script is idempotent. If your host has real CUDA installed
# (libcuda.so.1 in /usr/lib/x86_64-linux-gnu/), this script is a no-op
# — the stubs are not needed, and v4 mode still works because
# CUDA_VISIBLE_DEVICES="" hides the GPU from torch.
#
# Usage:
#   bash tools/setup_cuda_stubs.sh [--stub-dir <path>]
# Default stub-dir: $HOME/cuda_stubs

set -euo pipefail

STUB_DIR="${HOME}/cuda_stubs"
while [ $# -gt 0 ]; do
    case "$1" in
        --stub-dir) STUB_DIR="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--stub-dir <path>]"
            exit 0
            ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

# If real CUDA is present, no need to stub.
if ldconfig -p 2>/dev/null | grep -q 'libcuda\.so\.1'; then
    echo "Real CUDA detected at $(ldconfig -p | grep 'libcuda.so.1' | head -1)."
    echo "Stubs not needed. v4 mode hides the GPU via CUDA_VISIBLE_DEVICES=\"\"."
    echo "Stub dir would have been: $STUB_DIR  (not created)"
    exit 0
fi

# Find torch's bundled stub libraries. They ship with the cu128 wheel
# under torch/lib/ and torch/cuda/lib/.
TORCH_LIB="$(python3 -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))' 2>/dev/null || true)"
if [ -z "$TORCH_LIB" ] || [ ! -d "$TORCH_LIB" ]; then
    echo "FATAL: torch not importable. Run 'bash install.sh' first." >&2
    exit 1
fi

mkdir -p "$STUB_DIR"

# Symlink the stub libs torch already shipped with.
linked=0
for lib_name in libcuda.so.1 libcudart.so.12 libcudart.so libnvrtc.so.12 libnvrtc.so; do
    # Search the torch tree for a matching file (under torch/lib/, torch/cuda/lib/, etc).
    found="$(find "$TORCH_LIB" "$(dirname "$TORCH_LIB")" -name "$lib_name*" 2>/dev/null | head -1)"
    if [ -n "$found" ] && [ ! -e "$STUB_DIR/$lib_name" ]; then
        ln -sf "$found" "$STUB_DIR/$lib_name"
        linked=$((linked + 1))
    fi
done

# Also check nvidia-cuda-runtime / nvidia-cuda-cu* pip packages.
for nv_pkg in nvidia.cuda_runtime nvidia.cuda_cupti nvidia.cuda_nvrtc; do
    NV_PATH="$(python3 -c "import importlib.util; spec = importlib.util.find_spec('$nv_pkg'); print(spec.submodule_search_locations[0] if spec else '')" 2>/dev/null || true)"
    if [ -n "$NV_PATH" ] && [ -d "$NV_PATH/lib" ]; then
        for so in "$NV_PATH/lib"/*.so*; do
            [ -e "$so" ] || continue
            base="$(basename "$so")"
            if [ ! -e "$STUB_DIR/$base" ]; then
                ln -sf "$so" "$STUB_DIR/$base"
                linked=$((linked + 1))
            fi
        done
    fi
done

if [ "$linked" -eq 0 ]; then
    echo "WARN: no CUDA stub libs found in torch tree. v4 mode may fail to" >&2
    echo "      import torch. Try reinstalling torch from a CUDA wheel:" >&2
    echo "      pip install torch --index-url https://download.pytorch.org/whl/cu128" >&2
    exit 1
fi

echo "Created $linked symlinks in $STUB_DIR"
ls -la "$STUB_DIR" | head -10
echo ""
echo "v4 mode usage: source tools/activate_emulator.sh"
echo "  (sets LD_LIBRARY_PATH=\$STUB_DIR:... and other v4 env vars)"
