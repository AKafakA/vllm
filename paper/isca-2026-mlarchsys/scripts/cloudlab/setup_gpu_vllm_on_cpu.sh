#!/bin/bash
# Setup GPU vLLM (with CUDA mock) on CPU-only CloudLab host
# Uses uv for clean env, GPU PyTorch wheel, and our CUDA mock
set -e

echo "=== Setup GPU vLLM on CPU host (with CUDA mock) ==="
echo "Host: $(hostname)"

WORKDIR="${HOME}/vllm-emulator-gpu"
mkdir -p "${WORKDIR}"

# Install uv
if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi
uv --version

# Create clean venv with Python 3.12
echo "Creating venv..."
uv venv --python 3.12 "${WORKDIR}/venv"
source "${WORKDIR}/venv/bin/activate"

# Install GPU PyTorch (CUDA 12.8 wheel — runs on CPU too, just larger)
echo "Installing GPU PyTorch..."
uv pip install torch==2.10.0

# Install build deps
echo "Installing build deps..."
uv pip install "setuptools>=78" setuptools-scm cmake ninja numpy

# Install vLLM from our code (GPU target — same as Vast)
echo "Installing vLLM (GPU target)..."
cd ~/vllm-emulator/code
SETUPTOOLS_SCM_PRETEND_VERSION=0.18.1 uv pip install --no-build-isolation -e .

# Verify
echo ""
echo "=== Verification ==="
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
python3 -c "import vllm; print(f'vLLM: {vllm.__version__}')"
python3 -c "from vllm_emulator.cuda_mock import install; print(f'CUDA mock: ready')"

echo ""
echo "Setup complete. Venv: ${WORKDIR}/venv"
echo "To test: VLLM_EMULATOR_MOCK_CUDA=1 python3 -c 'import vllm_emulator.cuda_mock; import torch; print(torch.cuda.is_available())'"
