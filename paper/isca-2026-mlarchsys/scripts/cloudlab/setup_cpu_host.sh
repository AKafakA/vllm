#!/bin/bash
# Setup vLLM-Emulator on CloudLab CPU-only host (xl170, Ubuntu 22.04)
set -e

echo "=== Setting up vLLM-Emulator on CPU-only host ==="
echo "Host: $(hostname)"
echo "CPUs: $(nproc)"
echo "RAM: $(free -h | grep Mem | awk '{print $2}')"

WORKDIR="${HOME}/vllm-emulator"
mkdir -p "${WORKDIR}"

# Install Python 3.12
if ! python3.12 --version 2>/dev/null; then
    echo "Installing Python 3.12..."
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update
    sudo apt-get install -y python3.12 python3.12-venv python3.12-dev
fi
python3.12 --version

# Create venv
if [ ! -d "${WORKDIR}/venv" ]; then
    echo "Creating venv..."
    python3.12 -m venv "${WORKDIR}/venv"
fi
source "${WORKDIR}/venv/bin/activate"

# Install pip
pip install --upgrade pip setuptools wheel

# Install PyTorch CPU-only
echo "Installing PyTorch (CPU)..."
pip install torch --index-url https://download.pytorch.org/whl/cpu

echo ""
echo "=== Setup complete ==="
echo "Venv: ${WORKDIR}/venv"
echo "Python: $(python3 --version)"
echo "Torch: $(python3 -c 'import torch; print(torch.__version__)')"
echo ""
echo "Next steps:"
echo "  1. rsync the vllm-emulator code"
echo "  2. pip install -e . (CPU-only, no CUDA needed)"
echo "  3. Run EmulatorPlatform test"
