#!/bin/bash
# CSD3 A100 environment setup
# Run once in interactive mode to verify everything works
set -e

echo "Setting up vLLM-Emulator on CSD3 A100..."

# Use scratch space for large files
WORKDIR="/rds/user/wd312/hpc-work/vllm-emulator"
mkdir -p "${WORKDIR}"

# Create venv (don't use system Python)
if [ ! -d "${WORKDIR}/venv" ]; then
    echo "Creating venv..."
    python3.12 -m venv "${WORKDIR}/venv"
fi
source "${WORKDIR}/venv/bin/activate"

# Install vLLM
echo "Installing vLLM..."
VLLM_USE_PRECOMPILED=1 pip install -e /home/wd312/Code/llm/vllm-emulator

# Pin transformers
pip install transformers==4.50.3

# Verify
python3 -c "import vllm; print('vLLM', vllm.__version__)"
python3 -c "from vllm_emulator.hooks.gpu_hook import GpuWorkerHook; print('Emulator OK')"
python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

echo "Setup complete. Workdir: ${WORKDIR}"
