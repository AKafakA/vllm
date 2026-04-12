#!/bin/bash
# Copy ALL cached .so files and run quick test.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

LOG=/tmp/vllm_fix_and_test.log
CACHE="$HOME/.cache/uv/archive-v0/u1_2p-3xG1cqE7VZwuZto/vllm"

echo "=== Copy all .so and test at $(date) ===" > "$LOG"

# Copy all cached .so files
cp "$CACHE/_C.abi3.so" vllm/_C.abi3.so
cp "$CACHE/_moe_C.abi3.so" vllm/_moe_C.abi3.so
cp "$CACHE/cumem_allocator.abi3.so" vllm/cumem_allocator.abi3.so
cp "$CACHE/_flashmla_C.abi3.so" vllm/_flashmla_C.abi3.so
cp "$CACHE/_flashmla_extension_C.abi3.so" vllm/_flashmla_extension_C.abi3.so
mkdir -p vllm/vllm_flash_attn
cp "$CACHE/vllm_flash_attn/_vllm_fa2_C.abi3.so" vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so
cp "$CACHE/vllm_flash_attn/_vllm_fa3_C.abi3.so" vllm/vllm_flash_attn/_vllm_fa3_C.abi3.so
echo "Copied all 7 .so files from cache" >> "$LOG"

# Verify
python3 -c "import vllm._C; print('vllm._C OK')" >> "$LOG" 2>&1
python3 -c "from vllm.version import __version__; print(f'vllm={__version__}')" >> "$LOG" 2>&1
python3 -c "from vllm_emulator.profiler.trace_profiler import StepCycleTracer; print('emulator OK')" >> "$LOG" 2>&1
python3 -c "from vllm.vllm_flash_attn import _vllm_fa2_C; print('flash_attn OK')" >> "$LOG" 2>&1
echo "=== Install verified ===" >> "$LOG"

# Run quick smoke test
echo "Starting quick test at $(date)" >> "$LOG"
rm -rf results/RTX-8000-quick
bash tools/e2e/profile_and_eval.sh \
    --model Qwen/Qwen3-8B \
    --result-dir ./results/RTX-8000-quick \
    --rates '1 4 8' \
    --warmup-prompts 50 \
    --eval-prompts 100 \
    --profile-rates '1 4 8' \
    >> /tmp/vllm_quick_test.log 2>&1

echo "=== Quick test done at $(date) ===" >> "$LOG"
