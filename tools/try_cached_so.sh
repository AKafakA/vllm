#!/bin/bash
# Try the cached _C.abi3.so from uv archive
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

LOG=/tmp/vllm_try_cached.log
echo "=== Trying cached .so at $(date) ===" > "$LOG"

CACHED="$HOME/.cache/uv/archive-v0/u1_2p-3xG1cqE7VZwuZto/vllm/_C.abi3.so"
CURRENT="vllm/_C.abi3.so"

echo "Cached: $(ls -la $CACHED)" >> "$LOG"
echo "Current: $(ls -la $CURRENT)" >> "$LOG"

# Backup current and try cached
cp "$CURRENT" "${CURRENT}.bak"
cp "$CACHED" "$CURRENT"

echo "Testing cached .so:" >> "$LOG"
if python3 -c "import vllm._C; print('vllm._C OK')" >> "$LOG" 2>&1; then
    echo "=== CACHED .so WORKS ===" >> "$LOG"
    uv pip install pytest pytest-asyncio >> "$LOG" 2>&1
else
    echo "=== CACHED .so FAILED ===" >> "$LOG"
    # Restore
    cp "${CURRENT}.bak" "$CURRENT"
fi
rm -f "${CURRENT}.bak"

echo "=== Done at $(date) ===" >> "$LOG"
cat "$LOG"
