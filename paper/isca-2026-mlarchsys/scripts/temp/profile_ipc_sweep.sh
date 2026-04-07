#!/bin/bash
# Two-pass IPC overhead profiling:
#   Pass 1: N-sweep on real GPU → real_TTFT(N)
#   Pass 2: N-sweep on emulator (no IPC overhead) → emu_TTFT(N)
#   Compute: overhead(N) = max(0, real - emu) → store in profile
set -e
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"
PROFILE="${RESULT_DIR}/profiles/serving-1.5b-tp1-calibrated.json"
PROFILER="/workspace/vllm-emulator-v18/paper/isca-2026-mlarchsys/scripts/temp/profile_ipc_overhead.py"
PORT=8100
MAX_N=30

pkill -9 -f EngineCore 2>/dev/null || true
pkill -9 -f api_server 2>/dev/null || true
sleep 5

# Rebuild profile without IPC overhead first (initial profile for pass 2)
echo "=== Rebuild initial profile (no IPC overhead) ==="
python3 /workspace/vllm-emulator-v18/paper/isca-2026-mlarchsys/scripts/temp/build_serving_profile_2d.py \
    "${RESULT_DIR}/step_cycle_1.5b_full.jsonl" \
    "${RESULT_DIR}/profiles/sweep-1.5b-tp1-v14.json" \
    "$PROFILE" "$MODEL" "RTX-3060-12GB" > /dev/null 2>&1
echo "  Done"

warmup_server() {
    echo "  Warming up..."
    for i in $(seq 1 10); do
        curl -s --max-time 10 http://localhost:$PORT/v1/completions \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"$MODEL\",\"prompt\":\"warmup $i\",\"max_tokens\":5,\"temperature\":0}" > /dev/null
        sleep 0.2
    done
    python3 -m vllm.entrypoints.cli.main bench serve --model "$MODEL" --base-url http://localhost:$PORT \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 30 --request-rate 2 > /dev/null 2>&1
    sleep 3
    echo "  Warmup done"
}

# ===== PASS 1: Real GPU N-sweep =====
echo ""
echo "=== PASS 1: Real GPU N-sweep (N=1..${MAX_N}) ==="
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > /workspace/ipc_pass1_server.log 2>&1 &
for i in $(seq 1 120); do if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then break; fi; sleep 1; done
warmup_server

python3 "$PROFILER" --pass1 --port $PORT --model "$MODEL" \
    --output "${RESULT_DIR}/profiles/ipc_real_ttft.json" --max-n $MAX_N

pkill -f api_server 2>/dev/null || true; sleep 3
pkill -9 -f EngineCore 2>/dev/null || true; sleep 5

# ===== PASS 2: Emulator N-sweep (no IPC overhead) =====
echo ""
echo "=== PASS 2: Emulator N-sweep (N=1..${MAX_N}) ==="
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 4096 --port $PORT --trust-remote-code \
    > /workspace/ipc_pass2_server.log 2>&1 &
for i in $(seq 1 120); do if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then break; fi; sleep 1; done
grep -m1 ExecutorEmulatorHook /workspace/ipc_pass2_server.log || true
warmup_server

python3 "$PROFILER" --pass2 --port $PORT --model "$MODEL" \
    --output "${RESULT_DIR}/profiles/ipc_emu_ttft.json" --max-n $MAX_N

pkill -f api_server 2>/dev/null || true; sleep 3
pkill -9 -f EngineCore 2>/dev/null || true; sleep 5

# ===== Compute delta =====
echo ""
echo "=== Compute IPC overhead (real - emu) ==="
python3 "$PROFILER" --compute \
    --real "${RESULT_DIR}/profiles/ipc_real_ttft.json" \
    --emu "${RESULT_DIR}/profiles/ipc_emu_ttft.json" \
    --output "${RESULT_DIR}/profiles/ipc_overhead.json"

# Rebuild final profile WITH the computed IPC overhead
echo ""
echo "=== Rebuild final profile with IPC overhead ==="
python3 /workspace/vllm-emulator-v18/paper/isca-2026-mlarchsys/scripts/temp/build_serving_profile_2d.py \
    "${RESULT_DIR}/step_cycle_1.5b_full.jsonl" \
    "${RESULT_DIR}/profiles/sweep-1.5b-tp1-v14.json" \
    "$PROFILE" "$MODEL" "RTX-3060-12GB"

echo ""
echo "=== Summary ==="
python3 -c "
import json
data = json.load(open('${RESULT_DIR}/profiles/ipc_overhead.json'))
print(f'IPC overhead table: {len(data)} entries')
for d in data:
    if d['num_reqs'] in [1, 2, 3, 5, 10, 15, 20, 30]:
        print(f'  N={d[\"num_reqs\"]:3d}: real={d[\"real_ttft_us\"]/1000:.1f}ms  '
              f'emu={d[\"emu_ttft_us\"]/1000:.1f}ms  '
              f'overhead={d[\"overhead_us\"]/1000:.1f}ms')
"
echo "DONE"
