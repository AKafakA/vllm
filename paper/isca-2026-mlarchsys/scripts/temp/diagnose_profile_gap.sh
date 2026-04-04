#!/bin/bash
# Diagnose: trace actual step latencies at each rate and compare with profile
set -e
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"
PORT=8100

pkill -9 -f EngineCore 2>/dev/null || true; pkill -9 -f api_server 2>/dev/null || true; sleep 5

echo "=== Trace real step latencies at each rate ==="

# Run real server with BOTH step-cycle tracer AND execute_model tracer
rm -f "${RESULT_DIR}/diag_step_cycle.jsonl" "${RESULT_DIR}/diag_exec_trace.jsonl"

VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
VLLM_EMULATOR_STEP_TRACE_OUTPUT="${RESULT_DIR}/diag_step_cycle.jsonl" \
VLLM_EMULATOR_TRACE_PROFILE=1 \
VLLM_EMULATOR_TRACE_OUTPUT="${RESULT_DIR}/diag_exec_trace.jsonl" \
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 4096 --port ${PORT} --trust-remote-code \
    > /workspace/diag_server.log 2>&1 &

for i in $(seq 1 120); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then echo "  Ready ${i}s"; break; fi; sleep 1
done

for rate in 1 2 4; do
    echo "  Rate=${rate} (50 prompts)..."
    python3 -m vllm.entrypoints.cli.main bench serve --model "${MODEL}" --base-url http://localhost:${PORT} \
        --dataset-name random --random-input-len 256 --random-output-len 128 \
        --num-prompts 50 --request-rate ${rate} \
        --percentile-metrics ttft,tpot --metric-percentiles 50,99 2>&1 | grep -E "TPOT"
done

pkill -f api_server 2>/dev/null || true; sleep 2; pkill -9 -f EngineCore 2>/dev/null || true; sleep 5

echo ""
echo "=== Analysis: profile vs actual latencies ==="
python3 -c "
import json, statistics
from collections import defaultdict

# Load serving profile
profile = json.load(open('${RESULT_DIR}/profiles/serving-1.5b-tp1-v2.json'))
prof_map = {e['total_tokens']: e['latency_us'] for e in profile['forward_pass']}

# Load step-cycle trace (actual step times during serving)
step_records = []
for line in open('${RESULT_DIR}/diag_step_cycle.jsonl'):
    r = json.loads(line)
    if 'total_tokens' in r:
        step_records.append(r)

# Load exec trace (GPU-only times)
exec_records = []
for line in open('${RESULT_DIR}/diag_exec_trace.jsonl'):
    exec_records.append(json.loads(line))

print(f'Step-cycle records: {len(step_records)}')
print(f'Exec trace records: {len(exec_records)}')

# Group step-cycle by total_tokens
by_tt = defaultdict(list)
for r in step_records:
    by_tt[r['total_tokens']].append(r['step_cycle_us'])

# Group exec trace by total_tokens
exec_by_tt = defaultdict(list)
for r in exec_records:
    exec_by_tt[r['total_tokens']].append(r['latency_us'])

print(f'\n{\"tt\":>4} {\"profile\":>10} {\"step_med\":>10} {\"exec_med\":>10} {\"prof_err\":>10} {\"n\":>6}')
print('-' * 55)
for tt in sorted(by_tt):
    if tt <= 50 and len(by_tt[tt]) >= 3:
        step_med = statistics.median(by_tt[tt])
        exec_med = statistics.median(exec_by_tt.get(tt, [0])) if tt in exec_by_tt else 0
        prof_lat = prof_map.get(tt, 0)
        # Find nearest profile point if exact tt not in profile
        if prof_lat == 0 and prof_map:
            nearest = min(prof_map.keys(), key=lambda k: abs(k-tt))
            prof_lat = prof_map[nearest]
        err = (prof_lat - step_med) / step_med * 100 if step_med > 0 else 0
        print(f'{tt:>4} {prof_lat/1000:>10.1f} {step_med/1000:>10.1f} {exec_med/1000:>10.1f} {err:>+9.1f}% {len(by_tt[tt]):>6}')

print()
print('Key: profile = what emulator uses, step_med = actual step-cycle,')
print('     exec_med = GPU-only time, prof_err = profile vs step-cycle error')
print()
print('If prof_err is large negative → profile underestimates → emulator too fast')
print('If prof_err is large positive → profile overestimates → emulator too slow')
"
echo "DONE"
