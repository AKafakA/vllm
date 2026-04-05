#!/bin/bash
# Diagnose rate=1 TPOT and rate=4 TTFT issues
# Trace actual step latencies and batch sizes at each rate
set -e
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"
PROFILE="${RESULT_DIR}/profiles/serving-1.5b-tp1-v3.json"
PORT=8100

pkill -9 -f EngineCore 2>/dev/null || true; pkill -9 -f api_server 2>/dev/null || true; sleep 5

echo "=== Trace step latencies per rate ==="
rm -f "${RESULT_DIR}/diag_per_rate_r1.jsonl" "${RESULT_DIR}/diag_per_rate_r4.jsonl"

# Rate=1 trace
echo "  Rate=1 tracing..."
VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
VLLM_EMULATOR_STEP_TRACE_OUTPUT="${RESULT_DIR}/diag_per_rate_r1.jsonl" \
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 4096 --port ${PORT} --trust-remote-code \
    > /workspace/diag_r1_server.log 2>&1 &
for i in $(seq 1 120); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then break; fi; sleep 1
done
python3 -m vllm.entrypoints.cli.main bench serve --model "${MODEL}" --base-url http://localhost:${PORT} \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 50 --request-rate 1 --percentile-metrics ttft,tpot --metric-percentiles 50,99 2>&1 | grep "TPOT"
pkill -f api_server 2>/dev/null || true; sleep 2; pkill -9 -f EngineCore 2>/dev/null || true; sleep 5

# Rate=4 trace
echo "  Rate=4 tracing..."
VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
VLLM_EMULATOR_STEP_TRACE_OUTPUT="${RESULT_DIR}/diag_per_rate_r4.jsonl" \
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --max-model-len 4096 --port ${PORT} --trust-remote-code \
    > /workspace/diag_r4_server.log 2>&1 &
for i in $(seq 1 120); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then break; fi; sleep 1
done
python3 -m vllm.entrypoints.cli.main bench serve --model "${MODEL}" --base-url http://localhost:${PORT} \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 50 --request-rate 4 --percentile-metrics ttft,tpot --metric-percentiles 50,99 2>&1 | grep "TTFT\|TPOT"
pkill -f api_server 2>/dev/null || true; sleep 2; pkill -9 -f EngineCore 2>/dev/null || true; sleep 5

echo ""
echo "=== Analysis ==="
python3 -c "
import json, statistics
from collections import defaultdict

profile = json.load(open('${PROFILE}'))
prof_map = {e['total_tokens']: e['latency_us'] for e in profile['forward_pass']}

for rate, fname in [(1, 'diag_per_rate_r1.jsonl'), (4, 'diag_per_rate_r4.jsonl')]:
    records = []
    for line in open(f'${RESULT_DIR}/{fname}'):
        r = json.loads(line)
        if 'total_tokens' in r:
            records.append(r)

    by_tt = defaultdict(list)
    for r in records:
        by_tt[r['total_tokens']].append(r['step_cycle_us'])

    print(f'\\nRate={rate}: {len(records)} steps, tt range={min(by_tt)}-{max(by_tt)}')
    print(f'{\"tt\":>4} {\"actual_ms\":>10} {\"profile_ms\":>11} {\"error\":>8} {\"n\":>6}')
    for tt in sorted(by_tt):
        if len(by_tt[tt]) >= 3:
            actual = statistics.median(by_tt[tt])
            # Oracle uses piecewise interpolation — get the value it would return
            prof = prof_map.get(tt, 0)
            if prof == 0:
                # Interpolate from nearest
                tts = sorted(prof_map.keys())
                for i in range(len(tts)-1):
                    if tts[i] <= tt <= tts[i+1]:
                        ratio = (tt - tts[i]) / (tts[i+1] - tts[i])
                        prof = prof_map[tts[i]] + ratio * (prof_map[tts[i+1]] - prof_map[tts[i]])
                        break
            err = (prof - actual) / actual * 100 if actual > 0 and prof > 0 else 0
            marker = ' ✗' if abs(err) > 10 else ''
            print(f'{tt:>4} {actual/1000:>10.1f} {prof/1000:>11.1f} {err:>+7.1f}%{marker} {len(by_tt[tt]):>6}')

    # Show tt distribution
    total_steps = len(records)
    ranges = [(1,5), (5,10), (10,17), (17,30), (30,50)]
    print(f'  tt distribution:')
    for lo, hi in ranges:
        count = sum(len(by_tt[tt]) for tt in by_tt if lo <= tt < hi)
        pct = count/total_steps*100
        if pct > 1:
            print(f'    tt={lo}-{hi}: {count} steps ({pct:.0f}%)')
"
echo "DONE"
