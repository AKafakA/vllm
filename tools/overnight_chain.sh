#!/bin/bash
# Self-contained overnight orchestration:
#   1. Wait for v5 profile build to finish
#   2. Run v5 quick emu validation vs archive at r=2/8
#   3. Pick baseline (v5 if within ±3pp TPOT of archive, else archive)
#   4. Run F3/F5/F2 runtime ablations + F1/F4 build-step ablations
#   5. Summarize
#
# Launch: nohup bash tools/overnight_chain.sh > results/overnight_chain.log 2>&1 &
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

MASTER_LOG="/tmp/vllm_overnight_chain.log"
echo "=== Overnight chain start $(date) ===" > "$MASTER_LOG"
touch /tmp/vllm_overnight_chain.started

# Step 1: wait for v5 profile build to complete
echo "  [$(date +%T)] Step 1: waiting for /tmp/vllm_v5_oneround.done" >> "$MASTER_LOG"
for i in $(seq 1 180); do  # up to 90 min
    if [ -f /tmp/vllm_v5_oneround.done ]; then
        echo "  [$(date +%T)] v5 done, proceeding" >> "$MASTER_LOG"
        break
    fi
    sleep 30
done

if [ ! -f /tmp/vllm_v5_oneround.done ]; then
    echo "  [$(date +%T)] WARN: v5 not done after 90 min, proceeding with archive" >> "$MASTER_LOG"
    BASELINE_PROFILE="./results/_archive/serving-dense.json"
    BASELINE_TRACE="/nonexistent_archive_has_no_trace"
else
    # Step 2: v5 quick emu validation at r=2/8
    echo "" >> "$MASTER_LOG"
    echo "  [$(date +%T)] Step 2: v5 quick validation" >> "$MASTER_LOG"
    bash tools/validate_v5_quick.sh >> "$MASTER_LOG" 2>&1 || true

    # Step 3: pick baseline based on v5 vs archive accuracy
    # Archive reference:  r=2 TPOT -0.6  TTFT -32.5
    #                     r=8 TPOT -6.0  TTFT -28.5
    # Parse v5's summary.txt; compare; pick whichever passes.
    echo "" >> "$MASTER_LOG"
    echo "  [$(date +%T)] Step 3: baseline selection" >> "$MASTER_LOG"
    V5_SUMMARY="./results/RTX-8000-v5-quick/summary.txt"
    if [ -f "$V5_SUMMARY" ]; then
        cat "$V5_SUMMARY" >> "$MASTER_LOG"
        # Accept v5 if both r=2 and r=8 TPOT% are within ±5pp of archive reference.
        # This is a crude string-match check; full validation is in the result.
        python3 << 'PYEOF' >> "$MASTER_LOG" 2>&1
import re, sys, json, os
summary = open('./results/RTX-8000-v5-quick/summary.txt').read()
# Parse TPOT% rows for r=2 and r=8
accept = True
for rate in ('2', '8'):
    m = re.search(rf'^\s*{rate}\s*\|\s*([\-\+\.\d]+)', summary, re.M)
    if m:
        tpot = float(m.group(1))
        ref = {'2': -0.6, '8': -6.0}[rate]
        if abs(tpot - ref) > 5.0:
            print(f"  r={rate}: v5 TPOT {tpot}% vs archive {ref}% — delta {abs(tpot-ref):.1f}pp > 5pp threshold")
            accept = False
        else:
            print(f"  r={rate}: v5 TPOT {tpot}% vs archive {ref}% — delta {abs(tpot-ref):.1f}pp OK")
    else:
        print(f"  r={rate}: not found in summary, rejecting v5")
        accept = False
print("BASELINE_DECISION:", "v5" if accept else "archive")
open('/tmp/overnight_baseline.txt', 'w').write("v5" if accept else "archive")
PYEOF
    else
        echo "  v5 summary missing, falling back to archive" >> "$MASTER_LOG"
        echo "archive" > /tmp/overnight_baseline.txt
    fi

    DECISION=$(cat /tmp/overnight_baseline.txt 2>/dev/null || echo archive)
    if [ "$DECISION" = "v5" ]; then
        BASELINE_PROFILE="./results/RTX-8000-adaptive-v5-oneround/serving-full.json"
        BASELINE_TRACE="./results/RTX-8000-adaptive-v5-oneround/step_cycle_trace.jsonl"
    else
        BASELINE_PROFILE="./results/_archive/serving-dense.json"
        BASELINE_TRACE="/nonexistent_archive_has_no_trace"
    fi
    echo "  [$(date +%T)] BASELINE selected: $BASELINE_PROFILE" >> "$MASTER_LOG"
fi

# Step 4: overnight ablations
echo "" >> "$MASTER_LOG"
echo "  [$(date +%T)] Step 4: overnight ablations" >> "$MASTER_LOG"
BASELINE_PROFILE="$BASELINE_PROFILE" \
BASELINE_TRACE="$BASELINE_TRACE" \
    bash tools/overnight_ablations.sh >> "$MASTER_LOG" 2>&1

# Step 5: terminal marker
echo "" >> "$MASTER_LOG"
echo "=== Overnight chain DONE $(date) ===" >> "$MASTER_LOG"
touch /tmp/vllm_overnight_chain.done
