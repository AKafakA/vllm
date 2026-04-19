#!/bin/bash
# Self-contained v6 overnight orchestration:
#   Phase A: 5-round single-session reprofile
#   Phase B: emu validate v6 vs real baseline at 5 rates
#   Invariant gate: v6 within ±2pp TPOT / ±3pp TTFT of archive on each rate
#     PASS → Phase C uses v6 for all features
#     FAIL → Phase C uses archive (F3/F5/F2) + v4 trace (F1/F4)
#   Phase C: 5 features at r=2,8 x 1000 prompts each
#     F1: 4-variant outlier sweep (none/iqr/mad/winsor)
#     F3: sample_tokens_delay
#     F5: kNN K=1 vs K=3
#     F2: parallel surrogate
#     F4: 2d vs 3d axis
#   Phase D: consolidate into paper/apr_19/10_overnight_results.md
#
# Launch: nohup bash tools/overnight_chain_v6.sh > results/overnight_v6.log 2>&1 &
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

MASTER_LOG="/tmp/vllm_overnight_v6.log"
echo "=== v6 overnight chain start $(date) ===" > "$MASTER_LOG"
touch /tmp/vllm_overnight_v6.started

ARCHIVE_PROFILE="./results/_archive/serving-dense.json"
V4_TRACE="./results/RTX-8000-adaptive-v4/step_cycle_trace.jsonl"
V6_PROFILE="./results/RTX-8000-adaptive-v6-5r-single/serving-full.json"
V6_TRACE="./results/RTX-8000-adaptive-v6-5r-single/step_cycle_trace.jsonl"

# -------- Phase A: reprofile --------
echo "" >> "$MASTER_LOG"
echo "=== Phase A: reprofile $(date) ===" >> "$MASTER_LOG"
bash tools/adaptive_profile_v6_5r_single.sh >> "$MASTER_LOG" 2>&1 || true

if [ ! -f "$V6_PROFILE" ]; then
    echo "  FATAL: Phase A did not produce $V6_PROFILE" >> "$MASTER_LOG"
    # continue to Phase C fallback path using archive + v4 trace
    INVARIANT_STATUS="SKIPPED"
else
    # -------- Phase B: invariant validation --------
    echo "" >> "$MASTER_LOG"
    echo "=== Phase B: invariant validation $(date) ===" >> "$MASTER_LOG"
    bash tools/validate_v6_full.sh >> "$MASTER_LOG" 2>&1 || true

    # -------- Invariant gate decision --------
    echo "" >> "$MASTER_LOG"
    echo "=== Invariant gate $(date) ===" >> "$MASTER_LOG"
    python3 << 'PYEOF' >> "$MASTER_LOG" 2>&1
import re
# Archive reference from paper/apr_18/02_v3_baseline_AB.md
ARCH = {2: (-0.6, -32.5), 4: (-1.6, -30.7), 8: (-6.0, -28.5),
        16: (-5.6, -25.0), 32: (+4.9, +1.9)}
TPOT_TOL = 2.0
TTFT_TOL = 3.0

summary_path = './results/RTX-8000-v6-validate/summary.txt'
try:
    s = open(summary_path).read()
except Exception as e:
    print(f"INVARIANT_STATUS: MISSING ({e})")
    open('/tmp/vllm_v6_invariant.status','w').write('MISSING\n')
    raise SystemExit(0)

per_rate = {}
for r, (arch_tpot, arch_ttft) in ARCH.items():
    m = re.search(rf'^\s*{r}\s*\|\s*([\-\+\.\d]+)%\s*\|\s*([\-\+\.\d]+)%', s, re.M)
    if not m:
        print(f"r={r}: missing in summary, FAIL")
        per_rate[r] = False
        continue
    tpot = float(m.group(1)); ttft = float(m.group(2))
    dtpot = abs(tpot - arch_tpot); dttft = abs(ttft - arch_ttft)
    ok = (dtpot <= TPOT_TOL) and (dttft <= TTFT_TOL)
    per_rate[r] = ok
    print(f"r={r}: v6 tpot={tpot} ttft={ttft}  arch tpot={arch_tpot} ttft={arch_ttft}  "
          f"dtpot={dtpot:.2f}pp dttft={dttft:.2f}pp  {'PASS' if ok else 'FAIL'}")

all_pass = all(per_rate.values())
status = 'PASS' if all_pass else 'FAIL'
print(f"INVARIANT_STATUS: {status}")
open('/tmp/vllm_v6_invariant.status','w').write(status + '\n')
PYEOF
    INVARIANT_STATUS=$(cat /tmp/vllm_v6_invariant.status 2>/dev/null | head -1 || echo MISSING)
fi
echo "  Invariant status: $INVARIANT_STATUS" >> "$MASTER_LOG"

# Select Phase C baselines based on invariant gate.
if [ "$INVARIANT_STATUS" = "PASS" ]; then
    BASELINE_PROFILE_RUNTIME="$V6_PROFILE"
    BASELINE_TRACE_BUILD="$V6_TRACE"
else
    BASELINE_PROFILE_RUNTIME="$ARCHIVE_PROFILE"
    BASELINE_TRACE_BUILD="$V4_TRACE"
fi
echo "  Phase C runtime baseline: $BASELINE_PROFILE_RUNTIME" >> "$MASTER_LOG"
echo "  Phase C build-step trace: $BASELINE_TRACE_BUILD" >> "$MASTER_LOG"

# Pre-build F1 and F4 variants from the chosen trace.
if [ -f "$BASELINE_TRACE_BUILD" ]; then
    for FLT in iqr mad winsor; do
        OUT="./results/overnight_f1_${FLT}_profile.json"
        echo "  [$(date +%T)] building F1 $FLT profile..." >> "$MASTER_LOG"
        python3 vllm_emulator/profile/build_serving_profile_filtered.py \
            "$BASELINE_TRACE_BUILD" "$OUT" \
            --tt-bucket-width 1 --conc-bucket-width 5 \
            --outlier-filter "$FLT" >> "$MASTER_LOG" 2>&1 || echo "    F1 $FLT build failed" >> "$MASTER_LOG"
    done
    echo "  [$(date +%T)] building F4 3D profile..." >> "$MASTER_LOG"
    python3 vllm_emulator/profile/build_serving_profile_filtered.py \
        "$BASELINE_TRACE_BUILD" "./results/overnight_f4_3d_profile.json" \
        --tt-bucket-width 1 --conc-bucket-width 5 \
        --profile-axes 3d --new-reqs-bucket-width 4 >> "$MASTER_LOG" 2>&1 || echo "    F4 3D build failed" >> "$MASTER_LOG"
fi

# -------- Phase C: feature ablations --------
echo "" >> "$MASTER_LOG"
echo "=== Phase C: feature ablations $(date) ===" >> "$MASTER_LOG"

PHASE_C_RATES="2 8"
PHASE_C_PROMPTS=1000

run_feature() {
    local NAME="$1"
    local PROF_A="$2"
    local PROF_B="$3"
    local ENV_A="$4"
    local ENV_B="$5"

    echo "" >> "$MASTER_LOG"
    echo "=== Feature $NAME start $(date) ===" >> "$MASTER_LOG"
    if [ ! -f "$PROF_B" ]; then
        echo "  SKIP $NAME — profile_B missing: $PROF_B" >> "$MASTER_LOG"
        return
    fi
    PROFILE="$PROF_A" PROFILE_B="$PROF_B" FEATURE="$NAME" \
    EXTRA_ENV_A="$ENV_A" EXTRA_ENV_B="$ENV_B" \
    RATES="$PHASE_C_RATES" NUM_PROMPTS="$PHASE_C_PROMPTS" \
        bash tools/validate_feature_ab.sh >> "$MASTER_LOG" 2>&1 || true
    # Rename default dirs (validate_feature_ab.sh uses -apr18 suffix) to -apr19.
    for SIDE in off on; do
        OLD="./results/RTX-8000-${NAME}-${SIDE}-apr18"
        NEW="./results/RTX-8000-${NAME}-${SIDE}-apr19"
        [ -d "$OLD" ] && [ ! -d "$NEW" ] && mv "$OLD" "$NEW"
    done
    echo "=== Feature $NAME done $(date) ===" >> "$MASTER_LOG"
}

# F1 — 4-variant outlier sweep. Pass A is baseline (unfiltered); Pass B is each filter.
# We implement the sweep as 3 separate feature-A/Bs, one per filter, sharing baseline A.
for FLT in iqr mad winsor; do
    run_feature "f1_${FLT}" \
        "$BASELINE_PROFILE_RUNTIME" "./results/overnight_f1_${FLT}_profile.json" \
        "" ""
done

# F3 — sample_tokens_delay (runtime)
run_feature "f3" \
    "$BASELINE_PROFILE_RUNTIME" "$BASELINE_PROFILE_RUNTIME" \
    "" "VLLM_EMULATOR_SAMPLE_TOKENS_DELAY=1"

# F5 — kNN (runtime): K=1 off vs K=3 on
run_feature "f5" \
    "$BASELINE_PROFILE_RUNTIME" "$BASELINE_PROFILE_RUNTIME" \
    "VLLM_EMULATOR_ORACLE_K=1" "VLLM_EMULATOR_ORACLE_K=3"

# F2 — parallel surrogate (runtime)
run_feature "f2" \
    "$BASELINE_PROFILE_RUNTIME" "$BASELINE_PROFILE_RUNTIME" \
    "" "VLLM_EMULATOR_PARALLEL_SURROGATE=1"

# F4 — 3D axis (build-step): 2d vs 3d
run_feature "f4" \
    "$BASELINE_PROFILE_RUNTIME" "./results/overnight_f4_3d_profile.json" \
    "VLLM_EMULATOR_PROFILE_AXES=2d" "VLLM_EMULATOR_PROFILE_AXES=3d"

# -------- Phase D: consolidate --------
echo "" >> "$MASTER_LOG"
echo "=== Phase D: summarize $(date) ===" >> "$MASTER_LOG"
bash tools/overnight_summarize_v6.sh >> "$MASTER_LOG" 2>&1 || true

echo "" >> "$MASTER_LOG"
echo "=== v6 overnight chain DONE $(date) ===" >> "$MASTER_LOG"
touch /tmp/vllm_overnight_v6.done
