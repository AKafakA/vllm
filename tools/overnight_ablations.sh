#!/bin/bash
# Overnight Path-1 ablations. Runs F5, F2, F3 (runtime) and F1, F4 (build-step)
# against a chosen baseline profile. Sequential (one server at a time on PORT).
#
# Usage:
#   BASELINE_PROFILE=./results/RTX-8000-adaptive-v5-oneround/serving-full.json \
#   BASELINE_TRACE=./results/RTX-8000-adaptive-v5-oneround/step_cycle_trace.jsonl \
#   bash tools/overnight_ablations.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

BASELINE_PROFILE="${BASELINE_PROFILE:?set BASELINE_PROFILE}"
BASELINE_TRACE="${BASELINE_TRACE:?set BASELINE_TRACE}"
RATES="${RATES:-2 8 16}"
NUM_PROMPTS="${NUM_PROMPTS:-1000}"
MASTER_LOG="/tmp/vllm_overnight.log"

echo "=== Overnight ablations start $(date) ===" > "$MASTER_LOG"
echo "  BASELINE_PROFILE=$BASELINE_PROFILE" >> "$MASTER_LOG"
echo "  BASELINE_TRACE=$BASELINE_TRACE" >> "$MASTER_LOG"

if [ ! -f "$BASELINE_PROFILE" ]; then
    echo "FATAL: baseline profile not found" | tee -a "$MASTER_LOG"
    exit 1
fi

# Build F1 variant profile (IQR filter) from the baseline's raw trace.
F1_PROFILE="./results/overnight_f1_iqr_profile.json"
if [ -f "$BASELINE_TRACE" ] && [ ! -f "$F1_PROFILE" ]; then
    echo "  [$(date +%T)] building F1 IQR profile..." >> "$MASTER_LOG"
    python3 vllm_emulator/profile/build_serving_profile_filtered.py \
        "$BASELINE_TRACE" "$F1_PROFILE" \
        --tt-bucket-width 1 --conc-bucket-width 5 \
        --outlier-filter iqr >> "$MASTER_LOG" 2>&1 || echo "    F1 build failed" >> "$MASTER_LOG"
fi

# Build F4 variant profile (3D axis) from the baseline's raw trace.
F4_PROFILE="./results/overnight_f4_3d_profile.json"
if [ -f "$BASELINE_TRACE" ] && [ ! -f "$F4_PROFILE" ]; then
    echo "  [$(date +%T)] building F4 3D profile..." >> "$MASTER_LOG"
    python3 vllm_emulator/profile/build_serving_profile_filtered.py \
        "$BASELINE_TRACE" "$F4_PROFILE" \
        --tt-bucket-width 1 --conc-bucket-width 5 \
        --profile-axes 3d --new-reqs-bucket-width 4 >> "$MASTER_LOG" 2>&1 || echo "    F4 build failed" >> "$MASTER_LOG"
fi

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
    PROFILE="$PROF_A" \
    PROFILE_B="$PROF_B" \
    FEATURE="$NAME" \
    EXTRA_ENV_A="$ENV_A" \
    EXTRA_ENV_B="$ENV_B" \
    RATES="$RATES" \
    NUM_PROMPTS="$NUM_PROMPTS" \
        bash tools/validate_feature_ab.sh >> "$MASTER_LOG" 2>&1
    echo "=== Feature $NAME done $(date) ===" >> "$MASTER_LOG"
}

# F3 — sample_tokens delay (runtime env). Requires avg_sample_ms in profile.
run_feature "f3" \
    "$BASELINE_PROFILE" "$BASELINE_PROFILE" \
    "" "VLLM_EMULATOR_SAMPLE_TOKENS_DELAY=1"

# F5 — kNN oracle conditioning (runtime env).
run_feature "f5" \
    "$BASELINE_PROFILE" "$BASELINE_PROFILE" \
    "VLLM_EMULATOR_ORACLE_K=1" "VLLM_EMULATOR_ORACLE_K=3"

# F2 — parallel surrogate (runtime env).
run_feature "f2" \
    "$BASELINE_PROFILE" "$BASELINE_PROFILE" \
    "" "VLLM_EMULATOR_PARALLEL_SURROGATE=1"

# F1 — IQR outlier filter (build-step).
run_feature "f1" \
    "$BASELINE_PROFILE" "$F1_PROFILE" \
    "" ""

# F4 — 3D prefill axis (build-step).
run_feature "f4" \
    "$BASELINE_PROFILE" "$F4_PROFILE" \
    "VLLM_EMULATOR_PROFILE_AXES=2d" "VLLM_EMULATOR_PROFILE_AXES=3d"

echo "" >> "$MASTER_LOG"
echo "=== Overnight ablations DONE $(date) ===" >> "$MASTER_LOG"
touch "/tmp/vllm_overnight.done"
