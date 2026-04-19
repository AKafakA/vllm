#!/bin/bash
# Phase C re-runner on v6 baseline with all 5 features deployed from
# exp/combined-apr19. Trimmed to r=2, r=8 at 500 prompts each for time budget.
# Features: F1 (iqr/mad/winsor variants), F3, F5, F2, F4.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

MASTER_LOG="/tmp/vllm_phase_c_v6.log"
echo "=== Phase C v6 start $(date) ===" > "$MASTER_LOG"
touch /tmp/vllm_phase_c_v6.started

V6_PROFILE="./results/RTX-8000-adaptive-v6-5r-single/serving-full.json"

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
    RATES="2 8" NUM_PROMPTS="500" \
        bash tools/validate_feature_ab.sh >> "$MASTER_LOG" 2>&1 || true
    for SIDE in off on; do
        OLD="./results/RTX-8000-${NAME}-${SIDE}-apr18"
        NEW="./results/RTX-8000-${NAME}-${SIDE}-apr19"
        [ -d "$OLD" ] && [ ! -d "$NEW" ] && mv "$OLD" "$NEW"
    done
    echo "=== Feature $NAME done $(date) ===" >> "$MASTER_LOG"
}

# F1 — 3-variant outlier sweep
for FLT in iqr mad winsor; do
    run_feature "f1_${FLT}" \
        "$V6_PROFILE" "./results/overnight_f1_${FLT}_profile.json" \
        "" ""
done

# F3 — sample_tokens_delay (runtime). v6 profile has avg_sample_ms? check.
run_feature "f3" \
    "$V6_PROFILE" "$V6_PROFILE" \
    "" "VLLM_EMULATOR_SAMPLE_TOKENS_DELAY=1"

# F5 — kNN K=1 vs K=3 (runtime)
run_feature "f5" \
    "$V6_PROFILE" "$V6_PROFILE" \
    "VLLM_EMULATOR_ORACLE_K=1" "VLLM_EMULATOR_ORACLE_K=3"

# F2 — parallel surrogate (runtime)
run_feature "f2" \
    "$V6_PROFILE" "$V6_PROFILE" \
    "" "VLLM_EMULATOR_PARALLEL_SURROGATE=1"

# F4 — 3D axis (build-step)
run_feature "f4" \
    "$V6_PROFILE" "./results/overnight_f4_3d_profile.json" \
    "VLLM_EMULATOR_PROFILE_AXES=2d" "VLLM_EMULATOR_PROFILE_AXES=3d"

# Phase D: consolidate
echo "" >> "$MASTER_LOG"
echo "=== Phase D: summarize $(date) ===" >> "$MASTER_LOG"
bash tools/overnight_summarize_v6.sh >> "$MASTER_LOG" 2>&1 || true

echo "" >> "$MASTER_LOG"
echo "=== Phase C v6 DONE $(date) ===" >> "$MASTER_LOG"
touch /tmp/vllm_phase_c_v6.done
touch /tmp/vllm_overnight_v6.done
