#!/bin/bash
# Audit config-parity across all stages (real / profile / emu) of given
# result directories. Reads the "non-default args" line from every server
# log under each cell + its sibling profile pack dir, normalizes (strips
# the port — always differs), and reports any mismatch.
#
# Why this exists: Apr 22-24 we burned 3 days chasing a fake "A10
# saturation gap" that was actually `chain_clean_5rate_real_only.sh`
# omitting `--max-num-seqs 64` while the profile and emu validate both
# used 64. That single config mismatch made real hit conc=181 at r=32
# while emu was capped at 64 — looked like a real physics gap. See
# MEMORY.md / feedback_config_parity.md.
#
# Usage:
#   tools/parity_audit.sh <cell_dir> [<cell_dir> ...]
# Exit 0 if all OK, 1 if any violation.

set -u
EXIT=0

audit_cell() {
    local cell_dir="$1"
    [ -d "$cell_dir" ] || return 0

    local cell_name="$(basename "$cell_dir")"
    # Sibling profile dir: results/<HW>-adaptive-<cell_name>/...
    # Try common HW prefixes.
    local profile_dir=""
    for HW in RTX-8000 A10 L40S H100; do
        local cand="$(dirname "$cell_dir")/${HW}-adaptive-${cell_name}"
        if [ -d "$cand" ]; then
            profile_dir="$cand"
            break
        fi
    done

    # Collect all server_*.log files: bench logs (Stages 1+3) + profile
    # captures (Stage 2). Use depth 2 so logs/ subdirs are included.
    local logs
    logs="$(find "$cell_dir" -maxdepth 2 -name 'server_*.log' 2>/dev/null)"
    if [ -n "$profile_dir" ] && [ -d "$profile_dir" ]; then
        logs="$logs"$'\n'"$(find "$profile_dir" -maxdepth 2 -name 'server_*.log' 2>/dev/null)"
    fi

    local first_args=""
    local first_log=""
    local violations=""
    while IFS= read -r logfile; do
        [ -z "$logfile" ] && continue
        [ ! -f "$logfile" ] && continue
        local args
        # Strip the port — it always varies.
        args="$(grep 'non-default args' "$logfile" 2>/dev/null \
                | head -1 \
                | sed -E "s/.*non-default args: //" \
                | sed -E "s/'port':[^,]*, ?//")"
        [ -z "$args" ] && continue
        if [ -z "$first_args" ]; then
            first_args="$args"
            first_log="$logfile"
        elif [ "$args" != "$first_args" ]; then
            violations="$violations"$'\n  '"$(basename "$logfile"): $args"
            violations="$violations"$'\n  vs '"$(basename "$first_log"): $first_args"
            EXIT=1
        fi
    done <<< "$logs"

    if [ -z "$first_args" ]; then
        echo "PARITY SKIP $cell_name: no server logs with non-default args found"
    elif [ -z "$violations" ]; then
        echo "PARITY OK $cell_name: $first_args"
    else
        echo "PARITY VIOLATION $cell_name:$violations"
    fi
}

if [ $# -eq 0 ]; then
    echo "Usage: $0 <cell_dir> [<cell_dir> ...]" >&2
    exit 2
fi

for cell_dir in "$@"; do
    audit_cell "$cell_dir"
done

exit $EXIT
