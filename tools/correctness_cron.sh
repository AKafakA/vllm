#!/bin/bash
# Correctness sweep cron — runs every 30 min during the experiment window.
# Three checks per fire:
#
#   A. Config parity — every active cell dir's server logs share identical
#      non-default args (real / profile / emu). Violation → halt that cell.
#   B. Accuracy gates — completed cells' MEAN-deltas (TPOT / ITL / E2E /
#      TTFT / tput) against paper gates (<6% / <6% / <6% / <15% / <1%).
#      Violation → enqueue rerun action.
#   C. Auto-rerun queue — write the next-action plan to rerun_queue.md.
#      The cron NEVER mid-run kills a bench; it only observes + queues.
#
# Differential rules:
#   - R3 (--no-prefix-caching) and TRITON (--attention-backend) cells
#     reuse profile packs captured under their cell-specific server config.
#     If they violate accuracy, first hypothesis is "stale profile" →
#     suggested action: reprofile+rebench.
#   - Burstiness reuses M2's profile. If it violates but M2 passes at the
#     same rate, that's a real burstiness-axis result (NOT stale profile)
#     → suggested action: investigate; do NOT auto-reprofile.
#
# Usage:
#   bash tools/correctness_cron.sh [<cell_dir> ...]
# If no args, scans ./results/ for cells with run.log.

set -uo pipefail
cd ~/Code/llm/vllm-emulator

PARITY_LOG="paper/apr_25/parity_log.md"
RERUN_QUEUE="paper/apr_25/rerun_queue.md"
mkdir -p "$(dirname "$PARITY_LOG")"
[ ! -f "$PARITY_LOG" ] && echo "# Parity audit log (auto-populated)" > "$PARITY_LOG"
[ ! -f "$RERUN_QUEUE" ] && echo "# Rerun queue (auto-populated)" > "$RERUN_QUEUE"

NOW="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Discover cells to audit. If args given, use them; else scan results/.
if [ $# -gt 0 ]; then
    CELL_DIRS=("$@")
else
    mapfile -t CELL_DIRS < <(find ./results -maxdepth 2 -name run.log -type f 2>/dev/null \
        | xargs -r -n1 dirname | sort -u)
fi

if [ "${#CELL_DIRS[@]}" -eq 0 ]; then
    echo "[$NOW] no cells to audit" >> "$PARITY_LOG"
    exit 0
fi

# ---- Check A: config parity ----
echo "" >> "$PARITY_LOG"
echo "## $NOW" >> "$PARITY_LOG"
PARITY_FAIL=0
for cell in "${CELL_DIRS[@]}"; do
    out="$(bash tools/parity_audit.sh "$cell" 2>&1)"
    echo "$out" >> "$PARITY_LOG"
    if echo "$out" | grep -q 'PARITY VIOLATION'; then
        PARITY_FAIL=$((PARITY_FAIL + 1))
        cell_name="$(basename "$cell")"
        # Halt marker: cell scripts should poll for this.
        touch "/tmp/correctness_halt_${cell_name}.flag"
        # Queue rerun.
        printf -- "- [%s] **CONFIG VIOLATION** %s — fix scripts before re-running\n" \
            "$NOW" "$cell_name" >> "$RERUN_QUEUE"
    fi
done

# ---- Check B: accuracy gates ----
TPOT_GATE=6.0; ITL_GATE=6.0; E2E_GATE=6.0
TTFT_GATE=15.0; TPUT_GATE=1.0
ACCURACY_FAIL=0

for cell in "${CELL_DIRS[@]}"; do
    DELTAS_CSV="$cell/per_rate_deltas.csv"
    [ ! -f "$DELTAS_CSV" ] && continue
    cell_name="$(basename "$cell")"

    # Decide differential rule for this cell name.
    case "$cell_name" in
        *r3*|*prefix-off*|*triton*|*TRITON*) ACTION="reprofile+rebench" ;;
        *burst*|*Burst*)                     ACTION="investigate (burstiness-axis)" ;;
        *)                                   ACTION="rebench" ;;
    esac

    # Iterate rate rows.
    python3 - "$cell" "$DELTAS_CSV" "$TPOT_GATE" "$ITL_GATE" "$E2E_GATE" \
                       "$TTFT_GATE" "$TPUT_GATE" "$ACTION" "$NOW" "$RERUN_QUEUE" <<'PY'
import csv, sys
from pathlib import Path
cell, csv_path, tpot_g, itl_g, e2e_g, ttft_g, tput_g, action, now, rerun_q = sys.argv[1:]
tpot_g, itl_g, e2e_g, ttft_g, tput_g = map(float, (tpot_g, itl_g, e2e_g, ttft_g, tput_g))
cell_name = Path(cell).name
violations = []
with open(csv_path) as f:
    rdr = csv.DictReader(f)
    for row in rdr:
        rate = row["rate"]
        ttft = abs(float(row["ttft_mean_pct"]))
        tpot = abs(float(row["tpot_mean_pct"]))
        itl  = abs(float(row["itl_mean_pct"]))
        e2e  = abs(float(row["e2e_mean_pct"]))
        tput = abs(float(row["tput_pct"]))
        bad = []
        if tpot > tpot_g: bad.append(f"TPOT={tpot:.2f}%")
        if itl  > itl_g:  bad.append(f"ITL={itl:.2f}%")
        if e2e  > e2e_g:  bad.append(f"E2E={e2e:.2f}%")
        if ttft > ttft_g: bad.append(f"TTFT={ttft:.2f}%")
        if tput > tput_g: bad.append(f"tput={tput:.2f}%")
        if bad:
            violations.append((rate, ", ".join(bad)))
if violations:
    with open(rerun_q, "a") as f:
        for rate, detail in violations:
            f.write(f"- [{now}] **ACCURACY** {cell_name} r={rate}: "
                    f"{detail} → suggested action: {action}\n")
    sys.exit(7)  # signal violation count to outer
PY
    if [ $? -eq 7 ]; then
        ACCURACY_FAIL=$((ACCURACY_FAIL + 1))
    fi
done

# ---- Summary ----
echo "[$NOW] cron sweep done: ${#CELL_DIRS[@]} cells; $PARITY_FAIL parity violations; $ACCURACY_FAIL cells with accuracy violations"
exit 0
