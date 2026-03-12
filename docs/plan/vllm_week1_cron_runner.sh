#!/usr/bin/env bash
set -euo pipefail

REPO="/home/jinx/.openclaw/workspace/projects/vllm-emulator"
PLAN_DIR="$REPO/docs/plan"
DAILY_DIR="$PLAN_DIR/daily"
LOG_DIR="$PLAN_DIR/logs"

mkdir -p "$DAILY_DIR" "$LOG_DIR"

TODAY_UTC="$(date -u +%F)"
NOW_UTC="$(date -u +"%F %T UTC")"
FILE="$DAILY_DIR/${TODAY_UTC}.md"

if [[ ! -f "$FILE" ]]; then
  cat > "$FILE" <<EOF
# Daily vllm-emulator Log — ${TODAY_UTC}

## 10:00 UTC Checkpoint
- Timestamp: ${NOW_UTC}
- Focus:
  - [ ] Current sprint task progress
  - [ ] Test status
  - [ ] Blockers

## Main-session Acceptance
- Report reviewed:
- Artifacts checked:
- Next action:
EOF
fi

{
  echo ""
  echo "---"
  echo "Checkpoint ping: ${NOW_UTC}"
  echo "- Continue vllm-emulator sprint work"
} >> "$FILE"

{
  echo "[$NOW_UTC] vllm checkpoint file=$FILE"
} >> "$LOG_DIR/cron.log" 2>&1
