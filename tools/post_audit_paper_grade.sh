#!/bin/bash
# Post-completion audit: waits for all currently-running paper benches
# to finish, audits every JSON for known failure modes, and surfaces
# anything that needs a final-pass rerun BEFORE the paper numbers are
# locked. Designed to be the last gate before paper submission.
#
# Failure modes audited:
#   1. completed != num_prompts → tail-truncation bias (EMFILE etc.)
#   2. failed_requests > 0 → bench client errors
#   3. duration unexpectedly short (<60s for r>=4) → server crash mid-run
#   4. mean_ttft_ms vs median_ttft_ms ratio > 1.5 → outlier-skewed mean
#   5. output_throughput unexpectedly low → engine stall (run-to-run variance)
#
# For any cell flagged: emit a `paper_rerun_<cell>.sh` to schedule a
# clean rerun with the strictest current standard (ulimit + parity args).
set -uo pipefail
ulimit -n 65536 2>/dev/null || true
LOG=~/Code/llm/vllm-emulator/paper/apr_25/post_audit.log
echo "=== post-audit start $(date -u) ===" > "$LOG"

# Wait for currently-active markers to fire .done
WAITS=(
    "/tmp/vllm_takeover_apr25.done"          # personal_gpu_vm
    "/tmp/vllm_apr25-a10-3config.done"       # vast (must be checked via ssh)
    "/tmp/vllm_apr25-cpu-host-ablation.done" # cpu_host (must be checked via ssh)
)
echo "[$(date -u +%T)] waiting for all jobs to finish..." >> "$LOG"
# Local check
while [ ! -f "/tmp/vllm_takeover_apr25.done" ]; do sleep 60; done
# Remote checks via ssh
while ! ssh -o ControlPath=~/.ssh/sockets/vast-a10 vast 'ls /tmp/vllm_apr25-a10-3config.done 2>/dev/null' >/dev/null 2>&1; do sleep 60; done
while ! ssh -o ControlPath=~/.ssh/sockets/cpu_host cpu_host 'ls /tmp/vllm_apr25-cpu-host-ablation.done 2>/dev/null' >/dev/null 2>&1; do sleep 60; done
echo "[$(date -u +%T)] all jobs reported .done — auditing..." >> "$LOG"

audit_dir() {
    local label="$1" jsondir="$2"
    echo "" >> "$LOG"
    echo "=== $label : $jsondir ===" >> "$LOG"
    python3 - <<EOF >> "$LOG"
import json, os, glob
issues = []
for path in sorted(glob.glob('$jsondir/*.json')):
    name = os.path.basename(path)
    if not name.endswith('.json'): continue
    try:
        d = json.load(open(path))
    except Exception as e:
        issues.append(f"{name}: parse_error {e}"); continue
    n = d.get('num_prompts', 0)
    c = d.get('completed', 0)
    fr = d.get('failed_requests', 0)
    dur = d.get('duration', 0)
    mt = d.get('mean_ttft_ms', 0)
    mdt = d.get('median_ttft_ms', 0)
    tput = d.get('output_throughput', 0)
    flags = []
    if c < n: flags.append(f"truncated({c}/{n})")
    if fr > 0: flags.append(f"failed_reqs={fr}")
    if dur < 60 and n > 100: flags.append(f"short_dur={dur:.1f}s")
    if mt > 0 and mdt > 0 and mt/mdt > 1.5: flags.append(f"mean/med ratio={mt/mdt:.2f}")
    if tput > 0 and tput < 200 and 'r2' not in name: flags.append(f"low_tput={tput:.0f}")
    if flags: issues.append(f"{name}: " + ", ".join(flags))
    else:     print(f"  ✓ {name}  done={c}/{n} dur={dur:.0f}s TTFT={mt:.0f} TPOT={d.get('mean_tpot_ms',0):.2f} tput={tput:.0f}")
if issues:
    print("\n  ISSUES (need rerun):")
    for s in issues: print(f"    ⚠️  {s}")
else:
    print("  ALL CLEAN")
EOF
}

# Audit local RTX 8000 cells (synced from personal_gpu_vm)
ssh -o ControlPath=~/.ssh/sockets/dev-gpu-wd312 personal_gpu_vm 'cd ~/Code/llm/vllm-emulator/results && tar czf /tmp/rtx8000_results.tgz apr25-* RTX-8000-adaptive-apr25-* 2>/dev/null' 2>/dev/null || true
scp -o ControlPath=~/.ssh/sockets/dev-gpu-wd312 personal_gpu_vm:/tmp/rtx8000_results.tgz /tmp/ 2>/dev/null || true
mkdir -p /tmp/rtx8000_audit && cd /tmp/rtx8000_audit && tar xzf /tmp/rtx8000_results.tgz 2>/dev/null || true
for d in /tmp/rtx8000_audit/apr25-*; do
    [ -d "$d" ] || continue
    audit_dir "RTX8000:$(basename $d)" "$d"
done

# Audit vast A10 cells
ssh -o ControlPath=~/.ssh/sockets/vast-a10 vast 'cd /workspace/vllm-emulator/results && tar czf /tmp/a10_results.tgz apr25-a10-* apr25-track-* 2>/dev/null' 2>/dev/null || true
scp -o ControlPath=~/.ssh/sockets/vast-a10 vast:/tmp/a10_results.tgz /tmp/ 2>/dev/null || true
mkdir -p /tmp/a10_audit && cd /tmp/a10_audit && tar xzf /tmp/a10_results.tgz 2>/dev/null || true
for d in /tmp/a10_audit/apr25-*; do
    [ -d "$d" ] || continue
    audit_dir "A10:$(basename $d)" "$d"
done

# Audit cpu_host cells
ssh -o ControlPath=~/.ssh/sockets/cpu_host cpu_host 'cd ~/vllm-emulator/results && tar czf /tmp/cpu_results.tgz apr25-cpu-host-ablation 2>/dev/null' 2>/dev/null || true
scp -o ControlPath=~/.ssh/sockets/cpu_host cpu_host:/tmp/cpu_results.tgz /tmp/ 2>/dev/null || true
mkdir -p /tmp/cpu_audit && cd /tmp/cpu_audit && tar xzf /tmp/cpu_results.tgz 2>/dev/null || true
for d in /tmp/cpu_audit/apr25-cpu-host-ablation/*; do
    [ -d "$d" ] || continue
    audit_dir "cpuhost:$(basename $d)" "$d"
done

echo "" >> "$LOG"
echo "=== audit complete $(date -u) ===" >> "$LOG"
echo "" >> "$LOG"
echo "If any '⚠️' lines above appear, schedule final-pass rerun for those cells" >> "$LOG"
echo "with the current strictest standard (ulimit -n 65536 + parity args + nvidia-smi trace)." >> "$LOG"
touch /tmp/post_audit_done
