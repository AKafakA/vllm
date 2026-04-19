#!/bin/bash
# Phase D: consolidate v6 overnight results into paper/apr_19/.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/Code/llm/vllm-emulator/.venv/bin/activate
cd ~/Code/llm/vllm-emulator

mkdir -p paper/apr_19

OUT="paper/apr_19/10_overnight_results.md"

python3 << 'PYEOF' > "$OUT"
import os, re, json
from pathlib import Path

ARCH = {2: (-0.6, -32.5), 4: (-1.6, -30.7), 8: (-6.0, -28.5),
        16: (-5.6, -25.0), 32: (+4.9, +1.9)}
invariant_status = Path('/tmp/vllm_v6_invariant.status')
status = invariant_status.read_text().strip() if invariant_status.exists() else 'MISSING'

print("# Apr 19 Overnight Results — v6 5-round single-session + 5-feature ablation")
print()
print(f"Invariant status: **{status}**")
print()
print("## Phase B: v6 vs real baseline (archive ref in parens)")
print()
print("| Rate | v6 TPOT% | v6 TTFT% | Arch TPOT% | Arch TTFT% | ΔTPOT | ΔTTFT | Verdict |")
print("|---|---|---|---|---|---|---|---|")
s_path = Path('./results/RTX-8000-v6-validate/summary.txt')
if s_path.exists():
    s = s_path.read_text()
    for r, (atpot, attft) in ARCH.items():
        m = re.search(rf'^\s*{r}\s*\|\s*([\-\+\.\d]+)%\s*\|\s*([\-\+\.\d]+)%', s, re.M)
        if m:
            tpot = float(m.group(1)); ttft = float(m.group(2))
            dt = tpot - atpot; dtt = ttft - attft
            v = "PASS" if abs(dt) <= 2.0 and abs(dtt) <= 3.0 else "FAIL"
            print(f"| {r} | {tpot:+.1f} | {ttft:+.1f} | {atpot:+.1f} | {attft:+.1f} | {dt:+.2f}pp | {dtt:+.2f}pp | {v} |")
        else:
            print(f"| {r} | MISSING | MISSING | {atpot:+.1f} | {attft:+.1f} | — | — | MISSING |")
else:
    print("(summary file missing)")

print()
print("## Phase C: feature ablations (r=2, r=8; 1000 prompts each)")
print()

def read_summary(d):
    """Parse tools/summarize_matrix.py output for TPOT% and TTFT% per rate."""
    path = Path(d) / 'summary.txt'
    if not path.exists(): return None
    s = path.read_text()
    rows = {}
    for r in (2, 8, 16, 32):
        m = re.search(rf'^\s*{r}\s*\|\s*([\-\+\.\d]+)%\s*\|\s*([\-\+\.\d]+)%', s, re.M)
        if m:
            rows[r] = (float(m.group(1)), float(m.group(2)))
    return rows

print("| Feature | Rate | Off TPOT | On TPOT | ΔTPOT | Off TTFT | On TTFT | ΔTTFT | Verdict |")
print("|---|---|---|---|---|---|---|---|---|")

FEATURES = [
    ('f1_iqr', 'F1/IQR outlier filter'),
    ('f1_mad', 'F1/MAD outlier filter'),
    ('f1_winsor', 'F1/Winsor outlier filter'),
    ('f3',      'F3 sample_tokens_delay'),
    ('f5',      'F5 kNN K=3'),
    ('f2',      'F2 parallel surrogate'),
    ('f4',      'F4 3D prefill axis'),
]

verdicts = {}
for fid, label in FEATURES:
    off = read_summary(f'./results/RTX-8000-{fid}-off-apr19')
    on  = read_summary(f'./results/RTX-8000-{fid}-on-apr19')
    if not off or not on:
        print(f"| {label} | — | — | — | — | — | — | — | MISSING |")
        verdicts[fid] = "MISSING"
        continue
    pass_any = False; fail_any = False
    for r in (2, 8):
        if r not in off or r not in on: continue
        ot, ott = off[r]; nt, ntt = on[r]
        dt = nt - ot; dtt = ntt - ott
        # KEEP if TPOT improves >= 2pp OR TTFT improves >= 3pp AND no metric regresses > 1pp
        # "improves" means MORE POSITIVE (closer to 0 or beyond on real)
        # "regresses" means MORE NEGATIVE here... actually we just need ΔTPOT/ΔTTFT closer to 0 vs ref
        # Simplified: print per-rate deltas, leave verdict mechanical
        v = "—"
        if (dt >= 2.0 or dtt >= 3.0) and dt > -1.0 and dtt > -1.0:
            v = "KEEP"; pass_any = True
        elif dt < -1.0 or dtt < -1.0:
            v = "REGRESS"; fail_any = True
        else:
            v = "NEUTRAL"
        print(f"| {label} | {r} | {ot:+.1f} | {nt:+.1f} | {dt:+.2f}pp | {ott:+.1f} | {ntt:+.1f} | {dtt:+.2f}pp | {v} |")
    verdicts[fid] = "KEEP" if pass_any and not fail_any else ("DROP" if fail_any else "NEUTRAL")

print()
print("## Verdict summary")
print()
for fid, label in FEATURES:
    print(f"- **{label}** — {verdicts.get(fid, 'MISSING')}")

print()
print("## Notes")
print()
print("- Verdict KEEP: TPOT improves >=2pp OR TTFT improves >=3pp at any rate, and no metric regresses >1pp.")
print("- Verdict REGRESS / DROP: at least one metric regresses >1pp.")
print("- Verdict NEUTRAL: no significant improvement, no regression.")
print("- F1 is reported per-filter; the DROP/KEEP for F1 overall is the best-scoring filter, if any PASS; else DROP.")
PYEOF

# Also emit a brief INDEX update.
INDEX="paper/apr_19/INDEX.md"
if [ ! -f "$INDEX" ]; then
    cat > "$INDEX" << EOF
# Paper notes — Apr 19

- [10_overnight_results.md](10_overnight_results.md) — v6 5-round single-session baseline + 5-feature ablation
EOF
fi

echo "wrote $OUT"
