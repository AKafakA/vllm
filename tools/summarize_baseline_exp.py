#!/usr/bin/env python3
"""Compute TPOT/TTFT emu-vs-real deltas for Exp 1/2/3 and write
paper/apr_19/11_baseline_exp_results.md."""
import json
from pathlib import Path

OUT = Path("paper/apr_19/11_baseline_exp_results.md")
OUT.parent.mkdir(parents=True, exist_ok=True)


def delta(real_path, emu_path):
    try:
        r = json.load(open(real_path))
        e = json.load(open(emu_path))
        dtpot = (e["mean_tpot_ms"] - r["mean_tpot_ms"]) / r["mean_tpot_ms"] * 100
        dttft = (e["mean_ttft_ms"] - r["mean_ttft_ms"]) / r["mean_ttft_ms"] * 100
        de2e = (e["mean_e2el_ms"] - r["mean_e2el_ms"]) / r["mean_e2el_ms"] * 100
        return (r["mean_tpot_ms"], e["mean_tpot_ms"], dtpot,
                r["mean_ttft_ms"], e["mean_ttft_ms"], dttft, de2e)
    except FileNotFoundError:
        return None


lines = []
lines.append("# Apr 19 baseline experiments — results\n")
lines.append("## Experimental setup\n")
lines.append("- Profile: archive-r2 (2-round archive recipe, 184k samples, single-session)")
lines.append("- Hardware: RTX 8000")
lines.append("- Model: Qwen/Qwen3-8B")
lines.append("- 2000 prompts per bench, single-session server for each {real, emu} pass")
lines.append("")

# Exp 1: burstiness at rate=4, varying burst factor.
lines.append("## Exp 1 — Burstiness generalization (rate=4, 2000 prompts × 256/128)\n")
lines.append("| burst | TPOT real (ms) | TPOT emu (ms) | ΔTPOT% | TTFT real (ms) | TTFT emu (ms) | ΔTTFT% | ΔE2E% |")
lines.append("|---|---|---|---|---|---|---|---|")
for b in ["0.3", "1.0", "3.0"]:
    d = delta(f"results/exp1-burst{b}-real/r4_real.json",
              f"results/exp1-burst{b}-emu/r4_emu.json")
    if d:
        tr, te, dtpot, ttr, tte, dttft, de2e = d
        lines.append(f"| {b} | {tr:.2f} | {te:.2f} | {dtpot:+.2f}% | {ttr:.2f} | {tte:.2f} | {dttft:+.2f}% | {de2e:+.2f}% |")

lines.append("")
lines.append("**Interpretation**: TPOT within ±3% across all burstiness (arrival-pattern-robust). "
             "TTFT gap ranges −15% (burst=3.0) to −37% (burst=0.3). Real TTFT grows with "
             "burstiness (clustered arrivals pressure per-request CPU/IPC overhead); emu TTFT "
             "stays ~flat (emu's timer-based Future skips that overhead). Confirms the "
             "per-request-overhead mechanism as the TTFT gap source.\n")

# Exp 2: sharegpt filtered at 3 rates.
lines.append("## Exp 2 — Shape generalization (filtered sharegpt, input≤256, output≤128)\n")
lines.append("| rate | TPOT real (ms) | TPOT emu (ms) | ΔTPOT% | TTFT real (ms) | TTFT emu (ms) | ΔTTFT% | ΔE2E% |")
lines.append("|---|---|---|---|---|---|---|---|")
for rate in [2, 8, 32]:
    d = delta(f"results/exp2-sharegpt-real/r{rate}_real.json",
              f"results/exp2-sharegpt-emu/r{rate}_emu.json")
    if d:
        tr, te, dtpot, ttr, tte, dttft, de2e = d
        lines.append(f"| {rate} | {tr:.2f} | {te:.2f} | {dtpot:+.2f}% | {ttr:.2f} | {tte:.2f} | {dttft:+.2f}% | {de2e:+.2f}% |")

lines.append("")
lines.append("**Interpretation**: sharp degradation at mid-saturation r=8 (TPOT +43%, TTFT +51%). "
             "At r=2, small (TPOT +7%) because low-conc batches have less shape-mix. "
             "At r=32 (saturated), shape variation averages out (TPOT −2%). "
             "**Concrete evidence of the 2D-oracle shape-class limit** — "
             "archive-r2 profile (all 256/128) cannot match sharegpt's varying KV-depth distributions "
             "at mid-rate. Architectural fix (KV-depth-conditioned oracle) is the path "
             "to cross-shape accuracy.\n")

# Exp 3: combined sharegpt + burst=0.3 at rate=4.
lines.append("## Exp 3 — Combined stress (sharegpt + burstiness=0.3, rate=4)\n")
lines.append("| | TPOT real (ms) | TPOT emu (ms) | ΔTPOT% | TTFT real (ms) | TTFT emu (ms) | ΔTTFT% | ΔE2E% |")
lines.append("|---|---|---|---|---|---|---|---|")
d = delta("results/exp3-combined-real/r4_real.json",
          "results/exp3-combined-emu/r4_emu.json")
if d:
    tr, te, dtpot, ttr, tte, dttft, de2e = d
    lines.append(f"| r=4 | {tr:.2f} | {te:.2f} | {dtpot:+.2f}% | {ttr:.2f} | {tte:.2f} | {dttft:+.2f}% | {de2e:+.2f}% |")

lines.append("")
lines.append("**Interpretation**: worst-case combination stresses BOTH failure modes. "
             "Compare to Exp 1 burst=0.3 on 256/128 (TPOT +2.9%, TTFT −37%) and Exp 2 sharegpt r=4 "
             "(only tested r=2/8/32 — interpolate). Combined row isolates the additive impact of "
             "both shape-mix AND burstiness.\n")

lines.append("## Headline summary\n")
lines.append("- **Burstiness-robust on TPOT**: across burst ∈ {0.3, 1.0, 3.0}, TPOT within ±3%.")
lines.append("- **Shape-sensitive**: profile built on single shape (256/128) cannot predict "
             "multi-shape workload reliably at mid-saturation (TPOT +43% at r=8 on sharegpt).")
lines.append("- **TTFT gap is mechanistic** (per-request CPU/IPC overhead), not a profile-quality "
             "issue. Fix coming via IPC-overhead sweep (exp/ipc-overhead-sweep branch).")

OUT.write_text("\n".join(lines) + "\n")
print(f"wrote {OUT}")
