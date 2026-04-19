#!/usr/bin/env python3
"""Build 5-rate sharegpt matrix from exp2 (r=2/8/32) + supp (r=4/16)."""
import json
from pathlib import Path


def load(p):
    try:
        return json.load(open(p))
    except FileNotFoundError:
        return None


def delta(real_p, emu_p):
    r, e = load(real_p), load(emu_p)
    if r is None or e is None:
        return None
    dtpot = (e["mean_tpot_ms"] - r["mean_tpot_ms"]) / r["mean_tpot_ms"] * 100
    dttft = (e["mean_ttft_ms"] - r["mean_ttft_ms"]) / r["mean_ttft_ms"] * 100
    de2e = (e["mean_e2el_ms"] - r["mean_e2el_ms"]) / r["mean_e2el_ms"] * 100
    return (r["mean_tpot_ms"], e["mean_tpot_ms"], dtpot,
            r["mean_ttft_ms"], e["mean_ttft_ms"], dttft, de2e)


def pick(rate):
    if rate in (2, 8, 32):
        return (f"results/exp2-sharegpt-real/r{rate}_real.json",
                f"results/exp2-sharegpt-emu/r{rate}_emu.json")
    return (f"results/supp-sharegpt-r{rate}-real/r{rate}_real.json",
            f"results/supp-sharegpt-r{rate}-emu/r{rate}_emu.json")


out = Path("paper/apr_19/13_exp2_fullrate.md")
out.parent.mkdir(parents=True, exist_ok=True)
L = []
L.append("# Apr 19 — Exp 2 full 5-rate sharegpt matrix\n")
L.append("## Setup\n")
L.append("- Profile: archive-r2 (2-round archive, 184k samples)")
L.append("- Hardware: RTX 8000, Model: Qwen/Qwen3-8B")
L.append("- 2000 prompts per bench, filtered sharegpt (input≤256, output≤128)")
L.append("- r=2/8/32 from `exp2-sharegpt-*` (baseline Exp 2), r=4/16 from `supp-sharegpt-*` (gap-fill)\n")

L.append("## 5-rate sharegpt results\n")
L.append("| rate | TPOT real (ms) | TPOT emu (ms) | ΔTPOT% | TTFT real (ms) | TTFT emu (ms) | ΔTTFT% | ΔE2E% |")
L.append("|---|---|---|---|---|---|---|---|")
for rate in [2, 4, 8, 16, 32]:
    real_p, emu_p = pick(rate)
    d = delta(real_p, emu_p)
    if d:
        tr, te, dtpot, ttr, tte, dttft, de2e = d
        L.append(f"| {rate} | {tr:.2f} | {te:.2f} | {dtpot:+.2f}% | {ttr:.2f} | {tte:.2f} | {dttft:+.2f}% | {de2e:+.2f}% |")
    else:
        L.append(f"| {rate} | MISSING | | | | | | |")
L.append("")

L.append("## Interpretation\n")
L.append("- **r=2 light load**: TPOT gap modest (small KV-mix variance at low conc).")
L.append("- **r=4–8 mid-saturation**: worst TPOT gap — archive-r2 profile (built on 256/128 random) "
         "cannot predict sharegpt's variable KV-depth at moderate batching. Per-bucket sample "
         "mismatch amplifies at mid-concurrency.")
L.append("- **r=16–32 saturation**: gap shrinks — at high conc, many shape classes average out "
         "to profile-bucket statistics; saturation ceiling is shape-agnostic.")
L.append("- **Architectural verdict**: 2D oracle is KV-depth-blind; shape generalisation cap is "
         "structural, not a sample-density issue. Fix path: KV-depth-conditioned oracle "
         "(α-KV from model_config extension or 3D axis on sum_kv).\n")

L.append("## Comparison to random 256/128 (archive-r2 baseline)\n")
L.append("| rate | ΔTPOT sharegpt | ΔTPOT random | ΔTTFT sharegpt | ΔTTFT random |")
L.append("|---|---|---|---|---|")
# Archive-r2 random 256/128 baseline from previous validation.
random_baseline = {
    2: (-0.02, -31.65), 4: (-0.56, -29.16), 8: (-2.52, -23.47),
    16: (+0.04, -6.35), 32: (+1.11, -1.14),
}
for rate in [2, 4, 8, 16, 32]:
    real_p, emu_p = pick(rate)
    d = delta(real_p, emu_p)
    if d and rate in random_baseline:
        _, _, dtpot, _, _, dttft, _ = d
        rt, rtf = random_baseline[rate]
        L.append(f"| {rate} | {dtpot:+.2f}% | {rt:+.2f}% | {dttft:+.2f}% | {rtf:+.2f}% |")
L.append("")
L.append("**Shape-sensitivity signal**: sharegpt TPOT deltas are larger-magnitude at mid-rates, "
         "confirming 2D-oracle shape-class limit. TTFT deltas follow similar pattern because "
         "per-request IPC overhead was not yet applied at time of measurement.")

out.write_text("\n".join(L) + "\n")
print(f"wrote {out}")
