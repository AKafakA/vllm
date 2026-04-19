#!/usr/bin/env python3
"""Build TTFT-variant comparison table vs archive-r2 baseline + IPC-current."""
import json
from pathlib import Path

RATES = [2, 4, 8, 16, 32]

# Baselines from context (prior 5-rate validations).
archive_r2 = {
    2: (-0.02, -31.65), 4: (-0.56, -29.16), 8: (-2.52, -23.47),
    16: (+0.04, -6.35), 32: (+1.11, -1.14),
}
ipc_current = {
    2: (+6.92, -8.67), 4: (+16.76, -4.20), 8: (+15.95, -5.87),
    16: (+5.82, +8.15), 32: (+6.57, +3.01),
}


def delta(real_p, emu_p):
    r, e = json.load(open(real_p)), json.load(open(emu_p))
    dtpot = (e["mean_tpot_ms"] - r["mean_tpot_ms"]) / r["mean_tpot_ms"] * 100
    dttft = (e["mean_ttft_ms"] - r["mean_ttft_ms"]) / r["mean_ttft_ms"] * 100
    return (dtpot, dttft)


def variant_deltas(dirname):
    out = {}
    for r in RATES:
        real_p = Path(f"results/{dirname}/r{r}_real.json")
        emu_p = Path(f"results/{dirname}/r{r}_emu.json")
        if real_p.exists() and emu_p.exists():
            out[r] = delta(real_p, emu_p)
    return out


out = Path("paper/apr_19/14_ttft_variants.md")
out.parent.mkdir(parents=True, exist_ok=True)
L = []
L.append("# Apr 19 — TTFT overnight variant chain\n")
L.append("## Setup\n")
L.append("- Profile: archive-r2 with `sched_overhead_table` merged in (IPC sweep 19:17).")
L.append("- Variant v2-additive-divide: oracle adds `IPC_overhead(N)/num_new_reqs` to prefill-step")
L.append("  latency. Async chain-timer architecture preserved (no synchronous engine-thread sleep).")
L.append("- Hardware: RTX 8000, Model: Qwen/Qwen3-8B, 2000 prompts × 5 rates (random 256/128).\n")

variants = {
    "v2-additive-divide": variant_deltas("ttft-variant-v2-additive-divide"),
}

L.append("## Variant 5-rate matrix\n")
L.append("| rate | archive-r2 TPOT% | archive-r2 TTFT% | IPC-current TPOT% | IPC-current TTFT% | v2 TPOT% | v2 TTFT% |")
L.append("|---|---|---|---|---|---|---|")
for r in RATES:
    a_tp, a_tt = archive_r2.get(r, (None, None))
    i_tp, i_tt = ipc_current.get(r, (None, None))
    v = variants["v2-additive-divide"].get(r)
    v_tp = f"{v[0]:+.2f}%" if v else "—"
    v_tt = f"{v[1]:+.2f}%" if v else "—"
    L.append(f"| {r} | {a_tp:+.2f}% | {a_tt:+.2f}% | {i_tp:+.2f}% | {i_tt:+.2f}% | {v_tp} | {v_tt} |")
L.append("")

L.append("## Verdict\n")
L.append("**Winner: archive-r2 (no IPC injection) remains the best overall setting.**\n")
L.append("- **v2 vs IPC-current**: v2 mildly improves TPOT at mid-rates (r=4 +14.7 vs +16.8; r=8 "
         "+14.3 vs +16.0) and mildly improves TTFT (r=4 −7.8 vs −4.2; r=8 −11.1 vs −5.9). "
         "Division by num_new_reqs does *not* rescue the TPOT regression introduced by additive "
         "overhead in the oracle latency path.")
L.append("- **v2 vs archive-r2**: archive-r2 wins TPOT on every rate (|Δ|≤2.5% vs v2's +5 to +15%). "
         "v2 wins TTFT at r=2/4/8 (smaller magnitude gap) but overshoots TTFT at r=16 (+1%) and "
         "r=32 (+4%) where archive-r2 had it close to zero.")
L.append("- **Architectural implication**: additive oracle-latency injection contaminates scheduler "
         "feedback (downstream chain-timer cascades). The correct place for per-request IPC "
         "overhead is either (a) a synchronous block in the engine thread at prefill-step "
         "boundaries (blocks `num_output_placeholders` the way real IPC does), OR (b) left absent "
         "— accepting the TTFT gap as a known residual and investing that effort elsewhere.")
L.append("- **Tonight's decision**: revert oracle to pre-IPC state; keep `sched_overhead_table` "
         "captured in profile pack for future architectural-fix variants (Apr 20 Direction A).\n")

L.append("## Comparison to archive-r2 baseline shape-sensitivity (Exp 2)\n")
L.append("The TTFT gap of −32 to −23% at r=2/4/8 in archive-r2 is the *mechanistic TTFT gap*. "
         "Exp 2 on filtered sharegpt (paper/apr_19/13_exp2_fullrate.md) shows TPOT deltas up to "
         "+110% at r=16 from *shape generalisation*. The two failure modes are orthogonal — "
         "fixing one does not improve the other. Apr 20 focus should be shape-generalisation "
         "(α-KV model extension), since it affects more metrics and is a more fundamental "
         "oracle model error.")

out.write_text("\n".join(L) + "\n")
print(f"wrote {out}")
