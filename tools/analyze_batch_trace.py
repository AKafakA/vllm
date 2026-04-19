#!/usr/bin/env python3
"""Compare per-step batch composition between hook-on and hook-off emu runs.

Reads two executor-hook trace CSVs (columns: step, wall_s, tt, n_reqs,
n_decode, n_new, has_prefill, oracle_us, timer_delay_us). Outputs
histograms/summary to paper/apr_20/01_batch_composition_r16.md.
"""
import csv
import statistics
import sys
from pathlib import Path

HOOK = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/trace_hook.csv")
NOHOOK = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/tmp/trace_nohook.csv")
OUT = Path("paper/apr_20/01_batch_composition_r16.md")
OUT.parent.mkdir(parents=True, exist_ok=True)


def load(p):
    rows = []
    with open(p) as f:
        reader = csv.DictReader(f)
        for r in reader:
            for k in ("step", "tt", "n_reqs", "n_decode", "n_new", "has_prefill"):
                r[k] = int(r[k])
            for k in ("wall_s", "oracle_us", "timer_delay_us"):
                r[k] = float(r[k])
            rows.append(r)
    return rows


def summary(rows, label):
    if not rows:
        return {"label": label, "steps": 0}
    n_steps = len(rows)
    n_prefill_steps = sum(1 for r in rows if r["has_prefill"])
    n_new_per_step = [r["n_new"] for r in rows]
    n_reqs_per_step = [r["n_reqs"] for r in rows]
    n_decode_per_step = [r["n_decode"] for r in rows]
    tt_per_step = [r["tt"] for r in rows]
    oracle_us = [r["oracle_us"] for r in rows]

    pure_decode = sum(1 for r in rows if r["n_new"] == 0 and not r["has_prefill"])
    mixed = sum(1 for r in rows if r["n_new"] >= 1 and r["n_decode"] >= 1)
    pure_prefill = sum(1 for r in rows if r["n_new"] >= 1 and r["n_decode"] == 0)

    return {
        "label": label,
        "steps": n_steps,
        "prefill_steps": n_prefill_steps,
        "pure_decode_steps": pure_decode,
        "mixed_steps": mixed,
        "pure_prefill_steps": pure_prefill,
        "n_reqs_mean": statistics.mean(n_reqs_per_step),
        "n_reqs_p50": statistics.median(n_reqs_per_step),
        "n_reqs_max": max(n_reqs_per_step),
        "n_decode_mean": statistics.mean(n_decode_per_step),
        "n_new_mean": statistics.mean(n_new_per_step),
        "n_new_max": max(n_new_per_step),
        "tt_mean": statistics.mean(tt_per_step),
        "tt_p90": sorted(tt_per_step)[int(0.9 * len(tt_per_step))],
        "oracle_us_mean": statistics.mean(oracle_us),
    }


def hist_n_new(rows):
    """Distribution of n_new per step."""
    counts = {}
    for r in rows:
        counts[r["n_new"]] = counts.get(r["n_new"], 0) + 1
    total = sum(counts.values())
    return [(k, counts[k], counts[k] / total * 100) for k in sorted(counts.keys())]


L = []
L.append("# Apr 20 — Batch composition diagnostic, r=16 × 2000 prompts\n")
L.append("Compares per-step batch composition between emu with v3 arrival-delay "
         "hook (SCHEDULER_HOOK=1) and emu with hook disabled (SCHEDULER_HOOK=0), "
         "same random 256/128 workload, same profile (archive-r2).\n")

hook_rows = load(HOOK) if HOOK.exists() else []
nohook_rows = load(NOHOOK) if NOHOOK.exists() else []
h = summary(hook_rows, "hook_on")
n = summary(nohook_rows, "hook_off")

L.append("## Summary\n")
L.append("| metric | hook_on | hook_off | delta |")
L.append("|---|---|---|---|")
for key in ("steps", "prefill_steps", "pure_decode_steps", "mixed_steps",
            "pure_prefill_steps", "n_reqs_mean", "n_reqs_p50", "n_reqs_max",
            "n_decode_mean", "n_new_mean", "n_new_max", "tt_mean", "tt_p90",
            "oracle_us_mean"):
    ho = h.get(key, "—")
    no = n.get(key, "—")
    if isinstance(ho, (int, float)) and isinstance(no, (int, float)):
        fmt = "%.1f" if isinstance(ho, float) else "%d"
        d = ho - no
        L.append(f"| {key} | {fmt % ho} | {fmt % no} | {d:+.1f} |")
    else:
        L.append(f"| {key} | {ho} | {no} | — |")

L.append("")
L.append("## n_new distribution per step\n")
L.append("| n_new | hook_on count | hook_on % | hook_off count | hook_off % |")
L.append("|---|---|---|---|---|")
ho_hist = {k: (c, p) for k, c, p in hist_n_new(hook_rows)} if hook_rows else {}
no_hist = {k: (c, p) for k, c, p in hist_n_new(nohook_rows)} if nohook_rows else {}
all_keys = sorted(set(ho_hist.keys()) | set(no_hist.keys()))
for k in all_keys:
    hc, hp = ho_hist.get(k, (0, 0.0))
    nc, np_ = no_hist.get(k, (0, 0.0))
    L.append(f"| {k} | {hc} | {hp:.1f}% | {nc} | {np_:.1f}% |")

L.append("")
L.append("## Interpretation\n")

# Derive verdict.
if hook_rows and nohook_rows:
    ho_pure_pct = h["pure_decode_steps"] / h["steps"] * 100
    no_pure_pct = n["pure_decode_steps"] / n["steps"] * 100
    delta_pure_pct = ho_pure_pct - no_pure_pct

    ho_mixed_pct = h["mixed_steps"] / h["steps"] * 100
    no_mixed_pct = n["mixed_steps"] / n["steps"] * 100
    delta_mixed_pct = ho_mixed_pct - no_mixed_pct

    L.append(f"- hook_on: pure-decode steps = {ho_pure_pct:.1f}%, mixed = {ho_mixed_pct:.1f}%")
    L.append(f"- hook_off: pure-decode steps = {no_pure_pct:.1f}%, mixed = {no_mixed_pct:.1f}%")
    L.append(f"- Δ pure-decode: {delta_pure_pct:+.1f}pp")
    L.append(f"- Δ mixed: {delta_mixed_pct:+.1f}pp")
    L.append("")

    if abs(delta_pure_pct) < 2 and abs(delta_mixed_pct) < 2:
        L.append("**Verdict**: batch composition is **essentially identical** between hook_on "
                 "and hook_off (<2pp delta on pure-decode and mixed step fractions). "
                 "The r=16 TTFT drift is NOT from batch-composition shifting; it's likely "
                 "an oracle miscalibration or queue-drain artefact amplified by the hook's "
                 "per-step scheduling delay jitter.")
    elif delta_pure_pct > 3:
        L.append("**Verdict**: hook_on has significantly MORE pure-decode steps than hook_off "
                 f"({delta_pure_pct:+.1f}pp). This IS the mechanism I hypothesised earlier — "
                 "the admission delay creates 28ms windows where pending arrivals haven't "
                 "been drained yet, so the scheduler runs decode-only steps that are faster. "
                 "Over the course of r=16's saturated run, this accumulates into faster "
                 "queue drain and lower TTFT in emu. Fix: admit-at-t0 for scheduling pressure, "
                 "delay first-token emission downstream (architectural, Apr 21+).")
    else:
        L.append(f"**Verdict**: batch composition shows a moderate shift ({delta_pure_pct:+.1f}pp "
                 "pure-decode). Partial explanation for r=16 drift but not the dominant cause. "
                 "Likely interacts with other effects (variance, queue dynamics).")

OUT.write_text("\n".join(L) + "\n")
print(f"wrote {OUT}")
if hook_rows and nohook_rows:
    print(f"hook_on steps={h['steps']}, hook_off steps={n['steps']}")
