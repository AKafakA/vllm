"""Compare clean validation (new profile, emu hookOn) vs Phase 1 real.

Both sides use: fresh server per bench, common_warmup (100p @ r=4 seed=0),
default seed=0 for the 2000p bench. Deltas reflect pure emu vs real
differences given the new profile pack.

Targets: TPOT ≤ 6%, E2E ≤ 6%, TTFT ≤ 10% on every rate.
"""
import json
import sys
from pathlib import Path

REAL = Path("results/clean-5rate-default")
EMU  = Path("results/validate-apr20-clean")
RATES = [2, 4, 8, 16, 32]

METRICS = [
    ("TTFT", "median_ttft_ms", 10.0),
    ("TPOT", "median_tpot_ms", 6.0),
    ("ITL",  "median_itl_ms",  10.0),
    ("E2E",  "median_e2el_ms", 6.0),
]


def load(p):
    if not p.exists():
        return None
    return json.load(open(p))


def pct(new, ref):
    if ref in (None, 0):
        return float("nan")
    return (new - ref) / ref * 100.0


def main():
    rows_by_metric = {m: [] for m, _, _ in METRICS}
    for r in RATES:
        real = load(REAL / f"real_r{r}.json")
        emu  = load(EMU  / f"hookOn_r{r}.json")
        for m, key, threshold in METRICS:
            if real is None or emu is None:
                rows_by_metric[m].append((r, None, None, None, None, threshold))
                continue
            rv = real[key]
            ev = emu[key]
            d  = pct(ev, rv)
            rows_by_metric[m].append((r, rv, ev, d, abs(d) <= threshold, threshold))

    print("clean validation vs Phase 1 real (random 256/128, 2000p, seed=0)")
    print("=" * 78)
    for m, key, threshold in METRICS:
        print(f"\n--- {m} ({key}) — target |Δ| ≤ {threshold}% ---")
        print(f"{'rate':>4}  {'real':>10} {'emu':>10}  {'Δ%':>8}  {'verdict':>7}")
        for (r, rv, ev, d, ok, thr) in rows_by_metric[m]:
            if rv is None:
                print(f"r={r:>2}: MISSING")
                continue
            mark = "PASS" if ok else "**FAIL**"
            print(f"r={r:>2}  {rv:>10.2f} {ev:>10.2f}  {d:>+7.2f}%  {mark:>7}")

    # Summary
    print()
    print("=" * 78)
    print("Summary:")
    for m, _, threshold in METRICS:
        passes = sum(1 for (_, _, _, _, ok, _) in rows_by_metric[m]
                     if ok is True)
        tested = sum(1 for (_, _, _, _, ok, _) in rows_by_metric[m]
                     if ok is not None)
        print(f"  {m}: {passes}/{tested} rates pass ≤ {threshold}%")


if __name__ == "__main__":
    main()
