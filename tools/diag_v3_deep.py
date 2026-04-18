#!/usr/bin/env python3
"""Deep diagnostic: where in each round do v3's low-latency samples land?

Splits v3 trace by profiling_start/stop into rounds. For a target
(tt, conc) bucket, splits samples into position-within-round quartiles
and reports per-quartile latency mean. If variable-shape phases at
round-end contaminate buckets, the last quartile should have a
DIFFERENT (likely LOWER) mean than earlier quartiles.

Also checks the full distribution for bimodality.
"""
import json
import statistics
import sys
from collections import defaultdict

TRACE = "results/RTX-8000-adaptive-v3/step_cycle_trace.jsonl"
ARCHIVE = "results/_archive/serving-dense.json"

TARGETS = [
    (1, 2), (2, 2), (3, 2), (4, 2),  # decode-heavy at low conc
    (10, 12), (20, 22),  # moderate
    (100, 47), (200, 52),  # higher conc
]


def load_rounds():
    rounds = []
    current = None
    with open(TRACE) as f:
        for line in f:
            r = json.loads(line)
            if r.get("__marker__") == "profiling_start":
                current = []
                continue
            if r.get("__marker__") == "profiling_stop":
                if current is not None:
                    rounds.append(current)
                    current = None
                continue
            if r.get("_header"):
                continue
            if current is not None and "step_cycle_us" in r:
                current.append(r)
    return rounds


def bucketize(rec, tt_w=1, conc_w=5):
    tt = rec["total_tokens"]
    conc = rec.get("num_new_reqs", 0) + rec.get("num_decode_seqs", 0)
    if conc < 1:
        conc = 1
    ttb = (tt // tt_w) * tt_w + tt_w // 2
    cb = (conc // conc_w) * conc_w + conc_w // 2
    return (ttb, cb)


def quartile_mean(samples):
    if not samples:
        return None
    return sum(samples) / len(samples)


def analyse_bucket(rounds, tt_b, conc_b):
    # For each round, find samples matching this bucket + their position-in-round
    per_quartile = defaultdict(list)  # quartile (0-3) -> samples
    all_samples = []
    for round_recs in rounds:
        n = len(round_recs)
        if n == 0:
            continue
        for pos, rec in enumerate(round_recs):
            if bucketize(rec) == (tt_b, conc_b):
                q = min(3, (4 * pos) // n)  # 0..3 quartile
                per_quartile[q].append(rec["step_cycle_us"])
                all_samples.append(rec["step_cycle_us"])
    return per_quartile, all_samples


def main():
    rounds = load_rounds()
    print(f"Loaded {len(rounds)} rounds.\n")

    archive = json.load(open(ARCHIVE))
    archive_decode = {(b["tt"], b["conc"]): b for b in archive.get("decode_2d_distribution", [])}

    print(f"{'bucket':>12}  |  {'Q0_mean':>9} {'Q0_n':>6}  "
          f"{'Q1_mean':>9} {'Q1_n':>6}  "
          f"{'Q2_mean':>9} {'Q2_n':>6}  "
          f"{'Q3_mean':>9} {'Q3_n':>6}  |  "
          f"{'archive_mean':>12} {'archive_n':>9}")
    print("-" * 150)
    for tt, conc in TARGETS:
        per_q, all_s = analyse_bucket(rounds, tt, conc)
        a_bucket = archive_decode.get((tt, conc))
        a_samples = a_bucket.get("samples") if a_bucket else None
        a_mean = sum(a_samples) / len(a_samples) if a_samples else None
        a_n = len(a_samples) if a_samples else 0

        q_cells = []
        for q in range(4):
            s = per_q.get(q, [])
            if s:
                q_cells.append(f"{int(sum(s)/len(s)):>9} {len(s):>6}")
            else:
                q_cells.append(f"{'--':>9} {'0':>6}")

        a_cell = f"{int(a_mean):>12} {a_n:>9}" if a_mean else f"{'n/a':>12} {'n/a':>9}"
        print(f"  tt={tt:>3} c={conc:>3}  |  " + "  ".join(q_cells) +
              f"  |  {a_cell}")

    print("\n=== Bimodality check: distribution shape per bucket ===")
    for tt, conc in [(1, 2), (2, 2), (10, 12), (20, 22)]:
        per_q, all_s = analyse_bucket(rounds, tt, conc)
        if len(all_s) < 100:
            continue
        all_s.sort()
        n = len(all_s)
        # Percentiles
        pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        vals = [all_s[int(n * p / 100)] for p in pcts]
        print(f"  v3 ({tt}, {conc}) n={n}: " + " ".join(
            f"p{p}={int(v)}" for p, v in zip(pcts, vals)))
        a_bucket = archive_decode.get((tt, conc))
        if a_bucket:
            a_samples = sorted(a_bucket["samples"])
            m = len(a_samples)
            avals = [a_samples[int(m * p / 100)] for p in pcts]
            print(f"  archive ({tt}, {conc}) n={m}: " + " ".join(
                f"p{p}={int(v)}" for p, v in zip(pcts, avals)))


if __name__ == "__main__":
    main()
