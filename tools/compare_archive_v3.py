#!/usr/bin/env python3
"""Diagnostic: compare archive vs v3 profile for evidence of bug vs variance.

Prints per-bucket sample-count + distribution statistics for the
r=2/4/8/16/32 most-queried bucket regions. If means shift systematically,
v3 has a bug. If means match but variance widens (heavy tails), v3 has
a legitimate data-quality issue (variable-shape contamination or
fine-bucket undersampling).
"""
import json
import statistics
import sys

ARCHIVE = sys.argv[1] if len(sys.argv) > 1 else "results/_archive/serving-dense.json"
V3 = sys.argv[2] if len(sys.argv) > 2 else "results/RTX-8000-adaptive-v3/serving-full.json"


def load_table(path, key):
    p = json.load(open(path))
    return {(b["tt"], b["conc"]): b for b in p.get(key, [])}


def stats(samples):
    if not samples:
        return None
    sorted_s = sorted(samples)
    n = len(samples)
    return {
        "n": n,
        "mean": int(sum(samples) / n),
        "median": int(sorted_s[n // 2]),
        "p10": int(sorted_s[n // 10]),
        "p90": int(sorted_s[n * 9 // 10]),
        "min": int(sorted_s[0]),
        "max": int(sorted_s[-1]),
        "std": int(statistics.pstdev(samples)) if n > 1 else 0,
    }


def main():
    print(f"Archive: {ARCHIVE}")
    print(f"V3     : {V3}")

    archive_decode = load_table(ARCHIVE, "decode_2d_distribution")
    v3_decode = load_table(V3, "decode_2d_distribution")

    print(f"\narchive decode: {len(archive_decode)} buckets")
    print(f"v3      decode: {len(v3_decode)} buckets")

    # Find common buckets with most samples in both
    common = []
    for key, ab in archive_decode.items():
        vb = v3_decode.get(key)
        if not vb:
            continue
        a_n = len(ab.get("samples", []))
        v_n = len(vb.get("samples", []))
        if a_n > 20 and v_n > 20:
            common.append((key, a_n + v_n, ab, vb))
    common.sort(key=lambda x: -x[1])

    print(f"\nCommon buckets with >=20 samples each: {len(common)}")
    print(f"\n{'bucket':>12}  | {'archive':>55}  | {'v3':>55}  | {'Δmean':>8}")
    print("-" * 140)
    for (key, _, ab, vb) in common[:10]:
        a = stats(ab["samples"])
        v = stats(vb["samples"])
        if not a or not v:
            continue
        dmean = v["mean"] - a["mean"]
        pct = 100.0 * dmean / a["mean"] if a["mean"] else 0.0
        a_str = f"n={a['n']:5d} mean={a['mean']:6d} p10={a['p10']:6d} p90={a['p90']:6d} max={a['max']:7d}"
        v_str = f"n={v['n']:5d} mean={v['mean']:6d} p10={v['p10']:6d} p90={v['p90']:6d} max={v['max']:7d}"
        print(f"  tt={key[0]:>3} c={key[1]:>3} | {a_str}  | {v_str}  | {dmean:+6d}  ({pct:+.1f}%)")

    # Aggregate stats: is v3's mean systematically higher or lower?
    print("\n=== Aggregate (across all common buckets with >=20 samples) ===")
    diffs_pct = []
    for (key, _, ab, vb) in common:
        a = stats(ab["samples"])
        v = stats(vb["samples"])
        if a and v and a["mean"] > 0:
            diffs_pct.append(100.0 * (v["mean"] - a["mean"]) / a["mean"])
    if diffs_pct:
        diffs_pct.sort()
        n = len(diffs_pct)
        print(f"  buckets compared: {n}")
        print(f"  median Δmean%: {diffs_pct[n // 2]:+.2f}%")
        print(f"  mean   Δmean%: {sum(diffs_pct) / n:+.2f}%")
        print(f"  p10    Δmean%: {diffs_pct[n // 10]:+.2f}%")
        print(f"  p90    Δmean%: {diffs_pct[n * 9 // 10]:+.2f}%")

        # Check variance-expansion signature
        std_a_sum = 0
        std_v_sum = 0
        cnt = 0
        for (key, _, ab, vb) in common:
            a = stats(ab["samples"])
            v = stats(vb["samples"])
            if a and v:
                std_a_sum += a["std"]
                std_v_sum += v["std"]
                cnt += 1
        if cnt:
            print(f"\n  mean std (archive): {std_a_sum / cnt:.0f} us")
            print(f"  mean std (v3)     : {std_v_sum / cnt:.0f} us")
            ratio = std_v_sum / std_a_sum if std_a_sum else 0
            print(f"  variance-expansion ratio: {ratio:.2f}x  "
                  f"({'v3 NOISIER' if ratio > 1.1 else 'similar' if ratio > 0.9 else 'v3 TIGHTER'})")

    # Sparsity check
    print("\n=== Sparsity: samples per bucket distribution ===")
    a_sizes = [len(b.get("samples", [])) for b in archive_decode.values()]
    v_sizes = [len(b.get("samples", [])) for b in v3_decode.values()]
    for name, sizes in [("archive", a_sizes), ("v3", v_sizes)]:
        sizes.sort()
        n = len(sizes)
        if n:
            print(f"  {name}: n_buckets={n} min={sizes[0]} p10={sizes[n//10]} "
                  f"median={sizes[n//2]} p90={sizes[n*9//10]} max={sizes[-1]} "
                  f"total_samples={sum(sizes)}")


if __name__ == "__main__":
    main()
