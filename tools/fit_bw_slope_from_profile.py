#!/usr/bin/env python3
"""Fit BW slope from an existing profile's cross-concurrency centroids.

Replaces the single-sequence synthetic calibration. Inputs the
step_cycle_trace captured during a normal profile run. Per concurrency
level, computes (mean_sum_kv, mean_step_us). Regresses step_us over
mean_sum_kv across the conc buckets.

This sidesteps the single-sequence limitation where sum_kv varies by
only a few hundred tokens — here sum_kv spans 100→25 000 across conc
levels 1→64, giving a real signal.

Outputs the same JSON schema as profile_bw_calibration.py so the
merge tool works unchanged.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(trace_path):
    hdr = None
    rows = []
    with open(trace_path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("_header"):
                hdr = d
                continue
            if d.get("__marker__"):
                continue
            rows.append(d)
    return hdr, rows


def fit_linear(xs, ys):
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(slope), float(intercept), float(r2)


def compute_kv_per_token_bytes(header):
    L = header.get("num_hidden_layers")
    KV_H = header.get("num_key_value_heads")
    D = header.get("head_dim")
    if not all([L, KV_H, D]):
        return 0
    return 2 * KV_H * D * 2 * L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-path", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--hw-bw-gbs", type=float, default=None,
                    help="HW peak BW for the 'constant' slope variant.")
    ap.add_argument("--min-samples-per-conc", type=int, default=30)
    args = ap.parse_args()

    hdr, rows = load(args.trace_path)
    if hdr is None:
        print("no header found — aborting", file=sys.stderr)
        return 1
    decode = [
        r for r in rows
        if r.get("num_new_reqs", 0) == 0
        and r.get("num_decode_seqs", 0) > 0
        and "sum_kv" in r
        and "step_cycle_us" in r
    ]
    print(f"[fit] total rows={len(rows)} decode-only={len(decode)}", flush=True)

    by_conc = defaultdict(list)
    for r in decode:
        by_conc[r["num_decode_seqs"]].append(r)

    # Keep only conc buckets with enough samples for reliable mean.
    conc_centroids = []
    for c in sorted(by_conc):
        pts = by_conc[c]
        if len(pts) < args.min_samples_per_conc:
            continue
        mean_kv = float(np.mean([p["sum_kv"] for p in pts]))
        mean_us = float(np.mean([p["step_cycle_us"] for p in pts]))
        conc_centroids.append((c, len(pts), mean_kv, mean_us))

    print(f"[fit] usable conc buckets: {len(conc_centroids)}", flush=True)
    if len(conc_centroids) < 5:
        print("[fit] too few conc buckets — aborting", file=sys.stderr)
        return 1

    print(f"{'conc':>4}  {'n':>7}  {'mean_sum_kv':>12}  {'mean_step_us':>12}")
    for c, n, mkv, mus in conc_centroids:
        print(f"{c:>4}  {n:>7d}  {mkv:>12.1f}  {mus:>12.1f}")

    xs = [c[2] for c in conc_centroids]
    ys = [c[3] for c in conc_centroids]
    slope, intercept, r2 = fit_linear(xs, ys)

    # Overall mean sum_kv across ALL decode samples (weighted by bucket
    # size), used as the roofline-correction reference point: at query
    # sum_kv == mean, correction is 0, so within-profile behaviour is
    # preserved.
    all_kvs = [r["sum_kv"] for r in decode]
    overall_mean_sum_kv = float(np.mean(all_kvs)) if all_kvs else 0.0

    # Per-conc reference: mean sum_kv for each concurrency bucket.
    # The oracle uses these to make within-profile corrections ~zero,
    # eliminating the feedback-trap where a global reference makes
    # low-queue emulator states decode faster than real (which then
    # builds queue and flips to over-corrected slow regime).
    per_conc_mean_sum_kv = {
        str(c): float(np.mean([p["sum_kv"] for p in by_conc[c]]))
        for c in by_conc if len(by_conc[c]) >= args.min_samples_per_conc
    }

    kv_per_tok = compute_kv_per_token_bytes(hdr)
    implied_bw_gbs = (
        (kv_per_tok / (slope * 1e-6)) / 1e9 if slope > 0 and kv_per_tok > 0 else 0.0
    )

    slope_constant = None
    if args.hw_bw_gbs and kv_per_tok > 0:
        slope_constant = kv_per_tok / (args.hw_bw_gbs * 1e9) * 1e6

    result = {
        "gpu_name": hdr.get("gpu_name", "unknown"),
        "model_name": hdr.get("model_name", "unknown"),
        "kv_per_token_bytes": kv_per_tok,
        "bw_slope_measured_us_per_token": slope,
        "bw_slope_constant_us_per_token": slope_constant,
        "hw_bw_gbs_input": args.hw_bw_gbs,
        "bw_intercept_us": intercept,
        "bw_r_squared": r2,
        "n_conc_buckets_used": len(conc_centroids),
        "total_decode_samples": sum(c[1] for c in conc_centroids),
        "sum_kv_range_conc_means": [min(xs), max(xs)],
        "overall_mean_sum_kv": overall_mean_sum_kv,
        "bw_reference_sum_kv": overall_mean_sum_kv,
        "bw_reference_sum_kv_per_conc": per_conc_mean_sum_kv,
        "implied_sustained_bw_gbs_from_measured": implied_bw_gbs,
        "method": "cross_concurrency_centroids",
    }
    Path(args.out_json).write_text(json.dumps(result, indent=2))
    print()
    print(f"[fit] slope_measured = {slope:.4f} us/tok")
    print(f"[fit] intercept = {intercept:.1f} us")
    print(f"[fit] R² = {r2:.3f}")
    print(f"[fit] implied BW (measured) = {implied_bw_gbs:.1f} GB/s")
    if slope_constant is not None:
        print(f"[fit] slope_constant (from HW {args.hw_bw_gbs} GB/s) "
              f"= {slope_constant:.4f} us/tok")
    print(f"[fit] wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
