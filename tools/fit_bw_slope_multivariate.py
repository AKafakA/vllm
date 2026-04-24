#!/usr/bin/env python3
"""Multi-variable BW slope fit.

Instead of regressing step_time ~ sum_kv across conc centroids (which
confounds conc-scheduler-overhead with sum_kv-memory-cost), fit:

    step_us ≈ a + b * conc + c * sum_kv + d * (conc * sum_kv)

and report `c` as the KV-memory slope AFTER controlling for conc.
The interaction term `d * conc * sum_kv` captures any per-conc
amplification of KV cost (e.g., attention masking overhead).

Expected: `c` ≈ kv_per_token / sustained_BW, hardware-consistent
across different GPUs (unlike the naive measured slope, which
varies 4× between A10 and RTX 8000).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_decode_samples(trace_path):
    rows = []
    hdr = None
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
            if (
                d.get("num_new_reqs", 0) == 0
                and d.get("num_decode_seqs", 0) > 0
                and "sum_kv" in d
                and "step_cycle_us" in d
            ):
                rows.append(d)
    return hdr, rows


def fit_multivariate(rows):
    """Return coefficients and R² for
    step = a + b*conc + c*sum_kv + d*conc*sum_kv."""
    n = len(rows)
    X = np.zeros((n, 4))
    y = np.zeros(n)
    for i, r in enumerate(rows):
        c = r["num_decode_seqs"]
        k = r["sum_kv"]
        X[i] = [1.0, c, k, c * k]
        y[i] = r["step_cycle_us"]
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ coef
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return coef.tolist(), r2


def compute_kv_per_token_bytes(hdr):
    L = hdr.get("num_hidden_layers")
    KV_H = hdr.get("num_key_value_heads")
    D = hdr.get("head_dim")
    if not all([L, KV_H, D]):
        return 0
    return 2 * KV_H * D * 2 * L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-path", required=True)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    hdr, rows = load_decode_samples(args.trace_path)
    if hdr is None:
        print("no header found", file=sys.stderr)
        return 1
    print(f"# model={hdr.get('model_name')} gpu={hdr.get('gpu_name')}  rows={len(rows)}")

    coef, r2 = fit_multivariate(rows)
    a, b_conc, c_kv, d_interact = coef
    print(f"## multi-var fit: step_us = {a:.1f} + {b_conc:.1f}*conc "
          f"+ {c_kv:.4f}*sum_kv + {d_interact:.6f}*conc*sum_kv")
    print(f"   R² = {r2:.4f}")
    print(f"   intercept a           = {a:.1f} us")
    print(f"   conc slope b          = {b_conc:.1f} us/conc")
    print(f"   sum_kv slope c        = {c_kv:.4f} us/token   <== pure KV term")
    print(f"   interaction d         = {d_interact:.6f} us/(conc*token)")

    kv_per_tok = compute_kv_per_token_bytes(hdr)
    if c_kv > 0 and kv_per_tok > 0:
        bw_gbs = kv_per_tok / (c_kv * 1e-6) / 1e9
        print(f"   implied BW from c     = {bw_gbs:.1f} GB/s")

    # Also show what c+d*conc would give at validation conc=64.
    effective_slope_at_64 = c_kv + d_interact * 64
    print(f"   effective slope @conc=64  = {effective_slope_at_64:.4f} us/token")

    if args.out_json:
        out = {
            "model_name": hdr.get("model_name"),
            "gpu_name": hdr.get("gpu_name"),
            "kv_per_token_bytes": kv_per_tok,
            "multivariate": {
                "intercept_us": a,
                "conc_slope_us_per_conc": b_conc,
                "sum_kv_slope_us_per_token": c_kv,
                "interaction_us_per_conc_per_token": d_interact,
                "r_squared": r2,
                "effective_slope_at_conc_64": effective_slope_at_64,
            },
            "implied_sustained_bw_gbs_from_c": (
                kv_per_tok / (c_kv * 1e-6) / 1e9
                if c_kv > 0 and kv_per_tok > 0 else None
            ),
        }
        Path(args.out_json).write_text(json.dumps(out, indent=2))
        print(f"wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
