#!/usr/bin/env python3
"""Merge a BW calibration JSON into an existing serving profile JSON.

Reads the profile pack, decorates it with:
  - bw_calibration: (entire calibration JSON inline)
  - bw_reference_sum_kv: mean sum_kv across decode samples (computed
    from the profile's own decode distribution)

Writes to either --out or back in place.
"""

import argparse
import json
import sys
from pathlib import Path


def compute_mean_sum_kv(profile_pack: dict) -> float:
    """Use the profile's decode_2d_distribution tt as a proxy for sum_kv.

    The profile builder buckets by tt (total tokens in batch), which for
    decode-only steps equals num_decode_seqs (conc) — each decode step
    contributes 1 token. For the roofline correction we actually want
    the mean sum_kv the profile saw; if the profile doesn't store that
    explicitly we fall back to mean(tt * avg_seq_len), but since that
    requires the raw trace, the simplest honest choice is to set the
    reference to the sum_kv center where the calibration was done, so
    within-calibration-range queries get ≈0 correction.
    """
    # Attempt to read a stored bw_reference_sum_kv if the profile builder
    # already computed it (future-proof). Else return a conservative 0
    # which makes the correction purely additive from sum_kv=0 up.
    ref = profile_pack.get("bw_reference_sum_kv")
    if isinstance(ref, (int, float)) and ref > 0:
        return float(ref)
    return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True,
                    help="Path to serving-full.json (profile pack).")
    ap.add_argument("--calibration", required=True,
                    help="Path to bw calibration JSON.")
    ap.add_argument("--reference-sum-kv", type=float, default=None,
                    help="Override bw_reference_sum_kv. If omitted, "
                         "uses mean of calibration's sum_kv_range.")
    ap.add_argument("--out", default=None,
                    help="Output path. Default: overwrite --profile.")
    args = ap.parse_args()

    profile_path = Path(args.profile)
    calib_path = Path(args.calibration)
    out_path = Path(args.out) if args.out else profile_path

    profile = json.loads(profile_path.read_text())
    calib = json.loads(calib_path.read_text())

    if "error" in calib:
        print(f"calibration has error: {calib['error']}", file=sys.stderr)
        return 1

    profile["bw_calibration"] = calib

    if args.reference_sum_kv is not None:
        ref = float(args.reference_sum_kv)
    elif "bw_reference_sum_kv" in calib and calib["bw_reference_sum_kv"]:
        # Preferred: fit-emitted reference (overall mean sum_kv in profile).
        ref = float(calib["bw_reference_sum_kv"])
    else:
        lo, hi = calib.get("sum_kv_range_conc_means") or calib.get("sum_kv_range", [0, 0])
        ref = (lo + hi) / 2.0 if (lo or hi) else 0.0
    profile["bw_reference_sum_kv"] = ref

    out_path.write_text(json.dumps(profile, indent=2))
    print(f"merged calibration into {out_path}")
    print(f"  slope_measured = "
          f"{calib.get('bw_slope_measured_us_per_token'):.4f} us/tok")
    print(f"  slope_constant = "
          f"{calib.get('bw_slope_constant_us_per_token')} us/tok")
    print(f"  bw_reference_sum_kv = {ref:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
