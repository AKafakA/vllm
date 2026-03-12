#!/usr/bin/env python3
"""Minimal A/B comparison harness for vLLM emulator vs. profile ground truth.

Usage:
    python tests/integration/ab_comparison_harness.py \
        --profile-pack examples/profiles/a100-sxm-80gb.json

This script:
1. Loads a profile pack (the "ground truth" from real GPU measurements)
2. Creates an oracle from the profile
3. Queries the oracle at sample points and interpolation midpoints
4. Compares oracle estimates to expected values
5. Prints an accuracy report with per-query and aggregate error metrics

Exit code 0 if all errors are within the 15% threshold, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Allow running from repo root without install
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vllm_emulator.oracle import create_oracle_from_profile_pack
from vllm_emulator.profile.loader import load_profile_pack


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    phase: str           # "prefill" or "decode"
    input_value: int     # seq_len or active_seqs
    expected_us: float
    estimated_us: float
    is_exact: bool       # True if query is at an exact sample point

    @property
    def abs_error_us(self) -> float:
        return abs(self.estimated_us - self.expected_us)

    @property
    def rel_error(self) -> float:
        if self.expected_us == 0:
            return 0.0 if self.estimated_us == 0 else float("inf")
        return self.abs_error_us / self.expected_us


@dataclass
class ComparisonReport:
    results: list[QueryResult] = field(default_factory=list)
    error_threshold: float = 0.15  # 15%

    @property
    def max_rel_error(self) -> float:
        if not self.results:
            return 0.0
        return max(r.rel_error for r in self.results)

    @property
    def mean_rel_error(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.rel_error for r in self.results) / len(self.results)

    @property
    def passed(self) -> bool:
        return all(r.rel_error <= self.error_threshold for r in self.results)


# ---------------------------------------------------------------------------
# Interpolation helpers
# ---------------------------------------------------------------------------

def _lerp(x: float, x0: float, y0: float, x1: float, y1: float) -> float:
    """Linear interpolation between (x0,y0) and (x1,y1) at x."""
    if x1 == x0:
        return y0
    ratio = (x - x0) / (x1 - x0)
    return y0 + ratio * (y1 - y0)


def _build_prefill_queries(profile: dict) -> list[tuple[int, float, bool]]:
    """Build (seq_len, expected_us, is_exact) tuples for prefill phase."""
    samples = profile["prefill"]
    queries: list[tuple[int, float, bool]] = []

    # Exact sample points
    for s in samples:
        queries.append((s["seq_len"], s["latency_us"], True))

    # Midpoints between consecutive samples
    for i in range(len(samples) - 1):
        lo, hi = samples[i], samples[i + 1]
        mid_sl = (lo["seq_len"] + hi["seq_len"]) // 2
        mid_lat = _lerp(
            mid_sl, lo["seq_len"], lo["latency_us"],
            hi["seq_len"], hi["latency_us"],
        )
        queries.append((mid_sl, mid_lat, False))

    return queries


def _build_decode_queries(profile: dict) -> list[tuple[int, float, bool]]:
    """Build (active_seqs, expected_us, is_exact) tuples for decode phase."""
    samples = profile["decode"]
    queries: list[tuple[int, float, bool]] = []

    # Exact sample points
    for s in samples:
        queries.append((s["active_seqs"], s["latency_us_per_token"], True))

    # Midpoints
    for i in range(len(samples) - 1):
        lo, hi = samples[i], samples[i + 1]
        mid_aseq = (lo["active_seqs"] + hi["active_seqs"]) // 2
        mid_lat = _lerp(
            mid_aseq, lo["active_seqs"], lo["latency_us_per_token"],
            hi["active_seqs"], hi["latency_us_per_token"],
        )
        queries.append((mid_aseq, mid_lat, False))

    return queries


# ---------------------------------------------------------------------------
# Main comparison logic
# ---------------------------------------------------------------------------

def run_comparison(profile_path: str, threshold: float = 0.15) -> ComparisonReport:
    """Run A/B comparison and return report."""
    profile = load_profile_pack(profile_path)
    oracle = create_oracle_from_profile_pack(profile)

    report = ComparisonReport(error_threshold=threshold)

    # Prefill queries
    for seq_len, expected, is_exact in _build_prefill_queries(profile):
        estimated = oracle.estimate_prefill_latency_us(seq_len, batch_size=1)
        report.results.append(QueryResult(
            phase="prefill",
            input_value=seq_len,
            expected_us=expected,
            estimated_us=estimated,
            is_exact=is_exact,
        ))

    # Decode queries
    for active_seqs, expected, is_exact in _build_decode_queries(profile):
        estimated = oracle.estimate_decode_latency_us(active_seqs)
        report.results.append(QueryResult(
            phase="decode",
            input_value=active_seqs,
            expected_us=expected,
            estimated_us=estimated,
            is_exact=is_exact,
        ))

    return report


def print_report(report: ComparisonReport) -> None:
    """Print a human-readable accuracy report."""
    print("=" * 72)
    print("  vLLM Emulator A/B Comparison Report")
    print("=" * 72)
    print()

    header = f"{'Phase':<10} {'Input':>8} {'Expected(us)':>14} {'Estimated(us)':>14} {'Error':>8} {'Type':>6}"
    print(header)
    print("-" * len(header))

    for r in report.results:
        marker = "EXACT" if r.is_exact else "INTERP"
        err_str = f"{r.rel_error:.2%}"
        fail = " !!!" if r.rel_error > report.error_threshold else ""
        print(
            f"{r.phase:<10} {r.input_value:>8} {r.expected_us:>14.1f} "
            f"{r.estimated_us:>14.1f} {err_str:>8} {marker:>6}{fail}"
        )

    print()
    print(f"  Total queries : {len(report.results)}")
    print(f"  Max error     : {report.max_rel_error:.2%}")
    print(f"  Mean error    : {report.mean_rel_error:.2%}")
    print(f"  Threshold     : {report.error_threshold:.0%}")
    print(f"  Result        : {'PASS' if report.passed else 'FAIL'}")
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="A/B comparison harness for vLLM emulator oracle accuracy"
    )
    parser.add_argument(
        "--profile-pack",
        required=True,
        help="Path to JSON profile pack (ground truth)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.15,
        help="Max allowed relative error (default: 0.15 = 15%%)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON instead of human-readable table",
    )
    args = parser.parse_args()

    report = run_comparison(args.profile_pack, threshold=args.threshold)

    if args.json:
        output = {
            "profile_pack": args.profile_pack,
            "threshold": args.threshold,
            "passed": report.passed,
            "max_rel_error": report.max_rel_error,
            "mean_rel_error": report.mean_rel_error,
            "results": [
                {
                    "phase": r.phase,
                    "input_value": r.input_value,
                    "expected_us": r.expected_us,
                    "estimated_us": r.estimated_us,
                    "rel_error": r.rel_error,
                    "is_exact": r.is_exact,
                }
                for r in report.results
            ],
        }
        json.dump(output, sys.stdout, indent=2)
        print()
    else:
        print_report(report)

    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
