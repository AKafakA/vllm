"""Profile-driven GPU cost oracle for vLLM emulator."""

from __future__ import annotations

import math
from typing import Any

from .base import BaseGpuCostOracle


def _fit_power_law(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Fit y = a * x^b via least-squares in log-log space.

    Returns (a, b).  Falls back to (y[0]/x[0], 1.0) if the fit is
    degenerate (e.g. only one sample).
    """
    if len(xs) < 2:
        if xs[0] == 0:
            return (ys[0], 1.0)
        return (ys[0] / xs[0], 1.0)

    # log-log linear regression: log(y) = log(a) + b*log(x)
    log_xs = [math.log(max(x, 1)) for x in xs]
    log_ys = [math.log(max(y, 1e-6)) for y in ys]
    n = len(log_xs)
    sum_lx = sum(log_xs)
    sum_ly = sum(log_ys)
    sum_lx2 = sum(lx * lx for lx in log_xs)
    sum_lxly = sum(lx * ly for lx, ly in zip(log_xs, log_ys))

    denom = n * sum_lx2 - sum_lx * sum_lx
    if abs(denom) < 1e-12:
        # Degenerate (all same x) — fall back to linear
        return (ys[0] / max(xs[0], 1), 1.0)

    b = (n * sum_lxly - sum_lx * sum_ly) / denom
    log_a = (sum_ly - b * sum_lx) / n
    return (math.exp(log_a), b)


def _interpolate_linear(
    xs: list[float], ys: list[float], x: float
) -> float:
    """Piecewise linear interpolation (no clamping — caller handles
    extrapolation)."""
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            ratio = (x - xs[i]) / (xs[i + 1] - xs[i])
            return ys[i] + ratio * (ys[i + 1] - ys[i])
    # Shouldn't reach here when called correctly
    return ys[-1]


class ProfileGpuCostOracle(BaseGpuCostOracle):
    """GPU cost oracle that uses piecewise-linear interpolation within the
    profiled range and power-law extrapolation outside it.

    Prefill latency scales super-linearly with sequence length (roughly
    O(n^{1.5-2}) due to attention), so clamping at the max profiled
    seq_len would severely underestimate long-context workloads.
    Instead, we fit a power-law model from the profile samples and
    extrapolate beyond the profiled range.
    """

    def __init__(self, profile_pack: dict[str, Any]):
        self._profile = profile_pack
        self._gpu_model = profile_pack["gpu_model"]

        # Pre-sort samples by x-axis (seq_len / active_seqs) to guarantee
        # monotonic interpolation.
        self._prefill_samples = sorted(
            profile_pack["prefill"], key=lambda s: s["seq_len"]
        )
        self._decode_samples = sorted(
            profile_pack["decode"], key=lambda s: s["active_seqs"]
        )

        # Pre-compute power-law fits for extrapolation
        prefill_xs = [float(s["seq_len"]) for s in self._prefill_samples]
        prefill_ys = [float(s["latency_us"]) for s in self._prefill_samples]
        self._prefill_pw_a, self._prefill_pw_b = _fit_power_law(
            prefill_xs, prefill_ys
        )

        decode_xs = [float(s["active_seqs"]) for s in self._decode_samples]
        decode_ys = [
            float(s["latency_us_per_token"]) for s in self._decode_samples
        ]
        self._decode_pw_a, self._decode_pw_b = _fit_power_law(
            decode_xs, decode_ys
        )

    @property
    def gpu_model(self) -> str:
        return self._gpu_model

    def estimate_prefill_latency_us(
        self, prompt_tokens: int, batch_size: int
    ) -> float:
        """Estimate prefill latency.

        Within the profiled range: piecewise-linear interpolation.
        Outside: power-law extrapolation (captures super-linear attention
        scaling).
        """
        samples = self._prefill_samples
        seq_lens = [s["seq_len"] for s in samples]
        latencies = [s["latency_us"] for s in samples]

        if prompt_tokens <= 0:
            return 0.0

        if seq_lens[0] <= prompt_tokens <= seq_lens[-1]:
            return _interpolate_linear(
                [float(s) for s in seq_lens],
                [float(l) for l in latencies],
                float(prompt_tokens),
            )

        # Power-law extrapolation: y = a * x^b
        return self._prefill_pw_a * (prompt_tokens ** self._prefill_pw_b)

    def estimate_decode_latency_us(self, active_seqs: int) -> float:
        """Estimate per-token decode latency.

        Within the profiled range: piecewise-linear interpolation.
        Outside: power-law extrapolation.
        """
        samples = self._decode_samples
        seq_counts = [s["active_seqs"] for s in samples]
        latencies = [s["latency_us_per_token"] for s in samples]

        if active_seqs <= 0:
            return 0.0

        if seq_counts[0] <= active_seqs <= seq_counts[-1]:
            return _interpolate_linear(
                [float(s) for s in seq_counts],
                [float(l) for l in latencies],
                float(active_seqs),
            )

        return self._decode_pw_a * (active_seqs ** self._decode_pw_b)


def create_oracle_from_profile_pack(
    profile_pack: dict[str, Any],
) -> ProfileGpuCostOracle:
    """Factory function to create an oracle from a profile pack."""
    return ProfileGpuCostOracle(profile_pack)
