"""Profile-driven GPU cost oracle for vLLM emulator.

Minimal oracle: given (total_tokens, num_requests, has_prefill), find the
nearest populated 2D distribution bucket and sample from its raw samples.
No magic numbers, no synthetic fallbacks, no heuristic thresholds.
"""

from __future__ import annotations

import os
import random
from typing import Any

from .base import BaseGpuCostOracle


class ProfileGpuCostOracle(BaseGpuCostOracle):
    """GPU cost oracle that samples from profiled 2D (tt, concurrency)
    distributions.

    The only estimation method is nearest-neighbor lookup into
    (total_tokens, concurrency) buckets, then uniform random sampling
    from the raw latency samples in that bucket.  This preserves the
    real GPU's empirical distribution (mean, variance, tail) without
    any synthetic constants.
    """

    def __init__(self, profile_pack: dict[str, Any]):
        self._profile = profile_pack
        self._gpu_model = profile_pack["gpu_model"]

        # RNG for sampling from distribution buckets.
        self._rng = random.Random(42)

        # User-configurable percentile trim on raw samples.
        # Format: "lo,hi" e.g. "2,98" trims bottom 2% and top 2%.
        # Applied at sample time from the already-stored raw samples.
        trim_env = os.environ.get("VLLM_EMULATOR_SAMPLE_TRIM", "")
        if trim_env:
            parts = trim_env.split(",")
            self._trim_lo = int(parts[0])
            self._trim_hi = int(parts[1])
        else:
            self._trim_lo = 0
            self._trim_hi = 100

        # 2D distribution: (tt, conc) -> bucket with raw samples list.
        # Separated by step type: decode (CUDA graph) vs prefill (eager).
        # Combined distribution is the fallback.
        self._decode_2d_distribution: dict[tuple[int, int], dict] = {}
        for e in profile_pack.get("decode_2d_distribution", []):
            self._decode_2d_distribution[(e["tt"], e["conc"])] = e

        self._prefill_2d_distribution: dict[tuple[int, int], dict] = {}
        for e in profile_pack.get("prefill_2d_distribution", []):
            self._prefill_2d_distribution[(e["tt"], e["conc"])] = e

        self._combined_2d_distribution: dict[tuple[int, int], dict] = {}
        for e in profile_pack.get("step_cycle_2d_distribution", []):
            self._combined_2d_distribution[(e["tt"], e["conc"])] = e

        # F4: 3D axis tables (optional, v2.1 profiles). Keyed by
        # (tt, conc, new_reqs). Oracle uses them only when the env var
        # VLLM_EMULATOR_PROFILE_AXES=3d is set AND the tables are populated.
        # Rule is exact-match-only (no cross-bucket distance weighting
        # across the new axis) — on miss, falls back to the 2D path.
        self._decode_3d_distribution: dict[tuple[int, int, int], dict] = {}
        for e in profile_pack.get("decode_axis_distribution", []):
            self._decode_3d_distribution[(e["tt"], e["conc"], e["new_reqs"])] = e

        self._prefill_3d_distribution: dict[tuple[int, int, int], dict] = {}
        for e in profile_pack.get("prefill_axis_distribution", []):
            self._prefill_3d_distribution[(e["tt"], e["conc"], e["new_reqs"])] = e

        self._combined_3d_distribution: dict[tuple[int, int, int], dict] = {}
        for e in profile_pack.get("step_cycle_axis_distribution", []):
            self._combined_3d_distribution[(e["tt"], e["conc"], e["new_reqs"])] = e

        # F4 env read. Default 2d preserves current behaviour bit-identical.
        axes_env = os.environ.get("VLLM_EMULATOR_PROFILE_AXES", "2d").lower()
        if axes_env not in ("2d", "3d"):
            raise ValueError(
                f"VLLM_EMULATOR_PROFILE_AXES must be '2d' or '3d', got {axes_env!r}")
        self._profile_axes = axes_env
        # Once-only warning if 3d requested but no axis tables loaded.
        if self._profile_axes == "3d" and not (
                self._decode_3d_distribution or self._prefill_3d_distribution
                or self._combined_3d_distribution):
            print("[ProfileGpuCostOracle] VLLM_EMULATOR_PROFILE_AXES=3d but profile "
                  "has no *_axis_distribution tables; falling back to 2D lookup.")

        # Pre-trim samples in each bucket once at load time.
        for table in (self._decode_2d_distribution,
                      self._prefill_2d_distribution,
                      self._combined_2d_distribution,
                      self._decode_3d_distribution,
                      self._prefill_3d_distribution,
                      self._combined_3d_distribution):
            for key, bucket in table.items():
                raw = bucket.get("samples")
                if raw:
                    bucket["samples"] = self._trim_samples(raw)

    def _trim_samples(self, samples: list[float]) -> list[float]:
        """Apply percentile trim to a raw samples list."""
        if self._trim_lo == 0 and self._trim_hi == 100:
            return samples
        n = len(samples)
        if n == 0:
            return samples
        sorted_s = sorted(samples)
        lo_idx = int(n * self._trim_lo / 100)
        hi_idx = int(n * self._trim_hi / 100)
        if hi_idx <= lo_idx:
            return samples
        return sorted_s[lo_idx:hi_idx]

    @property
    def gpu_model(self) -> str:
        return self._gpu_model

    def _sample_2d_distribution(
        self, total_tokens: int, num_requests: int,
        has_prefill: bool = False,
    ) -> float | None:
        """Sample latency from 2D (tt, concurrency) distribution bucket.

        Nearest-neighbor lookup: finds the closest (tt, conc) bucket
        in the appropriate table (decode/prefill/combined) and returns
        a uniformly random sample from that bucket's raw latency list.

        Returns None only if no distribution data exists at all.
        """
        # Pick table: prefill vs decode vs combined.
        if has_prefill and self._prefill_2d_distribution:
            table = self._prefill_2d_distribution
        elif not has_prefill and self._decode_2d_distribution:
            table = self._decode_2d_distribution
        elif self._combined_2d_distribution:
            table = self._combined_2d_distribution
        else:
            return None

        tts_in_table = sorted(set(k[0] for k in table.keys()))
        if not tts_in_table:
            return None

        # Nearest tt bucket.
        if total_tokens <= tts_in_table[0]:
            tt_near = tts_in_table[0]
        elif total_tokens >= tts_in_table[-1]:
            tt_near = tts_in_table[-1]
        else:
            tt_near = min(tts_in_table, key=lambda t: abs(t - total_tokens))

        # Nearest concurrency bucket for this tt.
        available_concs = [c for (tt, c) in table.keys() if tt == tt_near]
        if not available_concs:
            return None
        conc_near = min(available_concs, key=lambda c: abs(c - num_requests))

        bucket = table.get((tt_near, conc_near))
        if not bucket:
            return None

        raw = bucket.get("samples")
        if raw:
            return float(self._rng.choice(raw))
        return None

    def _sample_3d_distribution(
        self, total_tokens: int, num_requests: int, num_new_reqs: int,
        has_prefill: bool = False,
    ) -> float | None:
        """F4: Exact-match 3D lookup; returns None on miss (caller falls back).

        Per review condition, NO cross-bucket distance weighting on the new
        axis — either an exact `(tt, conc, new_reqs)` bucket exists in the
        selected table, or we return None and 2D takes over.

        The existing tt-slice / conc-slice nearest-neighbor logic is still
        applied to the 2D sub-projection: we find the nearest (tt, conc)
        pair with the exact new_reqs bucket populated, then sample from it.
        """
        if has_prefill and self._prefill_3d_distribution:
            table = self._prefill_3d_distribution
        elif not has_prefill and self._decode_3d_distribution:
            table = self._decode_3d_distribution
        elif self._combined_3d_distribution:
            table = self._combined_3d_distribution
        else:
            return None

        # Filter to exact new_reqs bucket only (condition 1: no cross-
        # bucket weighting along the new axis).
        matching = [k for k in table.keys() if k[2] == num_new_reqs]
        if not matching:
            return None

        # Nearest tt bucket within this new_reqs slice.
        tts = sorted({k[0] for k in matching})
        if total_tokens <= tts[0]:
            tt_near = tts[0]
        elif total_tokens >= tts[-1]:
            tt_near = tts[-1]
        else:
            tt_near = min(tts, key=lambda t: abs(t - total_tokens))

        concs = [k[1] for k in matching if k[0] == tt_near]
        if not concs:
            return None
        conc_near = min(concs, key=lambda c: abs(c - num_requests))

        bucket = table.get((tt_near, conc_near, num_new_reqs))
        if not bucket:
            return None
        raw = bucket.get("samples")
        if raw:
            return float(self._rng.choice(raw))
        return None

    def estimate_step_latency_us(
        self, total_tokens: int,
        has_prefill: bool = False,
        num_requests: int = 0,
        num_new_reqs: int = 0,
        **kwargs,
    ) -> float:
        """Estimate latency for one forward pass via 2D or 3D sampling.

        Args:
            total_tokens: Total tokens in the batch.
            has_prefill: Whether batch contains new prefill requests.
            num_requests: Number of requests in batch.
            num_new_reqs: F4 third-axis input. Ignored when
                VLLM_EMULATOR_PROFILE_AXES != 3d.
            **kwargs: Ignored (backward compat for callers passing
                      oracle_mode, profile_section, avg_context_len).

        Returns:
            Sampled latency in microseconds, or 0 if no data.
        """
        if total_tokens <= 0:
            return 0.0

        # F4: try 3D exact-match first when gate is on.
        if self._profile_axes == "3d":
            result = self._sample_3d_distribution(
                total_tokens, max(num_requests, 1), num_new_reqs,
                has_prefill=has_prefill,
            )
            if result is not None:
                return result
            # Miss: fall back to 2D (per design doc §2 fallback rule).

        result = self._sample_2d_distribution(
            total_tokens, max(num_requests, 1),
            has_prefill=has_prefill,
        )
        if result is not None:
            return result

        # No distribution data at all -- return 0 and let caller handle.
        return 0.0


def create_oracle_from_profile_pack(
    profile_pack: dict[str, Any],
) -> ProfileGpuCostOracle:
    """Factory function to create an oracle from a profile pack."""
    return ProfileGpuCostOracle(profile_pack)
