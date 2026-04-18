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

        # F5: kNN conditioning. K=1 (default) keeps the current nearest-
        # neighbor code path byte-identical. K>1 uses Shepard (1968) p=2
        # inverse-distance weighting over range-normalised (tt, conc) axes.
        k_env = os.environ.get("VLLM_EMULATOR_ORACLE_K", "1")
        try:
            self._oracle_k = int(k_env)
        except (TypeError, ValueError):
            raise ValueError(
                f"VLLM_EMULATOR_ORACLE_K must be an integer, got {k_env!r}")
        if self._oracle_k < 1:
            raise ValueError(
                f"VLLM_EMULATOR_ORACLE_K must be >= 1, got {self._oracle_k}")

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

        # Pre-trim samples in each bucket once at load time.
        for table in (self._decode_2d_distribution,
                      self._prefill_2d_distribution,
                      self._combined_2d_distribution):
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

        Default (K=1, VLLM_EMULATOR_ORACLE_K=1 or unset): tt-slice then
        conc-slice nearest-neighbor, same as parent commit. Byte-identical
        RNG sequence.

        K>1 (F5): Shepard p=2 inverse-distance-weighted sampling over the
        K nearest range-normalised (tt, conc) buckets; delegates to
        _sample_knn_2d. Uses self._rng throughout (no secondary Random).

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

        if self._oracle_k > 1:
            return self._sample_knn_2d(table, total_tokens, num_requests)

        # K=1 path: unchanged from parent commit — byte-identical.
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

    def _sample_knn_2d(self, table, total_tokens, num_requests):
        """K>1 Shepard p=2 kNN sampler over range-normalised (tt, conc)."""
        keys = list(table.keys())
        if not keys:
            return None

        # Range normalisation (standard kNN input scaling).
        tts = [k[0] for k in keys]
        concs = [k[1] for k in keys]
        tt_range = (max(tts) - min(tts)) or 1
        conc_range = (max(concs) - min(concs)) or 1

        def _dist(k):
            dt = (k[0] - total_tokens) / tt_range
            dc = (k[1] - num_requests) / conc_range
            return (dt * dt + dc * dc) ** 0.5

        # K nearest buckets.
        ranked = sorted(((_dist(k), k) for k in keys), key=lambda x: x[0])
        top_k = ranked[: self._oracle_k]

        # Shepard 1968 exact-match short-circuit.
        for d, k in top_k:
            if d == 0.0:
                samples = table[k].get("samples") or []
                if samples:
                    return float(self._rng.choice(samples))
                return None

        # Shepard 1968 inverse-distance weighting with p=2.
        weights = [1.0 / (d * d) for d, _ in top_k]
        total_w = sum(weights)
        if total_w == 0:
            return None
        r = self._rng.random() * total_w
        cum = 0.0
        chosen_k = top_k[-1][1]
        for (d, k), w in zip(top_k, weights):
            cum += w
            if r <= cum:
                chosen_k = k
                break

        samples = table[chosen_k].get("samples") or []
        if samples:
            return float(self._rng.choice(samples))
        return None

    def estimate_step_latency_us(
        self, total_tokens: int,
        has_prefill: bool = False,
        num_requests: int = 0,
        **kwargs,
    ) -> float:
        """Estimate latency for one forward pass via 2D distribution sampling.

        Args:
            total_tokens: Total tokens in the batch.
            has_prefill: Whether batch contains new prefill requests.
            num_requests: Number of requests in batch.
            **kwargs: Ignored (backward compat for callers passing
                      oracle_mode, profile_section, avg_context_len).

        Returns:
            Sampled latency in microseconds, or 0 if no data.
        """
        if total_tokens <= 0:
            return 0.0

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
