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

        # KV-adjustment α: read from profile pack if present (profile was
        # built with --alpha-kv model). At query time, compute tt_eff = tt
        # + alpha_kv * sum_kv using current scheduler state, then bucket
        # lookup. When alpha_kv is absent or 0, behaviour is unchanged.
        self._alpha_kv = float(profile_pack.get("alpha_kv", 0.0))

        # IPC scheduling overhead table (from profile_ipc_overhead.py).
        # Measured TTFT minus profile prefill_step, per concurrency N.
        # Applied to prefill steps ONLY (decode steps don't pay per-request
        # overhead). Lookup by current num_requests; linear interpolation
        # between measured points. Based on commit 4d9983a0c (Apr 6).
        self._sched_overhead_table = profile_pack.get("sched_overhead_table", [])
        # Sort by num_reqs for correct interpolation.
        self._sched_overhead_table = sorted(
            self._sched_overhead_table, key=lambda e: e.get("num_reqs", 0))

        # Experimental: oracle aggregation mode. Default is "sample"
        # (IID random.choice). Alternatives are "median" and "mean" —
        # deterministic per-bucket estimator. Used to test whether
        # variance from sampling is load-bearing vs central-tendency
        # sufficient. No new knobs beyond this env-var gate.
        self._oracle_agg = os.environ.get("VLLM_EMULATOR_ORACLE_AGG", "sample").lower()
        if self._oracle_agg not in ("sample", "median", "mean"):
            raise ValueError(
                f"VLLM_EMULATOR_ORACLE_AGG must be 'sample', 'median', or 'mean'; "
                f"got {self._oracle_agg!r}")

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

    def _lookup_sched_overhead_us(self, num_reqs: int) -> float:
        """IPC scheduling overhead at concurrency N, interpolated from table.

        The table is per-concurrency measurements of TTFT - prefill_step on
        real hardware (profile_ipc_overhead.py). Applied only to prefill
        steps — matches the structure in commit 4d9983a0c.
        """
        table = self._sched_overhead_table
        if not table:
            return 0.0
        # Exact match first.
        for e in table:
            if e.get("num_reqs") == num_reqs:
                return float(e.get("overhead_us", 0))
        # Clamp below/above table range.
        if num_reqs <= table[0].get("num_reqs", 1):
            return float(table[0].get("overhead_us", 0))
        if num_reqs >= table[-1].get("num_reqs", 256):
            return float(table[-1].get("overhead_us", 0))
        # Linear interpolation between neighbouring table entries.
        for i in range(len(table) - 1):
            n_lo = table[i].get("num_reqs", 0)
            n_hi = table[i + 1].get("num_reqs", 0)
            if n_lo <= num_reqs <= n_hi and n_hi > n_lo:
                ratio = (num_reqs - n_lo) / (n_hi - n_lo)
                o_lo = float(table[i].get("overhead_us", 0))
                o_hi = float(table[i + 1].get("overhead_us", 0))
                return o_lo + ratio * (o_hi - o_lo)
        return float(table[-1].get("overhead_us", 0))

    def _aggregate(self, samples):
        """Return scalar latency for a sample array per the oracle agg mode.

        - sample: random.choice (default, current behaviour).
        - median: statistics.median (central order statistic).
        - mean:   arithmetic mean.
        """
        if not samples:
            return None
        if self._oracle_agg == "sample":
            return float(self._rng.choice(samples))
        if self._oracle_agg == "median":
            s = sorted(samples)
            n = len(s)
            return float(s[n // 2]) if n % 2 == 1 else 0.5 * (s[n // 2 - 1] + s[n // 2])
        # mean
        return float(sum(samples)) / len(samples)

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
            return self._aggregate(raw)
        return None

    def _sample_3d_distribution(
        self, total_tokens: int, num_requests: int, num_new_reqs: int,
        has_prefill: bool = False,
    ) -> float | None:
        """F4: Exact-match 3D lookup; returns None on miss (caller falls back)."""
        if has_prefill and self._prefill_3d_distribution:
            table = self._prefill_3d_distribution
        elif not has_prefill and self._decode_3d_distribution:
            table = self._decode_3d_distribution
        elif self._combined_3d_distribution:
            table = self._combined_3d_distribution
        else:
            return None

        matching = [k for k in table.keys() if k[2] == num_new_reqs]
        if not matching:
            return None

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
            return self._aggregate(raw)
        return None

    def _sample_knn_2d(self, table, total_tokens, num_requests):
        """F5: K>1 Shepard p=2 kNN sampler over range-normalised (tt, conc)."""
        keys = list(table.keys())
        if not keys:
            return None

        tts = [k[0] for k in keys]
        concs = [k[1] for k in keys]
        tt_range = (max(tts) - min(tts)) or 1
        conc_range = (max(concs) - min(concs)) or 1

        def _dist(k):
            dt = (k[0] - total_tokens) / tt_range
            dc = (k[1] - num_requests) / conc_range
            return (dt * dt + dc * dc) ** 0.5

        ranked = sorted(((_dist(k), k) for k in keys), key=lambda x: x[0])
        top_k = ranked[: self._oracle_k]

        for d, k in top_k:
            if d == 0.0:
                samples = table[k].get("samples") or []
                if samples:
                    return self._aggregate(samples)
                return None

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
            return self._aggregate(samples)
        return None

    def estimate_step_latency_us(
        self, total_tokens: int,
        has_prefill: bool = False,
        num_requests: int = 0,
        num_new_reqs: int = 0,
        sum_kv: int = 0,
        **kwargs,
    ) -> float:
        """Estimate latency for one forward pass via 2D or 3D sampling.

        Args:
            total_tokens: Total tokens in the batch.
            has_prefill: Whether batch contains new prefill requests.
            num_requests: Number of requests in batch.
            num_new_reqs: F4 third-axis input. Ignored when
                VLLM_EMULATOR_PROFILE_AXES != 3d.
            sum_kv: Sum of num_computed_tokens across scheduled requests.
                Used with alpha_kv (from profile pack) to compute tt_eff
                for bucket lookup. Ignored when profile has no alpha_kv.
            **kwargs: Ignored (backward compat for callers passing
                      oracle_mode, profile_section, avg_context_len).

        Returns:
            Sampled latency in microseconds, or 0 if no data.
        """
        if total_tokens <= 0:
            return 0.0

        # Apply α-adjustment if profile was built with it.
        tt_query = total_tokens
        if self._alpha_kv > 0 and sum_kv > 0:
            tt_query = total_tokens + self._alpha_kv * sum_kv

        # F4: try 3D exact-match first when gate is on.
        if self._profile_axes == "3d":
            result = self._sample_3d_distribution(
                tt_query, max(num_requests, 1), num_new_reqs,
                has_prefill=has_prefill,
            )
            if result is not None:
                return result
            # Miss: fall back to 2D (per design doc §2 fallback rule).

        result = self._sample_2d_distribution(
            tt_query, max(num_requests, 1),
            has_prefill=has_prefill,
        )
        # IPC overhead: add measured per-concurrency overhead ONLY to prefill
        # steps (per commit 4d9983a0c design). Decode steps don't pay it.
        if result is not None and has_prefill:
            result = result + self._lookup_sched_overhead_us(max(num_requests, 1))
        if result is not None:
            return result

        # No distribution data at all -- return 0 and let caller handle.
        return 0.0


def create_oracle_from_profile_pack(
    profile_pack: dict[str, Any],
) -> ProfileGpuCostOracle:
    """Factory function to create an oracle from a profile pack."""
    return ProfileGpuCostOracle(profile_pack)
