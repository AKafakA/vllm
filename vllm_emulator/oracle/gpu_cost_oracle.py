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

        # Group prefill samples by batch_size, sorted by seq_len within each
        self._prefill_by_bs: dict[int, list[dict]] = {}
        for s in profile_pack["prefill"]:
            bs = s["batch_size"]
            self._prefill_by_bs.setdefault(bs, []).append(s)
        for bs in self._prefill_by_bs:
            self._prefill_by_bs[bs].sort(key=lambda s: s["seq_len"])

        # Available batch sizes sorted for interpolation
        self._prefill_batch_sizes = sorted(self._prefill_by_bs.keys())

        # Fallback: flatten all samples for single-dim lookup
        self._prefill_samples = sorted(
            profile_pack["prefill"], key=lambda s: s["seq_len"]
        )

        self._decode_samples = sorted(
            profile_pack["decode"], key=lambda s: s["active_seqs"]
        )

        # Pre-compute power-law fits per batch_size group
        self._prefill_pw: dict[int, tuple[float, float]] = {}
        for bs, samples in self._prefill_by_bs.items():
            xs = [float(s["seq_len"]) for s in samples]
            ys = [float(s["latency_us"]) for s in samples]
            self._prefill_pw[bs] = _fit_power_law(xs, ys)

        # Global fallback power-law (batch_size=1 or flattened)
        if 1 in self._prefill_pw:
            self._prefill_pw_a, self._prefill_pw_b = self._prefill_pw[1]
        elif self._prefill_samples:
            xs = [float(s["seq_len"]) for s in self._prefill_samples]
            ys = [float(s["latency_us"]) for s in self._prefill_samples]
            self._prefill_pw_a, self._prefill_pw_b = _fit_power_law(xs, ys)
        else:
            # Serving profile with forward_pass only — no legacy sections
            self._prefill_pw_a, self._prefill_pw_b = (1.0, 1.0)

        decode_xs = [float(s["active_seqs"]) for s in self._decode_samples]
        decode_ys = [
            float(s["latency_us_per_token"]) for s in self._decode_samples
        ]
        if decode_xs:
            self._decode_pw_a, self._decode_pw_b = _fit_power_law(
                decode_xs, decode_ys)
        else:
            self._decode_pw_a, self._decode_pw_b = (1.0, 1.0)

        # Unified forward_pass profile (optional, preferred if available)
        self._forward_pass_samples = sorted(
            profile_pack.get("forward_pass", []),
            key=lambda s: s["total_tokens"],
        )
        if self._forward_pass_samples:
            fwd_xs = [float(s["total_tokens"]) for s in self._forward_pass_samples]
            fwd_ys = [float(s["latency_us"]) for s in self._forward_pass_samples]
            self._fwd_pw_a, self._fwd_pw_b = _fit_power_law(fwd_xs, fwd_ys)
        else:
            self._fwd_pw_a, self._fwd_pw_b = self._prefill_pw_a, self._prefill_pw_b

        # 2D forward pass profile (optional, most accurate)
        self._forward_pass_2d = profile_pack.get("forward_pass_2d", [])

        # Separate prefill/decode forward_pass (for step-type-aware estimation)
        self._prefill_forward_pass = sorted(
            profile_pack.get("prefill_forward_pass", []),
            key=lambda s: s["total_tokens"],
        )
        self._decode_forward_pass = sorted(
            profile_pack.get("decode_forward_pass", []),
            key=lambda s: s["total_tokens"],
        )

        # Offline forward_pass: decode step-cycle via LLM() path with CUDA
        # graphs. Used for offline (bench throughput) emulation where batch
        # sizes are large and step-cycle ≈ pure forward pass.
        self._offline_forward_pass = sorted(
            profile_pack.get("offline_forward_pass", []),
            key=lambda s: s["total_tokens"],
        )

        # 2D profile: per-request latency overhead from step-cycle data.
        self._2d_overhead_per_req_us = float(
            profile_pack.get("overhead_per_request_us", 0))

        # Concurrency correction table: bucketed correction by num_requests.
        # correction[N] = actual_step_cycle - oracle_1d_prediction at concurrency N.
        # Positive = profile underestimates, negative = overestimates.
        self._correction_table = profile_pack.get("correction_table", [])

        # 2D tables: indexed by (total_tokens, concurrency).
        # Split by step type: prefill (eager mode) vs decode (CUDA graph).
        # In vLLM V1, mixed batches (prefill) run eager, pure decode uses graphs.
        def _load_2d_table(key: str) -> tuple[dict, list]:
            table: dict[int, list[tuple[int, float]]] = {}
            for e in profile_pack.get(key, []):
                tt = e["tt"]
                table.setdefault(tt, []).append((e["conc"], e["latency_us"]))
            for tt in table:
                table[tt].sort()
            return table, sorted(table.keys())

        self._2d_table, self._2d_table_tts = _load_2d_table("step_cycle_2d_table")
        self._prefill_2d_table, self._prefill_2d_tts = _load_2d_table("prefill_2d_table")
        self._decode_2d_table, self._decode_2d_tts = _load_2d_table("decode_2d_table")

    def get_max_step_cycle_us(self, num_requests: int) -> float:
        """Get the maximum profiled step_cycle at a given concurrency.

        Used by chain drift correction (Option D): caps gpu_free_time
        drift to the maximum observed step duration at the current
        concurrency level, derived from the 2D table.
        """
        if not self._2d_table:
            return 0.0
        max_lat = 0.0
        for tt in self._2d_table_tts:
            result = self._lookup_2d_table(tt, num_requests)
            if result is not None and result > max_lat:
                max_lat = result
        return max_lat

    @property
    def gpu_model(self) -> str:
        return self._gpu_model

    def estimate_prefill_latency_us(
        self, prompt_tokens: int, batch_size: int
    ) -> float:
        """Estimate prefill latency for a batch of requests.

        If profile data exists for the given batch_size, interpolates
        within that group.  Otherwise, finds the two nearest batch_size
        groups and interpolates between them.  Falls back to power-law
        extrapolation outside the profiled range.
        """
        if prompt_tokens <= 0:
            return 0.0

        # Try exact batch_size match first
        if batch_size in self._prefill_by_bs:
            return self._estimate_prefill_for_bs(
                prompt_tokens, batch_size
            )

        # Interpolate between nearest batch_size groups
        if len(self._prefill_batch_sizes) >= 2 and batch_size > 0:
            bss = self._prefill_batch_sizes
            # Find bracketing batch sizes
            lo_bs, hi_bs = bss[0], bss[-1]
            for i in range(len(bss) - 1):
                if bss[i] <= batch_size <= bss[i + 1]:
                    lo_bs, hi_bs = bss[i], bss[i + 1]
                    break

            if batch_size <= bss[0]:
                return self._estimate_prefill_for_bs(
                    prompt_tokens, bss[0]
                )
            if batch_size >= bss[-1]:
                # Extrapolate from the two largest batch sizes
                lo_lat = self._estimate_prefill_for_bs(
                    prompt_tokens, bss[-2]
                )
                hi_lat = self._estimate_prefill_for_bs(
                    prompt_tokens, bss[-1]
                )
                if bss[-1] != bss[-2]:
                    ratio = (batch_size - bss[-2]) / (bss[-1] - bss[-2])
                    return lo_lat + ratio * (hi_lat - lo_lat)
                return hi_lat

            lo_lat = self._estimate_prefill_for_bs(prompt_tokens, lo_bs)
            hi_lat = self._estimate_prefill_for_bs(prompt_tokens, hi_bs)
            ratio = (batch_size - lo_bs) / (hi_bs - lo_bs)
            return lo_lat + ratio * (hi_lat - lo_lat)

        # Single batch_size group — use it regardless
        bs = self._prefill_batch_sizes[0] if self._prefill_batch_sizes else 1
        return self._estimate_prefill_for_bs(prompt_tokens, bs)

    def _estimate_prefill_for_bs(
        self, prompt_tokens: int, batch_size: int
    ) -> float:
        """Estimate prefill for a specific profiled batch_size."""
        samples = self._prefill_by_bs.get(batch_size, self._prefill_samples)
        seq_lens = [float(s["seq_len"]) for s in samples]
        latencies = [float(s["latency_us"]) for s in samples]

        if seq_lens[0] <= prompt_tokens <= seq_lens[-1]:
            return _interpolate_linear(seq_lens, latencies, float(prompt_tokens))

        # Power-law extrapolation
        a, b = self._prefill_pw.get(
            batch_size, (self._prefill_pw_a, self._prefill_pw_b)
        )
        return a * (prompt_tokens ** b)

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

    def estimate_step_latency_us(
        self, total_tokens: int, avg_context_len: int = 0,
        has_prefill: bool = False, profile_section: str = "online",
        num_requests: int = 0, oracle_mode: str = "step_cycle",
    ) -> float:
        """Estimate latency for one forward pass.

        Args:
            total_tokens: Total tokens in the batch
            avg_context_len: Average context length (for legacy 2D lookup)
            has_prefill: Whether batch contains new prefill requests
            profile_section: "online" or "offline" — selects profile data
            num_requests: Number of requests in batch (for 2d mode)
            oracle_mode: "step_cycle", "hybrid", or "2d"

        Profile selection:
          offline → offline_forward_pass (if available, else fall through)
          online + has_prefill → prefill_forward_pass
          online + no prefill → decode_forward_pass
          fallback → combined forward_pass
        """
        if total_tokens <= 0:
            return 0.0

        # 2D table mode: bilinear interpolation over (tt, concurrency)
        # Use separate prefill/decode tables when available (captures
        # CUDA graph vs eager mode difference in vLLM V1).
        if oracle_mode == "2d" and num_requests > 0:
            # Try step-type-specific 2D table first
            if has_prefill and self._prefill_2d_table:
                result = self._lookup_2d_table(
                    total_tokens, num_requests,
                    self._prefill_2d_table, self._prefill_2d_tts)
                if result is not None:
                    return result
            elif not has_prefill and self._decode_2d_table:
                result = self._lookup_2d_table(
                    total_tokens, num_requests,
                    self._decode_2d_table, self._decode_2d_tts)
                if result is not None:
                    return result
            # Fall back to combined 2D table
            if self._2d_table:
                result = self._lookup_2d_table(
                    total_tokens, num_requests,
                    self._2d_table, self._2d_table_tts)
                if result is not None:
                    return result
            # Fall through to 1D if no 2D table covers this range

        # Corrected mode: 1D base + bucketed correction by num_requests
        if oracle_mode == "corrected" and self._correction_table and num_requests > 0:
            base = self._estimate_1d(total_tokens, has_prefill, profile_section)
            correction = self._lookup_correction(num_requests)
            return max(1.0, base + correction)  # clamp to positive

        # Offline profile: step-cycles from LLM() path with CUDA graphs
        if profile_section == "offline" and self._offline_forward_pass:
            off_samples = self._offline_forward_pass
            xs = [float(s["total_tokens"]) for s in off_samples]
            ys = [float(s["latency_us"]) for s in off_samples]
            if xs[0] <= total_tokens <= xs[-1]:
                return _interpolate_linear(xs, ys, float(total_tokens))
            # Beyond range: extrapolate up, fall through for below
            if total_tokens > xs[-1] and len(xs) >= 2:
                a, b = _fit_power_law(xs, ys)
                return a * (total_tokens ** b)

        # 2D lookup (if available)
        if self._forward_pass_2d and avg_context_len > 0:
            return self._estimate_2d(total_tokens, avg_context_len)

        # Online profile: step-type-aware lookup
        samples = self._forward_pass_samples  # default (combined)
        if has_prefill and self._prefill_forward_pass:
            samples = self._prefill_forward_pass
        elif not has_prefill and self._decode_forward_pass:
            samples = self._decode_forward_pass

        if not samples:
            # Fall back to combined
            samples = self._forward_pass_samples
        if not samples:
            return self.estimate_prefill_latency_us(total_tokens, batch_size=1)

        xs = [float(s["total_tokens"]) for s in samples]
        ys = [float(s["latency_us"]) for s in samples]

        if xs[0] <= total_tokens <= xs[-1]:
            return _interpolate_linear(xs, ys, float(total_tokens))

        # Extrapolate using power-law from the combined profile
        return self._fwd_pw_a * (total_tokens ** self._fwd_pw_b)

    def _lookup_correction(self, num_requests: int) -> float:
        """Interpolate correction from bucketed correction table."""
        table = self._correction_table
        if not table:
            return 0.0

        # Find bracketing entries
        ns = [e["num_requests"] for e in table]
        cs = [e["correction_us"] for e in table]

        if num_requests <= ns[0]:
            return cs[0]
        if num_requests >= ns[-1]:
            return cs[-1]

        # Linear interpolation between brackets
        for i in range(len(ns) - 1):
            if ns[i] <= num_requests <= ns[i + 1]:
                frac = (num_requests - ns[i]) / (ns[i + 1] - ns[i])
                return cs[i] + frac * (cs[i + 1] - cs[i])

        return cs[-1]

    def _lookup_2d_table(self, total_tokens: int, num_requests: int,
                        table: dict | None = None,
                        table_tts: list | None = None) -> float | None:
        """Bilinear interpolation over a 2D (tt, concurrency) table.

        Args:
            table: {tt -> [(conc, latency_us), ...]} dict. Defaults to combined.
            table_tts: sorted tt keys. Defaults to combined.

        Returns None if the table doesn't cover this (tt, conc) range.
        """
        if table is None:
            table = self._2d_table
        if table_tts is None:
            table_tts = self._2d_table_tts
        tts = table_tts
        if not tts:
            return None

        # Clamp tt to table range
        if total_tokens < tts[0] or total_tokens > tts[-1]:
            return None

        def _interp_conc(conc_entries: list[tuple[int, float]], conc: int) -> float:
            """Interpolate latency for a given concurrency within one tt."""
            if len(conc_entries) == 1:
                return conc_entries[0][1]
            cs = [c for c, _ in conc_entries]
            ls = [l for _, l in conc_entries]
            if conc <= cs[0]:
                return ls[0]
            if conc >= cs[-1]:
                # Extrapolate from last two points
                if len(cs) >= 2:
                    slope = (ls[-1] - ls[-2]) / max(cs[-1] - cs[-2], 1)
                    return ls[-1] + slope * (conc - cs[-1])
                return ls[-1]
            return _interpolate_linear(
                [float(c) for c in cs], [float(l) for l in ls], float(conc))

        # Find bracketing tt values
        if total_tokens in table:
            return _interp_conc(table[total_tokens], num_requests)

        # Interpolate between two tt brackets
        lo_tt = tts[0]
        hi_tt = tts[-1]
        for i in range(len(tts) - 1):
            if tts[i] <= total_tokens <= tts[i + 1]:
                lo_tt, hi_tt = tts[i], tts[i + 1]
                break

        lo_lat = _interp_conc(table[lo_tt], num_requests)
        hi_lat = _interp_conc(table[hi_tt], num_requests)
        frac = (total_tokens - lo_tt) / max(hi_tt - lo_tt, 1)
        return lo_lat + frac * (hi_lat - lo_lat)

    def _estimate_1d(self, total_tokens: int, has_prefill: bool,
                     profile_section: str) -> float:
        """1D base latency estimation (no per-request overhead).
        Used by both step_cycle mode directly and 2d mode as the base."""
        return self.estimate_step_latency_us(
            total_tokens, has_prefill=has_prefill,
            profile_section=profile_section,
            oracle_mode="step_cycle")  # force 1D to avoid recursion

    def _estimate_2d(self, total_tokens: int, avg_context_len: int) -> float:
        """Bilinear interpolation over the 2D forward_pass profile.

        Groups 2D samples by total_tokens, interpolates avg_context_len
        within each group, then interpolates between groups.
        """
        # Group by total_tokens
        by_tt: dict[int, list[tuple[int, float]]] = {}
        for s in self._forward_pass_2d:
            tt = s["total_tokens"]
            ctx = s["avg_context_len"]
            lat = s["latency_us"]
            by_tt.setdefault(tt, []).append((ctx, lat))

        tts = sorted(by_tt.keys())
        if not tts:
            return self.estimate_prefill_latency_us(total_tokens, batch_size=1)

        def interp_context(points: list[tuple[int, float]], ctx: int) -> float:
            """Interpolate latency for a given context within one tt group."""
            points = sorted(points)
            xs = [float(p[0]) for p in points]
            ys = [float(p[1]) for p in points]
            if len(xs) == 1:
                return ys[0]
            if ctx <= xs[0]:
                return ys[0]
            if ctx >= xs[-1]:
                # Linear extrapolation from last two points
                if len(xs) >= 2:
                    slope = (ys[-1] - ys[-2]) / max(xs[-1] - xs[-2], 1)
                    return ys[-1] + slope * (ctx - xs[-1])
                return ys[-1]
            return _interpolate_linear(xs, ys, float(ctx))

        # Find bracketing total_tokens
        if total_tokens <= tts[0]:
            return interp_context(by_tt[tts[0]], avg_context_len)
        if total_tokens >= tts[-1]:
            return interp_context(by_tt[tts[-1]], avg_context_len)

        for i in range(len(tts) - 1):
            if tts[i] <= total_tokens <= tts[i + 1]:
                lo_lat = interp_context(by_tt[tts[i]], avg_context_len)
                hi_lat = interp_context(by_tt[tts[i + 1]], avg_context_len)
                ratio = (total_tokens - tts[i]) / max(tts[i + 1] - tts[i], 1)
                return lo_lat + ratio * (hi_lat - lo_lat)

        return interp_context(by_tt[tts[-1]], avg_context_len)


def create_oracle_from_profile_pack(
    profile_pack: dict[str, Any],
) -> ProfileGpuCostOracle:
    """Factory function to create an oracle from a profile pack."""
    return ProfileGpuCostOracle(profile_pack)
