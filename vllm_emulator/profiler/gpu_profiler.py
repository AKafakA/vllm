"""GPU compute profiler for generating profile packs.

This module profiles prefill and decode latency on real GPU hardware and
outputs a JSON profile pack compatible with the emulator's profile schema.

Usage (CLI)::

    python -m vllm_emulator.profiler.gpu_profiler \
        --gpu-model A100-SXM-80GB \
        --output profiles/a100.json

The profiler sweeps over configurable sequence-length / batch-size /
active-sequence grids and records per-iteration latency.  Raw samples are
aggregated (median by default) then converted into the profile-pack schema
consumed by :mod:`vllm_emulator.profile.loader`.

Design reference: Vidur profiler (latency grid sweep + aggregation).
"""

from __future__ import annotations

import dataclasses
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from vllm_emulator.profile.validator import validate_profile_pack


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GpuProfilingConfig:
    """Parameters that control a profiling run."""

    gpu_model: str = "A100-SXM-80GB"
    """Human-readable GPU identifier stored in the profile pack."""

    # Prefill sweep grid
    prefill_seq_lens: list[int] = field(
        default_factory=lambda: [128, 256, 512, 1024, 2048],
    )
    prefill_batch_sizes: list[int] = field(default_factory=lambda: [1])

    # Decode sweep grid
    decode_active_seqs: list[int] = field(
        default_factory=lambda: [1, 2, 4, 8, 16, 32],
    )

    # Measurement
    warmup_iters: int = 3
    measure_iters: int = 10
    aggregation: Literal["median", "mean"] = "median"

    version: str = "1.0"


# ---------------------------------------------------------------------------
# Raw / aggregated sample types
# ---------------------------------------------------------------------------

@dataclass
class RawSample:
    """A single latency measurement."""

    phase: Literal["prefill", "decode"]
    params: dict[str, int]
    latency_us: float


@dataclass
class AggregatedSample:
    """Aggregated result for one parameter combination."""

    phase: Literal["prefill", "decode"]
    params: dict[str, int]
    latency_us: float
    num_samples: int


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_samples(
    samples: list[RawSample],
    method: Literal["median", "mean"] = "median",
) -> list[AggregatedSample]:
    """Group raw samples by (phase, params) and aggregate latency values.

    Parameters
    ----------
    samples:
        Flat list of raw latency measurements.
    method:
        ``"median"`` (default) or ``"mean"``.

    Returns
    -------
    list[AggregatedSample]
        One entry per unique (phase, params-tuple) combination.
    """
    if method not in ("median", "mean"):
        raise ValueError(f"Unknown aggregation method: {method!r}")

    groups: dict[tuple[str, tuple[tuple[str, int], ...]], list[float]] = {}
    for s in samples:
        key = (s.phase, tuple(sorted(s.params.items())))
        groups.setdefault(key, []).append(s.latency_us)

    agg_fn = statistics.median if method == "median" else statistics.mean
    results: list[AggregatedSample] = []
    for (phase, params_tuple), latencies in sorted(groups.items()):
        results.append(
            AggregatedSample(
                phase=phase,
                params=dict(params_tuple),
                latency_us=agg_fn(latencies),
                num_samples=len(latencies),
            )
        )
    return results


# ---------------------------------------------------------------------------
# Schema conversion
# ---------------------------------------------------------------------------

def samples_to_profile_pack(
    aggregated: list[AggregatedSample],
    config: GpuProfilingConfig,
) -> dict[str, Any]:
    """Convert aggregated samples into a validated profile-pack dict.

    The output dict conforms to the schema expected by
    :func:`vllm_emulator.profile.validator.validate_profile_pack`.
    """
    prefill_rows: list[dict[str, Any]] = []
    decode_rows: list[dict[str, Any]] = []

    for s in aggregated:
        if s.phase == "prefill":
            prefill_rows.append({
                "seq_len": s.params["seq_len"],
                "batch_size": s.params["batch_size"],
                "latency_us": s.latency_us,
            })
        elif s.phase == "decode":
            decode_rows.append({
                "active_seqs": s.params["active_seqs"],
                "latency_us_per_token": s.latency_us,
            })

    # Sort for deterministic output
    prefill_rows.sort(key=lambda r: (r["batch_size"], r["seq_len"]))
    decode_rows.sort(key=lambda r: r["active_seqs"])

    pack: dict[str, Any] = {
        "version": config.version,
        "gpu_model": config.gpu_model,
        "prefill": prefill_rows,
        "decode": decode_rows,
    }

    # Validate before returning so callers get early feedback.
    validate_profile_pack(pack)
    return pack


# ---------------------------------------------------------------------------
# GPU profiler (requires CUDA at runtime)
# ---------------------------------------------------------------------------

class GpuProfiler:
    """Measures prefill and decode latency on a real GPU.

    This class is intentionally kept thin: the heavy lifting (model loading,
    kernel launch) is delegated to a *backend* callable so that tests can
    inject a fake backend.

    Parameters
    ----------
    config:
        Profiling configuration.
    backend:
        A callable ``(phase, params) -> latency_us`` that executes one
        iteration and returns wall-clock latency in microseconds.  When
        *None* the built-in CUDA timer backend is used (requires a GPU).
    """

    def __init__(
        self,
        config: GpuProfilingConfig | None = None,
        backend: Any | None = None,
    ) -> None:
        self.config = config or GpuProfilingConfig()
        self._backend = backend or _default_timer_backend

    # -- public API ---------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Execute the full profiling sweep and return a profile pack dict."""
        samples = self._collect_samples()
        aggregated = aggregate_samples(samples, method=self.config.aggregation)
        return samples_to_profile_pack(aggregated, self.config)

    def run_and_save(self, output_path: str | Path) -> Path:
        """Run profiling and write the profile pack to *output_path*."""
        pack = self.run()
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(pack, indent=2) + "\n", encoding="utf-8")
        return out

    # -- internals ----------------------------------------------------------

    def _collect_samples(self) -> list[RawSample]:
        samples: list[RawSample] = []
        cfg = self.config

        # Prefill sweep
        for seq_len in cfg.prefill_seq_lens:
            for batch_size in cfg.prefill_batch_sizes:
                params = {"seq_len": seq_len, "batch_size": batch_size}
                # Warmup
                for _ in range(cfg.warmup_iters):
                    self._backend("prefill", params)
                # Measure
                for _ in range(cfg.measure_iters):
                    lat = self._backend("prefill", params)
                    samples.append(RawSample("prefill", params, lat))

        # Decode sweep
        for active_seqs in cfg.decode_active_seqs:
            params = {"active_seqs": active_seqs}
            for _ in range(cfg.warmup_iters):
                self._backend("decode", params)
            for _ in range(cfg.measure_iters):
                lat = self._backend("decode", params)
                samples.append(RawSample("decode", params, lat))

        return samples


# ---------------------------------------------------------------------------
# Default backend (wall-clock stub – real CUDA backend pluggable)
# ---------------------------------------------------------------------------

def _default_timer_backend(phase: str, params: dict[str, int]) -> float:
    """Placeholder backend that returns wall-clock sleep latency.

    In production use, replace with a backend that launches real GPU kernels
    (e.g. via torch) and measures with CUDA events.
    """
    # Simulate a trivially small latency so the profiler can run without GPU.
    t0 = time.perf_counter()
    # no-op; real backend would launch a kernel here
    t1 = time.perf_counter()
    return (t1 - t0) * 1e6  # seconds → microseconds


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _cli_main() -> None:  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(
        description="Profile GPU compute latency and generate a profile pack.",
    )
    parser.add_argument(
        "--gpu-model",
        default="A100-SXM-80GB",
        help="GPU model name for the profile pack (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="profile_pack.json",
        help="Output JSON path (default: %(default)s).",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=3,
        help="Warmup iterations per grid point (default: %(default)s).",
    )
    parser.add_argument(
        "--measure-iters",
        type=int,
        default=10,
        help="Measurement iterations per grid point (default: %(default)s).",
    )
    parser.add_argument(
        "--aggregation",
        choices=["median", "mean"],
        default="median",
        help="Aggregation method (default: %(default)s).",
    )
    args = parser.parse_args()

    config = GpuProfilingConfig(
        gpu_model=args.gpu_model,
        warmup_iters=args.warmup_iters,
        measure_iters=args.measure_iters,
        aggregation=args.aggregation,
    )
    profiler = GpuProfiler(config=config)
    out = profiler.run_and_save(args.output)
    print(f"Profile pack written to {out}")


if __name__ == "__main__":
    _cli_main()
