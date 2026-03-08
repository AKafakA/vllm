"""GPU profiling script for emulator profile-pack generation.

This profiler runs lightweight vLLM generation workloads on a real GPU and
produces a profile-pack JSON matching the RFC schema:
- prefill: latency_us keyed by (seq_len, batch_size)
- decode: latency_us_per_token keyed by active_seqs
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProfileSample:
    phase: str
    key: int
    latency_us: float
    batch_size: int = 1


@dataclass(frozen=True)
class ProfilingConfig:
    model: str
    gpu_model: str
    output: Path
    seq_lens: tuple[int, ...] = (128, 256, 512, 1024)
    batch_sizes: tuple[int, ...] = (1, 2, 4)
    active_seqs: tuple[int, ...] = (1, 4, 8, 16)
    decode_tokens: int = 16
    warmup_runs: int = 1
    num_runs: int = 3
    dtype: str = "float16"


def _median(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot compute median of empty values")
    return float(statistics.median(values))


def records_to_profile_pack(
    *,
    gpu_model: str,
    prefill_samples: list[ProfileSample],
    decode_samples: list[ProfileSample],
    version: str = "1.0",
) -> dict[str, Any]:
    """Convert raw profile samples to profile-pack schema using median bucketing."""

    prefill_buckets: dict[tuple[int, int], list[float]] = {}
    for s in prefill_samples:
        if s.phase != "prefill":
            continue
        prefill_buckets.setdefault((int(s.key), int(s.batch_size)), []).append(float(s.latency_us))

    decode_buckets: dict[int, list[float]] = {}
    for s in decode_samples:
        if s.phase != "decode":
            continue
        decode_buckets.setdefault(int(s.key), []).append(float(s.latency_us))

    if not prefill_buckets:
        raise ValueError("No prefill samples collected")
    if not decode_buckets:
        raise ValueError("No decode samples collected")

    prefill_rows = [
        {
            "seq_len": seq_len,
            "batch_size": batch_size,
            "latency_us": _median(latencies),
        }
        for (seq_len, batch_size), latencies in sorted(prefill_buckets.items())
    ]

    decode_rows = [
        {
            "active_seqs": active_seqs,
            "latency_us_per_token": _median(latencies),
        }
        for active_seqs, latencies in sorted(decode_buckets.items())
    ]

    return {
        "version": version,
        "gpu_model": gpu_model,
        "prefill": prefill_rows,
        "decode": decode_rows,
    }


def _make_prompt(token_count: int) -> str:
    # Roughly stable, tokenizer-friendly synthetic prompt.
    return " ".join(["hello"] * max(1, token_count))


def run_gpu_profiling(config: ProfilingConfig) -> dict[str, Any]:
    """Run real-GPU profiling using vLLM and return a profile-pack dict."""

    from vllm import LLM, SamplingParams  # Imported lazily for unit-test friendliness.

    llm = LLM(model=config.model, dtype=config.dtype)

    prefill_samples: list[ProfileSample] = []
    decode_samples: list[ProfileSample] = []

    # Prefill proxy: max_tokens=1 so runtime is dominated by prompt processing.
    for seq_len in config.seq_lens:
        for batch_size in config.batch_sizes:
            prompts = [_make_prompt(seq_len) for _ in range(batch_size)]
            params = SamplingParams(max_tokens=1, temperature=0.0)

            for i in range(config.warmup_runs + config.num_runs):
                t0 = time.perf_counter()
                llm.generate(prompts, params)
                elapsed_us = (time.perf_counter() - t0) * 1e6
                if i >= config.warmup_runs:
                    prefill_samples.append(
                        ProfileSample(
                            phase="prefill",
                            key=seq_len,
                            batch_size=batch_size,
                            latency_us=elapsed_us,
                        ))

    # Decode proxy: short prompt + fixed generated tokens. Store per-token latency.
    for active_seqs in config.active_seqs:
        prompts = [_make_prompt(16) for _ in range(active_seqs)]
        params = SamplingParams(max_tokens=config.decode_tokens, temperature=0.0)

        for i in range(config.warmup_runs + config.num_runs):
            t0 = time.perf_counter()
            outputs = llm.generate(prompts, params)
            elapsed_us = (time.perf_counter() - t0) * 1e6

            if i >= config.warmup_runs:
                generated_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
                if generated_tokens <= 0:
                    raise RuntimeError("Profiler produced zero generated tokens")
                decode_samples.append(
                    ProfileSample(
                        phase="decode",
                        key=active_seqs,
                        latency_us=elapsed_us / generated_tokens,
                    ))

    return records_to_profile_pack(
        gpu_model=config.gpu_model,
        prefill_samples=prefill_samples,
        decode_samples=decode_samples,
    )


def _parse_int_csv(value: str) -> tuple[int, ...]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("Expected at least one integer")
    try:
        values = tuple(int(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("List must contain only integers") from exc
    if any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError("All values must be > 0")
    return values


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate vLLM emulator profile-pack from real GPU runs")
    parser.add_argument("--model", required=True, help="Model id/path loadable by vLLM")
    parser.add_argument("--gpu-model", required=True, help="GPU label written into profile pack (e.g., A100-SXM-80GB)")
    parser.add_argument("--output", required=True, type=Path, help="Output profile-pack JSON path")
    parser.add_argument("--seq-lens", type=_parse_int_csv, default=(128, 256, 512, 1024))
    parser.add_argument("--batch-sizes", type=_parse_int_csv, default=(1, 2, 4))
    parser.add_argument("--active-seqs", type=_parse_int_csv, default=(1, 4, 8, 16))
    parser.add_argument("--decode-tokens", type=int, default=16)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--dtype", default="float16")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = ProfilingConfig(
        model=args.model,
        gpu_model=args.gpu_model,
        output=args.output,
        seq_lens=args.seq_lens,
        batch_sizes=args.batch_sizes,
        active_seqs=args.active_seqs,
        decode_tokens=args.decode_tokens,
        warmup_runs=args.warmup_runs,
        num_runs=args.num_runs,
        dtype=args.dtype,
    )

    profile_pack = run_gpu_profiling(cfg)
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    cfg.output.write_text(json.dumps(profile_pack, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote profile pack: {cfg.output}")


if __name__ == "__main__":
    main()
