"""Trace-based profiler for execute_model() instrumentation.

Instruments the real GPU worker's execute_model() to record:
  - Input shape: (total_tokens, num_prefill_tokens, num_decode_seqs,
                   num_new_reqs, avg_seq_len, max_seq_len)
  - Output: wall-clock latency in microseconds (CUDA-synchronized)

The traces are saved as JSONL and can be converted into a profile
pack for the emulator oracle.

Usage:
    # Enable tracing via env var before starting vLLM:
    export VLLM_EMULATOR_TRACE_PROFILE=1
    export VLLM_EMULATOR_TRACE_OUTPUT=/path/to/trace.jsonl

    # Run real vLLM workload (tracing is passive, doesn't affect results)
    vllm bench throughput --model ... --num-prompts 100

    # Convert trace to profile pack:
    python -m vllm_emulator.profiler.trace_profiler \
        --trace /path/to/trace.jsonl \
        --output /path/to/profile.json
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput


# ---------------------------------------------------------------------------
# Trace collection (runs inside the GPU worker process)
# ---------------------------------------------------------------------------

TRACE_ENABLED_ENV = "VLLM_EMULATOR_TRACE_PROFILE"
TRACE_OUTPUT_ENV = "VLLM_EMULATOR_TRACE_OUTPUT"


class ExecuteModelTracer:
    """Records execute_model() input shapes and latencies.

    Installed as a lightweight wrapper around the real execute_model().
    Does NOT replace GPU execution — just measures it.
    """

    def __init__(self, output_path: str | None = None):
        self._output_path = output_path or os.environ.get(
            TRACE_OUTPUT_ENV, "/tmp/vllm_emulator_trace.jsonl"
        )
        self._records: list[dict[str, Any]] = []
        self._enabled = os.environ.get(
            TRACE_ENABLED_ENV, ""
        ).lower() in ("1", "true", "yes")
        self._step_count = 0

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def extract_shape(
        self, scheduler_output: "SchedulerOutput"
    ) -> dict[str, int]:
        """Extract the batch shape from a SchedulerOutput."""
        total_tokens = scheduler_output.total_num_scheduled_tokens

        # Count prefill tokens and requests
        num_new_reqs = len(scheduler_output.scheduled_new_reqs)
        num_prefill_tokens = 0
        seq_lens: list[int] = []
        for req in scheduler_output.scheduled_new_reqs:
            if req.prompt_token_ids:
                n = len(req.prompt_token_ids)
                num_prefill_tokens += n
                seq_lens.append(n)

        # Count decode sequences
        cached = scheduler_output.scheduled_cached_reqs
        num_decode_seqs = cached.num_reqs if cached.num_reqs > 0 else 0

        # Decode sequences each contribute 1 token
        num_decode_tokens = num_decode_seqs

        avg_seq_len = (
            sum(seq_lens) // len(seq_lens) if seq_lens else 0
        )
        max_seq_len = max(seq_lens) if seq_lens else 0

        return {
            "total_tokens": total_tokens,
            "num_prefill_tokens": num_prefill_tokens,
            "num_decode_tokens": num_decode_tokens,
            "num_new_reqs": num_new_reqs,
            "num_decode_seqs": num_decode_seqs,
            "avg_seq_len": avg_seq_len,
            "max_seq_len": max_seq_len,
        }

    def record(
        self,
        scheduler_output: "SchedulerOutput",
        latency_us: float,
    ) -> None:
        """Record one execute_model() invocation."""
        shape = self.extract_shape(scheduler_output)
        shape["latency_us"] = round(latency_us, 1)
        shape["step"] = self._step_count
        self._step_count += 1
        self._records.append(shape)

    def flush(self) -> None:
        """Write collected records to disk."""
        if not self._records:
            return
        path = Path(self._output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            for rec in self._records:
                f.write(json.dumps(rec) + "\n")
        count = len(self._records)
        self._records.clear()
        print(f"[TraceProfiler] Flushed {count} records to {path}")

    def flush_periodic(self, every: int = 100) -> None:
        """Flush every N steps to avoid memory buildup."""
        if len(self._records) >= every:
            self.flush()


# ---------------------------------------------------------------------------
# Trace → Profile Pack conversion
# ---------------------------------------------------------------------------

def load_trace(trace_path: str) -> list[dict[str, Any]]:
    """Load a JSONL trace file."""
    records = []
    with open(trace_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def trace_to_profile_pack(
    records: list[dict[str, Any]],
    gpu_model: str = "unknown",
    model_name: str = "unknown",
    bucket_size: int = 16,
) -> dict[str, Any]:
    """Convert trace records into a profile pack with forward_pass section.

    Groups records by total_tokens (bucketed) and computes median latency
    per bucket. Also generates prefill and decode sections for backward
    compatibility.

    Args:
        records: List of trace records from ExecuteModelTracer.
        gpu_model: GPU model name for the profile pack.
        model_name: Model name for the profile pack.
        bucket_size: Bucket size for total_tokens grouping.

    Returns:
        A profile pack dict compatible with the emulator oracle.
    """
    import statistics

    if not records:
        raise ValueError("No trace records to convert")

    # --- Forward pass: bucket by total_tokens ---
    fwd_buckets: dict[int, list[float]] = {}
    for rec in records:
        total = rec["total_tokens"]
        if total <= 0:
            continue
        # Round to nearest bucket
        bucket = max(1, (total + bucket_size // 2) // bucket_size * bucket_size)
        fwd_buckets.setdefault(bucket, []).append(rec["latency_us"])

    forward_pass = []
    for bucket in sorted(fwd_buckets):
        latencies = fwd_buckets[bucket]
        forward_pass.append({
            "total_tokens": bucket,
            "latency_us": round(statistics.median(latencies), 1),
            "num_samples": len(latencies),
        })

    # --- Prefill: bucket by (avg_seq_len, num_new_reqs) ---
    prefill_buckets: dict[tuple[int, int], list[float]] = {}
    for rec in records:
        if rec["num_prefill_tokens"] > 0 and rec["num_decode_seqs"] == 0:
            # Pure prefill step
            seq_len_bucket = max(1, rec["avg_seq_len"] // 64 * 64) or 64
            bs = rec["num_new_reqs"]
            prefill_buckets.setdefault((seq_len_bucket, bs), []).append(
                rec["latency_us"]
            )

    prefill = []
    for (seq_len, bs) in sorted(prefill_buckets):
        latencies = prefill_buckets[(seq_len, bs)]
        prefill.append({
            "seq_len": seq_len,
            "batch_size": bs,
            "latency_us": round(statistics.median(latencies), 1),
        })

    # --- Decode: bucket by num_decode_seqs ---
    decode_buckets: dict[int, list[float]] = {}
    for rec in records:
        if rec["num_prefill_tokens"] == 0 and rec["num_decode_seqs"] > 0:
            # Pure decode step
            decode_buckets.setdefault(rec["num_decode_seqs"], []).append(
                rec["latency_us"]
            )

    decode = []
    for n_seqs in sorted(decode_buckets):
        latencies = decode_buckets[n_seqs]
        decode.append({
            "active_seqs": n_seqs,
            "latency_us_per_token": round(statistics.median(latencies), 1),
        })

    # Ensure at least one entry in each section
    if not prefill:
        prefill = [{"seq_len": 128, "batch_size": 1, "latency_us": 1.0}]
    if not decode:
        decode = [{"active_seqs": 1, "latency_us_per_token": 1.0}]

    return {
        "version": "2.0",
        "gpu_model": gpu_model,
        "model_name": model_name,
        "profile_method": "trace",
        "num_trace_records": len(records),
        "prefill": prefill,
        "decode": decode,
        "forward_pass": forward_pass,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli_main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert execute_model() trace to profile pack.",
    )
    parser.add_argument(
        "--trace",
        required=True,
        help="Path to JSONL trace file.",
    )
    parser.add_argument(
        "--output", "-o",
        default="trace_profile_pack.json",
        help="Output JSON profile pack path.",
    )
    parser.add_argument(
        "--gpu-model",
        default="unknown",
        help="GPU model name for the profile pack.",
    )
    parser.add_argument(
        "--model-name",
        default="unknown",
        help="Model name for the profile pack.",
    )
    parser.add_argument(
        "--bucket-size",
        type=int,
        default=16,
        help="Bucket size for total_tokens grouping.",
    )
    parser.add_argument(
        "--print-stats",
        action="store_true",
        help="Print trace statistics.",
    )
    args = parser.parse_args()

    records = load_trace(args.trace)
    print(f"Loaded {len(records)} trace records from {args.trace}")

    if args.print_stats:
        import statistics
        latencies = [r["latency_us"] for r in records]
        tokens = [r["total_tokens"] for r in records]
        print(f"  Latency: median={statistics.median(latencies):.0f}us, "
              f"mean={statistics.mean(latencies):.0f}us, "
              f"p99={sorted(latencies)[int(len(latencies)*0.99)]:.0f}us")
        print(f"  Tokens/step: median={statistics.median(tokens):.0f}, "
              f"max={max(tokens)}")
        prefill_steps = sum(1 for r in records if r["num_prefill_tokens"] > 0)
        decode_steps = sum(1 for r in records if r["num_decode_seqs"] > 0 and r["num_prefill_tokens"] == 0)
        mixed_steps = sum(1 for r in records if r["num_prefill_tokens"] > 0 and r["num_decode_seqs"] > 0)
        print(f"  Steps: {len(records)} total, {prefill_steps} prefill, "
              f"{decode_steps} decode, {mixed_steps} mixed")

    pack = trace_to_profile_pack(
        records,
        gpu_model=args.gpu_model,
        model_name=args.model_name,
        bucket_size=args.bucket_size,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(pack, indent=2) + "\n", encoding="utf-8")
    print(f"Profile pack written to {out}")
    print(f"  forward_pass: {len(pack['forward_pass'])} buckets")
    print(f"  prefill: {len(pack['prefill'])} entries")
    print(f"  decode: {len(pack['decode'])} entries")


if __name__ == "__main__":
    _cli_main()
