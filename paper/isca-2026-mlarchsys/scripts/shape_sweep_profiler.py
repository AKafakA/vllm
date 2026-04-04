#!/usr/bin/env python3
"""Data-independent shape-sweep profiler for vLLM-Emulator.

Runs controlled workloads through real vLLM with execute_model() tracing
to build a profile that covers all relevant input shapes. The resulting
profile is workload-independent and can predict latency for any workload.

Shape dimensions swept:
  - Pure prefill: varied num_requests × seq_len
  - Pure decode: varied num_active_seqs
  - Mixed prefill+decode: combinations (requires chunked prefill)

The sweep is bounded by vLLM serving config:
  - max_num_batched_tokens: upper bound on total_tokens per step
  - max_num_seqs: upper bound on concurrent sequences

Usage:
    python shape_sweep_profiler.py \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --gpu-model RTX-3060-12GB \
        --output profiles/sweep-0.5b.json \
        --max-model-len 4096

The script:
  1. Starts vLLM with tracing enabled
  2. Runs a series of workloads designed to hit specific batch shapes
  3. Collects trace records from execute_model()
  4. Converts traces to a profile pack
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))


def run_sweep_workload(
    model: str,
    sweep_configs: list[dict],
    max_model_len: int = 4096,
    tp: int = 1,
    dtype: str = "auto",
    quantization: str | None = None,
    trace_output: str = "/tmp/shape_sweep_trace.jsonl",
    max_num_seqs: int = 256,
) -> str:
    """Run a series of workloads to sweep input shapes.

    Each sweep config specifies:
      - num_prompts: how many concurrent requests
      - input_len: prompt length per request
      - output_len: tokens to generate per request (controls decode steps)

    Returns path to the trace JSONL file.
    """
    # Enable tracing
    os.environ["VLLM_EMULATOR_TRACE_PROFILE"] = "1"
    os.environ["VLLM_EMULATOR_TRACE_OUTPUT"] = trace_output

    # Clear previous trace
    Path(trace_output).unlink(missing_ok=True)

    from vllm import LLM, SamplingParams

    print(f"Loading model: {model} (tp={tp})")
    llm_kwargs = dict(
        model=model,
        tensor_parallel_size=tp,
        dtype=dtype,
        trust_remote_code=True,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
    )
    if quantization:
        llm_kwargs["quantization"] = quantization

    llm = LLM(**llm_kwargs)
    print("Model loaded.\n")

    for i, cfg in enumerate(sweep_configs):
        num_prompts = cfg["num_prompts"]
        input_len = cfg["input_len"]
        output_len = cfg["output_len"]
        label = cfg.get("label", f"sweep-{i}")

        print(f"  [{i+1}/{len(sweep_configs)}] {label}: "
              f"{num_prompts} prompts × {input_len} in / {output_len} out")

        # Generate UNIQUE prompts to avoid prefix caching effects.
        # Each prompt has exact input_len tokens but different content.
        actual_len = min(input_len, max_model_len - output_len - 1)
        from vllm import TokensPrompt
        import random as _rng
        prompts = []
        for j in range(num_prompts):
            # Unique random token IDs per prompt — no prefix sharing
            ids = [_rng.randint(100, 30000) for _ in range(actual_len)]
            prompts.append(TokensPrompt(prompt_token_ids=ids))
        params = SamplingParams(
            max_tokens=output_len,
            temperature=0,
            ignore_eos=True,  # Force full output_len generation
        )

        # Warmup (1 run)
        llm.generate(prompts, params)

        # Measured run (2 runs for stability)
        for _ in range(2):
            llm.generate(prompts, params)

        print(f"    done")

    # Cleanup env
    del os.environ["VLLM_EMULATOR_TRACE_PROFILE"]
    del os.environ["VLLM_EMULATOR_TRACE_OUTPUT"]

    # Force flush via engine shutdown
    del llm

    return trace_output


def generate_sweep_configs(
    max_num_seqs: int = 256,
    max_num_batched_tokens: int = 8192,
    max_output_len: int = 256,
    max_model_len: int = 4096,
) -> list[dict]:
    """Generate sweep configurations covering the shape space.

    The sweep must cover ALL operating conditions the runtime workload
    may produce:
    - All batch sizes from 1 to max_num_seqs
    - All prompt lengths the workload uses
    - All decode context depths (prompt_len + output_len)

    Args:
        max_output_len: Maximum output tokens any workload request may
            generate. Decode configs generate at least this many tokens
            so the profile covers the full context depth range.
    """
    configs = []

    # --- Pure decode sweeps ---
    # Generate output_len >= max_output_len so decode steps reach the
    # same context depth as the real workload.
    # Dense coverage at low batch sizes (1-20) for online serving accuracy,
    # plus powers of 2 for larger batches (offline inference).
    decode_batch_sizes = (list(range(1, 21)) +
                          [24, 28, 32, 40, 48, 56, 64, 80, 96, 128])
    decode_batch_sizes = [b for b in decode_batch_sizes if b <= max_num_seqs]
    decode_prompt_lens = [128, 256, 512]

    for bs in decode_batch_sizes:
        for prompt_len in decode_prompt_lens:
            if bs * prompt_len > max_num_batched_tokens * 2:
                continue
            # Output must cover max workload output length
            output_len = min(max_output_len, max_model_len - prompt_len - 1)
            if output_len < 16:
                continue
            configs.append({
                "num_prompts": bs,
                "input_len": prompt_len,
                "output_len": output_len,
                "label": f"decode-{bs}seqs-{prompt_len}ctx-{output_len}out",
            })

    # --- Pure prefill sweeps ---
    # Short output (1 token) so we mostly measure prefill
    prefill_lens = [32, 64, 128, 256, 512, 1024]
    prefill_lens = [l for l in prefill_lens if l <= max_num_batched_tokens]
    prefill_batch_sizes = [1, 2, 4, 8]
    prefill_batch_sizes = [b for b in prefill_batch_sizes
                           if b <= max_num_seqs]

    for seq_len in prefill_lens:
        for bs in prefill_batch_sizes:
            if bs * seq_len > max_num_batched_tokens:
                continue
            configs.append({
                "num_prompts": bs,
                "input_len": seq_len,
                "output_len": 1,
                "label": f"prefill-{bs}x{seq_len}",
            })

    # --- Large prefill batches ---
    # The scheduler packs up to max_num_batched_tokens per step.
    # We need to cover total_tokens up to that limit.
    # Use many requests × moderate seq_len to hit high total_tokens.
    # Use single long sequences to force large total_tokens in one step
    # (avoids chunking that splits across iterations)
    large_seq_lens = [512, 1024, 2048, 4096]
    large_seq_lens = [l for l in large_seq_lens if l <= max_num_batched_tokens]
    for seq_len in large_seq_lens:
        configs.append({
            "num_prompts": 1,
            "input_len": seq_len,
            "output_len": 1,
            "label": f"large-single-{seq_len}tok",
        })

    # Use multiple requests that together fill the token budget
    # The key: all requests submit simultaneously, scheduler packs them
    large_total_tokens = [1024, 2048, 4096, max_num_batched_tokens]
    for target_tokens in large_total_tokens:
        # Short seq_len × many requests → all fit in one prefill step
        for seq_len in [64, 128, 256]:
            bs = max(1, target_tokens // seq_len)
            if bs <= max_num_seqs:
                configs.append({
                    "num_prompts": bs,
                    "input_len": seq_len,
                    "output_len": 1,
                    "label": f"large-batch-{bs}x{seq_len}={bs*seq_len}tok",
                })

    # --- Mixed prefill+decode sweeps ---
    # Long prompts with many requests → chunked prefill creates mixed steps
    # where some requests are prefilling while others are decoding
    mixed_configs = [
        {"num_prompts": 16, "input_len": 512, "output_len": max_output_len,
         "label": "mixed-16x512"},
        {"num_prompts": 8, "input_len": 1024, "output_len": max_output_len,
         "label": "mixed-8x1024"},
        {"num_prompts": 32, "input_len": 256, "output_len": max_output_len,
         "label": "mixed-32x256"},
        {"num_prompts": 16, "input_len": 1024, "output_len": max_output_len,
         "label": "mixed-16x1024"},
        # High-batch mixed configs to cover chunked prefill patterns
        {"num_prompts": 30, "input_len": 256, "output_len": max_output_len,
         "label": "mixed-30x256"},
        {"num_prompts": 32, "input_len": 256, "output_len": max_output_len,
         "label": "mixed-32x256"},
        {"num_prompts": 16, "input_len": 512, "output_len": max_output_len,
         "label": "mixed-16x512-long"},
        {"num_prompts": 64, "input_len": 128, "output_len": max_output_len,
         "label": "mixed-64x128"},
        # Large prompt configs to hit max_num_batched_tokens per step
        # (8 × 1024 = 8192 tokens packed into one step)
        {"num_prompts": 16, "input_len": 1024, "output_len": max_output_len,
         "label": "mixed-16x1024-long"},
        {"num_prompts": 8, "input_len": 2048, "output_len": max_output_len,
         "label": "mixed-8x2048"},
        {"num_prompts": 30, "input_len": 1024, "output_len": max_output_len,
         "label": "mixed-30x1024"},
    ]
    for cfg in mixed_configs:
        if cfg["num_prompts"] <= max_num_seqs:
            configs.append(cfg)

    return configs


def main():
    parser = argparse.ArgumentParser(
        description="Data-independent shape-sweep profiler"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpu-model", default="unknown-gpu")
    parser.add_argument("--output", "-o", default="sweep_profile.json")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--max-output-len", type=int, default=None,
        help="Max output tokens to sweep. Defaults to max_model_len/2. "
             "Sweep decode configs generate up to this many tokens "
             "to cover the full context depth range.")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--trace-output", default="/tmp/shape_sweep_trace.jsonl")
    args = parser.parse_args()

    print("=== Shape-Sweep Profiler ===")
    print(f"Model: {args.model}")
    print(f"GPU: {args.gpu_model}")
    print(f"TP: {args.tp}")
    print(f"Max model len: {args.max_model_len}")
    print(f"Max num seqs: {args.max_num_seqs}")
    print(f"Max batched tokens: {args.max_num_batched_tokens}")
    print()

    # Generate sweep configs
    max_output_len = args.max_output_len or args.max_model_len // 2
    configs = generate_sweep_configs(
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_output_len=max_output_len,
        max_model_len=args.max_model_len,
    )
    print(f"Generated {len(configs)} sweep configurations:")
    for cfg in configs:
        print(f"  {cfg['label']}: {cfg['num_prompts']} × "
              f"{cfg['input_len']}in / {cfg['output_len']}out")
    print()

    # Run sweeps
    trace_path = run_sweep_workload(
        model=args.model,
        sweep_configs=configs,
        max_model_len=args.max_model_len,
        tp=args.tp,
        dtype=args.dtype,
        quantization=args.quantization,
        trace_output=args.trace_output,
        max_num_seqs=args.max_num_seqs,
    )

    # Handle TP deduplication
    from vllm_emulator.profiler.trace_profiler import (
        load_trace,
        trace_to_profile_pack,
    )

    records = load_trace(trace_path)
    print(f"\nLoaded {len(records)} raw trace records")

    if args.tp > 1:
        # Deduplicate by step, taking max latency
        by_step = {}
        for r in records:
            step = r["step"]
            if step not in by_step or r["latency_us"] > by_step[step]["latency_us"]:
                by_step[step] = r
        records = sorted(by_step.values(), key=lambda x: x["step"])
        print(f"Deduped to {len(records)} records (TP={args.tp}, max per step)")

    # Print stats
    if records:
        latencies = [r["latency_us"] for r in records]
        tokens = [r["total_tokens"] for r in records]
        prefill_steps = sum(1 for r in records if r["num_prefill_tokens"] > 0 and r["num_decode_seqs"] == 0)
        decode_steps = sum(1 for r in records if r["num_prefill_tokens"] == 0 and r["num_decode_seqs"] > 0)
        mixed_steps = sum(1 for r in records if r["num_prefill_tokens"] > 0 and r["num_decode_seqs"] > 0)
        print(f"  Latency: median={statistics.median(latencies):.0f}us, "
              f"p99={sorted(latencies)[int(len(latencies)*0.99)]:.0f}us")
        print(f"  Tokens/step: min={min(tokens)}, median={statistics.median(tokens):.0f}, max={max(tokens)}")
        print(f"  Steps: {prefill_steps} prefill, {decode_steps} decode, {mixed_steps} mixed")

    # Convert to profile pack
    pack = trace_to_profile_pack(
        records,
        gpu_model=args.gpu_model,
        model_name=args.model,
        bucket_size=8,  # Finer buckets for sweep data
    )
    pack["profile_method"] = "shape_sweep"
    pack["sweep_configs"] = len(configs)
    pack["tensor_parallel"] = args.tp
    pack["max_num_seqs"] = args.max_num_seqs
    pack["max_num_batched_tokens"] = args.max_num_batched_tokens

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(pack, indent=2) + "\n")

    print(f"\nProfile pack written to {out_path}")
    print(f"  forward_pass: {len(pack['forward_pass'])} buckets")
    print(f"  prefill: {len(pack['prefill'])} entries")
    print(f"  decode: {len(pack['decode'])} entries")


if __name__ == "__main__":
    main()
