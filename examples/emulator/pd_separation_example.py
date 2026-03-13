#!/usr/bin/env python3
"""Example script demonstrating PD-separated scheduling in vLLM emulator.

This script shows how to use the emulator with disaggregated prefill/decode
(PD) separation, including CLI flags for enabling PD mode.

Usage:
    python pd_separation_example.py --profile profiles/a100.json --enable-pd-separation
    python pd_separation_example.py --profile profiles/a100.json --enable-pd-separation --policy prefill_first
    python pd_separation_example.py --profile profiles/a100.json  # joint scheduling (default)
"""

import argparse
import json
import sys
from pathlib import Path

from vllm_emulator.oracle import (
    ProfileGpuCostOracle,
    create_oracle_from_profile_pack,
    create_pd_separated_oracle,
)
from vllm_emulator.scheduler import (
    EmulatorScheduler,
    PDSchedulingPolicy,
    Request,
    create_scheduler,
)


def create_sample_profile() -> dict:
    """Create a sample profile pack for demonstration."""
    return {
        "version": "1.0",
        "gpu_model": "A100",
        "prefill": [
            {"seq_len": 128, "batch_size": 1, "latency_us": 15000},
            {"seq_len": 256, "batch_size": 1, "latency_us": 28000},
            {"seq_len": 512, "batch_size": 1, "latency_us": 50000},
            {"seq_len": 1024, "batch_size": 1, "latency_us": 95000},
            {"seq_len": 2048, "batch_size": 1, "latency_us": 185000},
        ],
        "decode": [
            {"active_seqs": 1, "latency_us_per_token": 500},
            {"active_seqs": 4, "latency_us_per_token": 1800},
            {"active_seqs": 8, "latency_us_per_token": 3500},
            {"active_seqs": 16, "latency_us_per_token": 6500},
            {"active_seqs": 32, "latency_us_per_token": 12000},
        ],
    }


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Demonstrate PD-separated scheduling in vLLM emulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--profile",
        type=Path,
        help="Path to profile pack JSON file. If not provided, uses built-in sample.",
    )
    
    parser.add_argument(
        "--enable-pd-separation",
        action="store_true",
        help="Enable PD-separated prefill/decode scheduling",
    )
    
    parser.add_argument(
        "--policy",
        type=str,
        choices=["prefill_first", "decode_first", "hybrid", "disaggregated"],
        default="hybrid",
        help="PD scheduling policy (default: hybrid)",
    )
    
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=32,
        help="Maximum batch size (default: 32)",
    )
    
    parser.add_argument(
        "--num-requests",
        type=int,
        default=5,
        help="Number of requests to simulate (default: 5)",
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=20,
        help="Max tokens per request (default: 20)",
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Load or create profile
    if args.profile:
        if not args.profile.exists():
            print(f"Error: Profile file not found: {args.profile}", file=sys.stderr)
            sys.exit(1)
        profile_pack = json.loads(args.profile.read_text())
        print(f"Loaded profile: {args.profile}")
    else:
        profile_pack = create_sample_profile()
        print("Using built-in sample profile")
    
    print(f"GPU model: {profile_pack['gpu_model']}")
    print(f"PD separation: {'enabled' if args.enable_pd_separation else 'disabled (joint)'}")
    if args.enable_pd_separation:
        print(f"Policy: {args.policy}")
    print(f"Max batch size: {args.max_batch_size}")
    print()
    
    # Create oracle
    base_oracle = create_oracle_from_profile_pack(profile_pack)
    
    # Wrap with PD-separated oracle if enabled
    if args.enable_pd_separation:
        oracle = create_pd_separated_oracle(base_oracle)
        print("Created PD-separated cost oracle")
    else:
        oracle = base_oracle
        print("Created base cost oracle (joint mode)")
    
    # Create scheduler
    scheduler = create_scheduler(
        oracle,
        enable_pd_separation=args.enable_pd_separation,
        policy=args.policy,
        max_batch_size=args.max_batch_size,
    )
    print(f"Created scheduler (PD: {scheduler.enable_pd_separation}, policy: {scheduler.policy.value})")
    print()
    
    # Create requests
    requests = []
    for i in range(args.num_requests):
        req = Request(
            request_id=f"req-{i:03d}",
            prompt_tokens=128 + (i * 64),  # Varying prompt lengths
            max_tokens=args.max_tokens,
            priority=i,
        )
        requests.append(req)
        scheduler.add_request(req)
    
    print(f"Added {len(requests)} requests to scheduler")
    print(f"Initial queue stats: {scheduler.get_queue_stats()}")
    print()
    
    # Simulate scheduling
    print("=" * 60)
    print("Scheduling simulation")
    print("=" * 60)
    
    total_prefill_time = 0.0
    total_decode_time = 0.0
    round_num = 0
    
    while True:
        decision = scheduler.schedule()
        
        # Stop if nothing to process
        if not decision.prefill_batch and not decision.decode_batch:
            break
        
        round_num += 1
        print(f"\n--- Round {round_num} ---")
        
        if decision.prefill_batch:
            print(f"Prefill batch: {len(decision.prefill_batch)} requests")
            if args.verbose:
                for req in decision.prefill_batch:
                    print(f"  - {req.request_id}: {req.prompt_tokens} tokens")
            print(f"  Time: {decision.prefill_time_us:.1f} µs")
            total_prefill_time += decision.prefill_time_us
            
            # Mark as prefill done and add to decode queue
            for req in decision.prefill_batch:
                req.is_prefill_done = True
        
        if decision.decode_batch:
            print(f"Decode batch: {len(decision.decode_batch)} requests")
            if args.verbose:
                for req in decision.decode_batch:
                    print(f"  - {req.request_id}: gen {req.generated_tokens}/{req.max_tokens}")
            print(f"  Time: {decision.decode_time_us:.1f} µs")
            total_decode_time += decision.decode_time_us
            
            # Advance tokens
            for req in decision.decode_batch:
                req.generated_tokens += 1
        
        # Check if all requests are done
        all_done = all(
            req.is_prefill_done and req.generated_tokens >= req.max_tokens
            for req in requests
        )
        if all_done:
            break
        
        # Safety limit
        if round_num > 100:
            print("Warning: Reached maximum rounds, stopping")
            break
    
    # Summary
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total rounds: {round_num}")
    print(f"Total prefill time: {total_prefill_time:.1f} µs ({total_prefill_time/1000:.2f} ms)")
    print(f"Total decode time: {total_decode_time:.1f} µs ({total_decode_time/1000:.2f} ms)")
    print(f"Total time: {(total_prefill_time + total_decode_time)/1000:.2f} ms")
    print(f"Final queue stats: {scheduler.get_queue_stats()}")


if __name__ == "__main__":
    main()
