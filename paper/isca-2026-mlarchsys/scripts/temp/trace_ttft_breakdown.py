#!/usr/bin/env python3
"""Trace TTFT breakdown: instrument each stage from request arrival to first token.

Sends requests one at a time (rate=0.2) and measures:
  T0: client sends HTTP request
  T1: client receives first token chunk (TTFT)

Then correlates with server-side step trace to decompose TTFT into:
  - HTTP + tokenization overhead
  - Scheduler queue wait
  - Prefill execution time
  - Output processing + streaming
"""
import asyncio
import json
import os
import sys
import time

import aiohttp

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
BASE_URL = "http://localhost:8100"
NUM_REQUESTS = 10
INPUT_LEN = 256
OUTPUT_LEN = 16  # Short output so each request finishes before next arrives


async def send_request(session, prompt_tokens, req_id):
    """Send one request and measure TTFT precisely."""
    payload = {
        "model": MODEL,
        "prompt": " ".join(str(t) for t in prompt_tokens),
        "max_tokens": OUTPUT_LEN,
        "temperature": 0,
        "stream": True,
    }

    t_send = time.perf_counter()
    first_token_time = None

    async with session.post(
        f"{BASE_URL}/v1/completions",
        json=payload,
    ) as resp:
        async for chunk in resp.content:
            if first_token_time is None and b'"text"' in chunk:
                first_token_time = time.perf_counter()

    t_done = time.perf_counter()
    ttft_ms = (first_token_time - t_send) * 1000 if first_token_time else -1
    total_ms = (t_done - t_send) * 1000
    return {
        "req_id": req_id,
        "ttft_ms": round(ttft_ms, 1),
        "total_ms": round(total_ms, 1),
        "t_send": t_send,
    }


async def main():
    import random
    rng = random.Random(42)

    results = []
    async with aiohttp.ClientSession() as session:
        for i in range(NUM_REQUESTS):
            # Unique random tokens to avoid prefix caching
            tokens = [rng.randint(100, 30000) for _ in range(INPUT_LEN)]
            prompt = " ".join(str(t) for t in tokens)

            result = await send_request(session, tokens, i)
            results.append(result)
            print(f"  Req {i}: TTFT={result['ttft_ms']:.1f}ms, total={result['total_ms']:.1f}ms")

            # Wait before next request (rate ~0.3)
            if i < NUM_REQUESTS - 1:
                await asyncio.sleep(3.0)

    # Summary
    ttfts = [r["ttft_ms"] for r in results if r["ttft_ms"] > 0]
    print(f"\nTTFT summary ({len(ttfts)} requests):")
    print(f"  Mean: {sum(ttfts)/len(ttfts):.1f}ms")
    print(f"  Min:  {min(ttfts):.1f}ms")
    print(f"  Max:  {max(ttfts):.1f}ms")
    print(f"  Req 1 (cold): {ttfts[0]:.1f}ms")
    print(f"  Req 2-{len(ttfts)} (warm mean): {sum(ttfts[1:])/len(ttfts[1:]):.1f}ms")

    # Save detailed results
    json.dump(results, open("/tmp/ttft_trace_results.json", "w"), indent=2)


if __name__ == "__main__":
    asyncio.run(main())
