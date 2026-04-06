"""Profile IPC scheduling overhead as a function of concurrent requests.

For each N=1..max_n:
  1. Start N-1 background requests (long output, keep engine busy)
  2. Send 1 measurement request via streaming
  3. Measure TTFT of the measurement request
  4. IPC_overhead(N) = measured_TTFT - profiled_prefill_step_time

This measures the real IPC overhead at steady-state concurrency N,
not burst arrival overhead.

Output: JSON array of {num_reqs, mean_ttft_us, overhead_us}.
Usage:
    python3 profile_ipc_overhead.py <port> <model> <profile_path> <output_path> [max_n]
"""
import json
import requests
import sys
import threading
import time

port = int(sys.argv[1])
model = sys.argv[2]
profile_path = sys.argv[3]
output_path = sys.argv[4]
max_n = int(sys.argv[5]) if len(sys.argv) > 5 else 50

base_url = f"http://localhost:{port}"

# Get profiled prefill step time at tt≈256
profile = json.load(open(profile_path))
prefill_step_us = 0
for section in ["prefill_forward_pass", "forward_pass"]:
    for e in profile.get(section, []):
        if 250 <= e["total_tokens"] <= 270:
            prefill_step_us = e["latency_us"]
            break
    if prefill_step_us > 0:
        break

print(f"Profiled prefill step: {prefill_step_us/1000:.1f}ms")
print(f"Sweeping N=1..{max_n}")


def send_background_request():
    """Send a request to keep the engine busy (same workload as benchmark)."""
    try:
        requests.post(
            f"{base_url}/v1/completions",
            json={"model": model, "prompt": "background " * 40,
                  "max_tokens": 128, "temperature": 0},
            timeout=60,
        )
    except Exception:
        pass


def measure_ttft(prompt_words=40):
    """Send a streaming request and measure TTFT in microseconds."""
    prompt = "measurement " * prompt_words
    t0 = time.perf_counter()
    try:
        r = requests.post(
            f"{base_url}/v1/completions",
            json={"model": model, "prompt": prompt,
                  "max_tokens": 5, "temperature": 0, "stream": True},
            stream=True, timeout=30,
        )
        for line in r.iter_lines():
            if line and b"text" in line:
                t1 = time.perf_counter()
                return (t1 - t0) * 1e6
    except Exception:
        pass
    return -1


results = []
NUM_MEASUREMENTS = 15  # More samples for stable median

for n in range(1, max_n + 1):
    # Start N-1 background requests staggered over time
    # to simulate different lifecycle stages (like real serving)
    bg_threads = []
    for i in range(n - 1):
        t = threading.Thread(target=send_background_request)
        t.start()
        bg_threads.append(t)
        time.sleep(0.3)  # Stagger: requests at different decode stages

    # Wait for background requests to reach steady-state decode
    # At 256 input tokens, prefill takes ~20ms. Decode starts immediately.
    # Wait 3s so background requests are deep in decode (not prefill).
    if n > 1:
        time.sleep(3.0)

    # Measure TTFT spread over time to capture different engine states
    ttft_samples = []
    for _ in range(NUM_MEASUREMENTS):
        ttft_us = measure_ttft()
        if ttft_us > 0:
            ttft_samples.append(ttft_us)
        time.sleep(0.3)  # Spread measurements across engine cycle states

    # Wait for background to finish
    for t in bg_threads:
        t.join(timeout=120)

    if ttft_samples:
        ttft_samples.sort()
        median_ttft_us = ttft_samples[len(ttft_samples) // 2]
        overhead_us = max(0, median_ttft_us - prefill_step_us)
        results.append({
            "num_reqs": n,
            "median_ttft_us": round(median_ttft_us, 0),
            "overhead_us": round(overhead_us, 0),
            "num_samples": len(ttft_samples),
        })
        print(f"  N={n:3d}: ttft={median_ttft_us/1000:.1f}ms  "
              f"overhead={overhead_us/1000:.1f}ms  "
              f"(n={len(ttft_samples)}, "
              f"p25={ttft_samples[len(ttft_samples)//4]/1000:.1f}ms, "
              f"p75={ttft_samples[3*len(ttft_samples)//4]/1000:.1f}ms)")
    else:
        print(f"  N={n:3d}: FAILED")

json.dump(results, open(output_path, "w"), indent=2)
print(f"\nSaved {len(results)} entries to {output_path}")
