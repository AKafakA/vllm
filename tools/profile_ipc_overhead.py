"""Profile IPC scheduling overhead as a function of concurrent requests.

For each N = 1..max_n:
  1. Start N-1 background requests (streaming, keep engine busy in decode).
  2. Wait 3s so backgrounds are deep in decode (past prefill).
  3. Send 1 "measurement" request via streaming; time HTTP POST → first
     streamed token = TTFT.
  4. IPC_overhead(N) = measured_TTFT_median - profile_prefill_step_us.
  Uses median of 15 samples per N, staggered 0.3s apart.

Output: JSON array [{num_reqs, median_ttft_us, overhead_us, num_samples}]
to be merged into the main profile JSON as `sched_overhead_table`.

Based on commit 4d9983a0c (Apr 6); adapted to current vllm-emulator layout.

Usage:
    python3 tools/profile_ipc_overhead.py \\
        <port> <model> <profile_path> <output_path> [max_n]
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

# Get profiled prefill step time at tt≈256 (input_len=256 prefills).
profile = json.load(open(profile_path))
prefill_step_us = 0.0
for section in ("prefill_2d_distribution", "prefill_forward_pass", "forward_pass"):
    for e in profile.get(section, []):
        tt = e.get("tt") or e.get("total_tokens")
        if tt is None:
            continue
        if 250 <= tt <= 270:
            samples = e.get("samples", [])
            if samples:
                samples_sorted = sorted(samples)
                prefill_step_us = float(samples_sorted[len(samples_sorted) // 2])
            elif "latency_us" in e:
                prefill_step_us = float(e["latency_us"])
            break
    if prefill_step_us > 0:
        break

if prefill_step_us <= 0:
    print(f"WARNING: no prefill sample at tt≈256; using 0 as baseline", file=sys.stderr)

print(f"Profiled prefill step (tt≈256): {prefill_step_us/1000:.1f}ms")
print(f"Sweeping N = 1..{max_n}")


def send_background_request():
    try:
        requests.post(
            f"{base_url}/v1/completions",
            json={"model": model, "prompt": "background " * 40,
                  "max_tokens": 128, "temperature": 0},
            timeout=120,
        )
    except Exception:
        pass


def measure_ttft():
    prompt = "measurement " * 40
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
NUM_MEASUREMENTS = 15

# Standard sweep points — covers low and high concurrency regimes.
# Dense at low N (TTFT most sensitive), sparse at high N.
sweep = [n for n in [1, 2, 3, 5, 8, 12, 20, 30, 50, 100, 150, 200, 256]
         if n <= max_n]

for n in sweep:
    bg_threads = []
    for _ in range(n - 1):
        t = threading.Thread(target=send_background_request)
        t.start()
        bg_threads.append(t)
        time.sleep(0.3)

    if n > 1:
        time.sleep(3.0)

    ttft_samples = []
    for _ in range(NUM_MEASUREMENTS):
        v = measure_ttft()
        if v > 0:
            ttft_samples.append(v)
        time.sleep(0.3)

    for t in bg_threads:
        t.join(timeout=120)

    if ttft_samples:
        ttft_samples.sort()
        median_ttft_us = ttft_samples[len(ttft_samples) // 2]
        mean_ttft_us = sum(ttft_samples) / len(ttft_samples)
        # `overhead_us` is the authoritative field used by oracle/hook.
        # Aggregation selectable via VLLM_IPC_OVERHEAD_AGG env var at table
        # build time; default = median (current behaviour). Raw samples
        # retained so downstream can rebuild with a different aggregation
        # without re-running the sweep.
        overhead_us = max(0.0, median_ttft_us - prefill_step_us)
        overhead_mean_us = max(0.0, mean_ttft_us - prefill_step_us)
        results.append({
            "num_reqs": n,
            "median_ttft_us": round(median_ttft_us, 0),
            "mean_ttft_us": round(mean_ttft_us, 0),
            "overhead_us": round(overhead_us, 0),
            "overhead_mean_us": round(overhead_mean_us, 0),
            "num_samples": len(ttft_samples),
            "raw_ttft_samples_us": [round(s, 0) for s in ttft_samples],
            "prefill_step_us": round(prefill_step_us, 0),
        })
        print(f"  N={n:3d}: ttft median={median_ttft_us/1000:.1f}ms "
              f"mean={mean_ttft_us/1000:.1f}ms  "
              f"overhead median={overhead_us/1000:.1f}ms "
              f"mean={overhead_mean_us/1000:.1f}ms  "
              f"(n={len(ttft_samples)}, "
              f"p25={ttft_samples[len(ttft_samples)//4]/1000:.1f}ms, "
              f"p75={ttft_samples[3*len(ttft_samples)//4]/1000:.1f}ms)")
    else:
        print(f"  N={n:3d}: FAILED")

json.dump(results, open(output_path, "w"), indent=2)
print(f"\nSaved {len(results)} entries to {output_path}")
