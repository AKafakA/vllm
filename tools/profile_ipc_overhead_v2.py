"""Profile IPC scheduling overhead: 2D (concurrency N, burst k) sweep.

Extends profile_ipc_overhead.py with:
  1. N sweep widened to cover saturation regime up to N=1024.
  2. Burst dimension: at each N, inject k simultaneous new requests
     (not just 1) and record all their TTFTs.
  3. Raw samples always retained so downstream can re-aggregate.

For each (N, k) cell:
  - Spawn N-1 background (decode-phase) requests.
  - Inject k measurement requests near-simultaneously (within ~10ms).
  - Collect k TTFTs per sample.
  - Repeat 5 samples per (N, k) cell.

Usage:
    python3 tools/profile_ipc_overhead_v2.py \\
        <port> <model> <profile_path> <output_path> \\
        [--burst-k "1 2 4 8"] [--max-n 1024]
"""
import argparse
import json
import requests
import sys
import threading
import time


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("port", type=int)
    p.add_argument("model")
    p.add_argument("profile_path")
    p.add_argument("output_path")
    p.add_argument("--burst-k", default="1 2 4 8",
                   help="Space-sep burst sizes to test, e.g. '1 2 4 8'")
    p.add_argument("--max-n", type=int, default=1024)
    p.add_argument("--samples-per-cell", type=int, default=5,
                   help="Independent samples per (N, k) cell")
    return p.parse_args()


args = parse_args()
base_url = f"http://localhost:{args.port}"

# Get profiled prefill step time at tt≈256.
profile = json.load(open(args.profile_path))
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

print(f"Profiled prefill step (tt≈256): {prefill_step_us/1000:.1f}ms")

BURST_KS = [int(x) for x in args.burst_k.split()]
# N sweep: dense at low N, coverage up to max_n (saturation regime).
N_SWEEP_FULL = [1, 2, 3, 5, 8, 12, 20, 30, 50, 100, 150, 200, 256,
                384, 512, 768, 1024]
N_SWEEP = [n for n in N_SWEEP_FULL if n <= args.max_n]
print(f"N sweep: {N_SWEEP}")
print(f"Burst k: {BURST_KS}")
print(f"Samples per cell: {args.samples_per_cell}")
print(f"Total cells: {len(N_SWEEP) * len(BURST_KS)}")


def send_background_request():
    try:
        requests.post(
            f"{base_url}/v1/completions",
            json={"model": args.model, "prompt": "background " * 40,
                  "max_tokens": 128, "temperature": 0},
            timeout=180,
        )
    except Exception:
        pass


def measure_ttft_once(slot_results, slot_idx):
    """Measure TTFT for one request in a burst; store into slot_results[slot_idx]."""
    prompt = f"measurement{slot_idx} " * 40
    t0 = time.perf_counter()
    try:
        r = requests.post(
            f"{base_url}/v1/completions",
            json={"model": args.model, "prompt": prompt,
                  "max_tokens": 5, "temperature": 0, "stream": True},
            stream=True, timeout=60,
        )
        for line in r.iter_lines():
            if line and b"text" in line:
                t1 = time.perf_counter()
                slot_results[slot_idx] = (t1 - t0) * 1e6
                return
    except Exception:
        pass
    slot_results[slot_idx] = -1


def measure_burst(k: int) -> list[float]:
    """Inject k simultaneous measurement requests, return their TTFTs (us)."""
    slot_results = [0.0] * k
    threads = []
    # Launch all k simultaneously; they'll race into the scheduler.
    for i in range(k):
        t = threading.Thread(target=measure_ttft_once, args=(slot_results, i))
        threads.append(t)
    # Start within a tight window.
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=90)
    return [s for s in slot_results if s > 0]


results = []

for n in N_SWEEP:
    # Spawn n-1 backgrounds once per N (reused across all k and all samples).
    bg_threads = []
    for _ in range(n - 1):
        t = threading.Thread(target=send_background_request)
        t.start()
        bg_threads.append(t)
        time.sleep(0.3)
    if n > 1:
        time.sleep(3.0)

    for k in BURST_KS:
        cell_samples = []
        for s_idx in range(args.samples_per_cell):
            burst = measure_burst(k)
            if burst:
                cell_samples.append({
                    "ttft_us_by_slot": [round(x, 0) for x in burst],
                    "mean_ttft_us": round(sum(burst) / len(burst), 0),
                    "median_ttft_us": round(sorted(burst)[len(burst) // 2], 0),
                })
            time.sleep(0.5)

        # Aggregate across samples: each sample contributes k TTFTs; combine all.
        all_ttfts = []
        for s in cell_samples:
            all_ttfts.extend(s["ttft_us_by_slot"])
        if all_ttfts:
            all_sorted = sorted(all_ttfts)
            median_all = all_sorted[len(all_sorted) // 2]
            mean_all = sum(all_ttfts) / len(all_ttfts)
            overhead_median = max(0.0, median_all - prefill_step_us)
            overhead_mean = max(0.0, mean_all - prefill_step_us)
            results.append({
                "num_reqs": n,
                "burst_k": k,
                "samples": cell_samples,
                "overhead_median_us": round(overhead_median, 0),
                "overhead_mean_us": round(overhead_mean, 0),
                "median_ttft_us": round(median_all, 0),
                "mean_ttft_us": round(mean_all, 0),
                "num_ttft_datapoints": len(all_ttfts),
                "prefill_step_us": round(prefill_step_us, 0),
            })
            print(f"  N={n:4d} k={k:2d}: n={len(all_ttfts):3d} "
                  f"median={median_all/1000:5.1f}ms "
                  f"mean={mean_all/1000:5.1f}ms "
                  f"oh_med={overhead_median/1000:5.1f}ms "
                  f"oh_mean={overhead_mean/1000:5.1f}ms")
        else:
            print(f"  N={n:4d} k={k:2d}: FAILED")

    # Join backgrounds.
    for t in bg_threads:
        t.join(timeout=180)

json.dump(results, open(args.output_path, "w"), indent=2)
print(f"\nSaved {len(results)} entries to {args.output_path}")
