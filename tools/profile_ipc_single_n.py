"""Measure IPC overhead for ONE (N, burst-k list) cell on an already-up server.

Designed to be invoked by an outer shell orchestrator that manages the
server lifecycle (cleanup → start → warmup → THIS SCRIPT → cleanup).
Expects the server to be up, warmed up, and quiet (running=0) at entry.

Appends one cell per (N, k) to the output file's JSON array. Safe to call
repeatedly across server restarts; the outer orchestrator concatenates cells.

Shares measurement logic with profile_ipc_overhead_v3.py — condition-based
waits via /metrics, no magic sleeps.

Usage:
    python3 tools/profile_ipc_single_n.py \\
        <port> <model> <profile_path> <output_path> <N> \\
        [--burst-k "1 2 4 8"] [--samples-per-cell 5] \\
        [--poll-ms 50] [--stable-polls 5]
"""
import argparse
import json
import os
import requests
import sys
import threading
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("port", type=int)
    p.add_argument("model")
    p.add_argument("profile_path",
                   help="Profile pack (only used for prefill_step_us baseline)")
    p.add_argument("output_path",
                   help="JSON array file; cells are appended")
    p.add_argument("N", type=int, help="Concurrency (N-1 bg + 1 measurement slot)")
    p.add_argument("--burst-k", default="1",
                   help="Space-sep burst sizes, e.g. '1 2 4 8'")
    p.add_argument("--samples-per-cell", type=int, default=5)
    p.add_argument("--poll-ms", type=int, default=50)
    p.add_argument("--stable-polls", type=int, default=5)
    return p.parse_args()


def get_metric_gauges(base_url: str) -> dict:
    try:
        r = requests.get(f"{base_url}/metrics", timeout=5)
        out = {}
        for line in r.text.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "{" in line:
                name = line.split("{", 1)[0]
            else:
                parts = line.split()
                name = parts[0] if parts else ""
            val_part = line.rsplit(" ", 1)[-1]
            try:
                out.setdefault(name, float(val_part))
            except ValueError:
                continue
        return out
    except Exception:
        return {}


def get_running_count(base_url: str) -> float:
    m = get_metric_gauges(base_url)
    return m.get("vllm:num_requests_running", -1)


def wait_for_running_at_least(base_url, target, poll_s, timeout_s=120):
    start = time.time()
    while time.time() - start < timeout_s:
        if get_running_count(base_url) >= target:
            return True
        time.sleep(poll_s)
    return False


def wait_for_stable(base_url, expected, stable_polls, poll_s, timeout_s=120):
    start = time.time()
    count = 0
    while time.time() - start < timeout_s:
        c = get_running_count(base_url)
        if c == expected:
            count += 1
            if count >= stable_polls:
                return True
        else:
            count = 0
        time.sleep(poll_s)
    return False


def send_background_request(base_url, model):
    try:
        requests.post(
            f"{base_url}/v1/completions",
            json={"model": model, "prompt": "background " * 40,
                  "max_tokens": 128, "temperature": 0},
            timeout=300,
        )
    except Exception:
        pass


def measure_ttft_single(base_url, model, slot_results, slot_idx):
    prompt = f"measurement{slot_idx} " * 40
    t0 = time.perf_counter()
    try:
        r = requests.post(
            f"{base_url}/v1/completions",
            json={"model": model, "prompt": prompt,
                  "max_tokens": 5, "temperature": 0, "stream": True},
            stream=True, timeout=120,
        )
        for line in r.iter_lines():
            if line and b"text" in line:
                t1 = time.perf_counter()
                slot_results[slot_idx] = (t1 - t0) * 1e6
                return
    except Exception:
        pass
    slot_results[slot_idx] = -1


def measure_burst(base_url, model, k):
    slot_results = [0.0] * k
    threads = [threading.Thread(target=measure_ttft_single,
                                 args=(base_url, model, slot_results, i))
               for i in range(k)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    return [s for s in slot_results if s > 0]


def load_prefill_step_us(profile_path):
    profile = json.load(open(profile_path))
    for section in ("prefill_2d_distribution", "prefill_forward_pass", "forward_pass"):
        for e in profile.get(section, []):
            tt = e.get("tt") or e.get("total_tokens")
            if tt is None:
                continue
            if 250 <= tt <= 270:
                samples = e.get("samples", [])
                if samples:
                    s = sorted(samples)
                    return float(s[len(s) // 2])
                if "latency_us" in e:
                    return float(e["latency_us"])
    return 0.0


def append_cells(output_path, new_cells):
    p = Path(output_path)
    if p.exists():
        existing = json.load(open(p))
        if isinstance(existing, dict) and "cells" in existing:
            existing["cells"].extend(new_cells)
        elif isinstance(existing, list):
            existing.extend(new_cells)
        else:
            existing = new_cells
    else:
        existing = new_cells
    with open(p, "w") as f:
        json.dump(existing, f, indent=2)


def main():
    args = parse_args()
    base_url = f"http://localhost:{args.port}"
    poll_s = args.poll_ms / 1000.0
    N = args.N
    burst_ks = [int(x) for x in args.burst_k.split()]

    prefill_step_us = load_prefill_step_us(args.profile_path)
    print(f"[N={N}] prefill_step_us={prefill_step_us/1000:.1f}ms  burst_ks={burst_ks}")

    if get_running_count(base_url) < 0:
        print("ERROR: /metrics not exposing vllm:num_requests_running", file=sys.stderr)
        sys.exit(1)

    if not wait_for_stable(base_url, 0, args.stable_polls, poll_s, timeout_s=60):
        cur = get_running_count(base_url)
        print(f"  WARN: server not quiet at entry (running={cur})")

    bg_threads = []
    for i in range(N - 1):
        t = threading.Thread(target=send_background_request,
                             args=(base_url, args.model))
        t.start()
        bg_threads.append(t)
        if not wait_for_running_at_least(base_url, i + 1, poll_s, timeout_s=30):
            print(f"  WARN: only {get_running_count(base_url)} running "
                  f"after spawning {i+1}")

    if N > 1:
        if not wait_for_stable(base_url, N - 1, args.stable_polls, poll_s, timeout_s=60):
            cur = get_running_count(base_url)
            print(f"  WARN: not stable at {N-1} (running={cur})")

    new_cells = []
    for k in burst_ks:
        cell_samples = []
        for _ in range(args.samples_per_cell):
            burst = measure_burst(base_url, args.model, k)
            if burst:
                cell_samples.append({
                    "ttft_us_by_slot": [round(x, 0) for x in burst],
                    "mean_ttft_us": round(sum(burst) / len(burst), 0),
                    "median_ttft_us": round(sorted(burst)[len(burst) // 2], 0),
                })
            wait_for_stable(base_url, N - 1, args.stable_polls, poll_s, timeout_s=60)

        all_ttfts = [t for s in cell_samples for t in s["ttft_us_by_slot"]]
        if all_ttfts:
            all_sorted = sorted(all_ttfts)
            median_all = all_sorted[len(all_sorted) // 2]
            mean_all = sum(all_ttfts) / len(all_ttfts)
            new_cells.append({
                "num_reqs": N,
                "burst_k": k,
                "samples": cell_samples,
                "overhead_median_us": round(max(0.0, median_all - prefill_step_us), 0),
                "overhead_mean_us":   round(max(0.0, mean_all - prefill_step_us), 0),
                "median_ttft_us": round(median_all, 0),
                "mean_ttft_us": round(mean_all, 0),
                "num_ttft_datapoints": len(all_ttfts),
                "prefill_step_us": round(prefill_step_us, 0),
            })
            print(f"  N={N:4d} k={k:2d}: n={len(all_ttfts):3d} "
                  f"median={median_all/1000:5.1f}ms "
                  f"oh_med={new_cells[-1]['overhead_median_us']/1000:5.1f}ms")

    append_cells(args.output_path, new_cells)
    print(f"Appended {len(new_cells)} cells to {args.output_path}")


if __name__ == "__main__":
    main()
