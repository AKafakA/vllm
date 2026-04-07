"""Two-pass IPC overhead profiling: measures the timer pipelining advantage.

Pass 1: N-sweep on REAL GPU → real_TTFT(N) for N=1..max_n
Pass 2: N-sweep on EMULATOR (same profile, no IPC overhead) → emu_TTFT(N)
Result: overhead(N) = max(0, real_TTFT(N) - emu_TTFT(N))

This captures the EXACT timer pipelining advantage at each concurrency
level. Rate-independent, profile once, works at any dynamic workload.

Usage:
    # Pass 1: real GPU server running on PORT
    python3 profile_ipc_overhead.py --pass1 --port 8100 --model MODEL --output real_ttft.json

    # Pass 2: emulator server running on PORT (with initial profile, no IPC overhead)
    python3 profile_ipc_overhead.py --pass2 --port 8100 --model MODEL --output emu_ttft.json

    # Compute delta:
    python3 profile_ipc_overhead.py --compute --real real_ttft.json --emu emu_ttft.json --output ipc_overhead.json
"""
import argparse
import json
import requests
import threading
import time


def send_background_request(base_url, model):
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


def measure_ttft(base_url, model, prompt_words=40):
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


def run_nsweep(base_url, model, max_n, num_measurements=15):
    """Run N-sweep: for each N, start N-1 background, measure 1 TTFT."""
    results = []

    for n in range(1, max_n + 1):
        # Start N-1 background requests staggered over time
        bg_threads = []
        for i in range(n - 1):
            t = threading.Thread(target=send_background_request,
                                 args=(base_url, model))
            t.start()
            bg_threads.append(t)
            time.sleep(0.3)

        # Wait for background to reach steady-state decode
        if n > 1:
            time.sleep(3.0)

        # Measure TTFT spread over time
        ttft_samples = []
        for _ in range(num_measurements):
            ttft_us = measure_ttft(base_url, model)
            if ttft_us > 0:
                ttft_samples.append(ttft_us)
            time.sleep(0.3)

        # Wait for background to finish
        for t in bg_threads:
            t.join(timeout=120)

        if ttft_samples:
            ttft_samples.sort()
            median_ttft_us = ttft_samples[len(ttft_samples) // 2]
            results.append({
                "num_reqs": n,
                "median_ttft_us": round(median_ttft_us, 0),
                "num_samples": len(ttft_samples),
            })
            print(f"  N={n:3d}: ttft={median_ttft_us/1000:.1f}ms  "
                  f"(n={len(ttft_samples)}, "
                  f"p25={ttft_samples[len(ttft_samples)//4]/1000:.1f}ms, "
                  f"p75={ttft_samples[3*len(ttft_samples)//4]/1000:.1f}ms)")
        else:
            print(f"  N={n:3d}: FAILED")

    return results


def compute_overhead(real_path, emu_path):
    """Compute overhead(N) = max(0, real_TTFT(N) - emu_TTFT(N))."""
    real_data = json.load(open(real_path))
    emu_data = json.load(open(emu_path))

    # Build lookup by num_reqs
    emu_by_n = {e["num_reqs"]: e["median_ttft_us"] for e in emu_data}

    results = []
    for r in real_data:
        n = r["num_reqs"]
        real_ttft = r["median_ttft_us"]
        emu_ttft = emu_by_n.get(n, real_ttft)  # Default: no overhead
        overhead = max(0, real_ttft - emu_ttft)
        results.append({
            "num_reqs": n,
            "overhead_us": round(overhead, 0),
            "real_ttft_us": round(real_ttft, 0),
            "emu_ttft_us": round(emu_ttft, 0),
        })
        print(f"  N={n:3d}: real={real_ttft/1000:.1f}ms  "
              f"emu={emu_ttft/1000:.1f}ms  "
              f"overhead={overhead/1000:.1f}ms")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pass1", action="store_true", help="Run N-sweep on real GPU")
    parser.add_argument("--pass2", action="store_true", help="Run N-sweep on emulator")
    parser.add_argument("--compute", action="store_true", help="Compute delta from pass1+pass2")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--real", help="Pass1 results (for --compute)")
    parser.add_argument("--emu", help="Pass2 results (for --compute)")
    parser.add_argument("--max-n", type=int, default=30, help="Max concurrent requests")
    args = parser.parse_args()

    if args.compute:
        print("=== Computing IPC overhead (real - emu) ===")
        results = compute_overhead(args.real, args.emu)
        json.dump(results, open(args.output, "w"), indent=2)
        print(f"\nSaved {len(results)} entries to {args.output}")
        return

    base_url = f"http://localhost:{args.port}"
    label = "REAL GPU" if args.pass1 else "EMULATOR"
    print(f"=== N-sweep on {label} (N=1..{args.max_n}) ===")

    results = run_nsweep(base_url, args.model, args.max_n)
    json.dump(results, open(args.output, "w"), indent=2)
    print(f"\nSaved {len(results)} entries to {args.output}")


if __name__ == "__main__":
    main()
