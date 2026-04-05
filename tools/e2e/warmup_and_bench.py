"""Warmup server then run bench serve at multiple rates.

Usage:
    python3 warmup_and_bench.py --port 8100 --model "Qwen/Qwen2.5-1.5B-Instruct" \
        --result-dir /path/to/results --prefix real --rates 1,2,4 --num-prompts 100
"""
import argparse
import os
import subprocess
import sys
import threading
import time

import requests


def warmup_server(base_url: str, model: str):
    """Warm up CUDA graphs by exercising various batch shapes."""
    print("  Warmup Phase 1: isolated requests (various prompt lengths)...")
    for plen in [32, 64, 128, 256, 512]:
        prompt = "warmup " * min(plen, 200)
        try:
            r = requests.post(
                f"{base_url}/v1/completions",
                json={"model": model, "prompt": prompt, "max_tokens": 5, "temperature": 0},
                timeout=30,
            )
        except Exception as e:
            print(f"    Warning: {e}")
        time.sleep(0.3)

    print("  Warmup Phase 2: concurrent requests...")

    def send_req(n_tokens):
        try:
            requests.post(
                f"{base_url}/v1/completions",
                json={"model": model, "prompt": "warmup " * 40, "max_tokens": n_tokens, "temperature": 0},
                timeout=60,
            )
        except Exception:
            pass

    for _ in range(3):
        threads = []
        for nt in [10, 20, 50]:
            t = threading.Thread(target=send_req, args=(nt,))
            t.start()
            threads.append(t)
            time.sleep(0.2)
        for t in threads:
            t.join()

    print("  Warmup Phase 3: bench-like workload (20 prompts at rate=2)...")
    subprocess.run(
        [
            sys.executable, "-m", "vllm.entrypoints.cli.main", "bench", "serve",
            "--model", model, "--base-url", base_url,
            "--dataset-name", "random", "--random-input-len", "256", "--random-output-len", "128",
            "--num-prompts", "20", "--request-rate", "2",
        ],
        capture_output=True,
        timeout=120,
    )
    time.sleep(2)
    print("  Warmup complete.")


def run_bench(base_url: str, model: str, rate: int, num_prompts: int,
              result_dir: str, result_filename: str):
    """Run bench serve and return the result."""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.cli.main", "bench", "serve",
        "--model", model, "--base-url", base_url,
        "--dataset-name", "random", "--random-input-len", "256", "--random-output-len", "128",
        "--num-prompts", str(num_prompts), "--request-rate", str(rate),
        "--percentile-metrics", "ttft,tpot", "--metric-percentiles", "50,99",
        "--save-result", "--result-dir", result_dir, "--result-filename", result_filename,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    # Extract TTFT/TPOT lines
    for line in result.stdout.split("\n"):
        if "TTFT" in line or "TPOT" in line:
            print(f"    {line.strip()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--prefix", required=True, help="Result file prefix (e.g. 'real' or 'emu')")
    parser.add_argument("--rates", default="1,2,4", help="Comma-separated request rates")
    parser.add_argument("--num-prompts", type=int, default=100)
    args = parser.parse_args()

    base_url = f"http://localhost:{args.port}"
    rates = [int(r) for r in args.rates.split(",")]

    # Wait for server
    print("Waiting for server...")
    for i in range(120):
        try:
            r = requests.get(f"{base_url}/health", timeout=2)
            if r.status_code == 200:
                print(f"  Server ready after {i+1}s")
                break
        except Exception:
            pass
        time.sleep(1)
    else:
        print("ERROR: Server not ready after 120s")
        sys.exit(1)

    warmup_server(base_url, args.model)

    os.makedirs(args.result_dir, exist_ok=True)
    for rate in rates:
        fn = f"{args.prefix}_rate{rate}.json"
        print(f"  Benchmarking rate={rate} ({args.num_prompts} prompts)...")
        run_bench(base_url, args.model, rate, args.num_prompts, args.result_dir, fn)

    print("Done.")


if __name__ == "__main__":
    main()
