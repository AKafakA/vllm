"""Send warmup requests to a running vLLM server.

Exercises all common batch shapes to pre-compile CUDA graphs.
"""
import argparse
import subprocess
import sys
import requests
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--heavy", action="store_true",
                        help="Run heavy warmup (bench-serve at target rate)")
    parser.add_argument("--rate", type=int, default=1,
                        help="Target rate for heavy warmup")
    args = parser.parse_args()

    base = f"http://localhost:{args.port}"

    # Phase 1: Simple requests to warm basic CUDA graphs
    print("  Warmup Phase 1: basic requests...")
    for i in range(10):
        try:
            requests.post(
                f"{base}/v1/completions",
                json={"model": args.model, "prompt": "warmup " * 40,
                      "max_tokens": 5, "temperature": 0},
                timeout=30,
            )
        except Exception as e:
            print(f"    {e}")
        time.sleep(0.3)

    if args.heavy:
        # Phase 2: Run bench-serve at target rate to pre-compile ALL
        # CUDA graph shapes that will appear during the actual benchmark
        print(f"  Warmup Phase 2: bench-serve at rate={args.rate} (30 prompts)...")
        subprocess.run(
            [
                sys.executable, "-m", "vllm.entrypoints.cli.main", "bench", "serve",
                "--model", args.model, "--base-url", base,
                "--dataset-name", "random", "--random-input-len", "256",
                "--random-output-len", "128",
                "--num-prompts", "30", "--request-rate", str(args.rate),
            ],
            capture_output=True,
            timeout=300,
        )
        time.sleep(2)

    print("  Warmup done.")


if __name__ == "__main__":
    main()
