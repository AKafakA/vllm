#!/usr/bin/env python3
"""Back-to-back real vs emulator evaluation for fair comparison."""
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

RESULT_DIR = "/workspace/eval_results/RTX-3060-12GB"
PROFILE = f"{RESULT_DIR}/profiles/sweep-1.5b-tp1-dense-v2.json"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
RATES = [1, 2, 4]
NUM_PROMPTS = 50
DECODE_OVERHEAD_US = 5000

def nuke_servers():
    for pat in ["EngineCore", "api_server", "vllm.entrypoints"]:
        os.system(f"pkill -9 -f '{pat}' 2>/dev/null")
    time.sleep(2)
    for dev in ["/dev/nvidia0", "/dev/nvidia1"]:
        result = subprocess.run(["fuser", dev], capture_output=True, text=True)
        if result.stdout.strip():
            for pid in result.stdout.strip().split():
                try: os.kill(int(pid.strip()), signal.SIGKILL)
                except: pass
    time.sleep(5)

def wait_server(port=8100, timeout=120):
    for i in range(timeout):
        try:
            urllib.request.urlopen(f"http://localhost:{port}/health", timeout=1)
            print(f"  Server ready after {i+1}s")
            return True
        except: time.sleep(1)
    print("  TIMEOUT: server didn't start")
    return False

def run_bench(label, rate, num_prompts=NUM_PROMPTS):
    result = subprocess.run(
        [sys.executable, "-m", "vllm.entrypoints.cli.main", "bench", "serve",
         "--model", MODEL,
         "--base-url", "http://localhost:8100",
         "--dataset-name", "random", "--random-input-len", "256",
         "--random-output-len", "128",
         "--num-prompts", str(num_prompts),
         "--request-rate", str(rate),
         "--percentile-metrics", "ttft,tpot,itl,e2el",
         "--metric-percentiles", "50,95,99",
         "--save-result", "--result-dir", f"{RESULT_DIR}/online",
         "--result-filename", f"{label}.json"],
        capture_output=True, text=True, timeout=600)
    for line in result.stdout.split("\n"):
        if any(m in line for m in ["TTFT", "TPOT", "Throughput"]):
            print(f"    {line.strip()}")
    return result.returncode == 0

def start_server(env_extra=None, log_file="server.log"):
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    proc = subprocess.Popen(
        [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
         "--model", MODEL, "--max-model-len", "4096",
         "--port", "8100", "--trust-remote-code"],
        env=env,
        stdout=open(f"/workspace/{log_file}", "w"),
        stderr=subprocess.STDOUT)
    return proc

def stop_server(proc):
    proc.terminate()
    try: proc.wait(timeout=30)
    except: proc.kill(); proc.wait()
    nuke_servers()

def main():
    os.makedirs(f"{RESULT_DIR}/online", exist_ok=True)
    nuke_servers()

    # Phase 1: Real baseline
    print("=" * 60)
    print("PHASE 1: Real baseline server")
    print("=" * 60)
    proc = start_server(log_file="b2b_real_server.log")
    if not wait_server():
        stop_server(proc); return

    for rate in RATES:
        print(f"\n  --- Real rate={rate} ---")
        run_bench(f"b2b_real_rate{rate}", rate=rate)

    stop_server(proc)
    print("\n  Real server stopped, GPU cleanup...")
    time.sleep(5)

    # Phase 2: Emulator (executor hook + decode overhead)
    print("\n" + "=" * 60)
    print(f"PHASE 2: Emulator (executor hook, decode_overhead={DECODE_OVERHEAD_US}us)")
    print("=" * 60)
    proc = start_server(
        env_extra={
            "VLLM_EMULATOR_ENABLE_ORACLE": "1",
            "VLLM_EMULATOR_PROFILE_PACK": PROFILE,
            "VLLM_EMULATOR_MODE": "realtime",
            "VLLM_EMULATOR_EXECUTOR_HOOK": "1",
            "VLLM_EMULATOR_DECODE_OVERHEAD_US": str(DECODE_OVERHEAD_US),
        },
        log_file="b2b_emu_server.log")
    if not wait_server():
        stop_server(proc); return

    # Verify executor hook
    os.system("grep 'ExecutorEmulatorHook' /workspace/b2b_emu_server.log | head -1")

    for rate in RATES:
        print(f"\n  --- Emulator rate={rate} ---")
        run_bench(f"b2b_emu_rate{rate}", rate=rate)

    stop_server(proc)

    # Phase 3: Compare
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)
    header = f"{'Config':<30} {'TTFT':>8} {'P99TTFT':>9} {'TPOT':>8} {'P99TPOT':>9}"
    print(header)
    print("-" * len(header))

    for rate in RATES:
        for prefix, label in [("Real", f"b2b_real_rate{rate}"),
                               ("Emu", f"b2b_emu_rate{rate}")]:
            path = f"{RESULT_DIR}/online/{label}.json"
            if os.path.exists(path):
                d = json.load(open(path))
                print(f"{prefix+' rate='+str(rate):<30} "
                      f"{d.get('mean_ttft_ms',0):>8.1f} {d.get('p99_ttft_ms',0):>9.1f} "
                      f"{d.get('mean_tpot_ms',0):>8.1f} {d.get('p99_tpot_ms',0):>9.1f}")

    # Error calculation
    print("\nERROR ANALYSIS:")
    for rate in RATES:
        rp = f"{RESULT_DIR}/online/b2b_real_rate{rate}.json"
        ep = f"{RESULT_DIR}/online/b2b_emu_rate{rate}.json"
        if os.path.exists(rp) and os.path.exists(ep):
            r, e = json.load(open(rp)), json.load(open(ep))
            print(f"\n  Rate={rate}:")
            for m in ["mean_ttft_ms", "p99_ttft_ms", "mean_tpot_ms", "p99_tpot_ms"]:
                if m in r and m in e and r[m] > 0:
                    err = (e[m] - r[m]) / r[m] * 100
                    ok = "✓" if abs(err) < 10 else "✗"
                    print(f"    {m:<20}: real={r[m]:>8.1f}  emu={e[m]:>8.1f}  err={err:>+6.1f}% {ok}")

    print("\nDONE")

if __name__ == "__main__":
    main()
