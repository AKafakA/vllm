#!/usr/bin/env python3
"""Back-to-back real vs emulator evaluation using serving profile.

Tests:
1. Multiple QPS rates (1, 2, 4) with fixed input/output lengths
2. Varied input lengths (128, 256, 512, 1024) at rate=2
3. Varied output lengths (64, 128, 256) at rate=2

All tests use 50 prompts for statistical stability.
"""
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

RESULT_DIR = "/workspace/eval_results/RTX-3060-12GB"
SWEEP_PROFILE = f"{RESULT_DIR}/profiles/sweep-1.5b-tp1-dense-v2.json"
SERVING_PROFILE = f"{RESULT_DIR}/profiles/serving-1.5b-tp1-step-cycle.json"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
NUM_PROMPTS = 50

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
    print("  TIMEOUT")
    return False

def run_bench(label, rate, input_len=256, output_len=128, num_prompts=NUM_PROMPTS):
    result = subprocess.run(
        [sys.executable, "-m", "vllm.entrypoints.cli.main", "bench", "serve",
         "--model", MODEL,
         "--base-url", "http://localhost:8100",
         "--dataset-name", "random",
         "--random-input-len", str(input_len),
         "--random-output-len", str(output_len),
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

def run_test_suite(prefix, rate_tests=True, varied_tests=True):
    """Run the full test suite. prefix is 'real' or 'emu'."""
    if rate_tests:
        for rate in [1, 2, 4]:
            print(f"\n  --- {prefix} rate={rate} (256in/128out) ---")
            run_bench(f"{prefix}_sp_rate{rate}", rate=rate)

    if varied_tests:
        # Varied input lengths at rate=2
        for il in [128, 512, 1024]:
            print(f"\n  --- {prefix} input={il} (rate=2, 128out) ---")
            run_bench(f"{prefix}_sp_in{il}", rate=2, input_len=il)

        # Varied output lengths at rate=2
        for ol in [64, 256]:
            print(f"\n  --- {prefix} output={ol} (rate=2, 256in) ---")
            run_bench(f"{prefix}_sp_out{ol}", rate=2, output_len=ol)

def main():
    os.makedirs(f"{RESULT_DIR}/online", exist_ok=True)
    nuke_servers()

    # Phase 1: Real baseline
    print("=" * 60)
    print("PHASE 1: Real baseline server")
    print("=" * 60)
    proc = start_server(log_file="sp_eval_real_server.log")
    if not wait_server():
        stop_server(proc); return
    run_test_suite("real")
    stop_server(proc)
    time.sleep(5)

    # Phase 2: Emulator with serving profile
    print("\n" + "=" * 60)
    print("PHASE 2: Emulator (serving profile, executor hook)")
    print("=" * 60)
    proc = start_server(
        env_extra={
            "VLLM_EMULATOR_ENABLE_ORACLE": "1",
            "VLLM_EMULATOR_PROFILE_PACK": SERVING_PROFILE,
            "VLLM_EMULATOR_MODE": "realtime",
            "VLLM_EMULATOR_EXECUTOR_HOOK": "1",
        },
        log_file="sp_eval_emu_server.log")
    if not wait_server():
        stop_server(proc); return

    os.system("grep 'ExecutorEmulatorHook' /workspace/sp_eval_emu_server.log | head -1")
    run_test_suite("emu")
    stop_server(proc)

    # Phase 3: Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    # Rate sweep results
    print("\n--- Rate Sweep (256in/128out) ---")
    print(f"{'Config':<25} {'TTFT':>8} {'P99TTFT':>9} {'TPOT':>8} {'P99TPOT':>9}")
    for rate in [1, 2, 4]:
        for pfx, label in [("Real", f"real_sp_rate{rate}"), ("Emu", f"emu_sp_rate{rate}")]:
            p = f"{RESULT_DIR}/online/{label}.json"
            if os.path.exists(p):
                d = json.load(open(p))
                print(f"{pfx+' rate='+str(rate):<25} "
                      f"{d.get('mean_ttft_ms',0):>8.1f} {d.get('p99_ttft_ms',0):>9.1f} "
                      f"{d.get('mean_tpot_ms',0):>8.1f} {d.get('p99_tpot_ms',0):>9.1f}")

    # Varied input results
    print("\n--- Varied Input Length (rate=2, 128out) ---")
    for il in [128, 256, 512, 1024]:
        if il == 256:
            rl, el = f"real_sp_rate2", f"emu_sp_rate2"
        else:
            rl, el = f"real_sp_in{il}", f"emu_sp_in{il}"
        for pfx, label in [("Real", rl), ("Emu", el)]:
            p = f"{RESULT_DIR}/online/{label}.json"
            if os.path.exists(p):
                d = json.load(open(p))
                print(f"{pfx+' in='+str(il):<25} "
                      f"{d.get('mean_ttft_ms',0):>8.1f} {d.get('p99_ttft_ms',0):>9.1f} "
                      f"{d.get('mean_tpot_ms',0):>8.1f} {d.get('p99_tpot_ms',0):>9.1f}")

    # Varied output results
    print("\n--- Varied Output Length (rate=2, 256in) ---")
    for ol in [64, 128, 256]:
        if ol == 128:
            rl, el = f"real_sp_rate2", f"emu_sp_rate2"
        else:
            rl, el = f"real_sp_out{ol}", f"emu_sp_out{ol}"
        for pfx, label in [("Real", rl), ("Emu", el)]:
            p = f"{RESULT_DIR}/online/{label}.json"
            if os.path.exists(p):
                d = json.load(open(p))
                print(f"{pfx+' out='+str(ol):<25} "
                      f"{d.get('mean_ttft_ms',0):>8.1f} {d.get('p99_ttft_ms',0):>9.1f} "
                      f"{d.get('mean_tpot_ms',0):>8.1f} {d.get('p99_tpot_ms',0):>9.1f}")

    # Error analysis
    print("\n--- ERROR ANALYSIS ---")
    tests = (
        [(f"rate={r}", f"real_sp_rate{r}", f"emu_sp_rate{r}") for r in [1,2,4]] +
        [(f"in={il}", f"real_sp_in{il}", f"emu_sp_in{il}") for il in [128,512,1024]] +
        [(f"out={ol}", f"real_sp_out{ol}", f"emu_sp_out{ol}") for ol in [64,256]]
    )
    for name, rk, ek in tests:
        rp = f"{RESULT_DIR}/online/{rk}.json"
        ep = f"{RESULT_DIR}/online/{ek}.json"
        if os.path.exists(rp) and os.path.exists(ep):
            r, e = json.load(open(rp)), json.load(open(ep))
            ttft_err = (e["mean_ttft_ms"] - r["mean_ttft_ms"]) / r["mean_ttft_ms"] * 100 if r["mean_ttft_ms"] > 0 else 0
            tpot_err = (e["mean_tpot_ms"] - r["mean_tpot_ms"]) / r["mean_tpot_ms"] * 100 if r["mean_tpot_ms"] > 0 else 0
            t_ok = "✓" if abs(ttft_err) < 10 else "✗"
            p_ok = "✓" if abs(tpot_err) < 10 else "✗"
            print(f"  {name:<12}: TTFT {ttft_err:>+6.1f}% {t_ok}  TPOT {tpot_err:>+6.1f}% {p_ok}")

    print("\nDONE")

if __name__ == "__main__":
    main()
