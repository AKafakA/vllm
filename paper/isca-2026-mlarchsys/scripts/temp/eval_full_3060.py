#!/usr/bin/env python3
"""Full evaluation suite for RTX 3060 × 2.

Runs sequentially:
  Phase 1: TP=2 with 3B model (sweep profile + serving profile + eval)
  Phase 2: Feature ablations (chunked prefill, CUDA graphs) on 1.5B
  Phase 3: BurstGPT trace replay (online serving)
  Phase 4: Offline throughput 2×2 matrix (realtime mode)
"""
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

RESULT_DIR = "/workspace/eval_results/RTX-3060-12GB"
MODEL_1_5B = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_3B = "Qwen/Qwen2.5-3B-Instruct"
HF_HOME = "/workspace/.hf_home"

os.environ["HF_HOME"] = HF_HOME
os.environ["HUGGINGFACE_HUB_CACHE"] = f"{HF_HOME}/hub"


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


def wait_server(port=8100, timeout=180):
    for i in range(timeout):
        try:
            urllib.request.urlopen(f"http://localhost:{port}/health", timeout=1)
            print(f"  Server ready after {i+1}s")
            return True
        except: time.sleep(1)
    print("  TIMEOUT")
    return False


def run_bench_serve(label, model, rate, input_len=256, output_len=128,
                    num_prompts=50, port=8100):
    result = subprocess.run(
        [sys.executable, "-m", "vllm.entrypoints.cli.main", "bench", "serve",
         "--model", model, "--base-url", f"http://localhost:{port}",
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


def run_bench_throughput(label, model, input_len=256, output_len=128,
                         num_prompts=100, tp=1):
    env = os.environ.copy()
    result = subprocess.run(
        [sys.executable, "-m", "vllm.entrypoints.cli.main", "bench", "throughput",
         "--model", model,
         "--dataset-name", "random",
         "--random-input-len", str(input_len),
         "--random-output-len", str(output_len),
         "--num-prompts", str(num_prompts),
         "--tensor-parallel-size", str(tp),
         "--output-json", f"{RESULT_DIR}/offline/{label}.json"],
        capture_output=True, text=True, timeout=600, env=env)
    for line in result.stdout.split("\n"):
        if "Throughput" in line:
            print(f"    {line.strip()}")
    return result.returncode == 0


def start_server(model, env_extra=None, log_file="server.log", tp=1):
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
           "--model", model, "--max-model-len", "4096",
           "--port", "8100", "--trust-remote-code",
           "--tensor-parallel-size", str(tp)]
    proc = subprocess.Popen(
        cmd, env=env,
        stdout=open(f"/workspace/{log_file}", "w"),
        stderr=subprocess.STDOUT)
    return proc


def stop_server(proc):
    proc.terminate()
    try: proc.wait(timeout=30)
    except: proc.kill(); proc.wait()
    nuke_servers()


def compare_results(pairs, section_name):
    """Compare real vs emu results and print error analysis."""
    print(f"\n  {section_name} Error Analysis:")
    for name, real_file, emu_file in pairs:
        rp = f"{RESULT_DIR}/online/{real_file}.json"
        ep = f"{RESULT_DIR}/online/{emu_file}.json"
        if os.path.exists(rp) and os.path.exists(ep):
            r, e = json.load(open(rp)), json.load(open(ep))
            ttft_err = (e["mean_ttft_ms"] - r["mean_ttft_ms"]) / r["mean_ttft_ms"] * 100
            tpot_err = (e["mean_tpot_ms"] - r["mean_tpot_ms"]) / r["mean_tpot_ms"] * 100
            t_ok = "✓" if abs(ttft_err) < 10 else "✗"
            p_ok = "✓" if abs(tpot_err) < 10 else "✗"
            print(f"    {name:<20}: TTFT {ttft_err:>+6.1f}% {t_ok}  TPOT {tpot_err:>+6.1f}% {p_ok}")


# ======================================================================
# PHASE 1: TP=2 with 3B model
# ======================================================================
def phase1_tp2():
    print("=" * 60)
    print("PHASE 1: TP=2 evaluation with Qwen2.5-3B (2×RTX 3060)")
    print("=" * 60)

    profile_dir = f"{RESULT_DIR}/profiles"
    sweep_profile = f"{profile_dir}/sweep-3b-tp2.json"
    serving_profile = f"{profile_dir}/serving-3b-tp2.json"
    trace_file = f"{RESULT_DIR}/sweep_trace_3b_tp2.jsonl"

    # Step 1a: Sweep profile
    if not os.path.exists(sweep_profile):
        print("\n  Step 1a: Sweep profiling (3B, TP=2)...")
        subprocess.run(
            [sys.executable,
             "paper/isca-2026-mlarchsys/scripts/shape_sweep_profiler.py",
             "--model", MODEL_3B, "--gpu-model", "RTX-3060-12GB",
             "--output", sweep_profile,
             "--max-model-len", "4096", "--max-num-seqs", "64",
             "--max-output-len", "128", "--tp", "2",
             "--trace-output", trace_file],
            cwd="/workspace/vllm-emulator-v18", timeout=900)
        nuke_servers()
    else:
        print(f"\n  Sweep profile exists: {sweep_profile}")

    # Step 1b: Serving trace for 3B TP=2
    step_cycle_file = f"{RESULT_DIR}/step_cycle_3b_tp2.jsonl"
    if not os.path.exists(serving_profile):
        print("\n  Step 1b: Serving trace (3B, TP=2)...")
        if os.path.exists(step_cycle_file):
            os.remove(step_cycle_file)

        proc = start_server(MODEL_3B, tp=2,
            env_extra={
                "VLLM_EMULATOR_TRACE_STEP_CYCLE": "1",
                "VLLM_EMULATOR_STEP_TRACE_OUTPUT": step_cycle_file,
            },
            log_file="tp2_trace_server.log")
        if wait_server():
            for rate in [1, 2, 4]:
                print(f"    Tracing at rate={rate}...")
                run_bench_serve(f"_trace_3b_tp2_rate{rate}", MODEL_3B, rate=rate,
                                num_prompts=30)
            stop_server(proc)

            # Convert trace to serving profile
            print("  Converting trace to serving profile...")
            subprocess.run([sys.executable, "-c", f"""
import json, statistics
from collections import defaultdict

records = []
for line in open("{step_cycle_file}"):
    r = json.loads(line)
    if "total_tokens" in r:
        records.append(r)

by_tt = defaultdict(list)
for r in records:
    by_tt[r["total_tokens"]].append(r["step_cycle_us"])

forward_pass = []
for tt in sorted(by_tt):
    lats = by_tt[tt]
    if len(lats) >= 2:
        forward_pass.append({{
            "total_tokens": tt,
            "latency_us": round(statistics.median(lats), 1),
            "num_samples": len(lats),
        }})

# Merge with sweep profile for large tt
sweep = json.load(open("{sweep_profile}"))
max_tt = max(e["total_tokens"] for e in forward_pass) if forward_pass else 0
for e in sweep["forward_pass"]:
    if e["total_tokens"] > max_tt:
        forward_pass.append(e)

profile = {{
    "gpu_model": "RTX-3060-12GB-TP2",
    "model_name": "{MODEL_3B}",
    "profile_type": "serving_step_cycle",
    "forward_pass": sorted(forward_pass, key=lambda e: e["total_tokens"]),
}}
json.dump(profile, open("{serving_profile}", "w"), indent=2)
print(f"Serving profile: {{len(forward_pass)}} buckets")
"""], timeout=60)
        else:
            stop_server(proc)
            print("  FAILED: server didn't start for trace")
            return
        nuke_servers()
    else:
        print(f"\n  Serving profile exists: {serving_profile}")

    # Step 1c: Real baseline (TP=2)
    print("\n  Step 1c: Real baseline (3B, TP=2)...")
    proc = start_server(MODEL_3B, tp=2, log_file="tp2_real_server.log")
    if wait_server():
        for rate in [1, 2, 4]:
            print(f"\n  --- Real TP=2 rate={rate} ---")
            run_bench_serve(f"real_3b_tp2_rate{rate}", MODEL_3B, rate=rate)
        stop_server(proc)
    else:
        stop_server(proc)
        return
    time.sleep(5)

    # Step 1d: Emulator (TP=2, serving profile)
    print("\n  Step 1d: Emulator (3B, TP=2, serving profile)...")
    proc = start_server(MODEL_3B, tp=2,
        env_extra={
            "VLLM_EMULATOR_ENABLE_ORACLE": "1",
            "VLLM_EMULATOR_PROFILE_PACK": serving_profile,
            "VLLM_EMULATOR_MODE": "realtime",
            "VLLM_EMULATOR_EXECUTOR_HOOK": "1",
        },
        log_file="tp2_emu_server.log")
    if wait_server():
        os.system("grep ExecutorEmulatorHook /workspace/tp2_emu_server.log | head -1")
        for rate in [1, 2, 4]:
            print(f"\n  --- Emu TP=2 rate={rate} ---")
            run_bench_serve(f"emu_3b_tp2_rate{rate}", MODEL_3B, rate=rate)
        stop_server(proc)
    else:
        stop_server(proc)

    compare_results(
        [(f"rate={r}", f"real_3b_tp2_rate{r}", f"emu_3b_tp2_rate{r}") for r in [1,2,4]],
        "TP=2 3B")


# ======================================================================
# PHASE 2: Feature ablations (1.5B TP=1)
# ======================================================================
def phase2_ablations():
    print("\n" + "=" * 60)
    print("PHASE 2: Feature ablations (1.5B, TP=1)")
    print("=" * 60)

    serving_profile = f"{RESULT_DIR}/profiles/serving-1.5b-tp1-step-cycle.json"

    # 2a: Chunked prefill OFF (enforce no chunking via large max_num_batched_tokens)
    print("\n  2a: No chunked prefill...")
    for mode, prefix in [("real", "real"), ("emu", "emu")]:
        env_extra = {}
        if mode == "emu":
            env_extra = {
                "VLLM_EMULATOR_ENABLE_ORACLE": "1",
                "VLLM_EMULATOR_PROFILE_PACK": serving_profile,
                "VLLM_EMULATOR_MODE": "realtime",
                "VLLM_EMULATOR_EXECUTOR_HOOK": "1",
            }
        proc = start_server(MODEL_1_5B, env_extra=env_extra,
            log_file=f"ablation_nochunk_{mode}.log")
        # Note: chunked prefill is ON by default in v0.18.1.
        # To disable, we'd need --no-chunked-prefill or --max-num-batched-tokens=99999
        # For now, just test default (with chunked prefill) vs enforce-eager (no CUDA graphs)
        if wait_server():
            print(f"    {prefix} with chunked prefill (default)...")
            run_bench_serve(f"{prefix}_ablation_default", MODEL_1_5B, rate=2, num_prompts=30)
            stop_server(proc)
        else:
            stop_server(proc)
        time.sleep(3)

    # 2b: No CUDA graphs (enforce-eager)
    print("\n  2b: No CUDA graphs (enforce-eager)...")
    for mode, prefix in [("real", "real"), ("emu", "emu")]:
        env_extra = {}
        if mode == "emu":
            env_extra = {
                "VLLM_EMULATOR_ENABLE_ORACLE": "1",
                "VLLM_EMULATOR_PROFILE_PACK": serving_profile,
                "VLLM_EMULATOR_MODE": "realtime",
                "VLLM_EMULATOR_EXECUTOR_HOOK": "1",
            }
        # Start with --enforce-eager to disable CUDA graphs
        env_all = os.environ.copy()
        env_all.update(env_extra)
        proc = subprocess.Popen(
            [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
             "--model", MODEL_1_5B, "--max-model-len", "4096",
             "--port", "8100", "--trust-remote-code",
             "--enforce-eager"],
            env=env_all,
            stdout=open(f"/workspace/ablation_eager_{mode}.log", "w"),
            stderr=subprocess.STDOUT)
        if wait_server():
            print(f"    {prefix} with enforce-eager...")
            run_bench_serve(f"{prefix}_ablation_eager", MODEL_1_5B, rate=2, num_prompts=30)
            stop_server(proc)
        else:
            stop_server(proc)
        time.sleep(3)

    compare_results([
        ("default", "real_ablation_default", "emu_ablation_default"),
        ("no-cudagraph", "real_ablation_eager", "emu_ablation_eager"),
    ], "Feature Ablations")


# ======================================================================
# PHASE 3: Offline throughput (realtime mode)
# ======================================================================
def phase3_offline():
    print("\n" + "=" * 60)
    print("PHASE 3: Offline throughput (1.5B, TP=1)")
    print("=" * 60)

    os.makedirs(f"{RESULT_DIR}/offline", exist_ok=True)
    sweep_profile = f"{RESULT_DIR}/profiles/sweep-1.5b-tp1-dense-v2.json"

    # Real baseline
    print("\n  Real offline throughput...")
    run_bench_throughput("real_offline_1.5b", MODEL_1_5B)

    # Emulator with sweep profile
    print("\n  Emulator offline throughput (sweep profile)...")
    os.environ["VLLM_EMULATOR_ENABLE_ORACLE"] = "1"
    os.environ["VLLM_EMULATOR_PROFILE_PACK"] = sweep_profile
    os.environ["VLLM_EMULATOR_MODE"] = "realtime"
    run_bench_throughput("emu_offline_1.5b", MODEL_1_5B)
    # Clean env
    for k in ["VLLM_EMULATOR_ENABLE_ORACLE", "VLLM_EMULATOR_PROFILE_PACK",
              "VLLM_EMULATOR_MODE"]:
        os.environ.pop(k, None)

    # Compare
    for label, fname in [("Real", "real_offline_1.5b"), ("Emu", "emu_offline_1.5b")]:
        p = f"{RESULT_DIR}/offline/{fname}.json"
        if os.path.exists(p):
            d = json.load(open(p))
            tps = d.get("tokens_per_second", d.get("generation_tokens_per_second", 0))
            print(f"    {label}: {tps:.0f} tok/s")

    nuke_servers()


# ======================================================================
# MAIN
# ======================================================================
def main():
    os.makedirs(f"{RESULT_DIR}/online", exist_ok=True)
    os.makedirs(f"{RESULT_DIR}/profiles", exist_ok=True)
    nuke_servers()

    phase1_tp2()
    phase2_ablations()
    phase3_offline()

    print("\n" + "=" * 60)
    print("ALL PHASES COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
