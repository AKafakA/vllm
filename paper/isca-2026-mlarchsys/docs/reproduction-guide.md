# Reproduction Guide: vLLM-Emulator Evaluation

This guide covers end-to-end reproduction of both offline inference and online serving emulation results from the paper. All steps assume a machine with at least one NVIDIA GPU and a working vLLM installation.

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Installation](#2-installation)
3. [Profiling Pipeline](#3-profiling-pipeline)
4. [Offline Inference Emulation](#4-offline-inference-emulation)
5. [Online Serving Emulation](#5-online-serving-emulation)
6. [Environment Variables Reference](#6-environment-variables-reference)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. Prerequisites

**Hardware:**
- NVIDIA GPU (tested on RTX 3060 12GB, A100 80GB, A30 24GB)
- Minimum 12GB VRAM for 1.5B models, 24GB for 7B/8B models

**Software:**
- Python 3.12
- CUDA 12.x or 13.x
- vLLM 0.18.1 (this repository)

---

## 2. Installation

```bash
# Create isolated environment
uv venv --python 3.12
source .venv/bin/activate

# Install vLLM with emulator extensions
# Option A: precompiled (fast, ~2 min)
VLLM_USE_PRECOMPILED=1 uv pip install -e .

# Option B: from source (slow, ~30 min, needed for C++ changes)
MAX_JOBS=4 uv pip install -e .

# Verify
python -c "import vllm; print(vllm.__version__)"
python -c "from vllm_emulator.hooks.gpu_hook import GpuWorkerHook; print('OK')"
```

---

## 3. Profiling Pipeline

The emulator requires a **profile pack** — a JSON file mapping batch shapes to latencies. Two profiling methods are available:

### 3.1 Sweep Profile (GPU forward pass only — for offline inference)

The sweep profiler runs controlled workloads to measure GPU forward pass latency at each batch size. This is **data-independent** — profiled once per GPU/model, tested on any workload.

```bash
python paper/isca-2026-mlarchsys/scripts/shape_sweep_profiler.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --gpu-model RTX-3060-12GB \
    --output profiles/sweep-1.5b-tp1.json \
    --max-model-len 4096 \
    --max-num-seqs 64 \
    --max-output-len 128 \
    --tp 1 \
    --trace-output profiles/sweep_trace_1.5b.jsonl
```

**Parameters:**
- `--model`: HuggingFace model name (downloads on first use)
- `--gpu-model`: label for the profile (human-readable)
- `--max-output-len`: should cover the longest expected output in your workload
- `--tp`: tensor parallelism degree (must match serving config)
- `--trace-output`: raw trace JSONL for debugging/reprocessing

**Output:** `profiles/sweep-1.5b-tp1.json` — profile pack with `forward_pass` section mapping `total_tokens → latency_us`.

**Duration:** ~5-10 minutes per model depending on GPU speed.

### 3.2 Serving Profile (step-cycle time — for online serving)

The serving profile captures the **full step cycle time** including GPU execution, scheduling, output processing, and GPU/CPU overlap. This is critical for online serving accuracy because the overhead varies with batch size.

**Step 1: Start the real server with step-cycle tracing**

```bash
VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
VLLM_EMULATOR_STEP_TRACE_OUTPUT=profiles/step_cycle_1.5b.jsonl \
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --max-model-len 4096 \
    --port 8100 \
    --trust-remote-code
```

**Step 2: Run a representative workload at multiple QPS rates**

```bash
# Run at several rates to cover different batch size regimes
for rate in 1 2 4 8; do
    python -m vllm.entrypoints.cli.main bench serve \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --base-url http://localhost:8100 \
        --dataset-name random \
        --random-input-len 256 --random-output-len 128 \
        --num-prompts 30 --request-rate $rate
done
```

**Step 3: Convert the step-cycle trace to a serving profile**

```python
import json, statistics
from collections import defaultdict

# Load step-cycle trace
records = []
for line in open("profiles/step_cycle_1.5b.jsonl"):
    r = json.loads(line)
    if "total_tokens" in r:
        records.append(r)

# Build profile: median step_cycle_us per total_tokens
by_tt = defaultdict(list)
for r in records:
    by_tt[r["total_tokens"]].append(r["step_cycle_us"])

forward_pass = []
for tt in sorted(by_tt):
    lats = by_tt[tt]
    if len(lats) >= 2:
        forward_pass.append({
            "total_tokens": tt,
            "latency_us": round(statistics.median(lats), 1),
            "num_samples": len(lats),
        })

# Merge: for large tt not seen during serving, fall back to sweep profile
sweep = json.load(open("profiles/sweep-1.5b-tp1.json"))
max_serving_tt = max(e["total_tokens"] for e in forward_pass)
for e in sweep["forward_pass"]:
    if e["total_tokens"] > max_serving_tt:
        forward_pass.append(e)

profile = {
    "gpu_model": "RTX-3060-12GB",
    "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
    "profile_type": "serving_step_cycle",
    "forward_pass": sorted(forward_pass, key=lambda e: e["total_tokens"]),
}
json.dump(profile, open("profiles/serving-1.5b-tp1.json", "w"), indent=2)
```

**Duration:** ~3-5 minutes (server startup + benchmark runs).

---

## 4. Offline Inference Emulation

Offline inference emulation uses the **sweep profile** with the **worker-level hook** to replace GPU forward passes with calibrated `time.sleep()`.

### 4.1 Run Real Baseline

```bash
python -m vllm.entrypoints.cli.main bench throughput \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --dataset-name random \
    --random-input-len 256 --random-output-len 128 \
    --num-prompts 100 \
    --output-json results/real_offline.json
```

### 4.2 Run Emulator

```bash
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK=profiles/sweep-1.5b-tp1.json \
VLLM_EMULATOR_MODE=realtime \
python -m vllm.entrypoints.cli.main bench throughput \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --dataset-name random \
    --random-input-len 256 --random-output-len 128 \
    --num-prompts 100 \
    --output-json results/emu_offline.json
```

### 4.3 Compare

```bash
python -c "
import json
real = json.load(open('results/real_offline.json'))
emu = json.load(open('results/emu_offline.json'))
real_tps = real['tokens_per_second']
emu_tps = emu['tokens_per_second']
err = (emu_tps - real_tps) / real_tps * 100
print(f'Real: {real_tps:.0f} tok/s')
print(f'Emu:  {emu_tps:.0f} tok/s')
print(f'Error: {err:+.1f}%')
"
```

**Expected accuracy:** <5% error on generation throughput (tok/s).

### Architecture Notes

The worker-level hook intercepts `execute_model()` in the GPU worker:

```
execute_model(scheduler_output):
    if emulator_hook.is_enabled:
        cost = oracle.estimate_step_latency_us(total_tokens)
        fake_output = create_fake_output(scheduler_output)
        time.sleep(cost / 1e6)       # simulate GPU time
        store pending_output
        return None                    # triggers sample_tokens path

sample_tokens(grammar_output):
    if pending_output: return pending_output
    return real_sample_tokens()
```

---

## 5. Online Serving Emulation

Online serving emulation uses the **serving profile** (step-cycle) with the **executor-level hook** for non-blocking operation.

### Why Executor Hook?

The worker-level hook uses `time.sleep()` which blocks the worker thread. At high QPS with large batches (30+ concurrent requests), the sleep duration exceeds the async scheduler's timeout, causing deadlocks. The executor hook uses `threading.Timer`-based Futures that don't block the worker thread.

### Why Serving Profile?

GPU-only profiles miss 4-7ms of per-step serving overhead (scheduling, output processing, detokenization). This overhead varies with batch size due to GPU/CPU pipelining:

| Batch size (tt) | GPU-only | Step cycle | Overhead |
|-----------------|----------|------------|----------|
| 1-7 (low QPS)   | ~13ms    | ~20ms      | +6.5ms   |
| 8-10 (CUDA graph sweet spot) | ~14ms | ~17ms | +2.5ms |
| 17+ (high QPS)  | ~28ms    | ~20ms      | -8ms (overlap) |

A constant overhead is rate-dependent. The serving profile captures this naturally.

### 5.1 Run Real Baseline

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --max-model-len 4096 --port 8100 --trust-remote-code &
SERVER_PID=$!

# Wait for server
until curl -s http://localhost:8100/health > /dev/null 2>&1; do sleep 1; done

# Benchmark at multiple rates
for rate in 1 2 4; do
    python -m vllm.entrypoints.cli.main bench serve \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --base-url http://localhost:8100 \
        --dataset-name random \
        --random-input-len 256 --random-output-len 128 \
        --num-prompts 50 --request-rate $rate \
        --percentile-metrics ttft,tpot,itl,e2el \
        --metric-percentiles 50,95,99 \
        --save-result --result-dir results/online \
        --result-filename real_rate${rate}.json
done

kill $SERVER_PID
wait $SERVER_PID 2>/dev/null
# Clean up GPU: wait for EngineCore to exit
sleep 5
pkill -9 -f EngineCore 2>/dev/null
sleep 3
```

### 5.2 Run Emulator (immediately after real baseline for fair comparison)

```bash
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK=profiles/serving-1.5b-tp1.json \
VLLM_EMULATOR_MODE=realtime \
VLLM_EMULATOR_EXECUTOR_HOOK=1 \
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --max-model-len 4096 --port 8100 --trust-remote-code &
SERVER_PID=$!

until curl -s http://localhost:8100/health > /dev/null 2>&1; do sleep 1; done

for rate in 1 2 4; do
    python -m vllm.entrypoints.cli.main bench serve \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --base-url http://localhost:8100 \
        --dataset-name random \
        --random-input-len 256 --random-output-len 128 \
        --num-prompts 50 --request-rate $rate \
        --percentile-metrics ttft,tpot,itl,e2el \
        --metric-percentiles 50,95,99 \
        --save-result --result-dir results/online \
        --result-filename emu_rate${rate}.json
done

kill $SERVER_PID
wait $SERVER_PID 2>/dev/null
pkill -9 -f EngineCore 2>/dev/null
```

### 5.3 Compare

```python
import json, os

print(f"{'Config':<25} {'TTFT':>8} {'P99TTFT':>9} {'TPOT':>8} {'P99TPOT':>9}")
print("-" * 62)

for rate in [1, 2, 4]:
    for prefix, label in [("Real", f"real_rate{rate}"),
                           ("Emu", f"emu_rate{rate}")]:
        path = f"results/online/{label}.json"
        if os.path.exists(path):
            d = json.load(open(path))
            print(f"{prefix+' rate='+str(rate):<25} "
                  f"{d['mean_ttft_ms']:>8.1f} {d['p99_ttft_ms']:>9.1f} "
                  f"{d['mean_tpot_ms']:>8.1f} {d['p99_tpot_ms']:>9.1f}")

print("\nError analysis:")
for rate in [1, 2, 4]:
    rp = f"results/online/real_rate{rate}.json"
    ep = f"results/online/emu_rate{rate}.json"
    if os.path.exists(rp) and os.path.exists(ep):
        r, e = json.load(open(rp)), json.load(open(ep))
        ttft_err = (e["mean_ttft_ms"] - r["mean_ttft_ms"]) / r["mean_ttft_ms"] * 100
        tpot_err = (e["mean_tpot_ms"] - r["mean_tpot_ms"]) / r["mean_tpot_ms"] * 100
        print(f"  Rate={rate}: TTFT {ttft_err:+.1f}%, TPOT {tpot_err:+.1f}%")
```

**Expected accuracy:** <5% error on Mean TTFT and Mean TPOT across all QPS rates.

### 5.4 Important Notes

1. **Always run real and emulator back-to-back.** Real baselines vary 10-30% between server startups due to GPU state, CUDA graph warmup, and OS scheduling. Comparing against stale baselines gives misleading error percentages.

2. **Use `--random-input-len` and `--random-output-len`**, not `--input-len`. The latter uses `RandomDataset` with `range_ratio` that inflates prompt length.

3. **Kill EngineCore explicitly** after stopping the API server. The EngineCore subprocess survives parent termination and holds GPU memory.

4. **Use `--trust-remote-code`** for models like Qwen that require it.

---

## 6. Environment Variables Reference

### Core Emulator

| Variable | Values | Description |
|----------|--------|-------------|
| `VLLM_EMULATOR_ENABLE_ORACLE` | `1` / `true` | Enable the emulator hook |
| `VLLM_EMULATOR_PROFILE_PACK` | path | Profile pack JSON file |
| `VLLM_EMULATOR_MODE` | `realtime` / `accelerated` | `realtime`: `time.sleep()`, `accelerated`: no blocking (virtual time) |
| `VLLM_EMULATOR_EXECUTOR_HOOK` | `1` / `true` | Use executor-level hook (non-blocking, for online serving) |

### Overhead Calibration (optional, not needed with serving profile)

| Variable | Values | Description |
|----------|--------|-------------|
| `VLLM_EMULATOR_STEP_OVERHEAD_US` | float | Per-step overhead added to all steps (microseconds) |
| `VLLM_EMULATOR_DECODE_OVERHEAD_US` | float | Per-step overhead added only to decode steps |

### Tracing / Profiling

| Variable | Values | Description |
|----------|--------|-------------|
| `VLLM_EMULATOR_TRACE_PROFILE` | `1` / `true` | Enable execute_model() trace profiler |
| `VLLM_EMULATOR_TRACE_OUTPUT` | path | Output JSONL for execute_model() traces |
| `VLLM_EMULATOR_TRACE_STEP_CYCLE` | `1` / `true` | Enable step-cycle tracer (full step time) |
| `VLLM_EMULATOR_STEP_TRACE_OUTPUT` | path | Output JSONL for step-cycle traces |

### Which Hook to Use

| Use case | Hook | Profile | Why |
|----------|------|---------|-----|
| Offline throughput (`bench throughput`) | Worker (default) | Sweep profile | Worker hook is simpler, no async issues |
| Online serving (`bench serve`) | Executor (`VLLM_EMULATOR_EXECUTOR_HOOK=1`) | Serving profile | Non-blocking, avoids worker deadlock at high QPS |

---

## 7. Troubleshooting

### Server hangs at high QPS with worker hook
**Symptom:** Server shows 0 throughput with many running requests.
**Cause:** `time.sleep()` in the worker thread blocks too long for large batches.
**Fix:** Use the executor hook: `VLLM_EMULATOR_EXECUTOR_HOOK=1`

### GPU memory not freed after server stop
**Symptom:** `nvidia-smi` shows high memory use but no processes listed.
**Cause:** The `VLLM::EngineCore` subprocess survives parent termination.
**Fix:** `pkill -9 -f EngineCore; sleep 3`

### Profile missing coverage at small batch sizes
**Symptom:** Interpolation artifacts at tt=2-15 in online serving.
**Cause:** `trace_to_profile_pack()` uses coarse bucketing (bucket_size=8).
**Fix:** Already fixed — per-token granularity for tt≤32. Ensure you use the latest code.

### High TTFT error but good TPOT
**Symptom:** TTFT error >10% while TPOT is <5%.
**Cause:** Comparing against stale baseline from a different server startup.
**Fix:** Run real and emulator back-to-back in the same session.

### "Free memory on device cuda:0 ... less than desired" error
**Symptom:** Server fails to start.
**Cause:** Previous process left zombie GPU memory.
**Fix:** `pkill -9 -f EngineCore; sleep 5` or use `CUDA_VISIBLE_DEVICES=1` for the other GPU.

---

## Quick Reference

### Minimal Offline Evaluation (5 minutes)

```bash
# 1. Profile
python paper/isca-2026-mlarchsys/scripts/shape_sweep_profiler.py \
    --model Qwen/Qwen2.5-1.5B-Instruct --gpu-model MY-GPU \
    --output profiles/sweep.json

# 2. Real baseline
python -m vllm.entrypoints.cli.main bench throughput \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 100 --output-json results/real.json

# 3. Emulator
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK=profiles/sweep.json \
VLLM_EMULATOR_MODE=realtime \
python -m vllm.entrypoints.cli.main bench throughput \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 100 --output-json results/emu.json
```

### Minimal Online Evaluation (10 minutes)

```bash
# 1. Profile (sweep + serving trace)
# ... (see Section 3.1 and 3.2 above)

# 2. Back-to-back evaluation
python paper/isca-2026-mlarchsys/scripts/temp/backtoback_eval.py
```
