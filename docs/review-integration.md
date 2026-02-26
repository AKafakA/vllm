# Component Review: Integration Points (Engine, API, Benchmark)

**Date:** 2026-02-26
**Purpose:** Understand how to integrate emulator and how to test

---

## 1. How Platform Selection Works

### 1.1 Platform Plugin System

vLLM uses a plugin system to select platform:

```python
# vllm/platforms/__init__.py
builtin_platform_plugins = {
    "tpu": tpu_platform_plugin,
    "cuda": cuda_platform_plugin,
    "rocm": rocm_platform_plugin,
    "xpu": xpu_platform_plugin,
    "cpu": cpu_platform_plugin,
}

# Load custom plugins via entry points
platform_plugins = load_plugins_by_group("vllm.platform_plugins")
```

**Selection Mechanism:**
1. Check `VLLM_PLUGINS` environment variable
2. Load all discovered plugins
3. Activate one platform (cannot have multiple)

### 1.2 How to Add Emulator Platform

**Option 1: Entry Point (Recommended)**
```python
# setup.py
entry_points={
    "vllm.platform_plugins": [
        "emulator = vllm_emulator.platform:register",
    ],
}
```

**Option 2: Direct Import**
```python
# Before importing vllm
import vllm_emulator.platform
```

### 1.3 Activation

```bash
# Via environment variable
VLLM_PLUGINS=emulator vllm serve ...
```

---

## 2. Engine Core

### 2.1 Key Classes

| Class | File | Purpose |
|-------|------|---------|
| `EngineCore` | core.py:78 | Abstract engine interface |
| `EngineCoreProc` | core.py:636 | Process-based engine |
| `EngineCoreActor` | core.py:1585 | Actor-based engine |

### 2.2 Request Flow

```
API Request
    │
    ▼
EngineCore.add_request()
    │
    ▼
Scheduler (schedules requests)
    │
    ▼
Worker.execute_model()  ← Hook point for B (GPU Compute)
    │
    ▼
Response
```

---

## 3. Scheduler

### 3.1 Location
- `vllm/v1/scheduler.py`

### 3.2 Key Responsibility
- Batch formation
- Request prioritization
- Preemption decisions

**Note:** Scheduler runs in Category A (run real) - should not be emulated

---

## 4. Benchmark Tools

### 4.1 Available Benchmarks

| File | Purpose |
|------|---------|
| `benchmarks/throughput.py` | Offline throughput benchmark |
| `benchmarks/latency.py` | Latency benchmark |
| `benchmarks/serve.py` | Online serving benchmark |

### 4.2 How to Run

```bash
# Throughput benchmark
python -m vllm.benchmarks.throughput \
    --model meta-llama/Llama-2-7b-hf \
    --dataset random \
    --num-prompts 1000

# Latency benchmark
python -m vllm.benchmarks.latency \
    --model meta-llama/Llama-2-7b-hf
```

---

## 5. Integration Strategy

### 5.1 Adding Emulator Platform

1. **Create Platform Plugin**
   - File: `vllm_emulator/platform.py`
   - Class: `EmulatorPlatform(Platform)`
   - Register via entry point

2. **Activate**
   ```bash
   VLLM_PLUGINS=emulator vllm serve <model>
   ```

### 5.2 Testing Approach

1. **Start real vLLM**
   ```bash
   vllm serve <model> --gpu-memory-utilization 0.8
   ```

2. **Start emulated vLLM**
   ```bash
   VLLM_PLUGINS=emulator vllm serve <model> \
       --extra-config emulator_profile_pack=/path/to/profile.json
   ```

3. **Run same workload**
   ```bash
   # Use benchmark tools with same dataset
   python benchmark_client.py --url http://localhost:8000 ...
   ```

4. **Compare metrics**
   - TTFT (Time to First Token)
   - TPOT (Time Per Output Token)
   - E2E Latency
   - Throughput

---

## 6. Task: Integration Points Review

### Review Items

| Item | Status | Notes |
|------|--------|-------|
| Platform selection | ✅ DONE | Via VLLM_PLUGINS env |
| Engine core | ✅ DONE | Request flow understood |
| Scheduler | ✅ DONE | Runs real (Category A) |
| CLI entrypoint | ✅ DONE | Uses EngineArgs |
| Benchmark tools | ✅ DONE | throughput.py, latency.py |

---

## 7. Open Questions

1. **How to pass profile pack to emulator?**
   - Via `--extra-config`?
   - Via environment variable?
   - Via config file?

2. **How to enable emulator mode conditionally?**
   - Always on when platform=emulator?
   - Or allow hybrid mode?

3. **How to collect metrics for comparison?**
   - Use existing vLLM metrics?
   - Add custom metrics?

4. **How to verify correctness?**
   - A/B test with same workload?
   - Acceptable error threshold?

---

## 8. Next Steps

1. [ ] Define integration test plan
2. [ ] Create test script for A/B comparison
3. [ ] Implement P0.2 (API Contract) - need to define how to pass profile pack
4. [ ] Implement P1.1 (GPU Cost Oracle)
5. [ ] Run integration tests
