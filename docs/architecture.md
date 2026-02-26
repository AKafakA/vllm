# vLLM v1 Architecture Analysis

**Version:** 1.0
**Date:** 2026-02-26
**Status:** IN PROGRESS

---

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     API Layer (A)                          │
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │  OpenAI API     │  │  FastAPI       │                  │
│  │  (/v1/chat/)   │  │  Server        │                  │
│  └────────┬────────┘  └────────┬────────┘                  │
│           │                     │                           │
│           └──────────┬──────────┘                           │
│                      ▼                                      │
│            ┌────────────────┐                              │
│            │   Engine Core   │  v1/engine/core.py          │
│            │  (Scheduler)    │                              │
│            └────────┬────────┘                              │
└─────────────────────┼───────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
┌───────────┐  ┌───────────┐  ┌───────────┐
│  Worker 0 │  │  Worker 1 │  │  Worker N │
│ GPU/Prefill│  │ GPU/Decode│  │ GPU/...  │
└─────┬─────┘  └─────┬─────┘  └─────┬─────┘
      │               │               │
      ▼               ▼               ▼
┌─────────────────────────────────────────────────────────────┐
│                   Device Layer (B/C/D)                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ GPU Compute  │  │ KV Offload   │  │  Network    │    │
│  │ (model exec) │  │ (CPU↔GPU)    │  │ (TP/PP/PD) │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Key Components

### 2.1 Engine Core (`vllm/v1/engine/core.py`)

**Responsibility:** Request lifecycle management, scheduling decisions

**Key Classes:**
- `EngineCore` - Main engine loop
- `EngineCoreProc` - Process-based engine
- `LLMEngine` - High-level API

**Category:** A (Run Real)

**Flow:**
1. Receive request via API
2. Add to request queue
3. Scheduler decides which requests to run
4. Dispatch to workers
5. Collect outputs and stream back

---

### 2.2 Worker (`vllm/v1/worker/gpu_worker.py`)

**Responsibility:** GPU model execution

**Key Methods:**
| Method | Purpose | Category |
|--------|---------|----------|
| `initialize_cache()` | KV cache allocation | C |
| `init_device()` | GPU initialization | A |
| `load_model()` | Load model weights | A |
| `execute_model()` | Run model forward | **B (GPU Compute)** |
| `profile()` | Profiling | A |

**execute_model() Flow:**
```python
def execute_model(self, scheduler_output):
    # 1. Get batch from scheduler
    # 2. Prepare input (tokenization, padding)
    # 3. Model forward pass  ← TARGET FOR EMULATION
    # 4. Sampling
    # 5. Return outputs
```

**Category:** B (GPU Compute - execute_model is the hook point)

---

### 2.3 Model Runner (`vllm/v1/worker/gpu_model_runner.py`)

**Responsibility:** Batch management, input preprocessing, model execution

**Key Classes:**
- `GPUModelRunner` - Main model runner
- `GPUInputBatch` - Input batch representation

**Key Methods:**
| Method | Purpose | Category |
|--------|---------|----------|
| `execute_model()` | Run model on batch | B |
| `_execute_model()` | Core forward logic | B |
| `_run_attention()` | Attention computation | B |
| `_decode_tokens()` | Decode next token | B |

**Category:** B (Model execution - all forward passes)

---

### 2.4 KV Offload (`vllm/v1/kv_offload/`)

**Responsibility:** KV cache management between GPU and CPU

**Key Files:**
| File | Purpose | Category |
|------|---------|----------|
| `backend.py` | Offload backend | C |
| `factory.py` | Backend factory | C |
| `cpu.py` | CPU offload | C |
| `lru_manager.py` | LRU eviction | C |

**Key Classes:**
- `KVCacheManager` - Main KV cache manager
- `KVOffloadBackend` - Base offload interface

**Category:** C (CPU↔GPU Interaction)

**Hook Points:**
- `KVCacheManager.allocate()` - KV allocation
- `KVCacheManager.evict()` - KV eviction
- `KVCacheManager.transfer()` - CPU↔GPU transfer

---

### 2.5 Distributed Communication (`vllm/distributed/`)

**Responsibility:** Tensor parallelism, pipeline parallelism, KV transfer

#### 2.5.1 Device Communicators (`distributed/device_communicators/`)

| File | Purpose | Category |
|------|---------|----------|
| `cuda_communicator.py` | NCCL communication | D |
| `custom_all_reduce.py` | Custom all-reduce | D |
| `pynccl.py` | PyNCCL wrapper | D |

**Category:** D (Network - all_reduce, send_recv)

#### 2.5.2 KV Transfer (`distributed/kv_transfer/`)

| File | Purpose | Category |
|------|---------|----------|
| `kv_connector/` | KV connector interfaces | D |
| `kv_transfer_state.py` | Transfer state | D |

**Category:** D (PD KV transfer)

---

### 2.6 Platform (`vllm/platforms/`)

**Responsibility:** Device abstraction, configuration

**Key Files:**
| File | Purpose |
|------|---------|
| `interface.py` | Platform base class |
| `cuda.py` | CUDA platform |
| `cpu.py` | CPU platform |

**Platform Plugin System:**
- Entry point: `vllm.platform_plugins`
- Example: `vllm-ascend` plugin

**Category:** A (Platform config - runs real)

---

## 3. Hook Points Summary

| Hook Point | File | Method | Category | Priority |
|------------|------|--------|----------|----------|
| GPU Compute | `gpu_worker.py` | `execute_model()` | B | P0 |
| Model Forward | `gpu_model_runner.py` | `execute_model()` | B | P0 |
| Attention | `gpu_model_runner.py` | `_run_attention()` | B | P1 |
| KV Offload | `kv_offload/backend.py` | `transfer()` | C | P1 |
| KV Evict | `kv_offload/lru_manager.py` | `evict()` | C | P1 |
| All-Reduce | `device_communicators/` | `all_reduce()` | D | P1 |
| Send/Recv | `device_communicators/` | `send/recv()` | D | P1 |
| KV Transfer | `kv_transfer/` | `transfer_kv()` | D | P1 |

---

## 4. Emulator Integration Strategy

### Option A: Platform Plugin (Recommended)
- Replace entire Platform class
- Pros: Clean interface, vLLM supported
- Cons: May not capture all hook points

### Option B: Worker Replacement
- Replace `gpu_worker.py` with emulator
- Pros: Direct control over execution
- Cons: More integration work

### Option C: Model Runner Patch
- Patch `gpu_model_runner.py`
- Pros: Granular control
- Cons: Fragile, may break with updates

### Option D: Oracle Hook (Hybrid)
- Keep real execution path
- Add oracle hooks for timing
- Pros: Easy to implement, real behavior preserved
- Cons: Performance overhead

**Recommended:** Start with Option A (Platform Plugin) + Option D (Oracle Hook)

---

## 5. Next Steps

1. [ ] Verify Platform plugin interface in v0.15.1
2. [ ] Design cost oracle interface
3. [ ] Define profile pack schema
4. [ ] Implement P0.1 (Cost Boundary) - DONE
5. [ ] Implement P0.2 (API Contract)
6. [ ] Implement P1.1 (GPU Cost Oracle)

---

## 6. Reference

- vLLM v0.15.1 Source: `/projects/vllm-emulator/vllm/`
- Platform Plugin: `vllm/platforms/interface.py`
- Worker: `vllm/v1/worker/gpu_worker.py`
- Model Runner: `vllm/v1/worker/gpu_model_runner.py`
