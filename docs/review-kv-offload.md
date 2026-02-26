# Component Review: KV Offload (Category C)

**Date:** 2026-02-26
**Component:** KV Offload
**Category:** C - CPU↔GPU Interaction

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Scheduler Layer                          │
│  ┌─────────────────────────────────────────────────────┐  │
│  │ OffloadingManager (abstract)                        │  │
│  │  ├── lookup()        - Check if blocks are offloaded│  │
│  │  ├── prepare_load()  - Prepare blocks for loading   │  │
│  │  ├── prepare_store() - Prepare blocks for storing   │  │
│  │  └── touch()         - Update LRU                   │  │
│  └─────────────────────────────────────────────────────┘  │
│                            │                               │
│                            ▼                               │
│  ┌─────────────────────────────────────────────────────┐  │
│  │ LRUOffloadingManager                                │  │
│  │  └── Uses Backend (CPU/GPU/...)                     │  │
│  └─────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      Worker Layer                          │
│  ┌─────────────────────────────────────────────────────┐  │
│  │ OffloadingWorker                                    │  │
│  │  ├── register_handler()                              │  │
│  │  ├── transfer_async()  - Initiate async transfer     │  │
│  │  └── get_finished()   - Get completed transfers     │  │
│  └─────────────────────────────────────────────────────┘  │
│                            │                               │
│                            ▼                               │
│  ┌─────────────────────────────────────────────────────┐  │
│  │ SingleDirectionOffloadingHandler                    │  │
│  │  └── Uses torch.cuda.Stream for async copy          │  │
│  │  └── Uses ops.swap_blocks() for actual transfer     │  │
│  └─────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Key Files

| File | Purpose | Lines |
|------|---------|-------|
| `kv_offload/abstract.py` | OffloadingManager interface | 100+ |
| `kv_offload/backend.py` | Backend (CPU/GPU) interface | 80+ |
| `kv_offload/lru_manager.py` | LRU eviction logic | 150+ |
| `kv_offload/worker/worker.py` | Worker-side transfer handler | 150+ |
| `kv_offload/worker/cpu_gpu.py` | CPU↔GPU transfer impl | 250+ |

---

## 3. Hook Points

### 3.1 OffloadingManager Level (Scheduler)

| Method | File | Purpose | Can Emulate? |
|--------|------|---------|--------------|
| `lookup()` | abstract.py | Check offloaded blocks | **Yes** |
| `prepare_load()` | abstract.py | Prepare for load | **Yes** |
| `prepare_store()` | abstract.py | Prepare for store | **Yes** |
| `touch()` | abstract.py | LRU update | No (cheap) |

### 3.2 OffloadingWorker Level (Worker)

| Method | File | Purpose | Can Emulate? |
|--------|------|---------|--------------|
| `transfer_async()` | worker.py | Start transfer | **Yes** |
| `get_finished()` | worker.py | Check completion | **Yes** |

### 3.3 Actual Transfer (cpu_gpu.py)

| Operation | Purpose | Can Emulate? |
|-----------|---------|--------------|
| `ops.swap_blocks()` | Actual CPU↔GPU copy | **Yes** |
| `torch.cuda.Stream` | Async transfer | **Yes** |

---

## 4. Emulation Strategy

### Option 1: Mock OffloadingManager

Replace `OffloadingManager` with a mock that:
- Returns fake hit counts
- Simulates latency based on bytes + bandwidth
- Tracks "offloaded" state without real copies

```python
class EmulatedOffloadingManager(OffloadingManager):
    def lookup(self, block_hashes):
        # Return fake hit count based on profile
        latency = self.oracle.get_lookup_latency(num_blocks)
        time.sleep(latency / 1e6)  # Convert us to seconds
        return hit_count
    
    def prepare_store(self, block_hashes):
        # Simulate eviction latency
        latency = self.oracle.get_evict_latency(num_blocks, bytes)
        time.sleep(latency / 1e6)
        return fake_prepare_store_output
```

### Option 2: Oracle Hook in Worker

Keep real OffloadingManager, add oracle hooks:

```python
# In OffloadingWorker.transfer_async()
def transfer_async(self, job_id, spec):
    # Add oracle hook
    bytes_transferred = calculate_bytes(spec)
    emulated_latency = oracle.get_transfer_latency(
        bytes=bytes_transferred,
        direction=self.direction,  # CPU->GPU or GPU->CPU
        concurrency=self.active_transfers
    )
    
    # Either sleep for emulated time OR skip real transfer
    if emulated_mode:
        return True  # Fake success
    
    # Real transfer
    return self._do_real_transfer(job_id, spec)
```

---

## 5. Profile Schema

For KV Offload, we need:

```json
{
  "lookup": [
    {"num_blocks": 1, "latency_us": 10},
    {"num_blocks": 8, "latency_us": 50},
    {"num_blocks": 64, "latency_us": 400}
  ],
  "transfer": {
    "cpu_to_gpu": [
      {"bytes": 4096, "latency_us": 100},
      {"bytes": 1048576, "latency_us": 5000}
    ],
    "gpu_to_cpu": [
      {"bytes": 4096, "latency_us": 100},
      {"bytes": 1048576, "latency_us": 5000}
    ]
  },
  "evict": [
    {"num_blocks": 1, "latency_us": 5},
    {"num_blocks": 8, "latency_us": 30}
  ]
}
```

**Dimensions:**
- `bytes` - Transfer size
- `direction` - CPU→GPU or GPU→CPU
- `num_blocks` - Number of KV blocks
- `concurrency` - Number of concurrent transfers

---

## 6. Task Definition

### Task P1.2: Offload Interaction Model

**Objective:** Create oracle for CPU↔GPU KV transfer

**Deliverables:**
1. `vllm_emulator/oracle/offload_cost_oracle.py`
   - `get_lookup_latency(num_blocks)` 
   - `get_transfer_latency(bytes, direction, concurrency)`
   - `get_evict_latency(num_blocks)`
2. Hook into `OffloadingWorker.transfer_async()` (conditional)
3. Profile pack loader for offload

**Acceptance Criteria:**
- [ ] Oracle interface defined
- [ ] Can load profile pack
- [ ] Transfer latency = f(bytes, direction, concurrency)
- [ ] Lookup latency = f(num_blocks)
- [ ] Can fallback to real execution

---

## 7. Dependencies

- P1.2 depends on: P0.1 (Cost Boundary)
- Related to: P1.1 (GPU Compute) - offload happens before/after GPU compute

---

## 8. Notes

- Offload is optional in vLLM (controlled by `--kv-transfer-config`)
- Two main backends: CPU, GPU (in-proc)
- Transfer uses CUDA streams for async
- Events track transfer completion
- `ops.swap_blocks()` is the actual CUDA call
