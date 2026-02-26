# Component Review: gpu_worker.py + gpu_model_runner.py

**Date:** 2026-02-26
**Component:** GPU Worker + Model Runner
**Files:**
- `vllm/v1/worker/gpu_worker.py` (965 lines)
- `vllm/v1/worker/gpu_model_runner.py` (5000+ lines)

---

## 1. Call Flow

```
EngineCore
    │
    ▼
Worker.execute_model(scheduler_output)
    │
    ▼ (line 626-627)
ModelRunner.execute_model(scheduler_output, intermediate_tensors)
    │
    ▼
ModelRunner._execute_model()
    │
    ▼
self.model()  ← Actual neural network forward (line 3013, 4697)
```

---

## 2. Key Methods

### 2.1 Worker Methods (gpu_worker.py)

| Method | Line | Purpose | Category |
|--------|------|---------|----------|
| `__init__` | 69 | Initialization | A |
| `initialize_cache` | 171 | KV cache allocation | C |
| `init_device` | 175 | GPU init | A |
| `load_model` | 270 | Load weights | A |
| `execute_model` | 582 | Main execution | **B** |
| `profile` | 656 | Profiling | A |
| `check_health` | 684 | Health check | A |

### 2.2 ModelRunner Methods (gpu_model_runner.py)

| Method | Line | Purpose | Category |
|--------|------|---------|----------|
| `execute_model` | 3274 | Main entry | **B** |
| `_execute_model` | ~3500 | Core forward | **B** |
| `_run_attention` | N/A | Attention computation | **B** |
| `model` (property) | 539 | Returns nn.Module | **B** |

---

## 3. Hook Points for Emulation

### Option 1: Replace at Worker Level
**File:** `gpu_worker.py`
**Method:** `execute_model()` (line 582)

```python
def execute_model(self, scheduler_output):
    # ... preprocessing ...
    
    with self.annotate_profile(scheduler_output):
        output = self.model_runner.execute_model(  # ← HOOK HERE
            scheduler_output, intermediate_tensors
        )
    
    # ... postprocessing ...
    return output
```

**Pros:** Clean interface, easy to swap
**Cons:** Need to handle all the preprocessing/postprocessing

### Option 2: Replace at ModelRunner Level
**File:** `gpu_model_runner.py`
**Method:** `execute_model()` (line 3274)

**Pros:** More granular control
**Cons:** More integration work, may break with updates

### Option 3: Replace at Model Forward Level
**File:** `gpu_model_runner.py`
**Method:** `self.model()` (line 3013, 4697)

**Pros:** Most granular control
**Cons:** Need to understand model architecture

---

## 4. Input/Output Analysis

### 4.1 scheduler_output (Input)

From `v1/core/sched/output.py`:
- `total_num_scheduled_tokens`: int
- `num_scheduled_tokens`: Dict[req_id, int]
- `preempted_req_ids`: List[str]
- Batch metadata

### 4.2 ModelRunnerOutput (Output)

From `v1/outputs.py`:
- `outputs`: List[RequestOutput]
- `num_scheduled_tokens`: int
- `finished_requests_ids`: List[str]

---

## 5. Profiling Data Available

From `execute_model()`:
- `scheduler_output.total_num_scheduled_tokens` - Total tokens in batch
- `scheduler_output.num_scheduled_tokens` - Per-request token count
- Batch composition (prefill vs decode)

From `gpu_model_runner.execute_model()`:
- `num_scheduled_tokens` - Total tokens
- `num_reqs` - Number of requests
- `max_num_scheduled_tokens` - Max tokens for any request
- `input_batch` - Full input batch details

---

## 6. Recommended Integration Strategy

### Phase 1: Oracle Hook at Worker Level

1. Create `GpuCostOracle` class
2. In `gpu_worker.execute_model()`, add timing oracle:
   ```python
   # Before model_runner.execute_model()
   start = time.perf_counter()
   
   output = self.model_runner.execute_model(...)
   
   # After execution
   elapsed = time.perf_counter() - start
   oracle.record(elapsed, scheduler_output)
   ```

3. Oracle returns emulated latency instead of real

### Phase 2: Full Replacement

After profiling data is collected:
1. Replace `model_runner.execute_model()` entirely
2. Return emulated `ModelRunnerOutput` based on lookup tables

---

## 7. Profile Pack Schema

For GPU Compute (Category B), we need:

```json
{
  "prefill": [
    {"seq_len": 128, "latency_us": 15000},
    {"seq_len": 256, "latency_us": 28000},
    ...
  ],
  "decode": [
    {"active_seqs": 1, "latency_us_per_token": 500},
    {"active_seqs": 8, "latency_us_per_token": 3500},
    ...
  ]
}
```

**Dimensions:**
- Prefill: `prompt_tokens` (primary), `batch_size` (secondary)
- Decode: `active_seqs` (primary), `kv_cache_hit` (secondary)

---

## 8. Task Definition

### Task P1.1: GPU Cost Oracle Interface

**Objective:** Create oracle interface that can be inserted at worker level

**Deliverables:**
1. `vllm_emulator/oracle/gpu_cost_oracle.py`
   - Interface: `get_prefill_latency(prompt_tokens, batch_size)`
   - Interface: `get_decode_latency(active_seqs)`
2. Hook into `gpu_worker.execute_model()` (conditional)
3. Profile pack loader

**Acceptance Criteria:**
- [ ] Oracle interface defined
- [ ] Can load profile pack
- [ ] Can fallback to real execution
- [ ] No behavioral change when disabled

---

## 9. Notes

- v0.15.1 uses "GPUModelRunner" not "GPUModelRunnerV2"
- `use_v2_model_runner` flag controls runner version
- KV transfer hooks are in `has_kv_transfer_group()` calls
- Need to handle both prefill and decode in same batch (mixed batch)
