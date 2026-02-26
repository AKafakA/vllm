# vLLM Emulator Cost Boundary

This document defines which components of vLLM run **real** (no modeling) vs which are **modeled** (emulated) in the emulator backend.

## Category A: Run Real (No Modeling)

These components execute real code with real CPU overhead:

### API Ingress
- `vllm/entrypoints/api_server.py` - FastAPI server, HTTP handling
- `vllm/entrypoints/openai/` - OpenAI-compatible API endpoints
- `vllm/entrypoints/grpc_server.py` - gRPC server

### Tokenization & Preprocessing
- `vllm/tokenizers/` - Tokenizer loading and inference
- `vllm/inputs/` - Input preprocessing, prompt parsing

### Scheduler Policy
- `vllm/v1/engine/core.py` - Core engine, request dispatch
- `vllm/v1/scheduler.py` - Scheduling logic, batch formation
- `vllm/v1/worker/` - Worker lifecycle management

### Request Lifecycle
- `vllm/engine/` - Engine initialization, request queuing
- `vllm/sequence.py` - Sequence management
- `vllm/outputs.py` - Output processing

### Housekeeping (Low Priority)
- Logging, metrics, stats collection

---

## Category B: Model - GPU Compute Cost

These components are modeled using lookup tables:

### Prefill Phase
- **Target:** `vllm/v1/worker/gpu_worker.py` - Model forward pass
- **Modeled:** T_prefill_gpu(prompt_len, batch_ctx, parallelism, dtype)
- **Profile:** Prefill latency by prompt token count

### Decode Phase
- **Target:** `vllm/v1/worker/gpu_worker.py` - Token generation
- **Modeled:** T_decode_gpu_per_token(active_seqs, kv_state, parallelism, dtype)
- **Profile:** Decode latency by concurrent sequence count

---

## Category C: Model - CPU↔GPU Interaction

These components are modeled with interaction cost:

### KV Cache Management
- **Target:** `vllm/v1/worker/gpu_worker.py` - KV cache operations
- **Modeled:** T_offload(bytes, direction, bw_eff, overlap_ratio, sync_penalty, concurrency)

### Memory Operations
- **Target:** `vllm/device_allocator/` - Device memory allocation
- **Modeled:** Copy operations between CPU/GPU memory

### Synchronization
- **Target:** CUDA synchronization primitives
- **Modeled:** Sync points, stream events

---

## Category D: Model - Network (PD Separation)

These components are modeled for prefill-decode separation:

### Tensor Parallelism
- **Target:** `vllm/distributed/` - All-reduce operations
- **Modeled:** T_all_reduce(op, bytes, world_size, topology)
- **Profile:** Per topology (NVLink, PCIe, IB)

### Pipeline Parallelism
- **Target:** `vllm/distributed/` - Send/recv operations
- **Modeled:** T_send_recv(op, bytes, src_dst, topology)

### PD KV Transfer
- **Target:** Custom PD dispatch logic
- **Modeled:** T_kv_transfer(bytes, concurrency, topology)

---

## Module Quick Reference

| Module Path | Category | Notes |
|-------------|----------|-------|
| `entrypoints/api_server.py` | A | HTTP server |
| `entrypoints/openai/` | A | OpenAI API |
| `tokenizers/` | A | Tokenization |
| `inputs/` | A | Preprocessing |
| `v1/engine/core.py` | A | Engine core |
| `v1/scheduler.py` | A | Scheduling |
| `v1/worker/gpu_worker.py` | B | GPU compute |
| `device_allocator/` | C | Memory ops |
| `distributed/` | D | Network ops |

---

## Implementation Notes

1. **Hook Points:** Replace GPU execution paths with oracle calls
2. **Profile Packs:** Separate JSON files per component
3. **Fallback:** Real execution when profile unavailable
4. **Validation:** A/B test against real GPU execution
