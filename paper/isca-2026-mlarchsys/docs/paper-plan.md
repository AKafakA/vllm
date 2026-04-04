# Paper Plan: NeurIPS 2025 (Full Paper)

**Venue:** NeurIPS 2025 (8 pages, single column)
**Deadline:** May 6, 2025 (2026 in our timeline)
**Decision:** Target NeurIPS over ISCA MLArchSys workshop — more visibility, attracts both systems and AI communities, tools/infra contributions welcomed.

---

## Framing: Key Contributions

### C1: Serving-Native Emulation (Forward-Compatible by Design)

Unlike Vidur (re-implements scheduler, fixed to vLLM v0.4) and REVATI (CUDA-level interception, hardware-coupled), our emulator is a **pluggable module** that hooks into the real vLLM engine:

- No scheduler mirroring — real vLLM scheduler runs natively
- No kernel modification — hooks at worker/executor level
- Profiling decoupled from model architecture, kernel backend, CUDA version
- Works with any vLLM feature: chunked prefill, CUDA graphs, prefix caching, TP/PP, PD disaggregation, KV offloading, gRPC, Realtime API
- Forward-compatible: new vLLM versions work with existing profiles

**Key differentiator:** Our emulator runs the REAL vLLM code — all scheduling decisions, batching heuristics, memory management, and API overhead happen for real. Only the GPU forward pass is replaced.

### C2: Step-Cycle Serving Profile (Closes the CPU Overhead Gap)

Previous work profiles GPU kernel time only. Real online serving TPOT includes 4-7ms of CPU overhead (scheduling, output processing, detokenization) that varies with batch size due to GPU/CPU pipelining.

- **Vidur admits this gap** in their paper (event-driven simulation misses CPU overhead)
- **REVATI doesn't address it** (offline inference only)
- **Our solution:** Profile the full step cycle time during a short serving trace. Captures GPU/CPU overlap naturally. Rate-independent, no calibration needed.

Result: <3% TTFT and TPOT error across all QPS rates, input/output lengths.

### C3: First Realtime Online Serving Emulation

We are the first to achieve accurate realtime emulation of online LLM serving:

- Real HTTP/gRPC server with real request arrivals
- Accurate TTFT, TPOT, P99 metrics (not simulated)
- Captures queueing, batching dynamics, head-of-line blocking
- Works with any API frontend (REST, gRPC, Realtime WebSocket)

REVATI: offline inference only. Vidur: event-driven simulation (not real serving).

### C4: Accelerated Mode for Capacity Planning

Virtual-time fast-forward simulation: run hours of serving workload in seconds.
Same profiling data, same accuracy, but without time.sleep() blocking.

---

## Feature Demo Matrix

### Validated (results in hand):
| Feature | Status | Result |
|---------|--------|--------|
| Chunked prefill (default) | ✅ Done | <1% error (default config) |
| CUDA graphs on/off | ✅ Done | +0.9% with graphs, -8% without |
| TP=2 (3B model) | ✅ Done | <5% error |
| Varied input lengths (128-1024) | ✅ Done | <3% error |
| Varied output lengths (64-256) | ✅ Done | <3% error |
| Multiple QPS rates (1-4) | ✅ Done | <3% error |
| Accelerated mode | ✅ Implemented | Virtual time tracking works |

### Planned (need implementation/testing):
| Feature | Effort | Priority | Notes |
|---------|--------|----------|-------|
| PD disaggregation | Medium | High | vLLM supports prefill/decode separation; need to test emulator with it |
| KV offloading (CPU↔GPU) | Medium | Medium | Need to model transfer overhead; vLLM has `--cpu-offload-gb` |
| gRPC serving | Low | High | Just start server with gRPC; emulator is API-independent |
| OpenAI Realtime API (WebSocket) | Low | High | WebSocket streaming; shows API-independence |
| Chunked prefill OFF | Low | Medium | `--no-chunked-prefill` or large `--max-num-batched-tokens` |
| BurstGPT trace replay | Medium | High | Real-world arrival patterns |
| Multi-model serving | Medium | Low | vLLM LoRA/multi-model support |
| Speculative decoding | High | Low | Would need spec decode profile |

### vLLM v0.18.1 Feature Availability (confirmed in codebase):
| Feature | Available? | Key files | CLI/Config |
|---------|-----------|-----------|------------|
| PD disaggregation | ✅ | `vllm/v1/worker/gpu/kv_connector.py` | KV connector config |
| CPU offloading | ✅ | `vllm/config/offload.py`, `vllm/model_executor/offloader/` | `--cpu-offload-gb` |
| gRPC serving | ✅ | `vllm/entrypoints/grpc_server.py` | `vllm serve --rpc grpc` |
| OpenAI Realtime API | ✅ | `vllm/entrypoints/openai/realtime/serving.py` | WebSocket endpoint |
| Chunked prefill | ✅ (default) | Built-in | `--enable-chunked-prefill`/`--no-chunked-prefill` |
| CUDA graphs | ✅ (default) | Built-in | `--enforce-eager` to disable |
| TP/PP | ✅ | Built-in | `--tensor-parallel-size`, `--pipeline-parallel-size` |

### Need to Investigate:
| Feature | Question |
|---------|----------|
| CPU-GPU transfer overhead | Captured implicitly in step-cycle profile? Or need explicit modeling for KV offload? |
| GPU-GPU (NVLink/PCIe) overhead | Captured in TP=2 serving profile implicitly |
| Network overhead (multi-node) | Need network oracle for multi-node TP/PP |
| PD disaggregation overhead | KV transfer between prefill/decode workers — profile captures it if server runs with disagg enabled |
| gRPC vs REST overhead | Should be minimal — test to confirm emulator is API-independent |
| Realtime WebSocket | WebSocket adds bidirectional streaming — test to confirm compatibility |

---

## Evaluation Plan

### GPU Classes (minimum 3 for NeurIPS):
| GPU | Source | Models | Status |
|-----|--------|--------|--------|
| RTX 3060 12GB | Vast | 0.5B, 1.5B, 3B (TP=2) | ✅ Done |
| A100 40GB | CloudLab (Apr 12-16) | 1.5B, 7B, 8B | Planned |
| A30 24GB | CloudLab (Apr 12-16) | 1.5B, 7B | Planned |
| RTX 8000 48GB | dev-gpu (Apr 6-10) | 7B, 8B | Planned |
| H100 80GB | Vast (~$6) | 1.5B, 7B | Nice-to-have |
| A100 80GB | CSD3 | 70B (TP=2/4) | Nice-to-have |
| V100 16GB | CloudLab (Apr 26-May 1) | 0.5B, 1.5B | Nice-to-have |

### Comparison Baselines:
| Baseline | What to show |
|----------|-------------|
| Vidur | Run their simulator on same workloads → show accuracy gap and forward-compat issue |
| REVATI | Compare claims from their paper (not open-sourced, so paper comparison only) |
| Real vLLM | Back-to-back comparison (our primary baseline) |

### Key Experiments for NeurIPS:
1. **Accuracy table**: TTFT/TPOT error across GPUs × models × rates (the 2×2 matrix)
2. **Feature sensitivity**: CUDA graphs, chunked prefill, PD disagg, KV offload
3. **Scalability**: TP=1, TP=2, TP=4, multi-node
4. **Workload generalization**: Random, BurstGPT, ShareGPT traces
5. **Accelerated mode speedup**: Wall time vs virtual time
6. **Profiling cost**: Time to profile vs time saved by emulation
7. **API independence**: REST vs gRPC vs Realtime WebSocket (same emulator, same accuracy)

---

## Timeline

- **Now → Apr 6**: Vast 3060 (BurstGPT, offline, accelerated test, gRPC/Realtime demo)
- **Apr 6-10**: RTX 8000 (7B/8B eval)
- **Apr 10-12**: Vidur comparison, paper draft outline
- **Apr 12-16**: CloudLab A100 + A30 (1 day borrowed)
- **Apr 17**: Vast H100 ($6, 4 hours)
- **Apr 18-May 5**: Paper writing, figures, related work
- **May 6**: NeurIPS submission
