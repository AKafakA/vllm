# Paper Outline: vLLM-Emulator (NeurIPS 2025)

**Title ideas:**
- "vLLM-Emulator: Accurate GPU-Free Emulation of LLM Serving Systems"
- "Serving-Native LLM Inference Emulation with <5% Error"
- "Profile Once, Emulate Anywhere: Forward-Compatible LLM Serving Emulation"

## 1. Introduction (~1 page)

- LLM serving is becoming the dominant GPU workload
- Researchers need to evaluate scheduling algorithms, batching strategies, hardware configs without GPU access
- Existing simulators (Vidur, REVATI) are either stale, closed-source, or miss critical CPU overhead
- We present vLLM-Emulator: a pluggable module for vLLM that replaces GPU computation with profiled latency predictions
- Key results: <3% TTFT/TPOT error on online serving across 3 model sizes, 2 parallelism configs, multiple QPS rates

## 2. Background & Motivation (~1 page)

### 2.1 LLM Serving Architecture
- Prefill vs decode phases
- Continuous batching, chunked prefill
- CUDA graphs, KV cache management
- Scheduling decisions: how batch composition affects latency

### 2.2 Why Emulation?
- GPU scarcity for research
- A/B testing scheduling algorithms
- Capacity planning across GPU types
- Hardware co-design exploration

### 2.3 Limitations of Existing Approaches
- **Vidur** (Microsoft): Re-implements scheduler, fixed to vLLM v0.4, event-driven (misses CPU overhead)
- **REVATI**: CUDA-level interception, hardware-coupled, not open-sourced, offline only
- **Analytical models**: Miss scheduling dynamics, batching heuristics
- Gap: no forward-compatible, serving-native emulator with accurate online serving metrics

## 3. Design (~2 pages)

### 3.1 Architecture: Pluggable Hooks
- Worker-level hook: intercepts execute_model(), replaces GPU with time.sleep()
- Executor-level hook: timer-based Futures for non-blocking online serving
- Both hooks: same oracle, same profile format, different timing mechanisms
- Forward compatibility: hooks at stable API boundaries, not kernel-level

### 3.2 Profiling Pipeline
- **Sweep profile**: data-independent, covers all batch sizes via controlled workloads
- **Serving profile**: step-cycle time (GPU + CPU overhead) measured during short serving trace
- Key insight: GPU-only profiles miss 4-7ms per-step overhead that varies with batch size due to GPU/CPU pipelining
- Profile format: JSON mapping total_tokens → latency_us

### 3.3 Step-Cycle Serving Profile
- Traces full _process_engine_step() duration including scheduling, execute_model, output processing
- Captures GPU/CPU overlap naturally (rate-independent)
- Per-token granularity for small batches (online serving operating range)
- Merged with sweep profile for large batches (offline coverage)

### 3.4 Accelerated Mode
- Virtual-time accumulation without time.sleep()
- Reports predicted GPU time vs wall time
- Useful for capacity planning, fast-forward simulation

## 4. Implementation (~0.5 pages)

- ~2000 lines of Python (vllm_emulator module)
- Plugin architecture: pip-installable, doesn't modify vLLM core
- Environment variables for configuration (no CLI changes needed)
- Works with all vLLM features: chunked prefill, CUDA graphs, TP, KV offloading

## 5. Evaluation (~2.5 pages)

### 5.1 Experimental Setup
- GPUs: RTX 3060 (12GB), [A100 40GB, A30 24GB, H100 80GB — planned]
- Models: Qwen2.5 0.5B/1.5B/3B, [7B/8B — planned]
- Workloads: Random (controlled), BurstGPT (real-world traces)
- Metrics: TTFT, TPOT, P99 variants, throughput (tok/s)

### 5.2 Online Serving Accuracy (Table 1)
Main result table: model × GPU × rate → TTFT error, TPOT error

RTX 3060 results:
| Model | Config | Rate=1 | Rate=2 | Rate=4 |
| 0.5B TP=1 | TTFT/TPOT | +0.0/+0.9% | +3.3/-0.0% | +0.1/+2.2% |
| 1.5B TP=1 | TTFT/TPOT | -0.1/-0.1% | -0.7/+0.2% | +0.2/+0.4% |
| 3B TP=2 | TTFT/TPOT | +4.7/+1.1% | -1.8/-1.3% | -0.6/+0.1% |

### 5.3 Workload Generalization (Table 2)
- Varied input lengths (128-1024): <3% error
- Varied output lengths (64-256): <1% error  
- BurstGPT traces: <4% error
- Shows emulator generalizes beyond profiling workload

### 5.4 Feature Sensitivity (Table 3)
| Feature | Real Impact | Emulator Error |
| CUDA graphs ON (default) | baseline | <1% |
| CUDA graphs OFF | 2.9× slower | -8% |
| Chunked prefill ON | baseline | <1% |
| Chunked prefill OFF | similar | <1% |
| KV offloading | captured in profile | <5% |

### 5.5 Accelerated Mode
- 36× faster than realtime emulation
- Virtual time predicts GPU utilization
- Enables fast-forward capacity planning

### 5.6 Comparison with Vidur
[Planned: run Vidur on same workloads, show accuracy gap + forward-compat issues]

## 6. Discussion (~0.5 pages)

### Limitations
- Offline throughput: time.sleep() overhead (-17%) → use accelerated mode
- enforce-eager (no CUDA graphs): -8% error due to higher variance
- PD disaggregation: requires NVLink, not tested on consumer GPUs
- Profile is hardware-specific: need to re-profile per GPU

### Future Work
- GPU-free EmulatorPlatform (no real GPU needed at emulation time)
- Network oracle for multi-node TP/PP
- Integration with cluster schedulers (Kubernetes, Ray Serve)
- Auto-profiling during model deployment

## 7. Related Work (~0.5 pages)
- Vidur (Agrawal et al., 2024)
- REVATI (Patel et al., 2024)
- Sarathi-Serve (Agrawal et al., 2024)
- DNN simulators: TimeLoop, Maestro, SCALE-Sim
- LLM benchmarking: FlexGen, vLLM benchmarks

## 8. Conclusion

## Appendix
- Environment variables reference
- Profile format specification
- Reproduction instructions
