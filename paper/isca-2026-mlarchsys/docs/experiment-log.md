# vLLM-Emulator Experiment Log

**Started:** 2026-04-02
**Vast Host:** 2x RTX 3060 12GB, Xeon E5-2680 v4, 64GB RAM, CUDA 13.0
**Vast Expiry:** 2026-04-04 ~5pm
**vLLM Version:** 0.18.1 (rebased from v0.18.1 tag)
**Install Method:** VLLM_USE_PRECOMPILED=1 with official PyPI wheel via VLLM_PRECOMPILED_WHEEL_LOCATION

---

## Terminology

**Emulator modes** (how GPU time is modeled):
- **Realtime mode**: `time.sleep(predicted_latency)` — wall clock matches predicted GPU time
- **Accelerated mode**: virtual-time fast-forward, no blocking — like REVATI/Vidur

**vLLM serving modes** (how requests arrive):
- **Online serving**: `vllm serve` — FastAPI/HTTP, async engine, real-time arrivals
- **Offline inference**: `LLM()` / `vllm bench throughput` — batch API, all requests at once

These are **orthogonal** — any combination is valid (2×2 matrix).

---

## Final Results (Session 2, v13)

### Data-Independent Sweep Profile Accuracy

| Model | Config | Real tok/s | Emulator tok/s | Error |
|-------|--------|-----------|---------------|-------|
| **Qwen2.5-1.5B** | TP=1, 30 prompts, 256in/128out | 2,302 | 2,229 | **-3.2%** ✅ |
| **Qwen2.5-0.5B** | TP=1, 30 prompts, 256in/128out | ~7,444 | ~7,603 | **~2.1%** ✅ |

Both results use **data-independent sweep profiles** — profiled once, tested on unseen workloads.

---

## Session 1: 2026-04-02 to 2026-04-03

### Infrastructure Setup

**Build Issues:**
1. Source build with MAX_JOBS=56 → OOM killed Vast host
2. Source build with MAX_JOBS=2 → 3.5 hours, got stuck in FA2 kernels
3. VLLM_USE_PRECOMPILED on old fork → ABI mismatch (torch 2.9.1 vs precompiled for 2.10.0)
4. **Solution:** Rebased to v0.18.1, VLLM_USE_PRECOMPILED=1 with official PyPI wheel

**Rebase:**
- Cherry-picked 20 emulator commits onto v0.18.1 tag
- Resolved 3 merge conflicts
- All 72 unit tests pass
- Force-pushed to `feature/emulator-backend`

**RTX 3060 Issues:**
- FP8 CUTLASS ops not available (sm_86 lacks FP8) → official PyPI wheel handles gracefully
- Emulator platform plugin: fixed to only activate when VLLM_EMULATOR_ENABLE_ORACLE=1

### Code Changes (Session 1)

1. **Fake token generation** — 1 token/req, deterministic RNG, avoids EOS, respects vocab
2. **Logprobs shape** — correct shape per request
3. **Thread safety** — locks on offload/network hook concurrency counters
4. **Power-law extrapolation** — replaces linear interpolation + clamping
5. **v0.18.1 async scheduler** — `_emulator_pending_output` + `sample_tokens()` returns stored output
6. **Prefill chunk handling** — 0 tokens for partial chunks, 1 for completed
7. **Unified oracle** — `estimate_step_latency_us(total_tokens)` with forward_pass profile
8. **Trace profiler** — instruments execute_model() for validation
9. **Rename** — online/offline → realtime/accelerated

---

## Session 2: 2026-04-03 (Accuracy Debugging)

### Iteration History (1.5B TP=1)

| Version | Profile | Error | Root Cause |
|---------|---------|-------|------------|
| v2 | Synthetic batch_size=1 | +183% | Oracle modeled as single giant sequence |
| v5 | Forward_pass unified | +19.2% | Synthetic profiler doesn't match batch composition |
| trace | Data-dependent | +2.4% | Overfits to workload (not data-independent) |
| sweep v1 | 1D total_tokens only | +30.5% | **Bug: prompt was half intended length** (`"hello " * (n//2+1)` = n/2 tokens) |
| sweep v7 | Fixed prompt length | +6.5% | Benchmark used `--random-input-len 256` = 256 tokens, profile matched |
| sweep v7 | Same, `--input-len 256` | +37% | **Benchmark produced 1024 tokens** (RandomDataset inflation), profile had 256 |
| sweep v11 | `max_output_len=128` | +10.4% | Sweep used identical prompts → prefix caching → scheduler chunked differently |
| **sweep v13** | **Unique random prompts** | **-3.2%** | **All issues fixed** |

### Key Bugs Found and Fixed

#### Bug 1: Prompt Token Count Mismatch
**Symptom:** 37% error despite "correct" profiling
**Root cause:** `"hello " * (input_len // 2 + 1)` generated half the intended tokens. "hello " is 1 token, not 2.
**Fix:** `"hello " * (input_len - 1)` or use `TokensPrompt` with exact token IDs
**Lesson:** Always verify actual token count, never assume text-to-token ratio

#### Bug 2: Benchmark --input-len vs --random-input-len
**Symptom:** Real baseline had 1024 tokens per prompt despite `--input-len 256`
**Root cause:** `vllm bench throughput --input-len 256` uses RandomDataset with default range_ratio that inflates prompt length. `--random-input-len 256` gives exactly 256 tokens.
**Fix:** Always use explicit `--random-input-len` and `--random-output-len` for controlled benchmarks
**Lesson:** Read benchmark tool documentation carefully; --input-len ≠ exact token count

#### Bug 3: Prefix Caching in Sweep Profiler
**Symptom:** Sweep produced max tt=2048 per step, but real workload hit tt=7425
**Root cause:** Sweep used identical `"hello " * 255` prompts for all requests. vLLM's prefix caching (enabled by default) detected shared prefixes and skipped prefilling cached tokens, resulting in much smaller `total_num_scheduled_tokens` per step.
**Fix:** Use unique random token IDs per prompt: `TokensPrompt(prompt_token_ids=[random IDs])`
**Lesson:** Profiler must use unique prompts to match real workload scheduler behavior. Prefix caching fundamentally changes how the scheduler packs batches.

#### Bug 4: Decode Output Length Mismatch
**Symptom:** 6.5% error instead of <5%
**Root cause:** Sweep decode configs used `output_len=64` but real workload generates 128 tokens. Shorter output = shorter KV cache during decode = faster decode steps.
**Fix:** Sweep `max_output_len` parameter, defaults to `max_model_len / 2` to cover full range.
**Lesson:** Profile must cover the full operating range of the workload — both input AND output dimensions.

### Misleading Investigations (Red Herrings)

These were explored but turned out NOT to be the real issues:

1. **GPU/CPU overlap** — Investigated extensively (deadline-based sleep, Future-based approach, threading.Timer). Not the issue because LLM() offline benchmark uses sequential step execution, not pipelined.

2. **2D oracle (total_tokens × context_length)** — Implemented bilinear interpolation over forward_pass_2d profile. The context dimension had minimal impact because the 1D profile accuracy was limited by the profiling bugs, not by missing the context dimension.

3. **KV cache reads / Vidur-style decomposition** — Computed total_kv_reads from scheduler's num_computed_tokens. Unnecessary complexity — 1D forward_pass is sufficient when profiling matches workload.

4. **CUDA graph capture overhead** — Profiled with warmup, not a significant source of error.

5. **Scheduler CPU overhead** — Real scheduler runs in both real and emulator, so it's already accounted for.

### Correct Sweep Profiler Design

The final working profiler (`shape_sweep_profiler.py v13`) requires:

1. **Unique random prompts** — `TokensPrompt(prompt_token_ids=[random IDs])` per prompt to avoid prefix caching
2. **Correct token counts** — verified via TokensPrompt, not text approximation
3. **Full output length coverage** — `max_output_len` covers the longest workload output
4. **Sufficient batch sizes** — configs that produce the same `total_num_scheduled_tokens` as real workloads
5. **Mixed prefill+decode configs** — multiple concurrent requests with output generation to create chunked-prefill mixed steps

### Architecture Notes

**execute_model() hook design (v0.18.1):**
```
execute_model(scheduler_output):
    if emulator_hook.is_enabled:
        cost = estimate_execution_cost(scheduler_output)
        fake_output = create_fake_output(scheduler_output)
        if should_block: time.sleep(cost)
        store pending_output
        return None  # triggers sample_tokens() path

sample_tokens(grammar_output):
    if pending_output: return pending_output
    return real_sample_tokens()
```

**Oracle: 1D forward_pass lookup**
- `estimate_step_latency_us(total_tokens)` 
- Piecewise linear interpolation within profiled range
- Power-law extrapolation outside range
- Profile section: `forward_pass: [{total_tokens, latency_us, num_samples}]`

**Trace profiler:**
- Instruments execute_model() with `torch.cuda.synchronize()` + `time.perf_counter()`
- Records: total_tokens, num_prefill_tokens, num_decode_seqs, latency_us per step
- Useful for validation and debugging, not for production profiles (adds sync overhead)

---

## Decisions Made

1. **Rebase to v0.18.1** — enables VLLM_USE_PRECOMPILED, matches latest vLLM
2. **1D forward_pass oracle** — sufficient with correct profiling; 2D is premature
3. **Unique random prompts in sweep** — critical to avoid prefix caching bias
4. **time.sleep() in execute_model** — simple, correct for offline inference
5. **Rename realtime/accelerated** — avoids collision with vLLM's online/offline serving
6. **Clean real GPU = primary baseline** — compare against non-traced benchmark
7. **`--random-input-len` / `--random-output-len`** — always use explicit token counts

---

## Session 3: 2026-04-03 (Online Serving Accuracy)

### Key Findings

**1. Per-token profile bucketing (tt≤32)**
The trace-to-profile converter was bucketing tt=2-7 → bucket 8, tt=9-15 → bucket 16 (bucket_size=8), losing dense coverage at online serving batch sizes. Fixed to use per-token granularity for tt≤32. This revealed a CUDA graph boundary at tt=17 where latency jumps from 15.6ms to 28.5ms (~2×).

**2. Decode-specific overhead**
Profile captures GPU forward pass only (~13ms for tt=1). Real TPOT includes ~4-6ms of output processing overhead (sampling, detokenization, scheduling). This overhead is cheaper with fake emulator outputs than real GPU outputs. Fix: `VLLM_EMULATOR_DECODE_OVERHEAD_US` applied only to steps with decode sequences, preserving prefill TTFT accuracy.

**3. Worker hook vs executor hook**
- Worker hook: runs inside real scheduler loop → accurate TTFT, but `time.sleep()` blocks worker thread → deadlocks at rate≥4 with large batches
- Executor hook: uses timer-based Futures → non-blocking, but TTFT depends on timer chain accuracy
- **Decision**: Use executor hook for online serving (non-blocking), worker hook for offline throughput

**4. Real baselines vary significantly**
Back-to-back measurements show real baseline TTFT can vary 30-60% between server startups (90ms vs 153ms at rate=1). Must always compare real vs emulator from the same session.

### Online Serving Results (preliminary, 4ms decode overhead)

Compared against **fresh baselines from same session**:

| Metric | Real rate=1 | Emu rate=1 | Error | Real rate=4 | Emu rate=4 | Error |
|--------|------------|-----------|-------|------------|-----------|-------|
| TTFT | 153.2ms | 146.4ms | -4.4% | 93.6ms | 95.1ms | +1.6% |
| TPOT | 20.6ms | 18.4ms | -10.5% | 18.4ms | 18.9ms | +2.7% |

Rate=4: TTFT +1.6%, TPOT +2.7% — both <5% ✓
Rate=1: TTFT -4.4% ✓, TPOT -10.5% (decode overhead calibration varies by rate)

### Back-to-Back Results (5ms decode overhead, 50 prompts)

| Rate | TTFT Error | TPOT Error | P99 TPOT Error |
|------|-----------|-----------|---------------|
| 1 | -1.0% ✓ | -6.5% ✓ | -7.6% ✓ |
| 2 | +5.9% ✓ | -5.3% ✓ | -6.9% ✓ |
| 4 | +12.6% | +6.7% ✓ | +2.0% ✓ |

TPOT <7% at all rates, but overhead is rate-dependent (too high at rate=4, too low at rate=1).

### Step-Cycle Serving Profile (BREAKTHROUGH)

**Key discovery**: GPU-only profiles miss per-step serving overhead that varies with batch size due to GPU/CPU pipelining. A constant `DECODE_OVERHEAD_US` is rate-dependent.

**Solution**: Profile the full step cycle time (GPU + scheduling + output processing) during a short serving trace. The step cycle captures GPU/CPU overlap naturally.

Step cycle vs GPU-only comparison (1.5B, RTX 3060):
```
tt=1-7:  cycle=20ms, GPU=13ms, overhead=+6.5ms (low batch, CPU dominates)
tt=8-10: cycle=17ms, GPU=14ms, overhead=+2.5ms (CUDA graph sweet spot)
tt≥17:   cycle=20ms, GPU=28ms, overhead=-8ms (GPU/CPU overlap hides CPU)
```

**Final results with serving profile (NO calibration constant):**

| Rate | TTFT Error | TPOT Error |
|------|-----------|-----------|
| **1** | **+2.5%** | **+1.0%** |
| **2** | **-2.8%** | **-0.2%** |
| **4** | **+0.8%** | **+0.5%** |

All metrics <3% across all rates. Rate-independent. No per-rate calibration needed.

### Comprehensive Serving Profile Evaluation (FINAL, back-to-back)

Qwen2.5-1.5B, RTX 3060, 50 prompts each, serving profile + executor hook.

**Rate sweep (256in/128out):**

| Rate | Real TTFT | Emu TTFT | Error | Real TPOT | Emu TPOT | Error |
|------|----------|---------|-------|----------|---------|-------|
| 1 | 144.1ms | 143.9ms | **-0.1%** | 21.0ms | 21.0ms | **-0.1%** |
| 2 | 85.1ms | 84.6ms | **-0.7%** | 20.2ms | 20.2ms | **+0.2%** |
| 4 | 84.2ms | 84.4ms | **+0.2%** | 18.9ms | 19.0ms | **+0.4%** |

**Varied input length (rate=2, 128out):**

| Input | Real TTFT | Emu TTFT | Error | Real TPOT | Emu TPOT | Error |
|-------|----------|---------|-------|----------|---------|-------|
| 128 | 82.9ms | 82.5ms | **-0.4%** | 20.0ms | 20.1ms | **+0.4%** |
| 256 | 85.1ms | 84.6ms | **-0.7%** | 20.2ms | 20.2ms | **+0.2%** |
| 512 | 159.2ms | 154.4ms | **-3.0%** | 22.3ms | 22.1ms | **-0.9%** |
| 1024 | 220.8ms | 220.0ms | **-0.4%** | 24.3ms | 24.4ms | **+0.5%** |

**Varied output length (rate=2, 256in):**

| Output | Real TTFT | Emu TTFT | Error | Real TPOT | Emu TPOT | Error |
|--------|----------|---------|-------|----------|---------|-------|
| 64 | 76.8ms | 76.3ms | **-0.6%** | 20.2ms | 20.1ms | **-0.3%** |
| 128 | 85.1ms | 84.6ms | **-0.7%** | 20.2ms | 20.2ms | **+0.2%** |
| 256 | 86.1ms | 85.6ms | **-0.5%** | 18.3ms | 18.4ms | **+0.4%** |

**All 16 metrics under 3%. Most under 1%. Paper-ready.**

### TP=2 Evaluation (3B model, 2×RTX 3060, enforce-eager)

Qwen2.5-3B-Instruct, TP=2, max-model-len=2048, enforce-eager (CUDA graphs OOM on 12GB).

**Key issues encountered:**
- CUDA OOM with default CUDA graphs on 2×12GB RTX 3060 → fixed with --enforce-eager
- First serving trace (30 prompts/rate) had too few samples → noisy profile, 5-9% TPOT error
- Outliers in step cycle data (0.7ms-9315ms) → added filtering: keep values >5ms and <3×median

**Final results (50 prompts/rate, outlier-filtered profile):**

| Rate | Real TTFT | Emu TTFT | Error | Real TPOT | Emu TPOT | Error |
|------|----------|---------|-------|----------|---------|-------|
| 1 | 916.0ms | 958.9ms | **+4.7%** | 193.5ms | 195.6ms | **+1.1%** |
| 2 | 417.8ms | 410.4ms | **-1.8%** | 161.0ms | 158.8ms | **-1.3%** |
| 4 | 459.8ms | 457.0ms | **-0.6%** | 184.1ms | 184.3ms | **+0.1%** |

All metrics under 5%. TP=2 works with the serving profile approach.

### Feature Ablation: CUDA Graphs (1.5B TP=1, rate=2)

| Config | Real TTFT | Emu TTFT | Error | Real TPOT | Emu TPOT | Error |
|--------|----------|---------|-------|----------|---------|-------|
| Default (CUDA graphs) | 161.9ms | 161.8ms | **-0.1%** | 22.0ms | 22.2ms | **+0.9%** |
| enforce-eager (no graphs) | 244.7ms | 227.3ms | -7.1% | 63.8ms | 58.6ms | -8.1% |

CUDA graphs provide 2.9× TPOT speedup (22ms vs 64ms). The emulator captures this speedup accurately with CUDA graphs (0.9% error) but underestimates enforce-eager by ~8%. This is because enforce-eager has higher latency variance — without deterministic CUDA graph execution, each kernel launch varies more with system state.

**Key insight for the paper**: the emulator's accuracy depends on the determinism of the GPU execution path. CUDA graphs make latency highly predictable → excellent emulation. Eager mode is less predictable → harder to emulate precisely.

### Chunked Prefill Ablation (1.5B TP=1, rate=2)

| Config | Real TTFT | Emu TTFT | Error | Real TPOT | Emu TPOT | Error |
|--------|----------|---------|-------|----------|---------|-------|
| Default (chunked ON) | 161.9ms | 161.8ms | **-0.1%** | 22.0ms | 22.2ms | **+0.9%** |
| No chunked prefill | 158.0ms | 157.2ms | **-0.5%** | 22.2ms | 22.1ms | **-0.6%** |

Both configs <1% error. Chunked prefill has minimal impact at input_len=256 (below the 2048 chunking threshold). For longer inputs where chunking matters, the serving profile approach captures the chunking overhead naturally.

### BurstGPT Cross-Workload Validation (1.5B TP=1)

Real-world trace replay using BurstGPT dataset (GPT-4 conversation logs with natural arrival patterns and variable input/output lengths).

| Workload | Real TTFT | Emu TTFT | Error | Real TPOT | Emu TPOT | Error |
|----------|----------|---------|-------|----------|---------|-------|
| BurstGPT rate=2 | 200.7ms | 202.7ms | **+1.0%** | 23.9ms | 23.8ms | **-0.3%** |
| BurstGPT rate=4 | 92.8ms | 96.2ms | **+3.7%** | 24.9ms | 25.8ms | **+3.9%** |
| Random rate=2 (control) | 153.8ms | 157.3ms | **+2.3%** | 22.2ms | 22.1ms | **-0.3%** |

All <5%. The emulator generalizes to unseen real-world workload patterns.

### Offline Throughput (1.5B TP=1)

| Config | Throughput | Error |
|--------|-----------|-------|
| Real | 5624 tok/s | — |
| Emu realtime (v13 profile) | 4661 tok/s | -17.1% |
| Emu realtime (dense-v2 profile) | 4588 tok/s | -18.4% |
| Emu accelerated (dense-v2) | 22310 tok/s | 3.9× faster (virtual time) |

**Why realtime offline is less accurate:** `time.sleep()` has OS scheduling jitter (~0.1-1ms per call) that compounds over 12,800+ decode steps. Real GPU runs kernels back-to-back without this overhead. This is an inherent limitation of sleep-based timing for batch workloads.

**Root cause investigation:**
- N=30 prompts: -3.8% error. Decode at tt≈30 (well-covered by profile)
- N=100 prompts: -16.1% error. Decode at tt≈100 (sparse profile coverage, CUDA graph boundary at tt=144 causes overestimation)
- The error scales with num_prompts because larger batches push into poorly-profiled tt ranges
- NOT primarily sleep overhead — it's profile accuracy at large batch sizes

**Fix needed:** Better sweep profile coverage at tt=30-128 (offline operating range).

**Accelerated mode virtual throughput:**
- N=30: virtual_time=13.7s for 11,520 tokens → virtual_throughput=842 tok/s vs real=2,312 tok/s (profile overestimates at large tt)
- Same root cause: profile latencies too high at tt=30-128

**Fix (v14 sweep profile):** Extended `decode_batch_sizes` in sweep profiler to `range(1,21) + [24,28,32,40,48,56,64,80,96,128]`. This covers the offline operating range (tt=30-128) with proper decode configurations instead of mismatched prefill data.

| Prompts | Error (v13) | Error (v14, fixed) |
|---------|------------|-------------------|
| N=30 | -3.8% | **+0.3%** |
| N=100 | -16.1% | **-4.4%** |

Root cause: v13 sweep only profiled decode batch sizes up to 32. For tt=50-100, the profile contained prefill/single-seq data with +48-55% overestimation vs actual 50-100 seq decode. The v14 sweep profiles actual decode at these batch sizes.

**This fix is data-independent** — no workload-specific traces needed. The throughput is computed as `total_tokens / virtual_gpu_time`.

### Accelerated Mode (Virtual Time)

Implemented virtual time tracking in both worker and executor hooks. In accelerated mode (`VLLM_EMULATOR_MODE=accelerated`):
- No `time.sleep()` — steps execute at CPU speed
- Cumulative predicted GPU time is tracked
- Summary reports: virtual GPU time, wall time, speedup factor
- 3.9× speedup over real execution on RTX 3060
- Useful for capacity planning simulations and offline throughput prediction

### KV Offloading Feature Demo (1.5B TP=1)

KV offloading (`--kv-offloading-size 2 --kv-offloading-backend native --disable-hybrid-kv-cache-manager`):

| | Real | Emu | Error |
|---|---|---|---|
| TTFT | 159.0ms | 165.4ms | **+4.0%** |
| TPOT | 22.1ms | 22.1ms | **-0.0%** |

The emulator works transparently with KV offloading — the serving profile captures KV transfer overhead as part of the step cycle time. No emulator code changes needed.

### PD Disaggregation (Not tested — hardware limitation)

PD disaggregation (P2pNcclConnector) failed on 2×RTX 3060: the P2P NCCL connector requires P2P DMA between GPUs which consumer GPUs on PCIe don't support. Needs NVLink-connected GPUs (A100, H100) — defer to CloudLab Apr 12-16.

### 0.5B Model Validation (TP=1, serving profile)

| Rate | Real TTFT | Emu TTFT | Error | Real TPOT | Emu TPOT | Error |
|------|----------|---------|-------|----------|---------|-------|
| 1 | 68.3ms | 68.3ms | **+0.0%** | 10.8ms | 10.8ms | **+0.9%** |
| 2 | 52.0ms | 53.7ms | **+3.3%** | 12.3ms | 12.3ms | **-0.0%** |
| 4 | 57.3ms | 57.4ms | **+0.1%** | 12.3ms | 12.6ms | **+2.2%** |

Three model sizes validated on RTX 3060: 0.5B, 1.5B, 3B — all <5% error with serving profile approach.

**PD disagg with NixlConnector attempted (v4):**
- Both instances initialized NIXL connector successfully
- Both got OOM killed: model (2.9GB) + KV cache + NIXL buffers > 12GB RTX 3060
- P2pNcclConnector also failed (needs P2P DMA, not available on consumer PCIe)
- **0.5B smoke test with NixlConnector**: Both instances started, NIXL initialized, served requests successfully. Step-cycle traces collected (800 prefill + 1400 decode records). Confirms the emulator architecture works with PD disagg.

**PD disagg emulator eval (0.5B, b2b):**

| | Real | Emu | Error |
|---|---|---|---|
| TTFT | 139.3ms | 142.3ms | **+2.1%** |
| TPOT | 34.3ms | 35.8ms | **+4.4%** |

Both under 5%. Emulator works with PD disaggregation.

**Conclusion**: PD disagg works with NIXL on small models. Full eval (1.5B+) needs ≥24GB GPUs (A30/A100). Defer to CloudLab.

The emulator's approach is inherently compatible with PD disagg because:
1. Each vLLM instance (prefill/decode) has its own worker hook
2. The serving profile captures per-instance step cycle time including KV transfer
3. No emulator code changes needed for PD disagg

---

## Remaining Work

### Priority 1: Re-run all experiments with fixed profiler
- [ ] 0.5B sweep v13 + all benchmarks
- [ ] 1.5B BurstGPT cross-workload validation
- [ ] TP=2 evaluation (both models)

### Priority 2: Feature demos
- [ ] With/without chunked prefill
- [ ] With/without CUDA graphs (enforce-eager)
- [ ] KV offloading demo
- [ ] PD disaggregation demo

### Priority 3: Online serving evaluation
- [ ] `vllm bench serve` with TTFT/TPOT/P99 metrics
- [ ] BurstGPT trace replay via online serving

### Priority 4: Accelerated mode
- [ ] Implement virtual-time accumulation
- [ ] Compare speedup vs real execution

### Priority 5: Paper
- [ ] Update draft with results
- [ ] Figures: accuracy comparison, capability table
- [ ] Related work positioning vs REVATI/Vidur
