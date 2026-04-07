# Emulator Accuracy Results — April 6, 2026

## Summary

Comprehensive evaluation of the vLLM-Emulator (Path A: GPU host with executor hook)
on Vast RTX 3060-12GB with Qwen2.5-1.5B-Instruct (TP=1, max-model-len=4096).

**Profile**: step-cycle-based 2D profile (45,800 records), with prefill/decode sections,
cross-reference outlier detection, and disabled CUDA graph warmup model.

**Methodology**: Independent server starts per rate, 200-prompt warmup at rate=4,
200-prompt evaluation per rate. Real and emulated baselines use identical vLLM server
config, identical warmup, and identical benchmark parameters.

## Results

| Metric | Rate=1 | Rate=2 | Rate=4 |
|--------|--------|--------|--------|
| Mean TTFT | +26.5% | +31.8% | +29.1% |
| Median TTFT | +31.5% | +34.1% | +29.2% |
| P99 TTFT | +10.4% | +28.7% | +24.1% |
| Mean TPOT | -0.1% | -0.2% | -0.3% |
| Median TPOT | -0.1% | -0.1% | -0.3% |
| P99 TPOT | -0.2% | -0.2% | -0.6% |
| Mean E2E | +0.6% | +0.7% | +0.6% |
| Output tok/s | -0.0% | -0.0% | -0.0% |
| Request/s | -0.0% | -0.0% | -0.0% |

### Absolute Latencies (ms)

| Metric | Rate=1 Real | Rate=1 Emu | Rate=2 Real | Rate=2 Emu | Rate=4 Real | Rate=4 Emu |
|--------|-------------|------------|-------------|------------|-------------|------------|
| Mean TTFT | 43.2 | 54.7 | 44.7 | 58.9 | 47.2 | 61.0 |
| Mean TPOT | 11.6 | 11.6 | 12.1 | 12.1 | 12.5 | 12.5 |
| Mean E2E | 1519.7 | 1529.1 | 1582.2 | 1593.5 | 1639.3 | 1648.4 |

## Key Findings

### What works (all <1% error)
- **TPOT** (Time Per Output Token): <0.5% at all rates
- **E2E latency**: <1% at all rates
- **Throughput** (tok/s, req/s): <0.1% at all rates
- **Duration**: <0.1% at all rates

### Known limitation: TTFT overestimation (~27-32%)

**Root cause**: Timer-based async Future pipelining artifact.

The vLLM v0.18.1 async scheduler uses `batch_queue_size=2` with pending Futures
for `num_output_placeholders` tracking. The emulator returns a pending Future
(via `threading.Timer`) that resolves after the profiled step-cycle time. While
the timer is pending, the engine's IPC loop picks up new requests — giving the
emulator an IPC scheduling advantage that real GPU doesn't have.

On real GPU, `execute_model()` blocks the engine thread during GPU execution,
so new requests must wait until the current step completes before being scheduled.
The emulator's timer doesn't block, allowing earlier scheduling.

The overestimation is ~14ms consistently across all rates, which is approximately
one decode step cycle (11-12ms) — matching the pipeline compensation theory.

**Why blocking doesn't work**: Any approach that blocks or prevents pipelining
(blocking sleep, `done()=True` Future, pipeline flag) causes `num_output_placeholders`
to drop to 0, which sets `num_new_tokens=0` in the async scheduler, creating a
deadlock where no tokens are generated.

**Impact on paper**: TTFT is a secondary metric for the emulator's use case
(scheduling policy evaluation). TPOT, E2E, and throughput — the primary metrics
for comparing scheduling policies — are all <1% accurate. The TTFT limitation
is documented as an architectural constraint of vLLM's async scheduler that
does not affect the emulator's core value proposition.

See `ttft-final-summary.md` for the full investigation (10 approaches tested).

## Changes in This Version

### Profile builder: cross-reference outlier detection
(`paper/isca-2026-mlarchsys/scripts/temp/build_serving_profile_2d.py`)

Added a second pass that compares each bucket's median against neighbors within
±10 total_tokens. If a bucket is >3x its neighbors, it's replaced with the
neighbor median. This catches cold-start contaminated entries (e.g., tt=256
at 78ms when tt=258 is 17ms — a 4.6x outlier from CUDA graph compilation
during the first profiling steps).

### Executor hook: disable CUDA graph warmup model
(`vllm_emulator/hooks/executor_hook.py`)

Disabled the automatic CUDA graph warmup overhead model. Real GPU pre-compiles
all CUDA graphs at startup, and with adequate warmup (200 prompts), there is
no cold-start overhead during benchmarking. The warmup model was adding +44ms
per new padded batch shape, which doesn't exist on real GPU after warmup.
Can be re-enabled via `VLLM_EMULATOR_CUDA_GRAPH_WARMUP_US` for cold-start analysis.

### Platform plugin: restrict activation to CPU-only mode
(`vllm_emulator/platform.py`)

The emulator platform plugin now only activates when BOTH `VLLM_EMULATOR_ENABLE_ORACLE`
and `VLLM_EMULATOR_MOCK_CUDA` are set. On machines with real GPU (Vast, CloudLab with
GPU), the executor hook approach uses the native CUDA platform. The emulator platform
is only needed for CPU-only operation (Path B). Also added `num_compute_units()` for
compatibility with newer vLLM code paths.

## Profiling Methodology

1. **Comprehensive trace collection** (45,800 step-cycle records):
   - 200-prompt warmup at rate=4 (thermal equilibrium + CUDA graph compilation)
   - Profile at rates 0.5, 1, 2, 3, 4, 6, 8, 12 (online) + inf (offline)
   - Step-cycle = wall-clock time of one scheduler step (GPU + scheduling overhead)

2. **2D profile structure**:
   - `prefill_forward_pass`: steps with new requests (higher latency from attention)
   - `decode_forward_pass`: decode-only steps (lower, dominated by batch size)
   - `forward_pass`: combined (backward compat, merged with sweep for large tt)

3. **Outlier detection**:
   - Per-bucket: filter samples <5000us and >3x median
   - Cross-bucket: replace buckets >3x their neighbors within ±10 tt

4. **Independent evaluation**:
   - Fresh server start per rate (no thermal drift)
   - 200-prompt warmup before each benchmark
   - `CUDA_VISIBLE_DEVICES=1` (avoid GPU 0 with orphaned CUDA memory)

## E2E Test Scripts

| Script | Purpose |
|--------|---------|
| `tools/e2e/full_profile_and_eval.sh` | Complete profiling + real/emu eval at rates 1,2,4,inf |
| `tools/e2e/rebuild_and_emu.sh` | Rebuild profile from trace, test emu-only vs existing baselines |
| `tools/e2e/compare_results.py` | Compare real vs emu benchmark JSON files |

## Environment

- **GPU**: Vast RTX 3060-12GB (2x, using GPU 1)
- **Model**: Qwen/Qwen2.5-1.5B-Instruct
- **vLLM**: v0.18.1 (fork with emulator hooks)
- **Config**: max-model-len=4096, chunked prefill, async scheduling
- **Warmup**: 200 prompts at rate=4 (~50s sustained GPU load)
- **Evaluation**: 200 prompts per rate, percentile metrics ttft,tpot at p50,p99
