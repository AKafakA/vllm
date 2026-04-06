# Session Handoff Document — April 6, 2026

## Project Overview

**vLLM-Emulator**: hooks into vLLM v0.18.1's execution path to replace real GPU computation with profiled latency predictions (`time.sleep` / `threading.Timer`), enabling GPU-free LLM serving evaluation.

**Paper deadline**: NeurIPS 2025, May 6, 2026 (30 days remaining).

**Branch**: `feature/emulator-backend` (56+ commits ahead of remote)

---

## Current Architecture

### Path A: GPU Host Emulation (Vast RTX 3060)
- Real GPU available, emulator hooks at executor level
- `ExecutorEmulatorHook` intercepts `execute_model()` → returns timer-based pending Future
- Profile: step-cycle-based serving profile with prefill/decode sections + CUDA graph warmup
- Env vars: `VLLM_EMULATOR_ENABLE_ORACLE=1`, `VLLM_EMULATOR_EXECUTOR_HOOK=1`, `VLLM_EMULATOR_PROFILE_PACK=<path>`

### Path B: CPU-Only Host Emulation (CloudLab)
- No GPU, uses CUDA mock + EmulatorPlatform plugin
- Status: **BLOCKED** — server fails to start on CloudLab (`libcuda.so.1` not found, `DP adjusted local rank` assertion)
- The `platform.py` file in code root was renamed to `emulator_platform.py` to fix a stdlib collision
- Needs debugging of the CUDA mock activation path

### Key Files
| File | Purpose |
|------|---------|
| `vllm_emulator/hooks/executor_hook.py` | Timer-based Future approach, CUDA graph warmup |
| `vllm/v1/executor/uniproc_executor.py` | Hook integration, sample_tokens dispatch |
| `vllm/v1/engine/core.py` | TTFT tracer, step-cycle tracer, batch queue logic |
| `vllm_emulator/oracle/gpu_cost_oracle.py` | Profile interpolation (prefill/decode/2D) |
| `vllm_emulator/profile/validator.py` | Serving profile format acceptance |
| `paper/isca-2026-mlarchsys/scripts/temp/build_serving_profile_2d.py` | Profile builder from step-cycle trace |
| `paper/isca-2026-mlarchsys/scripts/temp/profile_ipc_overhead.py` | Two-pass IPC overhead profiling |
| `tools/e2e/compare_results.py` | Full E2E metrics comparison (TTFT, TPOT, E2E, throughput) |
| `tools/e2e/smoke_test_all.sh` | Multi-rate smoke test |
| `tools/e2e/smoke_test_1k.sh` | 1000-prompt evaluation |
| `docs/benchmarking/ttft-investigation.md` | Root cause analysis of TTFT gap |
| `docs/benchmarking/ttft-fix-proposals.md` | 5 proposals to fix TTFT |

---

## Current Accuracy (Timer Approach, CUDA Graph Warmup Only)

Tested on RTX 3060 12GB, Qwen/Qwen2.5-1.5B-Instruct, input=256, output=128:

| Metric | Rate=1 | Rate=2 | Rate=4 |
|--------|--------|--------|--------|
| **TPOT** | <5% ✓ | <5% ✓ | <5% ✓ |
| **Throughput** | <1% ✓ | <1% ✓ | <1% ✓ |
| **E2E Latency** | <7% | <5% ✓ | <5% ✓ |
| **TTFT** | -28% to -39% ✗ | -15% ✗ | +9% |

**Offline (rate=inf)**: TPOT +50% — profile doesn't cover high concurrency (tt>30).

---

## The TTFT Problem (Fully Investigated)

### Root Cause
vLLM's `AsyncScheduler` (default since v0.14) requires pipelining via pending Futures in the batch queue. The timer approach creates pending Futures (`done()=False`) which:
1. Allow the async scheduler's `num_output_placeholders` to work correctly ✓
2. BUT cause the engine to return early from `step_with_batch_queue` (line ~596) ✗
3. During the early return, the engine processes IPC → picks up new requests faster
4. This gives the emulator a TTFT advantage of ~20-40ms at low rates

### Why Blocking Doesn't Work
Any approach that removes pipelining (blocking sleep, BlockingFuture, no-pipeline flag) causes the async scheduler to deadlock:
- Without pipelining: `num_output_placeholders` drops to 0 between steps
- Scheduling formula: `num_new_tokens = spec + 0 - computed = 0`
- All requests get 0 tokens → scheduler stops → deadlock at 20-30 concurrent requests
- Confirmed by extensive debugging with `[BQStep]` and `[SchedDebug]` logs

### REVATI Comparison
REVATI (arxiv:2601.00397) solves this with **virtual time** — no real waiting during GPU execution. Their approach is correct by construction but requires deep CUDA API interception and multi-process virtual time synchronization. Our timer approach is simpler but has the pipelining artifact.

### Proposed Fix (Not Yet Implemented)
**Timer Duration Compensation** (Proposal 2 from `ttft-fix-proposals.md`):
- Add `avg_decode_step_cycle` to prefill timer when concurrent requests exist (`num_reqs > 1`)
- This models the "waiting for previous pipeline step" that real GPU does
- Profile-driven, no manual tuning
- Only affects TTFT (prefill), not TPOT (decode)
- Concern: doesn't apply to single isolated requests (correct — no pipeline overlap for isolated requests)

---

## Infrastructure

### Vast GPU Host (RTX 3060 12GB × 2)
- SSH: `ssh vast` (config alias, socket at `~/.ssh/sockets/vast-*`)
- Code: `/workspace/vllm-emulator-v18/`
- Venv: `/workspace/vllm-v18-env/`
- Profiles: `/workspace/eval_results/RTX-3060-12GB/profiles/`
- Results: `/workspace/eval_results/RTX-3060-12GB/online/`
- Step-cycle trace: `/workspace/eval_results/RTX-3060-12GB/step_cycle_1.5b_full.jsonl`
- IPC overhead: `/workspace/eval_results/RTX-3060-12GB/profiles/ipc_overhead.json` (50 entries, N=1..50)
- IPC real TTFT: `/workspace/eval_results/RTX-3060-12GB/profiles/ipc_real_ttft.json`
- IPC emu TTFT: `/workspace/eval_results/RTX-3060-12GB/profiles/ipc_emu_ttft.json`
- **Vast has ~20 hours of GPU credits remaining** (as of Apr 5)

### CloudLab CPU Hosts
- hp140: `ssh -o ControlPath=~/.ssh/sockets/cloudlab-hp140 asdwb@hp140.utah.cloudlab.us`
- hp123: `ssh -o ControlPath=~/.ssh/sockets/cloudlab-hp123 asdwb@hp123.utah.cloudlab.us`
- 20 cores, 62GB RAM, no GPU
- Code: `~/vllm-emulator/code/`
- Venv: `~/vllm-emulator-gpu/venv/` (GPU PyTorch wheel on CPU host)
- Profile copied to: `~/eval_results/pathb/profiles/serving-1.5b-tp1-calibrated.json`
- **Path B is blocked** — CUDA mock not activating properly

### SSH Permissions
Project settings at `~/.claude/projects/-home-wd312-Code-llm-vllm-emulator/settings.json`:
```json
{"permissions": {"defaultMode": "dontAsk"}}
```
**Important**: Use literal paths (not `$HOME`) in SSH commands to avoid "simple_expansion" blocks. Always write scripts locally and rsync, never run Python directly via SSH.

---

## What's Working

1. **Timer approach with CUDA graph warmup** — TPOT <5%, throughput <1% at all online rates
2. **Profile builder** (`build_serving_profile_2d.py`) — builds calibrated profile from step-cycle trace
3. **TTFT tracer** (`_TTFTTracer` in core.py) — traces per-request timing breakdown
4. **E2E comparison tool** (`compare_results.py`) — full latency + throughput + E2E comparison
5. **IPC overhead profiling** (`profile_ipc_overhead.py`) — two-pass N-sweep (real vs emu)
6. **Scheduler debug logging** (`[BQStep]`, `[SchedDebug]`, `[FakeOutput]`) — extensive tracing

---

## What's Not Working / Needs Fix

### Priority 1: TTFT at Low Rates
- Timer pipelining causes -28% to -39% TTFT error at rate=1
- Proposed fix: Timer Duration Compensation (Proposal 2) — not yet implemented
- See `docs/benchmarking/ttft-fix-proposals.md`

### Priority 2: Path B (CPU-Only)
- CloudLab server fails to start with CUDA mock
- Error: `libcuda.so.1: cannot open shared object file` and `DP adjusted local rank 0 is out of bounds`
- `VLLM_EMULATOR_MOCK_CUDA=1` env var flagged as "Unknown"
- `platform.py` renamed to `emulator_platform.py` (was shadowing stdlib)
- Need to debug the CUDA mock activation + EmulatorPlatform plugin loading

### Priority 3: Offline Mode (rate=inf)
- TPOT +50% error — profile doesn't have data for tt>30
- Need to profile at higher rates (rate=16, 32) or extrapolate more carefully
- The step-cycle trace was captured at rates 0.5-8, covering tt≈1-25

### Priority 4: Feature Demos
- PD separation: `examples/emulator/pd_separation_example.py` — not tested
- CPU offload: `tools/e2e/run_cpu_offload_demo.sh` — not tested
- These need working Path B (CPU-only) first

### Priority 5: Multi-GPU / Multi-Model
- TP=2 (3B model) not retested with corrected code
- 0.5B model not retested
- BurstGPT trace evaluation not started

---

## Key Debugging Findings

### Async Scheduler Placeholder Mechanism
```python
# AsyncScheduler._post_schedule() — called during schedule()
request.num_output_placeholders += 1  # Expects in-flight batch

# AsyncScheduler._update_request_with_output() — called during update_from_output()  
request.num_output_placeholders -= len(new_token_ids)  # Batch completed

# Scheduling formula:
num_new_tokens = num_tokens_with_spec + num_output_placeholders - num_computed_tokens
```
- With pipelining: placeholders oscillate 1→2→1 → `num_new_tokens=1` ✓
- Without pipelining: placeholders oscillate 1→0 → `num_new_tokens=0` → **DEADLOCK**
- Confirmed: ANY approach without pipelining deadlocks at 20-30 concurrent requests

### CUDA Graphs
- Captured once at startup, never recompiled during serving
- Not the cause of TTFT gap (initially hypothesized, then disproved)
- CUDA graph warmup model (`cuda_graph_warmup_us`) still valid for startup overhead

### N-Sweep IPC Profiling
- Two-pass approach: real GPU TTFT vs emulator TTFT at each concurrency N=1..30
- Finding: emulator is actually SLOWER in N-sweep (58ms vs 37ms) — the pipelining advantage only appears during ONLINE serving, not isolated requests
- The N-sweep creates idealized conditions that don't replicate online serving dynamics

### Real GPU TTFT Variance
- Real TTFT at rate=1 varied from 84ms (early session) to 136ms (later session)
- Likely from GPU thermal state / Vast host contention
- Emulator TTFT is consistent (~63-75ms) regardless of session

---

## Uncommitted / Cleanup Needed

### Debug Logging Still in Code
- `[BQStep]` logging in `vllm/v1/engine/core.py` (step_with_batch_queue)
- `[SchedDebug]` logging in `vllm/v1/core/sched/scheduler.py`
- `[FakeOutput]` logging in `vllm_emulator/hooks/executor_hook.py`
- `[TTFT-TRACE]` tracer in `vllm/v1/engine/core.py`
- These should be removed or made conditional before production use

### Files on Vast Not Synced Back
- IPC profiling results: `ipc_overhead.json`, `ipc_real_ttft.json`, `ipc_emu_ttft.json`
- Various benchmark results in `/workspace/eval_results/RTX-3060-12GB/online/`
- Server logs for debugging: `/workspace/ttft_trace_*.log`, `/workspace/diag_*.log`

### Stale Test Scripts
Many one-off test scripts in `tools/e2e/` from debugging iterations:
- `diag_deadlock.sh`, `diag_single_req.sh`, `diag_warmup_prompt.sh`
- `rate1_thorough.sh`, `rate1_no_heavy_warmup.sh`, `rate1_a2a.sh`
- `rate_independent_test.sh`, `b2b_blocking.sh`
- Consider cleaning up before final evaluation

---

## Recommended Next Steps (In Priority Order)

### 1. Implement Timer Duration Compensation (Proposal 2)
- Add `pipeline_compensation_us` to prefill timer when `num_reqs > 1`
- Compute from profile's avg decode step cycle at low tt
- Test at rates 1, 2, 4 with 1000 prompts
- Expected: TTFT improves to <10% at rate=1 without affecting TPOT

### 2. Fix Path B on CloudLab
- Debug CUDA mock activation (`VLLM_EMULATOR_MOCK_CUDA=1` not recognized)
- Check if `emulator_platform.py` rename broke the plugin loading
- Test with `--enforce-eager` flag (no CUDA graphs on CPU)
- Run Path B smoke test at rates 1, 2, 4

### 3. Fix Offline Mode
- Profile at higher rates (rate=16, 32) to get step-cycle data for tt>30
- Or: add entries to forward_pass section for high tt from sweep profile
- Retest offline throughput

### 4. Clean Up and Commit
- Remove debug logging (or make conditional on env var)
- Remove stale test scripts
- Commit clean version for evaluation

### 5. Full Evaluation (1000 Prompts)
- Rate 1, 2, 3, 4 on Vast (Path A)
- Same on CloudLab (Path B, once fixed)
- Offline throughput
- Feature demos (PD separation, CPU offload)
- Multiple GPU classes (RTX 3060, A100 on CloudLab when available)

### 6. Paper Writing
- Results table: TPOT, TTFT, E2E, throughput at each rate
- TTFT limitation discussion (async scheduler constraint)
- Comparison with REVATI, Vidur
- Path A vs Path B accuracy comparison

---

## Git State

```
Branch: feature/emulator-backend
Latest commits:
  f5fc2626a docs: TTFT fix proposals — 5 approaches with code evidence
  a186e84e2 analysis: TTFT gap root cause — async scheduler requires pipelining
  8d508fb39 feat: TTFT trace diagnostic — reveals CUDA graph compilation as root cause
  4d9983a0c feat: measured IPC overhead profiling — profile-driven TTFT calibration
  36fece949 feat: auto-calibrated scheduling delay + CUDA graph warmup — rate=1 all metrics <5%
```

Key older commits (for reference):
- `14f97ec5f` fix: step_overhead=1000us — TPOT <4% all rates, TTFT <6% rate>=2
- `371dd4b1a` (referenced in conversation) — original proven timer approach

Modified files not yet committed:
- Debug logging in `core.py` and `scheduler.py`
- Various test scripts in `tools/e2e/`
