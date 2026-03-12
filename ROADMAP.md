# vLLM Emulator Backend - Project Roadmap

**Version:** 2.0 (RFC)
**Created:** 2026-02-26
**Status:** PENDING APPROVAL

---

## 1. Executive Summary

This roadmap implements the vLLM Emulator Backend as defined in `docs/rfc-design.md`.

**Goals:**
1. Enable rapid iteration on scheduling algorithms without GPU hardware
2. Support A/B testing of scheduling policies
3. Accelerate research on LLM serving systems

---

## 2. Cost Boundary Definition

| Category | Description | Behavior |
|----------|-------------|----------|
| **A** | API/Scheduler/Tokenization | Run Real |
| **B** | GPU Compute | Modeled |
| **C** | CPU↔GPU Interaction | Modeled |
| **D** | Network (TP/PP/PD) | Modeled |

---

## 3. Component Reviews

| Category | Component | Review File | Status |
|----------|-----------|-------------|--------|
| A | API/Scheduler | `docs/review-integration.md` | ✅ |
| B | GPU Worker | `docs/review-gpu-worker.md` | ✅ |
| C | KV Offload | `docs/review-kv-offload.md` | ✅ |
| D | Network | `docs/review-network.md` | ✅ |
| Integration | Engine/API/Benchmark | `docs/review-integration.md` | ✅ |

---

## 4. Project Structure

```
vllm-emulator/
├── vllm_emulator/
│   ├── __init__.py
│   ├── platform.py          # EmulatorPlatform
│   ├── oracle/
│   │   ├── base.py          # Base oracle interface
│   │   ├── gpu_cost_oracle.py    # Category B
│   │   ├── offload_cost_oracle.py # Category C
│   │   └── network_cost_oracle.py # Category D
│   ├── profile/
│   │   ├── loader.py        # Profile pack loader
│   │   └── validator.py    # Schema validation
│   └── hooks/
│       ├── gpu_hook.py     # Category B hooks
│       ├── offload_hook.py # Category C hooks
│       └── network_hook.py  # Category D hooks
├── tests/
│   ├── unit/
│   └── integration/
└── docs/
    ├── rfc-design.md        # This document
    └── ...
```

---

## 5. Task List

### Phase 1: Foundation (P0)

#### P0.1: Cost Boundary Documentation ✅ DONE
- **Task:** Write `docs/cost-boundary.md` defining A/B/C/D boundaries
- **Acceptance:** 
  - Document lists all modules in categories A/B/C/D
  - Clear explanation of what runs real vs modeled
  - Both English and Chinese versions
- **Review:** Based on component reviews
- **Commit ID:** TBD
- **Status:** ✅ DONE

#### P0.2: Platform Plugin Implementation ✅ DONE
- **Task:** Create `EmulatorPlatform` class and register via entry point
- **Acceptance:**
  - Platform loads via `VLLM_PLUGINS=emulator`
  - Device memory returns fake value
  - Basic initialization works
- **Deliverables:**
  1. `vllm_emulator/platform.py` - EmulatorPlatform class ✅
  2. `vllm_emulator/__init__.py` - Package init ✅
  3. `pyproject.toml` - Entry point registration ✅
- **Commit ID:** e567315
- **Status:** ✅ DONE

#### P0.3: Profile Pack System ✅ DONE
- **Task:** Define and implement profile pack loading
- **Acceptance:**
  - Can load JSON profile pack
  - Schema validation works
  - Example profile packs created
- **Deliverables:**
  1. `vllm_emulator/profile/loader.py` ✅
  2. `vllm_emulator/profile/validator.py` ✅
  3. Example profiles in `examples/profiles/` ✅
- **Commit ID:** TBD
- **Status:** ✅ DONE

#### P0.4: Profile Generation Scripts - REQUIRED ✅ DONE
- **Task:** Create profiling scripts to generate profile packs from real GPU runs
- **Acceptance:**
  - Script can run on real GPU and collect latency profiles
  - Output format matches profile pack schema
  - Supports different GPU models (A100, H100, etc.)
- **Deliverables:**
  1. `vllm_emulator/profiler/gpu_profiler.py` - GPU compute profiling ✅
  2. `vllm_emulator/profiler/offload_profiler.py` - Offload profiling (stub) ✅
  3. `vllm_emulator/profiler/network_profiler.py` - Network profiling (stub) ✅
- **Reference:** Vidur profiler design
- **Commit ID:** 241130a
- **Status:** ✅ DONE

---

### Phase 2: Core Oracles (P1)

**Note:** Category C (Offload) and D (Network) are marked as OPTIONAL. They can be added in Phase 2+ if needed.

#### P1.1: GPU Cost Oracle (Category B) - REQUIRED ✅ DONE
- **Component:** `vllm/v1/worker/gpu_worker.py`
- **Hook Point:** `Worker.execute_model()` (line 582)
- **Design:**
  - Oracle Hook at Worker level
  - Profile: prefill (prompt_tokens), decode (active_seqs)
  - Batch-level estimation (not per-request)
  - Supports two blocking modes: online (default) and offline (REVATI-like)
- **Acceptance:**
  - [x] Oracle interface defined
  - [x] Can load profile pack
  - [x] Prefill latency = f(prompt_tokens)
  - [x] Decode latency = f(active_seqs)
  - [x] Can fallback to real execution
  - [x] Batch-level timing (not per-request)
  - [x] Online blocking mode: time.sleep() for real-time simulation
  - [x] Offline mode: no blocking (virtual time)
  - [x] Warning for LLM.generate + online mode
- **Deliverables:**
  1. `vllm_emulator/oracle/base.py` - Base interface ✅
  2. `vllm_emulator/oracle/gpu_cost_oracle.py` ✅
  3. `vllm_emulator/hooks/gpu_hook.py` ✅
  4. `tests/unit/test_gpu_cost_oracle.py` ✅
  5. `vllm/entrypoints/llm.py` - Warning for offline path ✅
- **Commit ID:** 4ea9e32
- **Status:** ✅ DONE

#### P1.2: Offload Cost Oracle (Category C) - OPTIONAL
- **Component:** `vllm/v1/kv_offload/`
- **Status:** OPTIONAL - Can be added in Phase 2 if needed
- **Hook Points:**
  - `OffloadingManager.lookup()` (abstract.py)
  - `OffloadingWorker.transfer_async()` (worker.py)
  - `ops.swap_blocks()` (cpu_gpu.py)
- **Design:**
  - Mock OffloadingManager OR Oracle Hook
  - Profile: lookup, transfer, evict
- **Acceptance:**
  - [ ] Oracle interface defined
  - [ ] Can load profile pack
  - [ ] Lookup latency = f(num_blocks)
  - [ ] Transfer latency = f(bytes, direction, concurrency)
  - [ ] Can fallback to real execution
- **Deliverables:**
  1. `vllm_emulator/oracle/offload_cost_oracle.py`
  2. `vllm_emulator/hooks/offload_hook.py`
- **Commit ID:** TBD
- **Status:** NOT_STARTED

#### P1.3: Network Cost Oracle (Category D) - OPTIONAL
- **Component:** `vllm/distributed/device_communicators/`, `kv_transfer/`
- **Status:** OPTIONAL - Most research uses single-GPU, can be added later if needed
- **Hook Points:**
  - `cuda_communicator.all_reduce()` (line 130)
  - `cuda_communicator.send()` (line 240)
  - `cuda_communicator.recv()` (line 252)
  - `KVConnectorBase.start_load_kv()`, `save_kv_layer()`
- **Design:**
  - Oracle Hook at Communicator level
  - Per-topology profiles (NVLink, PCIe, IB)
  - Profile: all_reduce, send_recv, kv_transfer
- **Acceptance:**
  - [ ] Oracle interface defined
  - [ ] Can load topology-specific profiles
  - [ ] All-reduce latency = f(bytes, world_size, topology)
  - [ ] Send/Recv latency = f(bytes, topology)
  - [ ] KV transfer latency = f(bytes, direction, topology, concurrency)
  - [ ] Can fallback to real execution
- **Deliverables:**
  1. `vllm_emulator/oracle/network_cost_oracle.py`
  2. `vllm_emulator/hooks/network_hook.py`
- **Commit ID:** TBD
- **Status:** NOT_STARTED

---

### Phase 2: Advanced Features (Before Workshop Paper)

**Rationale:** Complete these before writing workshop paper to allow testing rounds before testbed expires (March 24, 2026).

#### P2.5.1: Cuda Graph Capture
- **Task:** Implement CUDA graph capture for accurate timing
- **Acceptance:**
  - Can capture CUDA graphs during profiling
  - Replays captured graphs in emulator
  - Improves timing accuracy
- **Deliverables:**
  1. CUDA graph capture utility
  2. Replay mechanism
- **Commit ID:** TBD
- **Status:** NOT_STARTED

#### P2.5.2: Prefill/Decode Separation Support
- **Task:** Add support for PD separation (vLLM v1 engine)
- **Acceptance:**
  - Can profile prefill and decode separately
  - Can emulate PD-separated scheduling
  - Accurate latency for both phases
- **Deliverables:**
  1. PD separation oracle hooks
  2. Scheduling emulation
- **Commit ID:** TBD
- **Status:** NOT_STARTED

#### P2.5.3: KV Cache Offload Integration
- **Task:** Integrate KV offload cost modeling
- **Acceptance:**
  - Accurate offload/recall timing
  - Works with profile packs
- **Deliverables:**
  1. Offload oracle integration
- **Commit ID:** TBD
- **Status:** NOT_STARTED

---

### Phase 3: Integration & Testing (P2)

#### P2.1: CLI Integration ✅ DONE
- **Task:** Add CLI flags for emulator mode
- **Acceptance:**
  - [x] `--emulator-mode` flag works (online/offline)
  - [x] `--profile-pack` flag works
  - [x] Help text updated (EmulatorConfig argument group)
  - [x] Environment variable fallback (VLLM_EMULATOR_BLOCKING_MODE, VLLM_EMULATOR_PROFILE_PACK)
  - [x] CLI flags propagate to env vars for worker processes
  - [x] Validation: --emulator-mode requires --profile-pack
- **Deliverables:**
  1. `vllm/engine/arg_utils.py` - EngineArgs.emulator_mode, profile_pack fields + _resolve_emulator_args() ✅
  2. `vllm/entrypoints/openai/cli_args.py` - Validation in validate_parsed_serve_args() ✅
- **Commit ID:** 4ea9e32
- **Status:** ✅ DONE

#### P2.2: Testing Infrastructure (MVP) ✅ DONE
- **Task:** Build A/B comparison pipeline and integration tests
- **Acceptance:**
  - [x] Oracle hook integration tests (env var init, fake output, enable/disable)
  - [x] Online blocking mode tested (time.sleep for estimated latency)
  - [x] Offline mode tested (no blocking / virtual time)
  - [x] Latency interpolation error < 15% threshold validated
  - [x] Boundary clamping, monotonicity, round-trip tests
  - [x] A/B comparison harness with accuracy report
- **Deliverables:**
  1. `tests/integration/test_oracle_hook.py` - Hook integration tests (14 tests) ✅
  2. `tests/integration/test_timing_accuracy.py` - Accuracy validation (21 tests) ✅
  3. `tests/integration/ab_comparison_harness.py` - A/B comparison script ✅
- **Commit ID:** 4ea9e32
- **Status:** ✅ DONE

---

### Test Plan (P1.1 - GPU Cost Oracle)

**Unit Tests:**
- `tests/unit/test_gpu_cost_oracle.py` - Oracle interpolation ✅ (exists)
- Profile pack validation tests

**Integration Tests:**
1. **Oracle Hook Integration** (`tests/integration/test_oracle_hook.py`)
   - Enable via env vars: `VLLM_EMULATOR_ENABLE_ORACLE=1`, `VLLM_EMULATOR_PROFILE_PACK=<path>`
   - Verify fake output is returned when enabled
   - Verify `time.sleep()` blocks for estimated latency
   - Verify fallback to real execution when disabled

2. **Timing Accuracy Test** (`tests/integration/test_timing_accuracy.py`)
   - Run emulator with known profile pack
   - Compare emulated latency vs profile pack values
   - Target: <15% error on decode path

3. **Continue Batching Test** (`tests/integration/test_continue_batching.py`)
   - Submit multiple requests with delays
   - Verify scheduler correctly waits for batch completion
   - Verify pending requests are properly queued

**Test Commands:**
```bash
# Unit tests
uv run pytest tests/unit/test_gpu_cost_oracle.py -v

# Integration tests (require GPU)
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK=examples/profiles/a100-sxm-80gb.json \
uv run pytest tests/integration/ -v

# Online mode (default): blocking for real-time simulation
VLLM_EMULATOR_BLOCKING_MODE=online

# Offline mode: no blocking (for LLM.generate batch inference)
VLLM_EMULATOR_BLOCKING_MODE=offline
```

#### P2.3: Documentation ✅ DONE
- **Task:** Complete user documentation
- **Acceptance:**
  - User guide complete
  - API documentation complete
  - Examples provided
- **Deliverables:**
  1. `docs/user-guide.md`
  2. `docs/api.md`
  3. `examples/`
- **Commit ID:** 0e7c026
- **Status:** NOT_STARTED

#### P2.4: Workshop Paper
- **Task:** Write workshop paper for ArXiv preprint
- **Acceptance:**
  - Paper follows ACM SIGCONF template
  - Includes evaluation methodology
  - Submitted to ArXiv
- **Deliverables:**
  1. Paper LaTeX
  2. ArXiv submission
- **Commit ID:** TBD
- **Status:** NOT_STARTED

---

## 6. Design Change Log

| Version | Date | Change | Reason | Approved By |
|---------|------|--------|--------|-------------|
| 1.0 | 2026-02-26 | Initial roadmap | - | David |
| 2.0 | 2026-02-26 | RFC design | Based on component reviews | Pending |

---

## 7. Acceptance Criteria Summary

| Phase | Task | Key Acceptance |
|-------|------|----------------|
| P0 | P0.1 Cost Boundary | Docs written |
| P0 | P0.2 Platform Plugin | Platform loads |
| P0 | P0.3 Profile System | Can load profiles |
| P1 | P1.1 GPU Oracle | Latency accuracy <15% |
| P1 | P1.2 Offload Oracle | Transfer accuracy |
| P1 | P1.3 Network Oracle | Network accuracy |
| P2 | P2.5.1 Cuda Graph | Capture/replay works |
| P2 | P2.5.2 PD Separation | Prefill/decode separate |
| P2 | P2.5.3 KV Offload | Offload modeling works |
| P2 | P2.1 CLI | Flags work |
| P2 | P2.2 Testing | A/B comparison works |
| P2 | P2.3 Docs | Complete |
| P2 | P2.4 Workshop Paper | ArXiv submitted |

## 8. Timeline

- **Testbed Expiry:** March 24, 2026
- **Goal:** Complete P2.5.1-2.5.3 before testbed expiry for potential testing round
- **Workshop Paper:** After advanced features, before July 2026

---

## 9. Next Steps

1. [x] Review and approve RFC design
2. [x] P0.1-P0.4: Foundation complete
3. [x] P1.1: GPU Cost Oracle complete (with timing simulation)
4. [x] P2.1: CLI Integration
5. [x] P2.2: Testing Infrastructure (run integration tests)
6. [ ] Continue with remaining roadmap tasks

### Quick Start (P1.1)
```bash
# Enable emulator mode (default: online blocking)
export VLLM_EMULATOR_ENABLE_ORACLE=1
export VLLM_EMULATOR_PROFILE_PACK=examples/profiles/a100-sxm-80gb.json

# For online serving (default): block for estimated latency
# VLLM_EMULATOR_BLOCKING_MODE=online  # (default)

# For offline batch inference: no blocking (faster)
export VLLM_EMULATOR_BLOCKING_MODE=offline

# Run vLLM as normal
```

---

## Workshop + vLLM Patch Scope (Q1-Q2 2026)

**Target:** Workshop paper + vLLM upstream patch

**Scope:**
- ✅ Online serving simulation (real-time blocking)
- ✅ Offline mode (REVATI-like, virtual time)
- ✅ GPU Cost Oracle (Category B)
- ✅ Chunked prefill support
- ❌ PD separation (defer to full paper)
- ❌ KV Offload Oracle (defer)
- ❌ Network Oracle (defer)

**Timeline:**
1. Week 1: Integration tests + accuracy validation
2. Week 2: CLI integration + polish
3. Week 3: Paper writing
4. Week 4: vLLM patch submission

---

## Full Paper Scope (Q3-Q4 2026, post-CARA)

**Target:** Full paper (MLSys/SOSP/OSDI)

**Scope (extension):**
- +KV Offload Oracle (Category C)
- +Network Oracle (Category D)  
- +PD separation support
- +SGLang version
- +Extensive evaluation (multiple models, workloads, policies)

**Trigger:** CARA testing window (est. ~2 months)
