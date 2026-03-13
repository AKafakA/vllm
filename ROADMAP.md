# vLLM Emulator Backend - Project Roadmap

**Version:** 3.0
**Created:** 2026-02-26
**Last Updated:** 2026-03-13
**Status:** Active Development

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
| **C** | CPU↔GPU Interaction (KV Offload) | Modeled |
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
│   ├── platform.py              # EmulatorPlatform
│   ├── oracle/
│   │   ├── base.py             # Base oracle interface
│   │   ├── gpu_cost_oracle.py  # Category B ✅
│   │   ├── pd_separated_oracle.py  # PD separation ✅
│   │   ├── offload_cost_oracle.py   # Category C (stub)
│   │   └── network_cost_oracle.py   # Category D (stub)
│   ├── profile/
│   │   ├── loader.py           # Profile pack loader
│   │   └── validator.py        # Schema validation
│   ├── profiler/
│   │   ├── gpu_profiler.py     # GPU profiling
│   │   ├── offload_profiler.py # Offload profiling (stub)
│   │   └── network_profiler.py # Network profiling (stub)
│   ├── scheduler/
│   │   └── emulator_scheduler.py  # PD scheduling ✅
│   └── hooks/
│       └── gpu_hook.py         # Category B hooks ✅
├── tests/
│   ├── unit/
│   │   ├── test_gpu_cost_oracle.py
│   │   ├── test_pd_separation.py
│   │   └── test_emulator_profile_pack.py
│   └── integration/
│       ├── test_oracle_hook.py
│       └── test_timing_accuracy.py
└── docs/
    ├── rfc-design.md
    ├── user-guide.md
    └── api.md
```

---

## 5. Task List

### Phase 1: Foundation (P0) ✅ COMPLETE

| Task | Status | Commit ID |
|------|--------|-----------|
| P0.1: Cost Boundary Documentation | ✅ DONE | c00aa36 |
| P0.2: Platform Plugin Implementation | ✅ DONE | e567315 |
| P0.3: Profile Pack System | ✅ DONE | ef01678 |
| P0.4: Profile Generation Scripts | ✅ DONE | 012a785 |

---

### Phase 2: Core Oracles (P1)

#### P1.1: GPU Cost Oracle (Category B) ✅ DONE
- **Component:** `vllm/v1/worker/gpu_worker.py`
- **Hook Point:** `Worker.execute_model()`
- **Design:**
  - Oracle Hook at Worker level
  - Profile: prefill (prompt_tokens), decode (active_seqs)
  - Batch-level estimation
  - Supports: online (default) and offline (REVATI-like) blocking modes
- **Deliverables:**
  1. `vllm_emulator/oracle/base.py` - Base interface ✅
  2. `vllm_emulator/oracle/gpu_cost_oracle.py` ✅
  3. `vllm_emulator/hooks/gpu_hook.py` ✅
  4. Tests ✅
- **Commit ID:** 4ea9e32
- **Status:** ✅ DONE

#### P1.2: Offload Cost Oracle (Category C) 📅 SCHEDULED
- **Component:** `vllm/v1/kv_offload/`
- **Hook Points:**
  - `OffloadingManager.lookup()` 
  - `OffloadingWorker.transfer_async()` 
  - `ops.swap_blocks()`
- **Design:**
  - Oracle Hook for lookup/transfer/evict latency
  - Profile: lookup = f(num_blocks), transfer = f(bytes, direction)
- **Deliverables:**
  1. `vllm_emulator/oracle/offload_cost_oracle.py`
  2. `vllm_emulator/hooks/offload_hook.py`
- **Status:** NOT_STARTED
- **Scheduled:** March 14, 2026 (Tomorrow)

#### P1.3: Network Cost Oracle (Category D) 📅 SCHEDULED
- **Component:** `vllm/distributed/device_communicators/`, `kv_transfer/`
- **Hook Points:**
  - `cuda_communicator.all_reduce()`
  - `cuda_communicator.send()/recv()`
  - `KVConnectorBase.start_load_kv()`, `save_kv_layer()`
- **Design:**
  - Oracle Hook at Communicator level
  - Per-topology profiles (NVLink, PCIe, IB)
- **Deliverables:**
  1. `vllm_emulator/oracle/network_cost_oracle.py`
  2. `vllm_emulator/hooks/network_hook.py`
- **Status:** NOT_STARTED
- **Scheduled:** March 15, 2026 (Day after tomorrow)

---

### Phase 2: Advanced Features (P2.5.x)

| Task | Status | Commit ID |
|------|--------|-----------|
| P2.5.1: CUDA Graph Capture | ❌ NOT_STARTED | - |
| P2.5.2: PD Separation Support | ✅ DONE | 0d4b003 |
| P2.5.3: KV Offload Integration | ❌ NOT_STARTED | - |

#### P2.5.2: Prefill/Decode Separation Support ✅ DONE
- **Deliverables:**
  1. `vllm_emulator/oracle/pd_separated_oracle.py` ✅
  2. `vllm_emulator/scheduler/emulator_scheduler.py` ✅
  3. `tests/unit/test_pd_separation.py` ✅
  4. `examples/emulator/pd_separation_example.py` ✅
- **Features:**
  - PDSeparatedCostOracle for phase-specific timing
  - Multiple scheduling policies: prefill_first, decode_first, hybrid
  - Backward compatibility with joint scheduling
- **Commit ID:** 0d4b003
- **Status:** ✅ DONE

---

### Phase 3: Integration & Testing (P2) ✅ COMPLETE

| Task | Status | Commit ID |
|------|--------|-----------|
| P2.1: CLI Integration | ✅ DONE | 4ea9e32 |
| P2.2: Testing Infrastructure | ✅ DONE | 4ea9e32 |
| P2.3: Documentation | ✅ DONE | 0e7c026 |

---

### Phase 4: Paper & Upstream (P3)

| Task | Status |
|------|--------|
| P2.4: Workshop Paper | ❌ NOT_STARTED |
| vLLM Patch Submission | ❌ NOT_STARTED |

---

## 6. Trigger Mechanism (Env Vars)

The emulator is triggered via environment variables:

| Env Variable | Description | Values |
|--------------|-------------|--------|
| `VLLM_EMULATOR_ENABLE_ORACLE` | Enable emulator mode | `1`, `true`, `yes` |
| `VLLM_EMULATOR_PROFILE_PACK` | Path to profile pack | `/path/to/profile.json` |
| `VLLM_EMULATOR_BLOCKING_MODE` | Blocking behavior | `online` (default), `offline` |
| `VLLM_EMULATOR_MEMORY` | Emulated GPU memory (bytes) | e.g., `85899345920` (80GB) |

**CLI flags also available:**
- `--emulator-mode online|offline`
- `--profile-pack /path/to/profile`

---

## 7. Quick Start

```bash
# Enable emulator mode (default: online blocking)
export VLLM_EMULATOR_ENABLE_ORACLE=1
export VLLM_EMULATOR_PROFILE_PACK=examples/profiles/a100-sxm-80gb.json

# Online mode (default): block for estimated latency
# VLLM_EMULATOR_BLOCKING_MODE=online

# Offline mode: no blocking (for LLM.generate batch inference)
export VLLM_EMULATOR_BLOCKING_MODE=offline

# Run vLLM as normal
```

---

## 8. Timeline

| Date | Task |
|------|------|
| Mar 13, 2026 | Code review + Roadmap update |
| **Mar 14, 2026** | **P1.2: Offload Cost Oracle** |
| **Mar 15, 2026** | **P1.3: Network Cost Oracle** |
| Mar 16-20, 2026 | Integration testing |
| Mar 21-24, 2026 | Final testing before testbed expiry |
| Post-expiry | Workshop paper writing |

---

## 9. Workshop Paper Scope (Q1-Q2 2026)

**Target:** Workshop paper + vLLM upstream patch

**Included:**
- ✅ Online serving simulation (real-time blocking)
- ✅ Offline mode (REVATI-like, virtual time)
- ✅ GPU Cost Oracle (Category B)
- ✅ Chunked prefill support
- ✅ PD separation support
- ❌ KV Offload Oracle (defer to full paper)
- ❌ Network Oracle (defer to full paper)

---

## 10. Full Paper Scope (Q3-Q4 2026)

**Target:** Full paper (MLSys/SOSP/OSDI)

**Additional:**
- +KV Offload Oracle (Category C)
- +Network Oracle (Category D)
- +SGLang version
- +Extensive evaluation

---

## 11. Next Steps

- [x] Review and approve RFC design
- [x] P0.1-P0.4: Foundation complete
- [x] P1.1: GPU Cost Oracle complete
- [x] P2.1: CLI Integration complete
- [x] P2.2: Testing Infrastructure complete
- [x] P2.3: Documentation complete
- [x] P2.5.2: PD Separation complete
- [ ] Mar 14: P1.2 Offload Cost Oracle
- [ ] Mar 15: P1.3 Network Cost Oracle
- [ ] P2.4: Workshop Paper
