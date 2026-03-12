# vLLM Emulator Backend - RFC Design Document

**Version:** 1.0 (RFC)
**Date:** 2026-02-26
**Status:** DRAFT - Pending Approval

---

## 1. Executive Summary

This document describes the design for a vLLM Emulator Backend that enables fast, accurate simulation of LLM inference workloads without requiring real GPU hardware. The emulator models GPU compute costs, CPU↔GPU interactions, and network operations to predict and throughput.

** latencyGoals:**
1. Enable rapid iteration on scheduling algorithms without GPU hardware
2. Support A/B testing of scheduling policies
3. Accelerate research on LLM serving systems

---

## 2. Problem Statement

### 2.1 Background
- Real GPU clusters are expensive and scarce
- Scheduling algorithm research requires extensive experimentation
- Current approaches: (1) simulation, (2) small-scale real testing

### 2.2 Related Work Landscape (Updated)

Recent work is no longer a single-baseline (Vidur-only) landscape. It now has three families:

1. **System simulators**: Vidur, LLMServingSim, Frontier, LLMServingSim 2.0, APEX
   - strong for broad what-if studies and search;
   - typically rely on explicit simulator abstractions and calibration.

2. **Code-path-preserving emulation**: REVATI
   - runs real serving framework control logic with GPU execution virtualized (time-warp);
   - strongest overlap with our original motivation.

3. **Configuration optimizer/estimator**: AIConfigurator
   - high-speed configuration recommendations under SLO constraints;
   - narrower than full runtime simulation/emulation, but strong on practical search productivity.

### 2.3 Positioning and Differentiation (Revised)

Given this landscape, this RFC should **not** claim novelty from "GPU-free exploration" alone.

**Our intended differentiation is:**
1. **vLLM-native decomposed emulation boundary (A/B/C/D)**:
   - A (API/scheduler/tokenization) runs real;
   - B/C/D are modeled explicitly via independent oracles.
2. **Mechanism-level explainability**:
   - prediction is decomposable into compute/offload/network contributions.
3. **Diagnosis-oriented workflow**:
   - errors can be localized by component (B/C/D) instead of only reporting aggregate TTFT/TPOT gaps.
4. **Research iteration support**:
   - pluggable oracles and profile packs for rapid scheduler what-if studies in vLLM-focused research.
5. **Online serving focus (key differentiator from REVATI)**:
   - **Primary LLM application is online serving**, not offline batch inference.
   - We support **real-time blocking simulation** for accurate temporal dynamics.
   - REVATI-style virtual time is available as an **option** for offline analysis.

**Non-goals / claims we avoid:**
- claiming to be the first code-path-preserving approach;
- claiming broad superiority over all simulators/emulators without direct evidence;
- claiming "strong compatibility guarantee" without cross-version measurements.

### 2.4 Key Risks and Impact on RFC

New related work (especially REVATI) changes risk profile:
- **Novelty collision risk**: high if we frame contribution as generic emulation speedup.
- **Reviewer risk**: high if related work stays Vidur-centric.
- **Evidence risk**: high if we only report aggregate accuracy without per-component diagnostics.

Impact on this RFC:
- keep architecture direction;
- tighten claims around explainability/diagnostics and vLLM-native integration;
- require explicit cross-baseline evaluation plan (simulator/emulator/optimizer families).

### 2.5 Comparative Limitations and Trade-offs (Revised)

| Axis | Our RFC (A/B/C/D decomposition + Online Blocking) | REVATI-style time-warp emulation | Simulator family (Vidur/LLMServingSim/Frontier/APEX) |
|------|----------------------------------|----------------------------------|-------------------------------------------------------|
| **Framework control-path fidelity** | High for A path; modeled for B/C/D | Very high (unchanged control plane) | Medium (abstraction/re-implementation dependent) |
| **Explainability of prediction error** | High (component-level) | Medium (often aggregate) | Medium (model-dependent) |
| **Search throughput** | Medium (depends on oracle efficiency) | Medium-high | High (often fastest at scale) |
| **Maintenance under fast framework evolution** | Medium-high (if hooks stable) | High (code-path inheritance) | Medium-low to medium |
| **Calibration dependence** | High | High | High |
| **Online serving simulation** | ✅ Full real-time blocking | ❌ Virtual time only | ❌ Not supported |
| **Offline batch inference** | ✅ Supported (offline mode) | ✅ Supported | ✅ Supported |
| **Primary risk** | overlap if claims too broad | interception coverage/corner cases | abstraction drift vs real systems |

**Key insight:** Online serving (real-time request arrival) is the primary application scenario for LLM inference systems. Our emulator supports both:
- **Online mode (default)**: Block for estimated latency to simulate real temporal dynamics
- **Offline mode**: No blocking (virtual time, like REVATI) for faster batch analysis

This table is a planning aid, not a superiority claim. Quantitative validation is required in Section 6.

---

## 3. Architecture Overview

### 3.1 Cost Boundary Definition

| Category | Description | Behavior |
|----------|-------------|----------|
| **A** | API/Scheduler/Tokenization | Run Real |
| **B** | GPU Compute | Modeled |
| **C** | CPU↔GPU Interaction | Modeled |
| **D** | Network (TP/PP/PD) | Modeled |

### 3.2 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     API Layer (A)                          │
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │  OpenAI API    │  │  FastAPI       │                  │
│  │  /v1/chat/    │  │  Server        │                  │
│  └────────┬────────┘  └────────┬────────┘                  │
│           │                     │                           │
│           └──────────┬──────────┘                           │
│                      ▼                                      │
│            ┌────────────────┐                              │
│            │   Engine Core   │  v1/engine/core.py          │
│            │  (Scheduler)    │                              │
│            └────────┬────────┘                              │
└─────────────────────┼───────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
┌───────────┐  ┌───────────┐  ┌───────────┐
│  Worker   │  │ KV Offload │  │  Network  │
│ (Category B)│  │(Category C) │  │(Category D)│
│  Oracle   │  │   Oracle   │  │   Oracle  │
└─────┬─────┘  └─────┬─────┘  └─────┬─────┘
      │              │              │
      ▼              ▼              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Cost Oracle Layer                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │GPU Cost Oracle│  │Offload Oracle│  │Network Oracle│    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Profile Pack Manager                     │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Component Design

### 4.1 Category A: Run Real (No Emulation)

**Components:**
- API Ingress: `entrypoints/api_server.py`, `entrypoints/openai/`
- Tokenization: `tokenizers/`
- Scheduler: `v1/engine/core.py`, `v1/scheduler.py`
- Request Lifecycle: `engine/`, `sequence.py`

**Design Decision:** No changes required. These run real code to preserve scheduling behavior.

### 4.2 Category B: GPU Compute Model

**Hook Point:** `v1/worker/gpu_worker.py::execute_model()` (line 582)

**Interface:**
```python
class GpuCostOracle:
    """Oracle for GPU compute latency."""
    
    def get_prefill_latency(
        self,
        prompt_tokens: int,
        batch_size: int = 1,
        parallelism: int = 1,
        dtype: str = "float16"
    ) -> float:
        """Get prefill phase latency in microseconds."""
        pass
    
    def get_decode_latency_per_token(
        self,
        active_seqs: int,
        kv_state: str = "hot",
        parallelism: int = 1,
        dtype: str = "float16"
    ) -> float:
        """Get decode phase latency per token in microseconds."""
        pass
```

**Profile Schema:**
```json
{
  "version": "1.0",
  "gpu_model": "A100",
  "prefill": [
    {"seq_len": 128, "batch_size": 1, "latency_us": 15000},
    {"seq_len": 256, "batch_size": 1, "latency_us": 28000},
    {"seq_len": 512, "batch_size": 1, "latency_us": 55000},
    {"seq_len": 1024, "batch_size": 1, "latency_us": 108000}
  ],
  "decode": [
    {"active_seqs": 1, "latency_us_per_token": 500},
    {"active_seqs": 4, "latency_us_per_token": 1800},
    {"active_seqs": 8, "latency_us_per_token": 3500},
    {"active_seqs": 16, "latency_us_per_token": 6800}
  ]
}
```

**Integration:**
1. Create `GpuCostOracle` class
2. In `gpu_worker.execute_model()`, add conditional oracle hook
3. If emulator mode enabled: return emulated latency
4. If disabled: use real GPU execution

**Blocking Modes (Critical for Online Serving):**

The emulator supports two timing modes, configured via `VLLM_EMULATOR_BLOCKING_MODE`:

| Mode | Behavior | Use Case |
|------|----------|----------|
| `online` (default) | Block for estimated latency using `time.sleep()` | Online serving with real-time request arrival |
| `offline` | No blocking (virtual time, like REVATI) | Offline batch inference analysis |

**Why this matters:**
- **Online serving is the primary application** for LLM inference systems
- When request arrival time matters (t=0: A arrives, t=100ms: B arrives), we must simulate real temporal dynamics
- The scheduler's decisions depend on actual timing: when a batch completes, pending requests see new state
- Virtual-time emulation (REVATI-style) cannot capture this because time doesn't advance between operations

**Timing Precision:**
- Online mode: uses `time.sleep()` with 1ms threshold (sub-ms delays ignored)
- For finer precision, consider ctypes-based spin-wait in future work

**LLM.generate() Warning:**
- When using offline batch interface (LLM.generate()) with online blocking mode, a warning is issued
- Recommends switching to offline mode for better performance

### 4.3 Category C: CPU↔GPU Interaction Model

**Hook Points:**
- `v1/kv_offload/abstract.py::OffloadingManager`
- `v1/kv_offload/worker/worker.py::OffloadingWorker.transfer_async()`
- `v1/kv_offload/worker/cpu_gpu.py::ops.swap_blocks()`

**Interface:**
```python
class OffloadCostOracle:
    """Oracle for CPU↔GPU interaction latency."""
    
    def get_lookup_latency(self, num_blocks: int) -> float:
        """Get KV lookup latency in microseconds."""
        pass
    
    def get_transfer_latency(
        self,
        bytes: int,
        direction: str,  # "cpu_to_gpu" or "gpu_to_cpu"
        concurrency: int = 1
    ) -> float:
        """Get transfer latency in microseconds."""
        pass
    
    def get_evict_latency(self, num_blocks: int) -> float:
        """Get KV eviction latency in microseconds."""
        pass
```

**Profile Schema:**
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

### 4.4 Category D: Network Model

**Hook Points:**
- `distributed/device_communicators/cuda_communicator.py::all_reduce()` (line 130)
- `distributed/device_communicators/cuda_communicator.py::send()` (line 240)
- `distributed/device_communicators/cuda_communicator.py::recv()` (line 252)
- `distributed/kv_transfer/kv_connector/v1/base.py::start_load_kv()`, `save_kv_layer()`

**Interface:**
```python
class NetworkCostOracle:
    """Oracle for network operation latency."""
    
    def get_all_reduce_latency(
        self,
        bytes: int,
        world_size: int,
        topology: str  # "nvlink", "pcie", "ib"
    ) -> float:
        """Get all-reduce latency in microseconds."""
        pass
    
    def get_send_latency(
        self,
        bytes: int,
        topology: str
    ) -> float:
        """Get send latency in microseconds."""
        pass
    
    def get_recv_latency(
        self,
        bytes: int,
        topology: str
    ) -> float:
        """Get recv latency in microseconds."""
        pass
    
    def get_kv_transfer_latency(
        self,
        bytes: int,
        direction: str,
        topology: str,
        concurrency: int = 1
    ) -> float:
        """Get KV transfer latency in microseconds."""
        pass
```

**Profile Schema:**
```json
{
  "all_reduce": {
    "nvlink": [
      {"bytes": 4096, "world_size": 2, "latency_us": 50},
      {"bytes": 1048576, "world_size": 2, "latency_us": 200},
      {"bytes": 1048576, "world_size": 8, "latency_us": 800}
    ],
    "pcie": [...],
    "ib": [...]
  },
  "send_recv": {...},
  "kv_transfer": {...}
}
```

## 4.5 Configuration

**Environment Variables:**

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `VLLM_EMULATOR_ENABLE_ORACLE` | `1`, `true`, `yes` | - | Enable emulator mode |
| `VLLM_EMULATOR_PROFILE_PACK` | path | - | Path to profile pack JSON |
| `VLLM_EMULATOR_BLOCKING_MODE` | `online`, `offline` | `online` | Timing simulation mode |

**Usage Examples:**

```bash
# Online serving (default): block for real-time simulation
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK=/path/to/profile.json \
vllm serve <model>

# Offline batch: no blocking (faster)
VLLM_EMULATOR_ENABLE_ORACLE=1 \
VLLM_EMULATOR_PROFILE_PACK=/path/to/profile.json \
VLLM_EMULATOR_BLOCKING_MODE=offline \
python -c "from vllm import LLM; llm = LLM(<model>); llm.generate(<inputs>)"
```

---

## 5. Platform Integration

### 5.1 Platform Plugin

Create `EmulatorPlatform` class:

```python
# vllm_emulator/platform.py

class EmulatorPlatform(Platform):
    """Platform for emulator backend."""
    
    _name = "emulator"
    _enum = PlatformEnum.OOT
    
    def __init__(self, vllm_config: "VllmConfig" = None):
        super().__init__()
        self.profile_pack_path = vllm_config.extra_config.get("emulator_profile_pack")
    
    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        return 80 * 1024**3  # Fake 80GB
    
    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return "Emulator GPU"
```

### 5.2 Registration

```python
# setup.py
entry_points={
    "vllm.platform_plugins": [
        "emulator = vllm_emulator.platform:register",
    ],
}

# Register function
def register():
    return "vllm_emulator.platform.EmulatorPlatform"
```

### 5.3 Activation

```bash
# Via environment variable
VLLM_PLUGINS=emulator vllm serve <model> \
    --extra-config emulator_profile_pack=/path/to/profile.json
```

---

## 6. Testing Strategy

### 6.1 A/B Comparison

1. **Real Mode**
   ```bash
   vllm serve <model> --gpu-memory-utilization 0.8
   ```

2. **Emulator Mode**
   ```bash
   VLLM_PLUGINS=emulator vllm serve <model> \
       --extra-config emulator_profile_pack=/path/to/profile.json
   ```

3. **Workload**
   - Use same benchmark dataset
   - Record: TTFT, TPOT, E2E latency, throughput

4. **Metrics**
   - P50 error < 10%
   - P95 error < 15%

### 6.2 Benchmark Tools

- `benchmarks/throughput.py` - Throughput comparison
- `benchmarks/latency.py` - Latency comparison
- Custom benchmark client for controlled testing

---

## 7. Project Structure

```
vllm-emulator/
├── vllm_emulator/
│   ├── __init__.py
│   ├── platform.py          # EmulatorPlatform
│   ├── oracle/
│   │   ├── __init__.py
│   │   ├── base.py          # Base oracle interface
│   │   ├── gpu_cost_oracle.py    # Category B
│   │   ├── offload_cost_oracle.py # Category C
│   │   └── network_cost_oracle.py # Category D
│   ├── profile/
│   │   ├── __init__.py
│   │   ├── loader.py        # Profile pack loader
│   │   └── validator.py    # Schema validation
│   └── hooks/
│       ├── __init__.py
│       ├── gpu_hook.py     # Category B hooks
│       ├── offload_hook.py # Category C hooks
│       └── network_hook.py  # Category D hooks
├── tests/
│   ├── unit/
│   │   ├── test_oracles.py
│   │   └── test_hooks.py
│   └── integration/
│       ├── test_ab_comparison.py
│       └── test_accuracy.py
├── docs/
│   ├── cost-boundary.md
│   ├── cost-boundary-zh.md
│   ├── architecture.md
│   ├── review-gpu-worker.md
│   ├── review-kv-offload.md
│   ├── review-network.md
│   └── review-integration.md
├── examples/
│   └── emulator_backend/
└── setup.py
```

---

## 8. Roadmap

### Phase 1: Foundation (P0) - REQUIRED

#### P0.1: Cost Boundary Documentation ✅ DONE
- [x] Write `docs/cost-boundary.md`
- [x] Write `docs/cost-boundary-zh.md`
- **Status:** ✅ DONE

#### P0.2: Platform Plugin
- [ ] Create `EmulatorPlatform` class
- [ ] Register via entry point
- [ ] Test basic activation
- **Status:** IN PROGRESS

#### P0.3: Profile Pack System
- [ ] Define profile pack schema
- [ ] Create loader/validator
- [ ] Create example profile packs
- **Status:** NOT_STARTED

#### P0.4: Profile Generation Scripts - REQUIRED
- [ ] Create profiling scripts to generate profile packs from real GPU runs
- [ ] Script can run on real GPU and collect latency profiles
- [ ] Output format matches profile pack schema
- [ ] Supports different GPU models (A100, H100, etc.)
- **Reference:** Vidur profiler design
- **Status:** NOT_STARTED

### Phase 2: Core Oracle (P1) - REQUIRED

#### P1.1: GPU Cost Oracle (Category B) - REQUIRED ✅ DONE
- [x] Create `GpuCostOracle` interface
- [x] Implement lookup logic (batch-level)
- [x] Hook into `gpu_worker.execute_model()`
- [x] Add fallback to real execution
- [x] Add online/offline blocking modes
- [x] Add warning for LLM.generate() + online mode
- **Focus:** Batch-level estimation, supports online serving simulation
- **Status:** ✅ DONE

### Phase 3: Optional Extensions (P2) - OPTIONAL

#### P2.1: Offload Cost Oracle (Category C) - OPTIONAL
- [ ] Create `OffloadCostOracle` interface
- [ ] Implement lookup/transfer/evict logic
- [ ] Hook into offload paths
- **Note:** Most research uses single-GPU, can be added later if needed
- **Status:** NOT_STARTED

#### P2.2: Network Cost Oracle (Category D) - OPTIONAL
- [ ] Create `NetworkCostOracle` interface
- [ ] Implement all_reduce/send_recv/kv_transfer
- [ ] Add topology support
- **Note:** Most research uses single-GPU, can be added later if needed
- **Status:** NOT_STARTED

### Phase 4: Integration (P3)

#### P3.1: CLI Integration
- [ ] Add `--emulator-mode` flag
- [ ] Add `--profile-pack` flag
- [ ] Update help text
- **Status:** NOT_STARTED

#### P3.2: Testing Infrastructure
- [ ] Create A/B comparison script
- [ ] Add accuracy metrics
- [ ] Create test profile packs
- **Status:** NOT_STARTED

#### P3.3: Documentation
- [ ] User guide
- [ ] API documentation
- [ ] Examples
- **Status:** NOT_STARTED

---

## 9. Open Questions

1. **Hybrid mode?** Should we allow partial emulation (only some categories)?

2. **Profile collection?** How to generate profile packs from real GPU runs?

3. **Accuracy targets?** Are P50<10%, P95<15% acceptable?

4. **Model support?** Should we limit to specific model architectures?

5. **Version compatibility?** How to handle vLLM version upgrades?

---

## 10. References

Implementation references:
- vLLM Source: `vllm/v1/`
- Platform Plugin: `vllm/platforms/`

Related work references (must-cover set):
- Vidur: A Large-Scale Simulation Framework for LLM Inference
- LLMServingSim: A HW/SW Co-Simulation Infrastructure for LLM Inference Serving at Scale
- Frontier: Simulating the Next Generation of LLM Inference Systems
- LLMServingSim 2.0: A Unified Simulator for Heterogeneous and Disaggregated LLM Serving Infrastructure
- REVATI: Transparent GPU-Free Time-Warp Emulation for LLM Serving
- AIConfigurator: Lightning-Fast Configuration Optimization for Multi-Framework LLM Serving
- APEX: An Extensible and Dynamism-Aware Simulator for Automated Parallel Execution in LLM Serving

Local analysis artifacts:
- `docs/paper-list.md`
- `docs/sim-emu-deep-dive-2026-03-06-v2.md`

---

## 11. Approval

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Author | Jinx | 2026-02-26 | |
| Reviewer | | | |
| Approver | David | | |
