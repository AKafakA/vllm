# vLLM Emulator — E2E Test Plan (Offline/Online + CPU Offload + PD Separation)

Last updated: 2026-04-02

## Goals
1. **Offline vs Online emulator modes**: quantify prediction accuracy and performance differences.
2. **CPU offloading demo**: show emulator can model offload cost paths (feature not in typical serving simulators).
3. **Prefill/Decode (PD) separation demo**: show emulator can model phase-aware scheduling.
4. **Realistic workload**: use **BurstGPT** trace as a realistic serving workload.

## Target cluster layouts (CloudLab)
- **A100-only**: 2 nodes × 4 GPU
- **A30-only**: 8 nodes × 1 GPU
- **Heterogeneous**: combined A100 + A30 (simulate mixed cluster)

## Models
- **Qwen2.5**: 7B + 14B (or 32B if resources allow)
- **Llama**: 3-8B + 3-70B (or 3.1-8B + 3.1-70B)

> Keep total matrix small for 4-page workshop.

## Workload
- **BurstGPT trace** (open dataset): https://github.com/HPMLL/BurstGPT
- Use a **scaled subset** (e.g., 1 day or sampled window) for manageable runtime.

## Metrics
- **Throughput**
- **P99 latency** (or P95 if noisy)
- **Emulator prediction error** (relative %)

---

## Phase A — Profile packs (single-node)
**Goal:** calibrate A100/A30 baseline profile packs.

1. Run real vLLM on single A100 and single A30 with fixed prompts.
2. Collect prefill/decode latency samples.
3. Generate profile packs:
   - `examples/profiles/a100-*.json`
   - `examples/profiles/a30-*.json`

---

## Phase B — Offline vs Online (homogeneous)
**Goal:** compare emulator behavior in offline vs online modes on homogeneous clusters.

For each model + cluster type:
- **Offline mode** (no blocking; REVATI-like):
  - `VLLM_EMULATOR_BLOCKING_MODE=offline`
- **Online mode** (blocking):
  - `VLLM_EMULATOR_BLOCKING_MODE=online`

Compare emulator metrics vs real serving trace replay (BurstGPT subset).

---

## Phase C — Heterogeneous cluster demo
**Goal:** show emulator captures scheduling and cost differences in mixed A100/A30.

Run with mixed profile packs:
- A100 profile for A100 nodes
- A30 profile for A30 nodes

Compare:
- emulator predictions
- real serving behavior (if feasible)

---

## Feature Demos (must-have in workshop)

### 1) CPU offloading demo
- Script: `tools/e2e/run_cpu_offload_demo.sh`
- Based on example: `examples/others/lmcache/cpu_offload_lmcache.py`
- Demonstrate CPU↔GPU transfer cost modeling via emulator.

### 2) PD separation demo
- Script: `tools/e2e/run_pd_separation_demo.sh`
- Based on example: `examples/emulator/pd_separation_example.py`
- Demonstrate phase-aware cost modeling and scheduling.

---

## Scripts (location)
All runnable scripts are stored under:
```
/tools/e2e/
```

- `run_offline_online_emulator.sh`
- `run_cpu_offload_demo.sh`
- `run_pd_separation_demo.sh`

Each script has placeholders for:
- model name
- profile pack path
- trace path

---

## Deliverables for 4-page workshop
1. **Figure 1**: System diagram (emulator + profile pack + offline/online switch)
2. **Figure 2**: Offline vs online accuracy (A100-only, A30-only)
3. **Figure 3**: Heterogeneous A100+A30 case
4. **Table 1**: CPU offload + PD separation feature demo summary

---

## Notes
- Keep evaluation matrix small; prioritize **clarity and evidence**, not exhaustive sweeps.
- Avoid large model runs if they threaten timeline; focus on convincing signal.
