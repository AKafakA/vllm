# Simulation/Emulation Related Work Deep Dive (2026-03-06)

## Scope lock
- **Target project:** `projects/vllm-emulator`
- **Claim scope:** simulation / emulation / performance prediction for LLM serving configuration exploration (heterogeneous + disaggregated)
- **Paper set:** Vidur, LLMServingSim, Frontier, LLMServingSim 2.0, REVATI, AIConfigurator, APEX (2411.17651)

## Artifacts downloaded
Stored under: `projects/vllm-emulator/references/papers/`
- `vidur_mlsys24.pdf`
- `llmservingsim_2024_2408.05499.pdf`
- `frontier_2025_2508.03148.pdf`
- `llmservingsim2_2026_2602.23036.pdf`
- `revati_2026_2601.00397.pdf`
- `aiconfigurator_2026_2601.06288.pdf`
- `parallelism_sim_2024_2411.17651.pdf`

---

## 1) Executive synthesis
- The space splits into three technical families:
  1. **Discrete/system simulators** (Vidur, LLMServingSim, Frontier, LLMServingSim2.0, APEX)
  2. **Code-preserving emulation** (REVATI)
  3. **Config optimization with learned/empirical estimators** (AIConfigurator)
- For vLLM Emulator positioning, the strongest threat/comparison axis is:
  - **REVATI** on “run real framework logic, virtualize GPU execution”
  - **LLMServingSim2.0/Frontier** on heterogeneity + disaggregation modeling breadth
  - **AIConfigurator/APEX** on search efficiency and practical tuning workflows

## 2) Per-paper mechanism cards (concise)
### Vidur
- Objective: high-fidelity, extensible simulation for LLM inference deployment search.
- Granularity: operator/system-level simulation with profiling + predictive models.
- Strength: strong simulation baseline and search framing.
- Limits: not code-path-faithful emulation of real framework runtime.

### LLMServingSim (v1)
- Objective: HW/SW co-simulation for serving at scale.
- Granularity: co-model hardware/software behavior.
- Strength: explicit system co-design viewpoint.
- Limits: earlier generation; less explicit disaggregated/heterogeneous unification than later variants.

### Frontier
- Objective: high-fidelity simulation for next-gen serving (MoE + disaggregation).
- Granularity: hierarchical system simulation.
- Strength: targets modern distributed complexity.
- Limits: simulator complexity and reproducibility/portability risks.

### LLMServingSim 2.0
- Objective: unified simulator for heterogeneous + disaggregated serving.
- Granularity: runtime interaction between HW and system software.
- Strength: explicit unification of heterogeneity and disaggregation.
- Limits: still simulation (not direct execution of production framework code paths).

### REVATI
- Objective: transparent GPU-free time-warp emulation.
- Granularity: executes real serving framework code, intercepts CUDA, virtual time-jumps.
- Strength: avoids simulator re-implementation drift; strong “framework-faithful” narrative.
- Limits: accuracy/maintenance depends on interception + timing model calibration.

### AIConfigurator
- Objective: lightning-fast configuration optimization across frameworks.
- Granularity: performance estimation + large config-space search.
- Strength: practical optimization workflow and multi-framework support.
- Limits: not a full simulator/emulator platform; estimator quality tied to data/mode assumptions.

### APEX (2411.17651)
- Objective: simulation-based automated parallel execution planning.
- Granularity: dynamism-aware simulator focused on parallelism strategy search.
- Strength: very relevant to config/parallelism optimization.
- Limits: narrower scope than full-stack serving simulator/emulator infrastructure.

## 3) Cross-paper trade-off matrix (short)
- **Control-plane realism:** REVATI > (AIConfigurator for config) > simulators
- **Architecture breadth (hetero+disagg):** LLMServingSim2.0 / Frontier strongest
- **Search productivity:** AIConfigurator / APEX strongest
- **Code-path faithfulness:** REVATI strongest, simulators weaker
- **Reproducibility risk:** learned/empirical estimators and calibration-heavy systems need careful setup disclosure

## 4) vLLM Emulator positioning
- Borrow directly:
  - From REVATI: code-path-preservation narrative + virtualization framing
  - From LLMServingSim2.0/Frontier: hetero/disagg modeling dimensions and benchmark scenarios
  - From AIConfigurator/APEX: config-search interfaces and SLA-constrained optimization outputs
- Adapt:
  - Cost-oracle decomposition (compute/memory/network) with explicit confidence bands
- Reject:
  - Over-claiming simulator-equivalent fidelity without calibration disclosures
- Differentiation risk:
  - If novelty is only “fast no-GPU estimate,” overlap risk with REVATI/AIConfigurator is high.

## 5) Model-backed hypotheses + test plan
- H1: code-path-preserving emulation wins under frequent framework evolution.
- H2: heterogeneity/disaggregation gains require explicit network+state transfer modeling.
- H3: estimator-only approaches can outperform emulation in search speed but degrade OOD robustness.

Validation ladder:
1. Synthetic microbench calibration (kernel classes)
2. Trace-backed replay on disaggregated workloads
3. Cross-framework holdout (vLLM/SGLang/TRT-LLM where possible)
4. Report confidence intervals and failure regions

## 6) Design section starter outline
1. Problem and motivation
2. Cost-boundary decomposition (A/B/C/D)
3. Runtime architecture and hooks
4. Calibration and profile-pack system
5. SLA-oriented config exploration loop
6. Evaluation and ablation

## 7) Motivation draft stubs
- D1: “Real-GPU configuration search is too expensive; we need faithful, cheap exploration.”
- D2: “Simulator re-implementation lags framework evolution; emulation closes this gap.”
- D3: “Heterogeneous/disaggregated serving requires jointly modeling compute, transfer, and scheduling.”

## 8) Risk register + rebuttal hooks
- Risk: overlap with REVATI/AIConfigurator.
  - Rebuttal: emphasize distinct fidelity/scope axis and validated win region.
- Risk: calibration fragility.
  - Rebuttal: provide per-workload error envelopes + OOD tests.
- Risk: benchmark cherry-picking.
  - Rebuttal: publish fixed suite covering homogeneous, heterogeneous, and disaggregated scenarios.

---

## Note
Attempted to use built-in `pdf` analyzer tool for full native deep analysis, but the tool failed with provider-side 404 in this session. This doc is produced from downloaded PDFs and extracted text-based evidence in-session.