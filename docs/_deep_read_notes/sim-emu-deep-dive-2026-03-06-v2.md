# Simulation/Emulation Related Work — Deep Dive v2 (Skill-compliant)

Date: 2026-03-06  
Target: `projects/vllm-emulator`  
Scope: simulation / emulation / performance prediction for LLM serving config exploration (heterogeneous + disaggregated).

Paper set:
1. Vidur (MLSys’24)
2. LLMServingSim (arXiv:2408.05499)
3. Frontier (arXiv:2508.03148)
4. LLMServingSim 2.0 (arXiv:2602.23036)
5. REVATI (arXiv:2601.00397)
6. AIConfigurator (arXiv:2601.06288)
7. APEX (arXiv:2411.17651)

---

## 1) Executive synthesis
This space now has **three distinct technical families**:

- **System simulators** (Vidur, LLMServingSim, Frontier, LLMServingSim2.0, APEX):
  strong for broad what-if evaluation and offline search; weaker on exact framework code-path fidelity.
- **Code-path-preserving emulation** (REVATI):
  strongest narrative on avoiding simulator re-implementation drift as vLLM/SGLang evolve.
- **Configuration optimizer/estimator** (AIConfigurator):
  strongest practical throughput for recommendation workflows; not a full serving runtime simulator.

For vLLM Emulator positioning, highest-overlap comparators are:
- **REVATI** (fidelity-by-execution claim),
- **LLMServingSim2.0 / Frontier** (heterogeneous + disaggregated system modeling breadth),
- **AIConfigurator/APEX** (configuration search productivity).

---

## 2) Per-paper mechanism cards

### 2.1 Vidur
**Objective**
- Large-scale high-fidelity simulation + deployment search for LLM inference.

**Mechanism**
- Profiles operators and uses predictive models; simulates end-to-end metrics (TTFT/throughput-like serving metrics).
- Includes Vidur-Search for cost-aware configuration search.

**Evaluation setup (reported)**
- Reports <9% inference latency estimation error.
- Emphasizes massive cost/time savings vs brute-force GPU exploration.

**Strengths**
- Early strong baseline for simulator-driven deployment search.
- Practical search framing with explicit cost comparison.

**Limits / risk**
- Still simulator abstraction, not direct execution of latest framework control path.
- Vulnerable to lag when serving framework internals evolve rapidly.

---

### 2.2 LLMServingSim (v1)
**Objective**
- HW/SW co-simulation infrastructure for serving systems at scale.

**Mechanism**
- Iteration-granularity simulation to handle autoregressive serving dynamics.
- Exploits decoder block redundancy and reuse to cut simulation overhead.
- Plugin-style integration with accelerator compile/sim stacks.

**Evaluation setup (reported)**
- <14.7% error vs real GPU behavior.
- ~91.5× faster than existing accelerator simulators (as claimed).

**Strengths**
- Explicit co-design lens (software + hardware modeling).
- Better dynamic serving treatment than many static simulators.

**Limits / risk**
- v1 generation scope; less unified treatment of newer disaggregated+heterogeneous runtime interactions than newer systems.

---

### 2.3 Frontier
**Objective**
- High-fidelity simulation for next-gen serving paradigms (MoE + disaggregation).

**Mechanism**
- Argues replica-centric abstractions are insufficient.
- Introduces workflow-centric/hierarchical orchestration model for multi-stage distributed serving.
- Explicitly models inter-cluster routing/data transfer/synchronization issues.

**Evaluation setup (reported)**
- Early/preliminary style evidence with multi-GPU setup; targets realistic complexity (MoE + disagg).

**Strengths**
- Correctly targets emerging system-of-systems complexity.
- Strong architecture argument against old replica-only abstractions.

**Limits / risk**
- Complexity of simulator stack can increase calibration/maintenance burden.
- Need careful reproducibility details to defend fidelity claims.

---

### 2.4 LLMServingSim 2.0
**Objective**
- Unified simulator for heterogeneous and disaggregated infrastructure.

**Mechanism**
- Embeds serving decisions + hardware behavior in one runtime loop.
- Interaction-aware modeling of batching/routing/placement/offloading/memory/power.
- Profile-based extensibility toward emerging accelerator/memory systems.

**Evaluation setup (reported)**
- Claims ~0.97% average error across key performance/memory/power metrics.
- Practical simulation time around minutes for complex configs.

**Strengths**
- Strong breadth across heterogeneity + disaggregation + power.
- Explicit interaction-aware narrative beyond static decomposition.

**Limits / risk**
- Still simulation rather than framework code-path execution.
- Low-error claims require transparent benchmark/trace coverage to avoid overfitting concerns.

---

### 2.5 REVATI
**Objective**
- GPU-free time-warp emulation that runs real serving framework logic.

**Mechanism**
- Intercepts CUDA APIs, virtualizes device runtime, and performs virtual-time jumps instead of kernel execution.
- Distributed time coordination protocol for causality-preserving jumps.
- Targets direct compatibility with framework code paths (vLLM/SGLang).

**Evaluation setup (reported)**
- <5% prediction error on multiple models/parallelism settings.
- 5–17× faster than real GPU execution.

**Strengths**
- Most direct answer to simulator re-implementation drift.
- Very strong positioning for rapidly changing serving frameworks.

**Limits / risk**
- Depends on robust predictor calibration + interception coverage.
- Potential edge-case drift in asynchronous/runtime corner behaviors.

---

### 2.6 AIConfigurator
**Objective**
- Lightning-fast multi-framework inference configuration optimization.

**Mechanism**
- Decomposes inference into primitives (GEMM/attention/comm/memory).
- Uses calibrated kernel-level database and abstraction layer to map framework/backend launch options.
- Searches large config spaces quickly under SLA constraints.

**Evaluation setup (reported)**
- Up to +40% (dense) / +50% (MoE) vs baseline configs.
- Search completion in ~tens of seconds.
- Uses MAPE for TTFT/TPOT fidelity.

**Strengths**
- Production-facing optimizer workflow and backend diversity.
- Strong usability for rapid recommendation loops.

**Limits / risk**
- Not a full serving simulator/emulator runtime.
- Fidelity bound to profile database coverage and mode assumptions.

---

### 2.7 APEX (2411.17651)
**Objective**
- Dynamism-aware simulator for automated parallel execution plan selection.

**Mechanism**
- Iteration-level batching simulation + repetitive-structure reduction for tractable search.
- Plan search across DP/PP/TP with latency + energy trade-offs.

**Evaluation setup (reported)**
- Avg relative error ~10.7%.
- Finds plans up to ~3.37× faster than heuristics; significant energy reductions.

**Strengths**
- Strong bridge between simulation and actionable parallelism planning.
- Explicitly models serving dynamism rather than static batch assumptions.

**Limits / risk**
- Scope is narrower (parallel execution planning) than full-stack serving simulator/emulator infrastructure.

---

## 3) Cross-paper trade-off matrix (condensed)

| Dimension | Strongest Works | Notes |
|---|---|---|
| Code-path fidelity to real serving framework | REVATI | Direct execution + CUDA interception; key emulation differentiator |
| Heterogeneous+disaggregated system breadth | LLMServingSim2.0, Frontier | Better explicit modeling of runtime interactions and multi-stage workflows |
| Search productivity / recommendation speed | AIConfigurator, APEX | Best for rapid config exploration under SLA constraints |
| Historical baseline + deployment search framing | Vidur | Canonical early simulator+search baseline |
| HW/SW co-sim foundation | LLMServingSim v1 | Important bridge to newer unified simulators |

Reproducibility risk axis:
- **Higher risk areas:** heavy calibration dependence, opaque trace coverage, or under-specified workload diversity.

---

## 4) vLLM Emulator positioning statement (project-linked)

### Borrow directly
- From REVATI: code-path-preserving value proposition; “avoid re-implementation drift” framing.
- From LLMServingSim2.0/Frontier: heterogeneity/disaggregation runtime-interaction dimensions for eval matrix.
- From AIConfigurator/APEX: SLA-centric search/reporting surface and practical optimization outputs.

### Adapt with constraints
- Oracle decomposition should expose confidence/coverage metadata (not just point predictions).
- Must separately report in-distribution vs out-of-distribution workload error.

### Reject
- Broad “simulator-level completeness” claims without explicit unsupported-path inventory.

### Differentiation risks
- If we only claim “fast no-GPU prediction,” overlap is high with REVATI + AIConfigurator.
- Differentiation should emphasize one of:
  1) better controllability/transparency of cost decomposition,
  2) stronger vLLM-native integration surface for research iteration,
  3) better uncertainty quantification and failure diagnostics.

---

## 5) Model-backed hypotheses + validation plan

### H1
Code-path-preserving emulation yields lower maintenance cost under fast framework evolution than pure simulator re-implementation.
- Test: multi-version replay (vLLM versions) and maintenance delta tracking.

### H2
Heterogeneous/disaggregated serving benefits are mis-estimated unless network/state-transfer interactions are explicitly modeled.
- Test: ablate transfer/coordination models and measure TTFT/TPOT bias.

### H3
Estimator-only optimizers are fastest for search, but generalization drops under unseen model/workload/hardware combinations.
- Test: holdout hardware + workload classes; compare ranking stability and SLA violation rates.

### Validation ladder
1. Micro-kernel/profile calibration
2. Trace-backed replay
3. Cross-framework holdout (where feasible)
4. Publish error envelopes + failure regions + confidence tags

---

## 6) Design section starter outline (for paper writing)
1. Why current exploration is too expensive/incomplete
2. Design goals: fidelity, speed, maintainability, reproducibility
3. Runtime architecture and hook points
4. Cost decomposition and calibration pipeline
5. Configuration exploration loop + SLA constraints
6. Evaluation methodology (ID/OOD, hetero/disagg, ablations)
7. Limitations and safe interpretation scope

---

## 7) Motivation paragraph drafts

### Draft A (fidelity + cost)
Exploring LLM serving configurations on real GPU clusters is prohibitively expensive and slow, yet pure simulators often lag behind rapidly evolving framework internals. This creates a practical gap between fast-but-stale modeling and faithful-but-costly deployment testing. We target this gap by building a serving exploration stack that preserves critical runtime behavior while reducing evaluation cost enough for iterative design-space search.

### Draft B (heterogeneity/disaggregation)
Modern LLM serving increasingly combines heterogeneous hardware and disaggregated execution paths, where performance is dominated by interactions among scheduling, communication, and memory transfer rather than isolated kernel speed. Existing tools usually optimize only one layer of this stack. We argue that robust configuration exploration requires a unified treatment of these interactions with explicit uncertainty and calibration boundaries.

### Draft C (research tooling)
A useful serving exploration tool should do more than predict a single latency number: it must expose where predictions are reliable, how decisions change under workload shifts, and why specific configurations win. We therefore frame the system as both a predictor and a research instrument, coupling fast what-if analysis with interpretable decomposition and reproducible validation.

---

## 8) Risk register + rebuttal hooks
- **Risk:** overlap with REVATI/AIConfigurator claims.
  - **Hook:** clarify novelty axis (integration depth, decomposition transparency, uncertainty, or evaluation scope).
- **Risk:** calibration overfitting.
  - **Hook:** publish cross-workload/hardware holdout and error envelopes.
- **Risk:** unsupported path bias.
  - **Hook:** enumerate unsupported features and provide fallback policy.
- **Risk:** benchmark cherry-picking.
  - **Hook:** fixed benchmark suite spanning homogeneous, heterogeneous, disaggregated scenarios.

---

## Evidence note
This v2 document is based on local full-paper text extraction (`pdftotext`) and structured per-paper notes under:
`projects/vllm-emulator/docs/_deep_read_notes/`.
