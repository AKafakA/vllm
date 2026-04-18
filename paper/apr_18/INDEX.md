# Apr 18 — 5-Feature Ablation Study

Active branch: `refactor/clean-emulator-v2` at `a102eed27`. Five feature branches `exp/fN-*` created throughout the day, each following design → review → implement → ablate → squash workflow.

## Current status (~13:45 UTC Apr 18)

| Slot | Status | Notes |
|---|---|---|
| S0 — v3 profiling + auto A/B | RUNNING | Round 5 at r=1 on `personal_gpu_vm`; ETA ~14:30 BST + 60 min auto A/B. Cron `bebd21f5` monitoring. |
| S1 — init `paper/apr_18/` | DONE | INDEX, progress, 00_session_context, 01_ablation_study_plan, 02_v3_baseline_AB scaffold, 03_experiment_tracker. |
| F1 design + review + implement | DONE | Byte-identity smoke tests pass. Branch `exp/f1-outlier-filter` @ `f7b1b8983`. |
| F3 design + review + implement | DONE | Branch `exp/f3-sample-tokens-delay` @ `2427e1bb0`. |
| F5 design + review + implement | DONE | Unit tests pass. Branch `exp/f5-knn-conditioning` @ `8d53683f9`. |
| F4 design + review + implement | DONE | Schema v2.0 / v2.1 byte-identity verified. Branch `exp/f4-prefill-axis` @ `d3115ded6`. |
| F2 design + review + implement | DONE | First draft REJECTED, rewrite APPROVED. Branch `exp/f2-parallel-surrogate` @ `6b86cb9a1`. |
| Generic A/B harness | DONE | `tools/validate_feature_ab.sh` committed on F2 branch. |
| F1–F5 A/B runs | PENDING | Blocked on v3 finish + baseline lock. |
| Combined build + full validation + conflict matrix | PENDING | Starts after all 5 feature A/Bs complete. |
| Handoff | PENDING | Slot S14 at end of day. |

## Session artefacts

- [00_session_context.md](00_session_context.md) — frozen snapshot + 6 binding requirements
- [01_ablation_study_plan.md](01_ablation_study_plan.md) — per-feature spec overview
- [02_v3_baseline_AB.md](02_v3_baseline_AB.md) — fallback-rule scaffold, fills when A/B done
- [03_experiment_tracker.md](03_experiment_tracker.md) — live run tracker + conflict matrix
- [progress.md](progress.md) — append-only checkpoint log

## Per-feature docs

- [04_f1_outlier_filter.md](04_f1_outlier_filter.md) — design + review verdict (APPROVED WITH CONDITIONS ×1)
- [05_f3_sample_tokens_delay.md](05_f3_sample_tokens_delay.md) — design + review verdict (APPROVED WITH CONDITIONS ×1)
- [06_f5_knn_conditioning.md](06_f5_knn_conditioning.md) — design + review verdict (APPROVED WITH CONDITIONS ×1)
- [07_f2_parallel_surrogate.md](07_f2_parallel_surrogate.md) — design (REJECTED → rewrite → APPROVED)
- [08_f4_prefill_axis.md](08_f4_prefill_axis.md) — design + review verdict (APPROVED WITH CONDITIONS ×1)

All five review verdicts cite a specific agent id and enumerate the conditions applied in the implementation commit.

## Plan + external refs

- Plan file: `~/.claude/plans/it-looks-good-let-unified-steele.md`
- Emulator implementation doc: `../emulator-docs/executor-hook-implementation.md`
- Canonical working-wiring commit: `182a75877`
- Fallback archived profile: `results/_archive/serving-dense.json` (108k, clean, doc §4.2 baseline)

## 6 binding requirements (applied to every feature review)

1. No augmentation (no emu-vs-real gap-fitting).
2. No new knobs beyond the gate listed in the plan.
3. No heuristic (profile data / model_config / named textbook default / user gate only).
4. No magic numbers.
5. Generic for models and hardware.
6. No vllm-core edits beyond the 2 existing hook lines.
