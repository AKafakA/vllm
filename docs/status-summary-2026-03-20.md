# vLLM Emulator — Status Summary (2026-03-20)

## Current status
- Repository path is `projects/vllm-emulator` (earlier typo checkout `projects/vllm-enumlator` was removed).
- Core emulator backend is now beyond the skeleton stage and includes working GPU-path integration plus newly wired offload/network emulator entry points.
- Main blocker is no longer basic implementation but evaluation strategy: earlier CloudLab node availability changed, so validation needs a new test path.

## Completed progress
- P0 foundation completed: cost-boundary docs, platform plugin, profile pack system, profile generation scripts.
- P1.1 GPU Cost Oracle completed and wired into `vllm/v1/worker/gpu_worker.py`.
- P1.2 Offload Cost Oracle now has oracle + hook + vLLM worker integration.
- P1.3 Network Cost Oracle now has oracle + hook + communicator integration.
- P2.1 CLI integration completed.
- P2.2 testing infrastructure completed.
- P2.3 documentation completed.
- P2.5.2 prefill/decode separation support completed.

## What is now true technically
The repo has usable emulator plumbing for:
- batch-level GPU timing emulation,
- online/offline blocking modes,
- CLI-driven use,
- PD-separation experiments,
- offload transfer simulation via `OffloadingWorker`,
- network simulation for core `all_reduce/send/recv` paths via `CudaCommunicator`.

## Still incomplete / caveats
- Offload integration is not exhaustive: manager-level lookup/swap paths are not fully modeled.
- Network integration is partial rather than universal: `reduce_scatter`, `all_gather`, `broadcast`, and KV connector-specific paths are not yet hooked.
- Offload/network env vars are still separate from the main emulator CLI toggle flow.
- Workshop paper / upstream submission are not started.

## Practical interpretation
This project is now best described as:
- **core emulator backend implemented**,
- **GPU path complete**,
- **offload/network support integrated for key paths but not yet total-system complete**,
- **evaluation still pending new local/CloudLab/HPC validation path**.

## Immediate next-step question
Because the original CloudLab path is no longer stable, the next decision should be evaluation-first:
1. local 4070 validation,
2. another CloudLab reservation/manifest,
3. Cambridge HPC-based partial validation,
4. paper-first writeup using current emulator scope.
