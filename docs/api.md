# Emulator Backend — API Notes

This document captures the **current public surface** used by the emulator backend. It is **not** an exhaustive API reference; it enumerates the pieces that are wired today.

## CLI / Engine Args
- `EngineArgs.emulator_mode` (string or None)
- `EngineArgs.profile_pack` (string or None)

Resolution rules (from `vllm/engine/arg_utils.py`):
1. CLI flag > env var fallback.
2. If `--emulator-mode` is set, `--profile-pack` is required.
3. When enabled, the CLI writes env vars for worker processes:
   - `VLLM_EMULATOR_ENABLE_ORACLE=1`
   - `VLLM_EMULATOR_BLOCKING_MODE=online|offline`
   - `VLLM_EMULATOR_PROFILE_PACK=/path/to/profile.json`

## Emulator Hooks
### GPU Cost Oracle
- Module: `vllm_emulator/oracle/gpu_cost_oracle.py`
- Hook: `vllm_emulator/hooks/gpu_hook.py`
- Integration point: `vllm/v1/worker/gpu_worker.py`
- Modes:
  - **online**: `time.sleep()` to simulate latency
  - **offline**: virtual time (no blocking)

### Oracle Interface
- Base: `vllm_emulator/oracle/base.py`
- Implementations use profile packs with prefill/ decode latency curves.

## Test Interfaces
- `tests/integration/test_oracle_hook.py`
- `tests/integration/test_timing_accuracy.py`
- `tests/integration/ab_comparison_harness.py`

These tests validate:
- Oracle enable/disable via env vars
- Blocking vs non‑blocking modes
- Timing accuracy thresholds

## Related files
- Profile loader/validator: `vllm_emulator/profile/`
- Profiler scripts: `vllm_emulator/profiler/`
