# E2E Evaluation Scripts — vLLM Emulator

## Key Scripts

### Profiling + Evaluation
| Script | Purpose |
|--------|---------|
| `full_profile_and_eval.sh` | **Primary**: comprehensive profiling (8 rates + offline) + independent real/emu evaluation at rates 1,2,4,inf |
| `rebuild_and_emu.sh` | Rebuild profile from existing trace with outlier fix, test emu-only vs existing real baselines |
| `compare_results.py` | Compare real vs emu benchmark JSON files, compute % error for all metrics |

### Quick Tests
| Script | Purpose |
|--------|---------|
| `quick_200.sh` | 200-prompt A2A test at rates 1,2,4 |
| `quick_rate1.sh` | Rate=1 only with verbose error checking |
| `smoke_test_all.sh` | Comprehensive smoke test across all rates |
| `smoke_test_1k.sh` | 1000-prompt formal evaluation |

### Diagnostic
| Script | Purpose |
|--------|---------|
| `diag_single_req.sh` | Start emu server, send 2 requests, check logs |
| `diag_deadlock.sh` | Test for deadlock under high concurrency |
| `diag_ttft_measurement.sh` | Compare bench serve TTFT vs manual streaming TTFT |
| `diag_warmup_prompt.sh` | Test if longer prompts cause timeout |

### Feature Demos
| Script | Purpose |
|--------|---------|
| `run_offline_online_emulator.sh` | Compare offline vs online emulator modes |
| `run_cpu_offload_demo.sh` | CPU offloading path demo |
| `run_pd_separation_demo.sh` | Prefill/decode separation demo |
| `pathb_smoke_test.sh` | Path B: CPU-only emulator on CloudLab |

## Environment Variables

```bash
VLLM_EMULATOR_ENABLE_ORACLE=1          # Enable emulator
VLLM_EMULATOR_PROFILE_PACK=<path>      # Profile JSON path
VLLM_EMULATOR_MODE=realtime            # realtime or accelerated
VLLM_EMULATOR_EXECUTOR_HOOK=1          # Use executor-level hook
VLLM_EMULATOR_CUDA_GRAPH_WARMUP_US=0   # CUDA graph warmup (0=disabled)
VLLM_EMULATOR_MOCK_CUDA=1              # CPU-only mode (Path B only)
```

## Usage (Vast RTX 3060)

```bash
# Full profiling + evaluation (~30 min)
source /workspace/vllm-v18-env/bin/activate
export CUDA_VISIBLE_DEVICES=1
bash tools/e2e/full_profile_and_eval.sh

# Quick emu-only retest after profile changes
bash tools/e2e/rebuild_and_emu.sh
```

## Methodology

1. **Warmup**: 200 prompts at rate=4 for thermal equilibrium + CUDA graph compilation
2. **Independent server starts**: Fresh server per rate to avoid thermal drift
3. **Profile**: 45,800 step-cycle records across 8 online rates + offline
4. **Outlier detection**: Cross-reference neighboring buckets (±10 tt, 3x threshold)

See `docs/benchmarking/results-apr6.md` for full results.
