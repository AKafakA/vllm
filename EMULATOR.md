# vLLM Emulator (Paper Artifact)

A profile-driven, hardware-independent emulator for production vLLM. Captures per-step latency distributions on a real GPU once, then reproduces matching online-serving latency on any host — including no-GPU CPU-only hosts.

This document covers installation, the v4 (CUDA-invisible) emulator path, and how to migrate to a newer base vLLM version.

---

## Two install modes

| Mode | Host | Real bench | Emu bench |
|------|------|------------|-----------|
| **A — Real GPU** | RTX 8000 / A10 / H100 / etc. | runs on the real GPU | runs alongside, with `CUDA_VISIBLE_DEVICES=""` to hide the GPU |
| **B — No-GPU CPU host** | Any Linux Xeon/EPYC | not possible (no GPU) | runs against a profile pack captured on Mode A; CUDA APIs satisfied by stubs + `cuda_mock.py` |

Both modes share the same emulator code path (the v4 path). Only the install differs in whether `tools/setup_cuda_stubs.sh` runs.

---

## Install (one command)

```bash
git clone <repo> vllm-emulator
cd vllm-emulator
bash install.sh
```

This:

1. Creates `.venv` (uses `uv venv` if available, else `python3 -m venv`)
2. Installs vLLM precompiled (`VLLM_USE_PRECOMPILED=1 pip install -e .`) plus the emulator package
3. Detects mode (A or B) and creates `~/cuda_stubs/` if Mode B
4. Runs `tools/check_vllm_compat.sh` to verify the patched API surface (executor + worker + model-runner) is present
5. Verifies imports

**Override**: `bash install.sh --vllm 0.18.1 --mode no-gpu` (force a vllm version + mode).

---

## Use

```bash
source tools/activate_emulator.sh                            # sets v4 env
export VLLM_EMULATOR_PROFILE_PACK=./results/<tag>/serving-full.json
python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-8B --max-model-len 4096 \
    --port 8100 --trust-remote-code
```

`activate_emulator.sh` exports the v4 invariants:

```
CUDA_VISIBLE_DEVICES=""                # hide any real GPU
LD_LIBRARY_PATH=$STUB_DIR:...          # cuda_mock + stub libcuda
VLLM_EMULATOR_ENABLE_ORACLE=1
VLLM_EMULATOR_MODE=realtime
VLLM_EMULATOR_EXECUTOR_HOOK=1          # the only hook
VLLM_EMULATOR_SCHEDULER_HOOK=0
VLLM_EMULATOR_IPC_POSITION=disabled
VLLM_EMULATOR_PREP_SURROGATE=0
VLLM_EMULATOR_ORACLE_AGG=sample
VLLM_EMULATOR_ORACLE_K=auto
VLLM_EMULATOR_ORACLE_MIN_SAMPLES=30
VLLM_EMULATOR_BW_SLOPE_SOURCE=disabled
ulimit -n 65536                        # bench-client EMFILE workaround
```

These are paper-grade defaults. Don't change them.

---

## Capture a profile pack (Mode A only)

```bash
source tools/_bench_common.sh
TAG=my-cell BENCH_MODEL=Qwen/Qwen3-8B \
    bash tools/adaptive_profile_capture.sh
```

Produces `./results/<HW>-adaptive-<TAG>/serving-full.json`. Takes ~3 h on RTX 8000 for full ShareGPT.

---

## Run a 3-stage cell (real bench + profile + emu validate)

```bash
bash tools/run_v4_cell.sh apr26-m2-main Qwen/Qwen3-8B
# or with cell-specific server config:
bash tools/run_v4_cell.sh apr26-r3-prefix-off Qwen/Qwen3-8B "--no-prefix-caching"
bash tools/run_v4_cell.sh apr26-triton    Qwen/Qwen3-8B "--attention-backend TRITON_ATTN"
```

Reuse an existing profile (Stage 2 skipped):

```bash
REUSE_PROFILE=./results/<existing-tag>/serving-full.json \
    bash tools/run_v4_cell.sh my-tag Qwen/Qwen3-8B
```

After each cell, the runner appends per-rate MEAN deltas to
`results/<tag>/per_rate_deltas.csv` and writes `run.log`.

---

## Parity invariant (read this before launching anything)

The Apr 22-24 campaign reset was caused by a single config-parity bug:
the real-bench script omitted `--max-num-seqs 64` while the profile-capture
and emu scripts used it. With matched configs, all paper cells pass at
< 6% mean TPOT/ITL/E2E.

Therefore: **all 3 stages (real bench, profile capture, emu validate)
must use IDENTICAL `non-default args` lines** (modulo port).

The runner enforces this:

- `run_one_full_sharegpt_cell.sh` refuses `--max-num-seqs` in `EXTRA_SERVER_ARGS`
- After Stage 2, asserts profile args == real args
- After Stage 3, warns if emu args != real args
- `tools/parity_audit.sh <cell_dir>` audits any time
- `tools/correctness_cron.sh` runs every 30 min, appends to
  `paper/apr_25/parity_log.md` and queues reruns in
  `paper/apr_25/rerun_queue.md` on violation

---

## Migrating to a newer base vLLM

The v4 hook surface against vllm public APIs is small (~210 LoC):

- `vllm/v1/executor/uniproc_executor.py` (~7 lines: lazy import + 3
  call sites in `execute_model` / `sample_tokens` / `shutdown`)
- `vllm/v1/worker/gpu_worker.py` (~30 lines, mostly inert in v4)
- `vllm_emulator/cuda_mock.py` meta-path patcher (`Worker` and
  `GPUModelRunner` method patches)

Migration steps:

1. `bash install.sh --vllm <new-version>`
2. `bash tools/check_vllm_compat.sh` — flags any missing methods on
   `Worker` / `GPUModelRunner`
3. If anything breaks, the diff surface above is small enough that a
   coding agent can usually patch it by reading the new vllm signatures

---

## Layout

```
vllm_emulator/                 # the emulator package (~3K LoC)
  hooks/
    executor_hook.py          # the only hook — intercepts execute_model
                              # at the executor level; returns timer
                              # Future after profile-driven sleep
  oracle/
    base.py                   # BaseGpuCostOracle
    gpu_cost_oracle.py        # 2D (tt, conc) lookup + sample
  profile/
    build_serving_profile_filtered.py  # builds the profile pack
    loader.py, validator.py
  profiler/
    gpu_profiler.py, trace_profiler.py
  cuda_mock.py                # CUDA-invisible mode shims
  platform.py                 # vllm platform plugin entry point
  worker_prep_surrogate.py    # CPU-side input-prep surrogate

vllm/                          # vendored vllm with the ~210 LoC hook
                              # patches (executor + worker)

tools/                         # cell runners, profile capture,
                              # parity audit, correctness cron

install.sh                     # one-shot installer
tools/activate_emulator.sh     # source-able env activation
tools/setup_cuda_stubs.sh      # CUDA stub generator (Mode B)
tools/check_vllm_compat.sh     # vllm API drift detector
tools/run_v4_cell.sh           # 3-stage cell wrapper
tools/parity_audit.sh          # config parity audit
tools/correctness_cron.sh      # 30-min monitor + auto-queue reruns
```
