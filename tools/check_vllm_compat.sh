#!/bin/bash
# Verify the installed vllm version exposes the public APIs the v4 hook
# patches. Run this after `pip install vllm==<version>` to detect API
# drift before launching a long bench. Migrating to a newer vllm should
# be mostly automatable by a coding agent — this script is the diff
# surface.
#
# Required APIs:
#   1. vllm.v1.executor.uniproc_executor.UniProcExecutor.execute_model
#      — the executor hook intercepts here (see vllm patch lines ~50-56)
#   2. vllm.v1.worker.gpu_worker.Worker (class)
#      — cuda_mock.py patches load_model / init_device / profile_run via
#        meta-path patcher
#   3. vllm.v1.worker.gpu_model_runner.GPUModelRunner (class)
#      — cuda_mock.py patches synchronize_input_prep / load_model
#   4. vllm.platforms.interface.Platform, PlatformEnum
#      — vllm_emulator.platform inherits Platform
#
# Usage:
#   bash tools/check_vllm_compat.sh
# Exit 0 if all targets present; exit 1 with diff if any missing.

set -u

python3 - <<'PY'
import importlib
import sys

errors = []

def check(module_path, names):
    try:
        m = importlib.import_module(module_path)
    except Exception as e:
        errors.append(f"  cannot import {module_path}: {e}")
        return
    for name in names:
        if not hasattr(m, name):
            errors.append(f"  {module_path}.{name} MISSING")

check("vllm.v1.executor.uniproc_executor", ["UniProcExecutor"])
check("vllm.v1.worker.gpu_worker", ["Worker"])
check("vllm.v1.worker.gpu_model_runner", ["GPUModelRunner"])
check("vllm.platforms.interface", ["Platform", "PlatformEnum"])
check("vllm.v1.core.sched.scheduler", ["Scheduler"])

# Check critical methods on Worker / GPUModelRunner that cuda_mock patches.
try:
    from vllm.v1.worker.gpu_worker import Worker
    for method in ("load_model", "init_device", "determine_available_memory",
                   "get_kv_cache_spec", "get_supported_tasks",
                   "initialize_from_config", "get_model"):
        if not hasattr(Worker, method):
            errors.append(f"  Worker.{method} MISSING")
except Exception as e:
    errors.append(f"  Worker introspection failed: {e}")

try:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    for method in ("synchronize_input_prep", "load_model", "profile_run",
                   "_dummy_run", "get_model"):
        if not hasattr(GPUModelRunner, method):
            errors.append(f"  GPUModelRunner.{method} MISSING")
except Exception as e:
    errors.append(f"  GPUModelRunner introspection failed: {e}")

import vllm
vllm_ver = getattr(vllm, "__version__", "unknown")

if errors:
    print(f"vllm {vllm_ver} COMPAT CHECK FAILED:")
    for e in errors:
        print(e)
    print("")
    print("If you upgraded vllm, the v4 hook patches need updating. The")
    print("public-API patch surface is in:")
    print("  - vllm/v1/executor/uniproc_executor.py (~50 lines)")
    print("  - vllm_emulator/cuda_mock.py meta-path patchers (Worker/Runner)")
    sys.exit(1)
else:
    print(f"vllm {vllm_ver} COMPAT CHECK PASSED — all v4 hook targets present.")
PY
