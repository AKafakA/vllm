#!/bin/bash
# Verify the installed vllm version exposes the public APIs the v4 hook
# patches. Run this after `pip install vllm==<version>` to detect API
# drift before launching a long bench. Migrating to a newer vllm should
# be mostly automatable by a coding agent — this script is the diff
# surface.
#
# Required APIs (verified by AST + class introspection — no imports of
# CUDA-bound modules, so this works on no-GPU hosts too):
#   1. vllm.v1.executor.uniproc_executor.UniProcExecutor.execute_model
#   2. vllm.v1.worker.gpu_worker.Worker (class + method set)
#   3. vllm.v1.worker.gpu_model_runner.GPUModelRunner (class + method set)
#   4. vllm.platforms.interface.Platform, PlatformEnum
#   5. vllm.v1.core.sched.scheduler.Scheduler
#
# Usage:
#   bash tools/check_vllm_compat.sh
# Exit 0 if all targets present; exit 1 with diff if any missing.

set -u

python3 - <<'PY'
import ast
import importlib.util
import sys

errors = []

def find_module_path(module_path):
    """Resolve a dotted module to its source file path (no import)."""
    try:
        spec = importlib.util.find_spec(module_path)
        if spec is None or spec.origin is None:
            return None
        return spec.origin
    except (ImportError, ValueError):
        return None

def parse_classes_methods(filepath):
    """Return {classname: set(method_names)} via AST (no execution)."""
    if not filepath:
        return {}
    try:
        with open(filepath) as f:
            tree = ast.parse(f.read(), filepath)
    except (SyntaxError, OSError):
        return {}
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            methods = set()
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.add(child.name)
            out[node.name] = methods
    return out

def check(module_path, class_methods):
    """class_methods: {ClassName: [required_method, ...] | True (just exists)}"""
    src = find_module_path(module_path)
    if not src:
        errors.append(f"  {module_path}: module not found")
        return
    klasses = parse_classes_methods(src)
    for cls, want in class_methods.items():
        if cls not in klasses:
            errors.append(f"  {module_path}.{cls} class MISSING")
            continue
        if want is True:
            continue
        for m in want:
            if m not in klasses[cls]:
                errors.append(f"  {module_path}.{cls}.{m} MISSING")

# Critical class + method targets. Method lists track what cuda_mock.py
# patches via meta-path patcher (Worker, GPUModelRunner) and what
# uniproc_executor + scheduler edits hook into.
check("vllm.v1.executor.uniproc_executor", {"UniProcExecutor": ["execute_model"]})
check("vllm.v1.worker.gpu_worker", {"Worker": [
    "load_model", "init_device", "determine_available_memory",
    "get_kv_cache_spec", "get_supported_tasks", "initialize_from_config",
    "get_model"]})
check("vllm.v1.worker.gpu_model_runner", {"GPUModelRunner": [
    "synchronize_input_prep", "load_model", "profile_run",
    "_dummy_run", "get_model"]})
check("vllm.platforms.interface", {"Platform": True, "PlatformEnum": True})
check("vllm.v1.core.sched.scheduler", {"Scheduler": True})

try:
    import vllm
    vllm_ver = getattr(vllm, "__version__", "unknown")
except ImportError:
    vllm_ver = "(import failed — likely no flash_attn ext, but AST checks ran anyway)"

if errors:
    print(f"vllm {vllm_ver} COMPAT CHECK FAILED:")
    for e in errors:
        print(e)
    print("")
    print("If you upgraded vllm, the v4 hook patches need updating. The")
    print("public-API patch surface is in:")
    print("  - vllm/v1/executor/uniproc_executor.py (~7 lines: lazy import")
    print("    + 3 call sites in execute_model / sample_tokens / shutdown)")
    print("  - vllm_emulator/cuda_mock.py (meta-path patcher for Worker")
    print("    and GPUModelRunner methods)")
    sys.exit(1)
else:
    print(f"vllm {vllm_ver} COMPAT CHECK PASSED — all v4 hook targets present.")
PY
