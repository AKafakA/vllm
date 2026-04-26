# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
vLLM Emulator - A backend emulator for vLLM that enables rapid iteration
on scheduling algorithms without GPU hardware.
"""

__version__ = "0.1.0"

# Auto-install cuda_mock module at vllm_emulator package import. Fires in
# every Python process that imports vllm_emulator, including EngineCore
# subprocesses spawned by vLLM. cuda_mock self-gates internally:
#   - install() (CUDA shims) only runs when VLLM_EMULATOR_MOCK_CUDA=1 AND
#     real CUDA is unavailable (cpu_host case).
#   - install_model_stubs() (skip load_model/profile_run/etc.) runs
#     whenever VLLM_EMULATOR_ENABLE_ORACLE=1 — applies on both cpu_host
#     and real-GPU hosts so the GPU isn't locked by resident weights.
import os as _os
if (_os.environ.get("VLLM_EMULATOR_MOCK_CUDA", "").lower() in ("1", "true", "yes")
        or _os.environ.get("VLLM_EMULATOR_ENABLE_ORACLE", "").lower() in ("1", "true", "yes")):
    try:
        from . import cuda_mock as _cuda_mock  # noqa: F401  (auto-installs on import)
    except Exception:
        pass

# Keep top-level platform import lightweight so utility modules (e.g.
# profile tooling) can be used without full vLLM/torch runtime dependencies.
try:
    from .platform import EmulatorPlatform, emulator_platform_plugin

    __all__ = [
        "EmulatorPlatform",
        "emulator_platform_plugin",
    ]
except Exception:  # pragma: no cover - optional dependency path
    __all__ = []
