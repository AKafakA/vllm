# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
vLLM Emulator - A backend emulator for vLLM that enables rapid iteration
on scheduling algorithms without GPU hardware.
"""

__version__ = "0.1.0"

# Keep top-level import lightweight so utility modules (e.g. profile tooling)
# can be used without full vLLM/torch runtime dependencies.
try:
    from .platform import EmulatorPlatform, emulator_platform_plugin

    __all__ = [
        "EmulatorPlatform",
        "emulator_platform_plugin",
    ]
except Exception:  # pragma: no cover - optional dependency path
    __all__ = []
