# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
vLLM Emulator Platform - A platform plugin for emulating GPU inference.
"""

from typing import TYPE_CHECKING

import torch

from vllm.platforms.interface import Platform, PlatformEnum

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None

# Default emulator device memory (80GB, similar to H100)
DEFAULT_EMULATOR_MEMORY = 80 * 1024**3  # 80 GB


class EmulatorPlatform(Platform):
    """
    Emulator platform that simulates GPU behavior without actual GPU hardware.
    
    This platform is useful for:
    - Rapid iteration on scheduling algorithms
    - Testing without GPU hardware
    - A/B testing of scheduling policies
    """
    
    _enum = PlatformEnum.OOT
    device_name = "Emulator"
    device_type: str = "cpu"  # Use CPU device so torch.device() works
    dispatch_key: str = "CPU"
    ray_device_key: str = ""  # Emulator doesn't support Ray
    
    # Override supported dtypes to exclude fp8 (not emulated)
    @property
    def supported_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16, torch.float32]
    
    @classmethod
    def get_device_capability(cls, device_id: int = 0):
        """Return fake device capability (8.0 for A100-like emulation)."""
        from vllm.platforms.interface import DeviceCapability
        return DeviceCapability(major=8, minor=0)
    
    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        """Return emulator device name."""
        return "Emulator Device"
    
    @classmethod
    def get_device_uuid(cls, device_id: int = 0) -> str:
        """Return fake UUID for emulator."""
        return f"emulator-{device_id}-0000-000000000000"
    
    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        """Return configured emulator memory (default 80GB)."""
        import os
        return int(os.environ.get("VLLM_EMULATOR_MEMORY", DEFAULT_EMULATOR_MEMORY))
    
    @classmethod
    def set_device(cls, device: torch.device) -> None:
        """No-op for emulator (no actual device)."""
        pass
    
    @classmethod
    def get_current_memory_usage(cls, device=None) -> float:
        """Return fake memory usage (always 0 for now)."""
        return 0.0
    
    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        """Update config for emulator mode."""
        # Disable compilation that requires real GPU
        vllm_config.compilation_config.enable = False
        # Use CPU for all computation
        vllm_config.device_config.device = "cpu"
    
    @classmethod
    def is_pin_memory_available(cls) -> bool:
        """Pin memory not available in emulator."""
        return False
    
    @classmethod
    def use_custom_allreduce(cls) -> bool:
        """Custom allreduce not available in emulator."""
        return False
    
    @classmethod
    def support_static_graph_mode(cls) -> bool:
        """Static graph mode not supported in emulator."""
        return False


def emulator_platform_plugin() -> str | None:
    """
    Platform plugin entry point.
    Only activates when VLLM_EMULATOR_ENABLE_ORACLE is set,
    so real GPU profiling/serving works normally when the
    emulator package is installed but not enabled.
    """
    import os
    if os.environ.get("VLLM_EMULATOR_ENABLE_ORACLE", "").lower() in ("1", "true", "yes"):
        return "vllm_emulator.platform.EmulatorPlatform"
    return None
