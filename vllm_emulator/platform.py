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


def _load_gpu_config_from_profile() -> dict:
    """Load GPU metadata from the profile pack (if available).

    The profile pack's model_config.gpu section contains GPU properties
    auto-collected during profiling on real hardware. This allows Path B
    (CPU-only emulation) to accurately mimic the profiled GPU.
    """
    import os
    profile_path = os.environ.get("VLLM_EMULATOR_PROFILE_PACK", "")
    if not profile_path or not os.path.exists(profile_path):
        return {}
    try:
        import json
        with open(profile_path) as f:
            pack = json.load(f)
        return pack.get("model_config", {}).get("gpu", {})
    except Exception:
        return {}


# Cached at class level after first access
_GPU_CONFIG_CACHE: dict | None = None


def _get_gpu_config() -> dict:
    global _GPU_CONFIG_CACHE
    if _GPU_CONFIG_CACHE is None:
        _GPU_CONFIG_CACHE = _load_gpu_config_from_profile()
    return _GPU_CONFIG_CACHE


class EmulatorPlatform(Platform):
    """
    Emulator platform that simulates GPU behavior without actual GPU hardware.

    This platform is useful for:
    - Rapid iteration on scheduling algorithms
    - Testing without GPU hardware
    - A/B testing of scheduling policies

    GPU properties (SM count, compute capability, memory, name) are read
    from the profile pack's model_config.gpu section when available,
    falling back to A100-like defaults if no profile is loaded.
    """

    _enum = PlatformEnum.OOT
    device_name = "cuda"  # Must match torch device name
    # device_type determines torch.device() — must be "cuda" when using
    # CUDA mock, "cpu" otherwise. Set dynamically in check_and_update_config.
    device_type: str = "cuda"
    dispatch_key: str = "CPU"
    ray_device_key: str = ""  # Emulator doesn't support Ray

    # Override supported dtypes to exclude fp8 (not emulated)
    @property
    def supported_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16, torch.float32]

    @classmethod
    def get_device_capability(cls, device_id: int = 0):
        """Return device capability from profile pack, or A100-like default."""
        from vllm.platforms.interface import DeviceCapability
        gpu = _get_gpu_config()
        cc = gpu.get("gpu_compute_capability", [8, 0])
        return DeviceCapability(major=cc[0], minor=cc[1])

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        """Return device name from profile pack, or generic default."""
        gpu = _get_gpu_config()
        return gpu.get("gpu_name", "Emulator Device")

    @classmethod
    def get_device_uuid(cls, device_id: int = 0) -> str:
        """Return fake UUID for emulator."""
        return f"emulator-{device_id}-0000-000000000000"

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        """Return GPU memory from profile pack, env override, or default."""
        import os
        env_mem = os.environ.get("VLLM_EMULATOR_MEMORY")
        if env_mem:
            return int(env_mem)
        gpu = _get_gpu_config()
        return gpu.get("gpu_memory_bytes", DEFAULT_EMULATOR_MEMORY)

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
        # Keep device as "cuda" — the CUDA mock handles the actual calls.
        # This ensures the GPU worker uses the same code path as real GPU.
        # Without CUDA mock (CPU-only vLLM build), set device = "cpu".
        import os
        if os.environ.get("VLLM_EMULATOR_MOCK_CUDA", "").lower() in ("1", "true"):
            vllm_config.device_config.device = "cuda"
        else:
            vllm_config.device_config.device = "cpu"
        # Use GPU worker class (same code path as real GPU serving)
        # The emulator hooks intercept execute_model() before any GPU work
        if vllm_config.parallel_config.worker_cls == "auto":
            vllm_config.parallel_config.worker_cls = (
                "vllm.v1.worker.gpu_worker.Worker"
            )
    
    @classmethod
    def check_if_supports_dtype(cls, dtype: "torch.dtype") -> None:
        """Emulator supports all dtypes."""
        pass

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        """Pin memory not available in emulator."""
        return False
    
    @classmethod
    def use_custom_allreduce(cls) -> bool:
        """Custom allreduce not available in emulator."""
        return False
    
    @classmethod
    def num_compute_units(cls, device_id: int = 0) -> int:
        """Return SM count from profile pack, or A100-like default."""
        gpu = _get_gpu_config()
        return gpu.get("gpu_sm_count", 108)

    @classmethod
    def support_static_graph_mode(cls) -> bool:
        """Static graph mode not supported in emulator."""
        return False


def emulator_platform_plugin() -> str | None:
    """
    Platform plugin entry point.
    Only activates when BOTH VLLM_EMULATOR_ENABLE_ORACLE and
    VLLM_EMULATOR_MOCK_CUDA are set. On machines with real GPU,
    the executor hook approach uses the native CUDA platform —
    the emulator platform is only needed for CPU-only (Path B).
    """
    import os
    if (os.environ.get("VLLM_EMULATOR_ENABLE_ORACLE", "").lower() in ("1", "true", "yes")
            and os.environ.get("VLLM_EMULATOR_MOCK_CUDA", "").lower() in ("1", "true", "yes")):
        return "vllm_emulator.platform.EmulatorPlatform"
    return None
