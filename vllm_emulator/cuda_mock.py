"""Minimal CUDA mock for running GPU vLLM on CPU-only hosts.

When VLLM_EMULATOR_MOCK_CUDA=1, this module patches torch.cuda to return
fake values, allowing GPU vLLM code paths to execute on CPU-only hosts.
The emulator hook intercepts execute_model() before any real GPU work,
so the mock only needs to fool startup, scheduling, and memory management.

Usage:
    export VLLM_EMULATOR_MOCK_CUDA=1
    # Then import before vllm:
    import vllm_emulator.cuda_mock  # patches torch.cuda

Or call vllm_emulator.cuda_mock.install() explicitly.
"""

import os
import sys
from unittest.mock import MagicMock

# Default fake GPU memory (12GB, configurable)
_FAKE_GPU_MEMORY = int(os.environ.get("VLLM_EMULATOR_MEMORY",
                                        12 * 1024**3))


def _load_gpu_config():
    """Load GPU metadata from profile pack for accurate CUDA mocking."""
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


def install():
    """Install CUDA mock if enabled via environment variable.

    GPU properties (name, memory, SM count, compute capability) are read
    from the profile pack's model_config.gpu section when available,
    falling back to env vars or A100-like defaults.
    """
    if os.environ.get("VLLM_EMULATOR_MOCK_CUDA", "").lower() not in ("1", "true"):
        return False

    import torch

    if torch.cuda.is_available():
        # Real CUDA available — no need to mock
        return False

    gpu_cfg = _load_gpu_config()

    # Resolve GPU properties: env var > profile pack > defaults
    fake_memory = int(os.environ.get("VLLM_EMULATOR_MEMORY",
                                     gpu_cfg.get("gpu_memory_bytes",
                                                 _FAKE_GPU_MEMORY)))
    fake_name = os.environ.get("VLLM_EMULATOR_GPU_NAME",
                               gpu_cfg.get("gpu_name", "Emulator Device"))
    fake_cc = gpu_cfg.get("gpu_compute_capability", [8, 0])
    fake_sm = gpu_cfg.get("gpu_sm_count", 108)
    fake_gpu_count = int(os.environ.get("VLLM_EMULATOR_NUM_GPUS",
                                        gpu_cfg.get("gpu_count", 1)))

    print(f"[EmulatorCudaMock] Installing CUDA mock for CPU-only host")

    # Patch core CUDA functions
    torch.cuda.is_available = lambda: True
    torch.cuda.device_count = lambda: fake_gpu_count
    torch.cuda.current_device = lambda: 0
    torch.cuda.set_device = lambda *a, **kw: None
    torch.cuda.synchronize = lambda *a, **kw: None
    torch.cuda.empty_cache = lambda: None
    torch.cuda.reset_peak_memory_stats = lambda *a, **kw: None
    torch.cuda.reset_max_memory_allocated = lambda *a, **kw: None
    torch.cuda.reset_max_memory_cached = lambda *a, **kw: None
    torch.cuda.mem_get_info = lambda device=None: (
        fake_memory,  # free
        fake_memory,  # total
    )
    torch.cuda.memory_allocated = lambda device=None: 0
    torch.cuda.max_memory_allocated = lambda device=None: 0
    torch.cuda.memory_reserved = lambda device=None: 0

    # Mock device properties (from profile pack or defaults)
    class FakeDeviceProps:
        name = fake_name
        major = fake_cc[0]
        minor = fake_cc[1]
        total_memory = fake_memory
        multi_processor_count = fake_sm
        max_threads_per_multi_processor = 2048

    torch.cuda.get_device_properties = lambda device=None: FakeDeviceProps()
    torch.cuda.get_device_name = lambda device=None: FakeDeviceProps.name
    torch.cuda.get_device_capability = lambda device=None: (
        fake_cc[0], fake_cc[1])

    # Mock Stream and Event
    class FakeStream:
        def __init__(self, *a, **kw): pass
        def synchronize(self): pass
        def wait_event(self, *a): pass
        def record_event(self, *a): return FakeEvent()
        def __enter__(self): return self
        def __exit__(self, *a): pass

    class FakeEvent:
        def __init__(self, *a, **kw): pass
        def record(self, *a): pass
        def synchronize(self): pass
        def wait(self, *a): pass
        def elapsed_time(self, other): return 0.0
        def query(self): return True

    torch.cuda.Stream = FakeStream
    torch.cuda.Event = FakeEvent
    torch.cuda.current_stream = lambda device=None: FakeStream()
    torch.cuda.default_stream = lambda device=None: FakeStream()

    # Mock CUDA RNG functions
    torch.cuda.manual_seed = lambda seed: None
    torch.cuda.manual_seed_all = lambda seed: None
    torch.cuda.seed = lambda: None
    torch.cuda.seed_all = lambda: None
    torch.cuda.initial_seed = lambda: 0

    # Mock C-level CUDA init to prevent "no NVIDIA driver" error
    if hasattr(torch._C, '_cuda_init'):
        torch._C._cuda_init = lambda: None
    if hasattr(torch._C, '_cuda_getDeviceCount'):
        torch._C._cuda_getDeviceCount = lambda: int(os.environ.get(
            "VLLM_EMULATOR_NUM_GPUS", "1"))
    if hasattr(torch._C, '_cuda_getDevice'):
        torch._C._cuda_getDevice = lambda: 0
    if hasattr(torch._C, '_cuda_setDevice'):
        torch._C._cuda_setDevice = lambda x: None

    # Also mock torch.accelerator (used by newer vLLM code)
    if hasattr(torch, 'accelerator'):
        torch.accelerator.device_count = torch.cuda.device_count
        torch.accelerator.current_device_index = lambda: 0
        torch.accelerator.is_available = lambda: True
        torch.accelerator.synchronize = lambda *a, **kw: None
        torch.accelerator.set_device_index = lambda *a, **kw: None

    print(f"[EmulatorCudaMock] Fake GPU: {FakeDeviceProps.name}, "
          f"{fake_memory // (1024**3)}GB, SM={fake_sm}, "
          f"CC={fake_cc[0]}.{fake_cc[1]}, "
          f"device_count={torch.cuda.device_count()}")

    return True


# Auto-install if env var is set at import time
if os.environ.get("VLLM_EMULATOR_MOCK_CUDA", "").lower() in ("1", "true"):
    install()
