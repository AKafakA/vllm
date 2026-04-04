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


def install():
    """Install CUDA mock if enabled via environment variable."""
    if os.environ.get("VLLM_EMULATOR_MOCK_CUDA", "").lower() not in ("1", "true"):
        return False

    import torch

    if torch.cuda.is_available():
        # Real CUDA available — no need to mock
        return False

    print("[EmulatorCudaMock] Installing CUDA mock for CPU-only host")

    # Save original functions
    _orig_is_available = torch.cuda.is_available
    _orig_device_count = torch.cuda.device_count

    # Patch core CUDA functions
    torch.cuda.is_available = lambda: True
    torch.cuda.device_count = lambda: int(os.environ.get(
        "VLLM_EMULATOR_NUM_GPUS", "1"))
    torch.cuda.current_device = lambda: 0
    torch.cuda.set_device = lambda *a, **kw: None
    torch.cuda.synchronize = lambda *a, **kw: None
    torch.cuda.empty_cache = lambda: None
    torch.cuda.reset_peak_memory_stats = lambda *a, **kw: None
    torch.cuda.reset_max_memory_allocated = lambda *a, **kw: None
    torch.cuda.reset_max_memory_cached = lambda *a, **kw: None
    torch.cuda.mem_get_info = lambda device=None: (
        _FAKE_GPU_MEMORY,  # free
        _FAKE_GPU_MEMORY,  # total
    )
    torch.cuda.memory_allocated = lambda device=None: 0
    torch.cuda.max_memory_allocated = lambda device=None: 0
    torch.cuda.memory_reserved = lambda device=None: 0

    # Mock device properties
    class FakeDeviceProps:
        name = os.environ.get("VLLM_EMULATOR_GPU_NAME", "Emulator Device")
        major = 8
        minor = 0
        total_memory = _FAKE_GPU_MEMORY
        multi_processor_count = 108  # A100-like
        max_threads_per_multi_processor = 2048

    torch.cuda.get_device_properties = lambda device=None: FakeDeviceProps()
    torch.cuda.get_device_name = lambda device=None: FakeDeviceProps.name
    torch.cuda.get_device_capability = lambda device=None: (8, 0)

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

    print(f"[EmulatorCudaMock] Fake GPU: {FakeDeviceProps.name}, "
          f"{_FAKE_GPU_MEMORY // (1024**3)}GB, "
          f"device_count={torch.cuda.device_count()}")

    return True


# Auto-install if env var is set at import time
if os.environ.get("VLLM_EMULATOR_MOCK_CUDA", "").lower() in ("1", "true"):
    install()
