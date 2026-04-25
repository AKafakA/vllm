"""Hook system for vLLM emulator.

The executor-level hook is the only supported hook in the v4 (CUDA-invisible)
path. Worker/scheduler/network/offload hooks were removed in the Apr 25 cleanup
since their concrete implementations are out of scope for the paper artifact.
"""

from .executor_hook import ExecutorEmulatorHook, get_executor_hook

__all__ = ["ExecutorEmulatorHook", "get_executor_hook"]
