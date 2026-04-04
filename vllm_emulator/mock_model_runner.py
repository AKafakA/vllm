"""Mock model runner for emulator mode.

Provides the minimal interface that the vLLM engine/executor
queries during initialization, without loading any model or
allocating GPU memory.
"""

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class MockModelRunner:
    """Minimal mock that satisfies engine queries without any GPU ops."""

    def __init__(self, vllm_config: "VllmConfig", device: torch.device):
        self.vllm_config = vllm_config
        self.device = device
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.model = None
        self.model_memory_usage = 0  # No real weights loaded
        self.input_registry = None

    def get_kv_cache_spec(self):
        """Return fake KV cache spec based on model config."""
        from vllm.v1.worker.gpu_model_runner import KVCacheSpec

        # Compute spec from model config without loading model
        num_layers = getattr(self.model_config.hf_config, 'num_hidden_layers', 32)
        num_kv_heads = getattr(self.model_config.hf_config, 'num_key_value_heads',
                               getattr(self.model_config.hf_config, 'num_attention_heads', 32))
        head_dim = getattr(self.model_config.hf_config, 'hidden_size', 4096) // \
                   getattr(self.model_config.hf_config, 'num_attention_heads', 32)

        block_size = self.cache_config.block_size or 16

        from vllm.v1.kv_cache_interface import FullAttentionSpec

        specs = {}
        for i in range(num_layers):
            specs[f"model.layers.{i}.self_attn.attn"] = FullAttentionSpec(
                num_kv_heads=num_kv_heads,
                head_size=head_dim,
                dtype=self.model_config.dtype,
                block_size=block_size,
            )
        return specs

    def load_model(self, **kwargs):
        """No-op — emulator doesn't load real weights."""
        pass

    def initialize_kv_cache(self, kv_cache_config):
        """No-op — emulator doesn't allocate KV cache."""
        pass

    def update_max_model_len(self, max_model_len):
        """Update max model len."""
        pass

    def init_fp8_kv_scales(self):
        """No-op."""
        pass

    def profile_run(self):
        """No-op — return 0 peak memory."""
        return 0

    def reset_mm_cache(self):
        pass

    def get_mm_cache(self):
        return None

    def get_encoder_timer_stats(self):
        return {}

    def __getattr__(self, name):
        """Catch-all for any other methods the engine might call."""
        def no_op(*args, **kwargs):
            return None
        return no_op
