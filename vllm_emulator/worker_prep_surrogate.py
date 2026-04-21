"""GPU-free surrogate for worker-side CPU preparation overhead.

Runs CPU operations equivalent to what vLLM's worker performs between
scheduler.step() and GPU forward: request bookkeeping, position/index
array construction, attention metadata, per-layer metadata, and
sampling-output bookkeeping.  The wall-clock cost is returned so the
caller can subtract it from the profiled step_cycle (avoiding
double-count).

All model parameters come from model_config with NO defaults; missing
keys raise KeyError loudly. Arrays are allocated dynamically per step.
"""

import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput


# vLLM's default CUDA-graph capture batch sizes. Framework constant,
# not a tuning knob. If vLLM changes these, update here.
_VLLM_CAPTURE_SIZES = (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128,
                       160, 192, 224, 256, 320, 384, 448, 512)


class WorkerPrepSurrogate:
    """Lightweight CPU-side surrogate for worker prep overhead."""

    def __init__(self, model_config: dict):
        # All fields required; no fallbacks.
        self._num_layers = model_config["num_hidden_layers"]
        self._hidden_size = model_config["hidden_size"]
        self._num_heads = model_config["num_attention_heads"]
        self._head_dim = self._hidden_size // max(self._num_heads, 1)
        self._vocab_size = model_config["vocab_size"]
        self._max_model_len = model_config["max_model_len"]
        self._block_size = model_config["block_size"]

        # Persistent request state (mimics input_batch tracking).
        self._active_req_ids: set = set()
        self._req_positions: dict = {}
        self._num_computed: dict = {}

        self._step_count = 0

    def run_prep_surrogate(
        self, scheduler_output: "SchedulerOutput"
    ) -> float:
        t0 = time.perf_counter()

        num_scheduled = scheduler_output.num_scheduled_tokens
        new_reqs = scheduler_output.scheduled_new_reqs
        total_tokens = scheduler_output.total_num_scheduled_tokens
        num_reqs = len(num_scheduled)

        if num_reqs == 0:
            return 0.0

        # Phase 1: _update_states
        for req in new_reqs:
            rid = req.req_id
            self._active_req_ids.add(rid)
            self._req_positions[rid] = 0
            self._num_computed[rid] = 0

        # Phase 2: _prepare_inputs
        req_ids = list(num_scheduled.keys())
        tokens_per_req = np.array(
            [num_scheduled[rid] for rid in req_ids], dtype=np.int32)

        req_indices = np.repeat(
            np.arange(num_reqs, dtype=np.int32), tokens_per_req)
        query_start = np.zeros(num_reqs + 1, dtype=np.int32)
        query_start[1:] = np.cumsum(tokens_per_req)

        positions = np.empty(total_tokens, dtype=np.int64)
        for i, rid in enumerate(req_ids):
            start = int(query_start[i])
            end = int(query_start[i + 1])
            pos = self._num_computed.get(rid, 0)
            positions[start:end] = np.arange(
                pos, pos + int(tokens_per_req[i]), dtype=np.int64)

        seq_lens = np.array(
            [self._num_computed.get(rid, 0) + int(tokens_per_req[i])
             for i, rid in enumerate(req_ids)],
            dtype=np.int32)

        # Phase 3: _determine_batch_execution (padding + capture-size pick)
        is_uniform = bool(np.all(tokens_per_req == 1))
        if is_uniform:
            padded_total = total_tokens
        else:
            padded_total = ((total_tokens + 7) // 8) * 8
        graph_size = padded_total
        for s in _VLLM_CAPTURE_SIZES:
            if s >= padded_total:
                graph_size = s
                break

        # Phase 4: _build_attention_metadata (slot mapping + block table)
        max_seq_len = int(seq_lens.max()) if num_reqs > 0 else 0
        max_num_blocks = max_seq_len // self._block_size + 1
        block_table = np.zeros((num_reqs, max_num_blocks), dtype=np.int32)
        slot_mapping = np.empty(total_tokens, dtype=np.int64)
        for i, rid in enumerate(req_ids):
            n_blocks = (int(seq_lens[i]) + self._block_size - 1) // self._block_size
            block_table[i, :n_blocks] = np.arange(n_blocks, dtype=np.int32)
            start = int(query_start[i])
            end = int(query_start[i + 1])
            base_slot = self._num_computed.get(rid, 0)
            slot_mapping[start:end] = np.arange(
                base_slot, base_slot + int(tokens_per_req[i]), dtype=np.int64)

        # Phase 5: _bookkeeping_sync
        for i, rid in enumerate(req_ids):
            self._num_computed[rid] = self._num_computed.get(rid, 0) + int(tokens_per_req[i])
            self._req_positions[rid] = self._num_computed[rid]

        # Phase 5b: Per-layer attention metadata (real worker iterates
        # over kv_cache_groups × attn_groups per layer).
        for _ in range(self._num_layers):
            _layer_q = query_start.copy()
            _layer_k = seq_lens.cumsum()
            _layer_blocks = block_table[:, :max_num_blocks]

        # Phase 6: sample + output bookkeeping
        output_token_ids = []
        for i, rid in enumerate(req_ids):
            tok = (self._step_count * 7 + i * 13) % self._vocab_size
            output_token_ids.append(tok)

        req_output_map = {}
        for i, rid in enumerate(req_ids):
            req_output_map[rid] = {
                "token_id": output_token_ids[i],
                "position": self._num_computed.get(rid, 0),
                "seq_len": int(seq_lens[i]),
            }

        # Placeholder logprobs — the model doesn't read these, they exist only
        # to match the output tensor shape/dtype profile. Use a deterministic
        # numpy Generator seeded from the step counter so runs are bit-identical
        # across reproductions (process-global np.random would otherwise diverge).
        if not hasattr(self, "_logprob_rng"):
            self._logprob_rng = np.random.default_rng(seed=0xC0FFEE)
        _logprob_placeholder = self._logprob_rng.random(num_reqs, dtype=np.float32)
        _top_k = np.argsort(_logprob_placeholder)

        # Retire finished requests.
        finished = set()
        for rid in self._active_req_ids:
            if self._num_computed.get(rid, 0) >= self._max_model_len:
                finished.add(rid)
        for rid in finished:
            self._active_req_ids.discard(rid)
            self._req_positions.pop(rid, None)
            self._num_computed.pop(rid, None)

        self._step_count += 1
        return time.perf_counter() - t0
