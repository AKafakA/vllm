"""GPU-free surrogate for worker-side CPU preparation overhead.

On real GPU, worker.execute_model() spends ~2-4ms on CPU-side input
preparation before GPU forward pass. The executor hook skips this entirely.
This surrogate performs equivalent CPU work to recover the timing gap
without requiring GPU memory or model weights.

The surrogate time is subtracted from the timer delay to avoid double
counting (profiled step_cycle already includes this prep time).

Key functions replicated (CPU-equivalent work only):
- _update_states: request lifecycle bookkeeping (~600us)
- _prepare_inputs: position/index array construction (~700us)
- _determine_batch_execution: batch shape dispatch (~200us)
- _build_attention_metadata: metadata object construction (~400us)
- _bookkeeping_sync equivalent: output token tracking (~600us)
"""

import time
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput


class WorkerPrepSurrogate:
    """Lightweight CPU-side surrogate for worker prep overhead."""

    def __init__(self, model_config: dict):
        """Initialize with model config (from HuggingFace or profile).

        Args:
            model_config: dict with keys like num_layers, hidden_size,
                         num_attention_heads, vocab_size, max_model_len
        """
        self._num_layers = model_config.get("num_hidden_layers", 28)
        self._hidden_size = model_config.get("hidden_size", 1536)
        self._num_heads = model_config.get("num_attention_heads", 12)
        self._head_dim = self._hidden_size // max(self._num_heads, 1)
        self._vocab_size = model_config.get("vocab_size", 151936)
        self._max_model_len = model_config.get("max_model_len", 4096)
        self._block_size = model_config.get("block_size", 16)

        # Persistent state (mimics model_runner's input_batch tracking)
        self._active_req_ids: set = set()
        self._req_positions: dict = {}  # req_id -> current position
        self._num_computed: dict = {}   # req_id -> num computed tokens

        # Pre-allocated arrays for reuse (avoids allocation overhead)
        self._max_batch = 512
        self._positions_buf = np.zeros(self._max_batch, dtype=np.int64)
        self._seq_lens_buf = np.zeros(self._max_batch, dtype=np.int32)
        self._query_start_buf = np.zeros(self._max_batch + 1, dtype=np.int32)
        self._slot_mapping_buf = np.zeros(self._max_batch, dtype=np.int64)
        self._block_table_buf = np.zeros(
            (self._max_batch, self._max_model_len // self._block_size + 1),
            dtype=np.int32)

        self._step_count = 0

    def run_prep_surrogate(
        self, scheduler_output: "SchedulerOutput"
    ) -> float:
        """Run CPU-equivalent of worker prep work. Returns wall-clock time in seconds.

        This performs the same class of CPU operations as the real worker:
        - Request state updates (dict/set operations)
        - NumPy array construction (positions, indices, seq_lens)
        - Metadata object building (block tables, slot mappings)
        - Output bookkeeping (token tracking)

        The operations are designed to take approximately the same CPU time
        as the real worker prep for the same batch composition.
        """
        t0 = time.perf_counter()

        num_scheduled = scheduler_output.num_scheduled_tokens
        new_reqs = scheduler_output.scheduled_new_reqs
        total_tokens = scheduler_output.total_num_scheduled_tokens
        num_reqs = len(num_scheduled)

        if num_reqs == 0:
            return 0.0

        # Phase 1: _update_states equivalent (~600us)
        # Request lifecycle: add new, track active
        new_req_ids = set()
        for req in new_reqs:
            rid = req.req_id
            new_req_ids.add(rid)
            self._active_req_ids.add(rid)
            self._req_positions[rid] = 0
            self._num_computed[rid] = 0

        # Phase 2: _prepare_inputs equivalent (~700us)
        # Build position arrays, request indices, cumulative sums
        req_ids = list(num_scheduled.keys())
        tokens_per_req = [num_scheduled[rid] for rid in req_ids]

        # np.repeat equivalent (maps tokens to request indices)
        if total_tokens <= self._max_batch:
            req_indices = np.repeat(
                np.arange(num_reqs, dtype=np.int32),
                tokens_per_req)

            # Cumulative token positions
            cum_tokens = np.cumsum(tokens_per_req)
            self._query_start_buf[:num_reqs + 1] = 0
            self._query_start_buf[1:num_reqs + 1] = cum_tokens

            # Position computation
            for i, rid in enumerate(req_ids):
                start = self._query_start_buf[i]
                end = self._query_start_buf[i + 1]
                pos = self._num_computed.get(rid, 0)
                self._positions_buf[start:end] = np.arange(
                    pos, pos + tokens_per_req[i], dtype=np.int64)

            # Seq lens
            self._seq_lens_buf[:num_reqs] = [
                self._num_computed.get(rid, 0) + tokens_per_req[i]
                for i, rid in enumerate(req_ids)]

        # Phase 3: _determine_batch_execution equivalent (~200us)
        # Check uniform decode, compute padding
        is_uniform = all(t == 1 for t in tokens_per_req)
        if is_uniform:
            padded_total = total_tokens
        else:
            # Pad to next multiple of 8 (like real code)
            padded_total = ((total_tokens + 7) // 8) * 8

        # CUDA graph size selection
        capture_sizes = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128,
                        160, 192, 224, 256, 320, 384, 448, 512]
        graph_size = padded_total
        for s in capture_sizes:
            if s >= padded_total:
                graph_size = s
                break

        # Phase 4: _build_attention_metadata equivalent (~400us)
        # Build slot mappings and block table metadata
        if total_tokens <= self._max_batch:
            for i, rid in enumerate(req_ids):
                seq_len = self._seq_lens_buf[i]
                num_blocks = (seq_len + self._block_size - 1) // self._block_size
                # Simulate block table construction
                self._block_table_buf[i, :num_blocks] = np.arange(
                    num_blocks, dtype=np.int32)
                # Slot mapping
                start = self._query_start_buf[i]
                end = self._query_start_buf[i + 1]
                base_slot = self._num_computed.get(rid, 0)
                self._slot_mapping_buf[start:end] = np.arange(
                    base_slot, base_slot + tokens_per_req[i], dtype=np.int64)

        # Phase 5: _bookkeeping_sync equivalent (~600us)
        # Update computed tokens and track output
        for i, rid in enumerate(req_ids):
            self._num_computed[rid] = self._num_computed.get(rid, 0) + tokens_per_req[i]
            self._req_positions[rid] = self._num_computed[rid]

        # Phase 5b: Per-layer attention metadata construction
        # Real worker iterates over kv_cache_groups × attn_groups
        # building backend-specific metadata per layer. This is a
        # significant chunk of the CPU prep cost.
        for layer_idx in range(self._num_layers):
            # Simulate per-layer metadata: query offsets, key offsets
            if total_tokens <= self._max_batch:
                _layer_query_offsets = self._query_start_buf[:num_reqs + 1].copy()
                _layer_key_offsets = self._seq_lens_buf[:num_reqs].cumsum()
                # Simulate block table slice per layer (real worker does this)
                _layer_blocks = self._block_table_buf[:num_reqs, :max(1, (max(self._seq_lens_buf[:num_reqs]) if num_reqs > 0 else 1) // self._block_size + 1)]

        # Phase 5c: Slot mapping commit (real worker commits per-request)
        if total_tokens <= self._max_batch:
            for i, rid in enumerate(req_ids):
                start = self._query_start_buf[i]
                end = self._query_start_buf[i + 1]
                # Simulate slot mapping verification (bounds check per slot)
                _slots = self._slot_mapping_buf[start:end]
                _valid = _slots >= 0

        # Phase 6: sample_tokens equivalent (~1.25ms on real GPU)
        # Output token construction, bookkeeping sync, token tracking
        # Real worker: _sample() + _bookkeeping_sync()
        output_token_ids = []
        for i, rid in enumerate(req_ids):
            # Simulate sampled token ID selection and tracking
            tok = (self._step_count * 7 + i * 13) % self._vocab_size
            output_token_ids.append(tok)
            # Update prev_sampled_token_ids tracking (like real bookkeeping)
            self._req_positions[rid] = self._num_computed.get(rid, 0)

        # Simulate output object construction per request
        # Real worker builds ModelRunnerOutput with per-request token lists
        req_output_map = {}
        for i, rid in enumerate(req_ids):
            req_output_map[rid] = {
                "token_id": output_token_ids[i],
                "position": self._num_computed.get(rid, 0),
                "seq_len": self._seq_lens_buf[i] if i < len(self._seq_lens_buf) else 0,
            }

        # Simulate logprobs computation placeholder (real worker does GPU→CPU copy)
        if num_reqs > 0:
            # Array operations that mimic logprobs extraction
            _logprob_placeholder = np.random.random(num_reqs).astype(np.float32)
            _top_k = np.argsort(_logprob_placeholder)

        # Clean up finished requests (simulate)
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
