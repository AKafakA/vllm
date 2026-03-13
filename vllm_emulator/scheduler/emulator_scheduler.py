"""Emulator scheduler with Prefill/Decode (PD) separation support."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from vllm_emulator.oracle.base import BaseGpuCostOracle

logger = logging.getLogger(__name__)


class PDSchedulingPolicy(Enum):
    """PD separation scheduling policies."""
    PREFILL_FIRST = "prefill_first"  # Process all pending prefills before decode
    DECODE_FIRST = "decode_first"    # Process pending decodes before prefill
    HYBRID = "hybrid"                # Interleave prefill and decode based on priority
    DISAGGREGATED = "disaggregated"  # Full separation with KV cache transfer


@dataclass
class Request:
    """Represents a generation request in the emulator."""
    request_id: str
    prompt_tokens: int
    max_tokens: int = 100
    generated_tokens: int = 0
    is_prefill_done: bool = False
    priority: int = 0

    @property
    def pending_tokens(self) -> int:
        """Number of tokens still to generate."""
        return self.max_tokens - self.generated_tokens


@dataclass
class SchedulingDecision:
    """Result of a scheduling decision."""
    prefill_batch: list[Request] = field(default_factory=list)
    decode_batch: list[Request] = field(default_factory=list)
    prefill_time_us: float = 0.0
    decode_time_us: float = 0.0


class EmulatorScheduler:
    """Scheduler with optional PD separation support.
    
    Supports both traditional joint scheduling and disaggregated
    prefill/decode (PD) scheduling modes.
    """

    def __init__(
        self,
        cost_oracle: BaseGpuCostOracle,
        enable_pd_separation: bool = False,
        policy: PDSchedulingPolicy = PDSchedulingPolicy.HYBRID,
        max_batch_size: int = 32,
    ):
        """Initialize the emulator scheduler.
        
        Args:
            cost_oracle: Oracle for estimating prefill/decode latencies.
            enable_pd_separation: Enable PD-separated scheduling.
            policy: Scheduling policy when PD separation is enabled.
            max_batch_size: Maximum batch size for any phase.
        """
        self._oracle = cost_oracle
        self._enable_pd = enable_pd_separation
        self._policy = policy
        self._max_batch_size = max_batch_size
        
        # Queues for PD-separated scheduling
        self._prefill_queue: list[Request] = []
        self._decode_queue: list[Request] = []
        
        # Active requests that are being decoded
        self._active_decodes: list[Request] = []

    @property
    def enable_pd_separation(self) -> bool:
        """Return whether PD separation is enabled."""
        return self._enable_pd

    @property
    def policy(self) -> PDSchedulingPolicy:
        """Return the current scheduling policy."""
        return self._policy

    def add_request(self, request: Request) -> None:
        """Add a new request to the scheduler.
        
        Args:
            request: The request to add.
        """
        if self._enable_pd:
            # In PD mode, new requests start in prefill queue
            self._prefill_queue.append(request)
            logger.debug(f"Added request {request.request_id} to prefill queue")
        else:
            # In non-PD mode, we'd handle joint scheduling
            self._prefill_queue.append(request)
            logger.debug(f"Added request {request.request_id} (joint mode)")

    def schedule(self) -> SchedulingDecision:
        """Schedule the next batch of prefill/decode operations.
        
        Returns:
            SchedulingDecision containing the batches to process.
        """
        if not self._enable_pd:
            return self._schedule_joint()
        return self._schedule_pd_separated()

    def _schedule_joint(self) -> SchedulingDecision:
        """Schedule with joint prefill+decode (non-PD mode)."""
        # Simple joint scheduling: process pending requests together
        decision = SchedulingDecision()
        
        # Move all pending prefill requests to prefill batch
        if self._prefill_queue:
            batch = self._prefill_queue[:self._max_batch_size]
            decision.prefill_batch = batch
            
            # Calculate prefill time
            total_tokens = sum(r.prompt_tokens for r in batch)
            decision.prefill_time_us = self._oracle.estimate_prefill_latency_us(
                total_tokens, len(batch)
            )
            
            # Remove processed from queue
            self._prefill_queue = self._prefill_queue[len(batch):]
            
            # Mark these as prefill done
            for r in batch:
                r.is_prefill_done = True
            
            # Add to active decodes
            self._active_decodes.extend(batch)

        # Process decode batch
        if self._active_decodes:
            batch = self._active_decodes[:self._max_batch_size]
            decision.decode_batch = batch
            
            decision.decode_time_us = self._oracle.estimate_decode_latency_us(len(batch))
            
            # Remove processed
            self._active_decodes = self._active_decodes[len(batch):]

        return decision

    def _schedule_pd_separated(self) -> SchedulingDecision:
        """Schedule with PD-separated prefill and decode."""
        decision = SchedulingDecision()
        
        if self._policy == PDSchedulingPolicy.PREFILL_FIRST:
            # Process all pending prefills before any decode
            if self._prefill_queue:
                batch = self._prefill_queue[:self._max_batch_size]
                decision.prefill_batch = batch
                
                total_tokens = sum(r.prompt_tokens for r in batch)
                decision.prefill_time_us = self._oracle.estimate_prefill_latency_us(
                    total_tokens, len(batch)
                )
                
                self._prefill_queue = self._prefill_queue[len(batch):]
                
                for r in batch:
                    r.is_prefill_done = True
                self._active_decodes.extend(batch)
                
        elif self._policy == PDSchedulingPolicy.DECODE_FIRST:
            # Process pending decodes before new prefills
            if self._active_decodes:
                batch = self._active_decodes[:self._max_batch_size]
                decision.decode_batch = batch
                decision.decode_time_us = self._oracle.estimate_decode_latency_us(len(batch))
                self._active_decodes = self._active_decodes[len(batch):]
                
            # Then do prefill if decode queue is empty
            if not decision.decode_batch and self._prefill_queue:
                batch = self._prefill_queue[:self._max_batch_size]
                decision.prefill_batch = batch
                
                total_tokens = sum(r.prompt_tokens for r in batch)
                decision.prefill_time_us = self._oracle.estimate_prefill_latency_us(
                    total_tokens, len(batch)
                )
                
                self._prefill_queue = self._prefill_queue[len(batch):]
                
                for r in batch:
                    r.is_prefill_done = True
                self._active_decodes.extend(batch)
                
        elif self._policy == PDSchedulingPolicy.HYBRID:
            # Hybrid: interleave based on some heuristic
            # Simple approach: process decode first if there are requests
            # waiting, otherwise do prefill
            if self._active_decodes and len(self._active_decodes) >= len(self._prefill_queue):
                batch = self._active_decodes[:self._max_batch_size]
                decision.decode_batch = batch
                decision.decode_time_us = self._oracle.estimate_decode_latency_us(len(batch))
                self._active_decodes = self._active_decodes[len(batch):]
            else:
                if self._prefill_queue:
                    batch = self._prefill_queue[:self._max_batch_size]
                    decision.prefill_batch = batch
                    
                    total_tokens = sum(r.prompt_tokens for r in batch)
                    decision.prefill_time_us = self._oracle.estimate_prefill_latency_us(
                        total_tokens, len(batch)
                    )
                    
                    self._prefill_queue = self._prefill_queue[len(batch):]
                    
                    for r in batch:
                        r.is_prefill_done = True
                    self._active_decodes.extend(batch)
                    
        elif self._policy == PDSchedulingPolicy.DISAGGREGATED:
            # Full disaggregation: prefill and decode on separate "nodes"
            # For now, same as hybrid but with explicit phase tracking
            return self._schedule_pd_separated()  # Delegate to hybrid for now

        return decision

    def get_queue_stats(self) -> dict[str, Any]:
        """Get current queue statistics."""
        return {
            "prefill_queue_size": len(self._prefill_queue),
            "active_decodes": len(self._active_decodes),
            "pd_enabled": self._enable_pd,
            "policy": self._policy.value,
        }


def create_scheduler(
    cost_oracle: BaseGpuCostOracle,
    enable_pd_separation: bool = False,
    policy: str = "hybrid",
    **kwargs,
) -> EmulatorScheduler:
    """Factory function to create an emulator scheduler.
    
    Args:
        cost_oracle: The cost oracle to use.
        enable_pd_separation: Enable PD-separated scheduling.
        policy: Scheduling policy name (prefill_first, decode_first, hybrid, disaggregated).
        **kwargs: Additional scheduler configuration.
        
    Returns:
        Configured EmulatorScheduler instance.
    """
    policy_enum = PDSchedulingPolicy(policy)
    return EmulatorScheduler(
        cost_oracle=cost_oracle,
        enable_pd_separation=enable_pd_separation,
        policy=policy_enum,
        **kwargs,
    )
