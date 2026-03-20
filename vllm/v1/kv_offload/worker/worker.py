# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from dataclasses import dataclass

from vllm.logger import init_logger
from vllm.v1.kv_offload.abstract import LoadStoreSpec

# Optional emulator hook - lazy import to avoid hard dependency
_EMULATOR_OFFLOAD_HOOK_MODULE = None


def _get_emulator_offload_hook():
    """Lazy import emulator offload hook to avoid hard dependency."""
    global _EMULATOR_OFFLOAD_HOOK_MODULE
    if _EMULATOR_OFFLOAD_HOOK_MODULE is None:
        try:
            from vllm_emulator.hooks import OffloadWorkerHook
            _EMULATOR_OFFLOAD_HOOK_MODULE = OffloadWorkerHook
        except ImportError:
            _EMULATOR_OFFLOAD_HOOK_MODULE = False
    return _EMULATOR_OFFLOAD_HOOK_MODULE if _EMULATOR_OFFLOAD_HOOK_MODULE else None

# a single transfer spec (src_blocks_spec, dst_blocks_spec)
TransferSpec = tuple[LoadStoreSpec, LoadStoreSpec]
# transfers are forwarded to workers by (src_medium, dst_medium)
TransferType = tuple[str, str]

logger = init_logger(__name__)


@dataclass
class TransferResult:
    job_id: int
    success: bool
    transfer_size: int | None = None  # Size in bytes
    transfer_time: float | None = None
    transfer_type: TransferType | None = None


class OffloadingHandler(ABC):
    """
    OffloadingHandler class for managing asynchronous KV data transfers

    This class runs in the worker.
    It kicks off async KV data transfer requests, and allows
    collecting back completion statuses.

    The class provides the following primitives:
        transfer_async() - kicks off a new transfer job
        get_finished() - returns a list of newly finished job IDs.
    """

    @abstractmethod
    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        """
        Initiates an asynchronous transfer of KV data.

        Args:
            job_id: a unique ID that will be used when notifying back on
                transfer completion.
            spec: the (src, dst) spec of the KV data transfer.

        Returns:
            True if transfer was submitted successfully.
        """
        pass

    @abstractmethod
    def get_finished(self) -> list[TransferResult]:
        """
        Get transfers finished since last call.

        Returns:
            A list of (job_id, success) of transfers.
        """
        pass

    @abstractmethod
    def wait(self, job_ids: set[int]) -> None:
        """
        Wait for jobs to finish (blocking).
        Args:
            job_ids: The set of job IDs to wait for.
        """


class OffloadingWorker:
    """
    OffloadingWorker class for managing asynchronous KV data transfers
    using multiple OffloadingHandlers

    This class runs in the worker.
    It kicks off async KV data transfer requests, by delegating
    to one of its registered OffloadingHandlers, based on the transfer type.

    The class provides the following primitives:
        register_handler() - registers a new handler to handle
            a specific transfer type
        transfer_async() - kicks off a new transfer job
            using one of the registered handlers.
        get_finished() - returns a list of newly finished job IDs
            from all handlers.
    """

    def __init__(self):
        self.handlers: set[OffloadingHandler] = set()
        self.transfer_type_to_handler: dict[TransferType, OffloadingHandler] = {}

        # Optional emulator hook for offload cost estimation (lazy loaded)
        self._emulator_hook = None
        self._emulator_finished: list[TransferResult] = []
        hook_cls = _get_emulator_offload_hook()
        if hook_cls is not None:
            try:
                self._emulator_hook = hook_cls(self)
            except Exception as e:
                logger.warning(
                    "Failed to initialize emulator offload hook: %r",
                    e,
                    exc_info=True,
                )

    def register_handler(
        self,
        src_cls: type[LoadStoreSpec],
        dst_cls: type[LoadStoreSpec],
        handler: OffloadingHandler,
    ) -> None:
        """
        Registers a new handler.

        Args:
            src_cls: the source type of transfers handled by this handler.
            dst_cls: the destination type of transfers handled by this handler.
            handler: the handler that will handle transfers.
        """
        transfer_type = (src_cls.medium(), dst_cls.medium())
        assert transfer_type not in self.transfer_type_to_handler
        self.handlers.add(handler)
        self.transfer_type_to_handler[transfer_type] = handler

    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        """
        Initiates an asynchronous transfer of KV data.

        Args:
            job_id: a unique ID that will be used when notifying back on
                transfer completion.
            spec: the (src, dst) spec of the KV data transfer.

        Returns:
            True if transfer was submitted successfully.
        """
        src, dst = spec

        # Emulator mode: use oracle to estimate transfer cost
        # instead of running real transfers
        if (self._emulator_hook is not None
                and self._emulator_hook.is_enabled):
            cost = self._emulator_hook.estimate_transfer_cost(src, dst)
            logger.debug(
                "Emulator offload oracle: job=%d lookup=%.2fus "
                "transfer=%.2fus total=%.2fus",
                job_id,
                cost["lookup_latency_us"],
                cost["transfer_latency_us"],
                cost["total_estimated_us"],
            )
            self._emulator_hook.apply_oracle_delay(src, dst)
            # Record as immediately finished
            self._emulator_finished.append((job_id, True))
            return True

        transfer_type = (src.medium(), dst.medium())
        handler = self.transfer_type_to_handler.get(transfer_type)
        assert handler is not None
        try:
            success = handler.transfer_async(job_id, spec)
        except Exception as e:
            logger.warning(
                "Exception in %r transfer %d: %r",
                transfer_type,
                job_id,
                e,
                exc_info=True,
            )
            return False

        if not success:
            logger.warning("Failed to submit %r transfer %d", transfer_type, job_id)
        else:
            logger.debug("Submitted %r transfer %d: %r", transfer_type, job_id, spec)
        return success

    def get_finished(self) -> list[TransferResult]:
        """
        Get transfers finished since last call.

        Returns:
            A list of TransferResults
        """
        finished = []
        # Collect emulator-simulated completions
        if self._emulator_finished:
            finished.extend(self._emulator_finished)
            self._emulator_finished.clear()
        for handler in self.handlers:
            finished.extend(handler.get_finished())
        return finished

    def wait(self, job_ids: set[int]) -> None:
        """
        Wait for jobs to finish (blocking).

        Args:
            job_ids: The set of job IDs to wait for.
        """
        for handler in self.handlers:
            handler.wait(job_ids)
