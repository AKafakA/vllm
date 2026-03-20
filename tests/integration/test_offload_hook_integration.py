"""Integration tests for the offload worker emulator hook wiring.

Tests verify that OffloadWorkerHook is correctly integrated into
OffloadingWorker.transfer_async() and get_finished().
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from vllm_emulator.hooks.offload_hook import (
    BLOCKING_MODE_OFFLINE,
    OFFLOAD_BLOCKING_MODE_ENV,
    OFFLOAD_ORACLE_ENABLED_ENV,
    OFFLOAD_PROFILE_PATH_ENV,
    OffloadWorkerHook,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_transfer_spec(src_medium="CPU", dst_medium="GPU", num_blocks=4):
    """Create mock (src, dst) specs for transfer testing."""
    src = MagicMock()
    src.medium.return_value = src_medium
    src.block_ids = list(range(num_blocks))

    dst = MagicMock()
    dst.medium.return_value = dst_medium
    dst.block_ids = list(range(num_blocks))

    return (src, dst)


# ---------------------------------------------------------------------------
# OffloadWorkerHook unit-level tests (no OffloadingWorker needed)
# ---------------------------------------------------------------------------

class TestOffloadHookStandalone:
    """Test OffloadWorkerHook behavior directly."""

    def test_disabled_when_env_absent(self):
        """Hook is disabled when env vars are not set."""
        env_clear = {
            OFFLOAD_ORACLE_ENABLED_ENV: "",
            OFFLOAD_PROFILE_PATH_ENV: "",
        }
        with patch.dict(os.environ, env_clear, clear=False):
            os.environ.pop(OFFLOAD_ORACLE_ENABLED_ENV, None)
            os.environ.pop(OFFLOAD_PROFILE_PATH_ENV, None)
            hook = OffloadWorkerHook(MagicMock())
        assert not hook.is_enabled
        assert hook.oracle is None

    def test_estimate_returns_zero_when_disabled(self):
        """Disabled hook returns zero costs."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(OFFLOAD_ORACLE_ENABLED_ENV, None)
            hook = OffloadWorkerHook(MagicMock())
        src, dst = _make_transfer_spec()
        cost = hook.estimate_transfer_cost(src, dst)
        assert cost["total_estimated_us"] == 0

    def test_apply_oracle_delay_returns_false_when_disabled(self):
        """Disabled hook returns False from apply_oracle_delay."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(OFFLOAD_ORACLE_ENABLED_ENV, None)
            hook = OffloadWorkerHook(MagicMock())
        src, dst = _make_transfer_spec()
        assert hook.apply_oracle_delay(src, dst) is False


# ---------------------------------------------------------------------------
# OffloadingWorker integration tests
# ---------------------------------------------------------------------------

class TestOffloadingWorkerIntegration:
    """Test that OffloadingWorker correctly uses the emulator hook."""

    def test_worker_has_emulator_hook_attribute(self):
        """OffloadingWorker should have _emulator_hook attribute."""
        from vllm.v1.kv_offload.worker.worker import OffloadingWorker
        # When env vars are absent, hook should be None or disabled
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(OFFLOAD_ORACLE_ENABLED_ENV, None)
            worker = OffloadingWorker()
        assert hasattr(worker, '_emulator_hook')
        assert hasattr(worker, '_emulator_finished')

    def test_worker_emulator_finished_drains(self):
        """Emulator finished list is drained on get_finished()."""
        from vllm.v1.kv_offload.worker.worker import OffloadingWorker
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(OFFLOAD_ORACLE_ENABLED_ENV, None)
            worker = OffloadingWorker()
        # Manually push into the emulator finished list
        worker._emulator_finished.append((42, True))
        worker._emulator_finished.append((43, True))
        results = worker.get_finished()
        assert (42, True) in results
        assert (43, True) in results
        # Second call should be empty
        assert worker.get_finished() == []
