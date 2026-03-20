"""Integration tests for the network emulator hook wiring.

Tests verify that NetworkHook is correctly wired and that the
CudaCommunicator integration entry points exist.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from vllm_emulator.hooks.network_hook import (
    BLOCKING_MODE_OFFLINE,
    NETWORK_BLOCKING_MODE_ENV,
    NETWORK_ORACLE_ENABLED_ENV,
    NETWORK_PROFILE_PATH_ENV,
    NETWORK_TOPOLOGY_ENV,
    NetworkHook,
    install_network_hook,
)
from vllm_emulator.oracle import NetworkTopology, TransferDirection


# ---------------------------------------------------------------------------
# NetworkHook standalone tests
# ---------------------------------------------------------------------------

class TestNetworkHookStandalone:
    """Test NetworkHook behavior directly (no CudaCommunicator needed)."""

    def test_disabled_when_env_absent(self):
        """Hook is disabled when env vars are not set."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(NETWORK_ORACLE_ENABLED_ENV, None)
            hook = NetworkHook()
        assert not hook.is_enabled
        assert hook.oracle is None

    def test_topology_parsing(self):
        """Topology string is parsed correctly."""
        hook = NetworkHook.__new__(NetworkHook)
        assert hook._parse_topology("nvlink") == NetworkTopology.NVLINK
        assert hook._parse_topology("pcie") == NetworkTopology.PCIE
        assert hook._parse_topology("ib") == NetworkTopology.INFINIBAND
        assert hook._parse_topology("unknown") == NetworkTopology.NVLINK

    def test_estimate_returns_zero_when_disabled(self):
        """Disabled hook returns zero costs."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(NETWORK_ORACLE_ENABLED_ENV, None)
            hook = NetworkHook()
        assert hook.estimate_all_reduce_cost(1024, 2) == 0.0
        assert hook.estimate_send_cost(1024) == 0.0
        assert hook.estimate_recv_cost(1024) == 0.0
        assert hook.estimate_kv_transfer_cost(1024) == 0.0

    def test_apply_delay_returns_false_when_disabled(self):
        """Disabled hook returns False from apply_ methods."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(NETWORK_ORACLE_ENABLED_ENV, None)
            hook = NetworkHook()
        assert hook.apply_all_reduce_delay(1024, 2) is False
        assert hook.apply_send_delay(1024) is False
        assert hook.apply_recv_delay(1024) is False
        assert hook.apply_kv_transfer_delay(1024) is False


# ---------------------------------------------------------------------------
# CudaCommunicator integration point verification
# ---------------------------------------------------------------------------

class TestCudaCommunicatorIntegration:
    """Verify that CudaCommunicator has the emulator hook integration code."""

    def test_hook_loader_exists(self):
        """The lazy hook loader function should be importable."""
        from vllm.distributed.device_communicators.cuda_communicator import (
            _get_emulator_network_hook,
        )
        # When env vars are absent, should return None
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(NETWORK_ORACLE_ENABLED_ENV, None)
            # Reset the cached state for clean test
            import vllm.distributed.device_communicators.cuda_communicator as mod
            mod._EMULATOR_NETWORK_HOOK = None
            mod._EMULATOR_NETWORK_HOOK_LOADED = False
            result = _get_emulator_network_hook()
        assert result is None
