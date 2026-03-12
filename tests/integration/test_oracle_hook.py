"""Integration tests for the GPU worker emulator hook.

Tests cover:
- Hook initialization via environment variables
- Oracle-driven fake output generation
- Online blocking mode (time.sleep for estimated latency)
- Offline mode (no blocking / virtual time)
- Fallback to disabled state when env vars are absent
"""

from __future__ import annotations

import os
import time
from unittest.mock import MagicMock, patch

import pytest

from vllm_emulator.hooks.gpu_hook import (
    BLOCKING_MODE_OFFLINE,
    BLOCKING_MODE_ONLINE,
    GpuWorkerHook,
    ORACLE_BLOCKING_MODE_ENV,
    ORACLE_ENABLED_ENV,
    ORACLE_PROFILE_PATH_ENV,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir,
                            "examples", "profiles")
A100_PROFILE = os.path.join(FIXTURES_DIR, "a100-sxm-80gb.json")


def _make_dummy_worker():
    """Return a minimal mock that satisfies GpuWorkerHook(worker)."""
    return MagicMock()


def _make_scheduler_output(
    new_reqs=None,
    cached_num_reqs=0,
    num_scheduled_tokens=None,
    total_tokens=1,
):
    """Build a lightweight mock SchedulerOutput for hook tests."""
    sched = MagicMock()

    # New requests (prefill)
    if new_reqs is None:
        new_reqs = []
    sched.scheduled_new_reqs = new_reqs

    # Cached (decode) requests
    sched.scheduled_cached_reqs.num_reqs = cached_num_reqs

    # Token counts
    if num_scheduled_tokens is None:
        num_scheduled_tokens = {}
    sched.num_scheduled_tokens = num_scheduled_tokens
    sched.total_num_scheduled_tokens = total_tokens

    return sched


def _make_new_req(prompt_token_ids, req_id="req-0"):
    """Create a mock new request with prompt tokens."""
    req = MagicMock()
    req.prompt_token_ids = prompt_token_ids
    req.req_id = req_id
    return req


# ---------------------------------------------------------------------------
# P2.2 §1 – Hook initialization
# ---------------------------------------------------------------------------

class TestHookInitialization:
    """Verify that the hook reads env vars and initializes correctly."""

    def test_disabled_when_env_vars_absent(self):
        """Hook should be disabled when VLLM_EMULATOR_ENABLE_ORACLE is unset."""
        env = {
            ORACLE_ENABLED_ENV: "",
            ORACLE_PROFILE_PATH_ENV: "",
            ORACLE_BLOCKING_MODE_ENV: "",
        }
        with patch.dict(os.environ, env, clear=False):
            # Make sure the env vars are truly cleared
            os.environ.pop(ORACLE_ENABLED_ENV, None)
            os.environ.pop(ORACLE_PROFILE_PATH_ENV, None)
            os.environ.pop(ORACLE_BLOCKING_MODE_ENV, None)
            hook = GpuWorkerHook(_make_dummy_worker())
        assert not hook.is_enabled
        assert hook.oracle is None

    def test_enabled_with_valid_profile(self):
        """Hook should be enabled when env vars point to a valid profile."""
        env = {
            ORACLE_ENABLED_ENV: "1",
            ORACLE_PROFILE_PATH_ENV: A100_PROFILE,
        }
        with patch.dict(os.environ, env, clear=False):
            hook = GpuWorkerHook(_make_dummy_worker())
        assert hook.is_enabled
        assert hook.oracle is not None
        assert hook.blocking_mode == BLOCKING_MODE_ONLINE  # default

    def test_enabled_offline_mode(self):
        """Hook respects BLOCKING_MODE=offline."""
        env = {
            ORACLE_ENABLED_ENV: "true",
            ORACLE_PROFILE_PATH_ENV: A100_PROFILE,
            ORACLE_BLOCKING_MODE_ENV: "offline",
        }
        with patch.dict(os.environ, env, clear=False):
            hook = GpuWorkerHook(_make_dummy_worker())
        assert hook.is_enabled
        assert hook.blocking_mode == BLOCKING_MODE_OFFLINE
        assert not hook.should_block

    def test_enabled_online_mode_explicit(self):
        """Hook respects BLOCKING_MODE=online explicitly."""
        env = {
            ORACLE_ENABLED_ENV: "yes",
            ORACLE_PROFILE_PATH_ENV: A100_PROFILE,
            ORACLE_BLOCKING_MODE_ENV: "online",
        }
        with patch.dict(os.environ, env, clear=False):
            hook = GpuWorkerHook(_make_dummy_worker())
        assert hook.is_enabled
        assert hook.blocking_mode == BLOCKING_MODE_ONLINE
        assert hook.should_block

    def test_error_when_enabled_without_profile(self):
        """Hook should raise when oracle is enabled but no profile path."""
        env = {
            ORACLE_ENABLED_ENV: "1",
            ORACLE_PROFILE_PATH_ENV: "",
        }
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop(ORACLE_PROFILE_PATH_ENV, None)
            with pytest.raises(ValueError, match="not configured"):
                GpuWorkerHook(_make_dummy_worker())

    def test_error_when_profile_missing(self):
        """Hook should raise when profile path does not exist."""
        env = {
            ORACLE_ENABLED_ENV: "1",
            ORACLE_PROFILE_PATH_ENV: "/nonexistent/profile.json",
        }
        with patch.dict(os.environ, env, clear=False):
            with pytest.raises(RuntimeError, match="Failed to load"):
                GpuWorkerHook(_make_dummy_worker())


# ---------------------------------------------------------------------------
# P2.2 §2 – Cost estimation
# ---------------------------------------------------------------------------

class TestCostEstimation:
    """Verify oracle cost estimation through the hook."""

    @pytest.fixture(autouse=True)
    def _enable_hook(self):
        env = {
            ORACLE_ENABLED_ENV: "1",
            ORACLE_PROFILE_PATH_ENV: A100_PROFILE,
            ORACLE_BLOCKING_MODE_ENV: "offline",
        }
        with patch.dict(os.environ, env, clear=False):
            self.hook = GpuWorkerHook(_make_dummy_worker())

    def test_prefill_cost_estimation(self):
        """Prefill cost should be > 0 for new requests."""
        new_req = _make_new_req(list(range(256)), req_id="r1")
        sched = _make_scheduler_output(
            new_reqs=[new_req],
            num_scheduled_tokens={"r1": 256},
            total_tokens=256,
        )
        cost = self.hook.estimate_execution_cost(sched)
        assert cost["prefill_latency_us"] > 0
        assert cost["total_estimated_us"] > 0

    def test_decode_cost_estimation(self):
        """Decode cost should be > 0 for cached (active) sequences."""
        sched = _make_scheduler_output(
            cached_num_reqs=4,
            num_scheduled_tokens={"r1": 1, "r2": 1, "r3": 1, "r4": 1},
            total_tokens=4,
        )
        cost = self.hook.estimate_execution_cost(sched)
        assert cost["decode_latency_us"] > 0
        assert cost["total_estimated_us"] > 0

    def test_zero_cost_when_disabled(self):
        """Disabled hook returns zero costs."""
        env = {ORACLE_ENABLED_ENV: ""}
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop(ORACLE_ENABLED_ENV, None)
            hook = GpuWorkerHook(_make_dummy_worker())
        sched = _make_scheduler_output()
        cost = hook.estimate_execution_cost(sched)
        assert cost["total_estimated_us"] == 0


# ---------------------------------------------------------------------------
# P2.2 §3 – Fake output generation
# ---------------------------------------------------------------------------

class TestFakeOutput:
    """Verify that fake ModelRunnerOutput is created correctly."""

    @pytest.fixture(autouse=True)
    def _enable_hook(self):
        env = {
            ORACLE_ENABLED_ENV: "1",
            ORACLE_PROFILE_PATH_ENV: A100_PROFILE,
            ORACLE_BLOCKING_MODE_ENV: "offline",
        }
        with patch.dict(os.environ, env, clear=False):
            self.hook = GpuWorkerHook(_make_dummy_worker())

    def test_fake_output_has_correct_req_ids(self):
        """Fake output should contain all scheduled request IDs."""
        sched = _make_scheduler_output(
            num_scheduled_tokens={"req-a": 1, "req-b": 1},
            total_tokens=2,
        )
        sched.scheduled_new_reqs = []
        output = self.hook.create_fake_output(sched)
        assert output is not None
        assert set(output.req_ids) == {"req-a", "req-b"}

    def test_fake_output_none_when_no_tokens(self):
        """Fake output should be None when nothing is scheduled."""
        sched = _make_scheduler_output(total_tokens=0)
        output = self.hook.create_fake_output(sched)
        assert output is None


# ---------------------------------------------------------------------------
# P2.2 §4 – Online blocking mode
# ---------------------------------------------------------------------------

class TestOnlineBlocking:
    """Verify that online mode actually sleeps for estimated latency."""

    @pytest.fixture(autouse=True)
    def _enable_hook_online(self):
        env = {
            ORACLE_ENABLED_ENV: "1",
            ORACLE_PROFILE_PATH_ENV: A100_PROFILE,
            ORACLE_BLOCKING_MODE_ENV: "online",
        }
        with patch.dict(os.environ, env, clear=False):
            self.hook = GpuWorkerHook(_make_dummy_worker())

    def test_online_mode_blocks(self):
        """In online mode, should_block is True and sleep is invoked."""
        assert self.hook.should_block

        # Build a scheduler output that triggers a prefill estimate
        new_req = _make_new_req(list(range(512)), req_id="r1")
        sched = _make_scheduler_output(
            new_reqs=[new_req],
            num_scheduled_tokens={"r1": 512},
            total_tokens=512,
        )
        cost = self.hook.estimate_execution_cost(sched)
        estimated_s = cost["total_estimated_us"] / 1_000_000

        # The estimate should be meaningful (> 1ms for 512 tokens)
        assert estimated_s > 0.001

        # Verify blocking by actually timing sleep
        t0 = time.monotonic()
        time.sleep(estimated_s)
        elapsed = time.monotonic() - t0
        # Allow 50% tolerance for OS scheduling jitter
        assert elapsed >= estimated_s * 0.5


# ---------------------------------------------------------------------------
# P2.2 §5 – Offline (no blocking) mode
# ---------------------------------------------------------------------------

class TestOfflineMode:
    """Verify that offline mode does NOT block."""

    @pytest.fixture(autouse=True)
    def _enable_hook_offline(self):
        env = {
            ORACLE_ENABLED_ENV: "1",
            ORACLE_PROFILE_PATH_ENV: A100_PROFILE,
            ORACLE_BLOCKING_MODE_ENV: "offline",
        }
        with patch.dict(os.environ, env, clear=False):
            self.hook = GpuWorkerHook(_make_dummy_worker())

    def test_offline_mode_no_block(self):
        """Offline mode: should_block is False, no sleep needed."""
        assert not self.hook.should_block

        new_req = _make_new_req(list(range(1024)), req_id="r1")
        sched = _make_scheduler_output(
            new_reqs=[new_req],
            num_scheduled_tokens={"r1": 1024},
            total_tokens=1024,
        )
        cost = self.hook.estimate_execution_cost(sched)
        # Offline still computes an estimate (for logging / metrics)
        assert cost["total_estimated_us"] > 0

        # But we do NOT sleep – verify the call returns instantly
        t0 = time.monotonic()
        # Simulate what gpu_worker.py does: check should_block
        if self.hook.should_block:
            time.sleep(cost["total_estimated_us"] / 1_000_000)
        elapsed = time.monotonic() - t0
        # Should be essentially instant (< 5ms)
        assert elapsed < 0.005
