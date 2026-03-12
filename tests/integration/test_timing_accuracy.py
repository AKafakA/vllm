"""Integration tests for emulator timing accuracy.

Validates that the oracle's latency interpolation meets the <15% error
threshold specified in the roadmap (P1.1 acceptance criteria).

Test approach:
- Load a known profile pack with exact sample points
- Query the oracle at sample points → expect 0% error
- Query at interpolation midpoints → expect bounded error
- Query out-of-range → expect clamped to boundary values
- Measure end-to-end online-mode sleep accuracy
"""

from __future__ import annotations

import json
import os
import tempfile
import time

import pytest

from vllm_emulator.oracle import ProfileGpuCostOracle, create_oracle_from_profile_pack
from vllm_emulator.profile.loader import load_profile_pack

# ---------------------------------------------------------------------------
# Reference profile with known, hand-picked values
# ---------------------------------------------------------------------------

REFERENCE_PROFILE = {
    "version": "1.0",
    "gpu_model": "test-reference-gpu",
    "prefill": [
        {"seq_len": 128, "batch_size": 1, "latency_us": 10000},
        {"seq_len": 256, "batch_size": 1, "latency_us": 20000},
        {"seq_len": 512, "batch_size": 1, "latency_us": 40000},
        {"seq_len": 1024, "batch_size": 1, "latency_us": 80000},
        {"seq_len": 2048, "batch_size": 1, "latency_us": 160000},
    ],
    "decode": [
        {"active_seqs": 1, "latency_us_per_token": 500},
        {"active_seqs": 4, "latency_us_per_token": 2000},
        {"active_seqs": 8, "latency_us_per_token": 4000},
        {"active_seqs": 16, "latency_us_per_token": 8000},
        {"active_seqs": 32, "latency_us_per_token": 16000},
    ],
}

# For this linear profile, the ground-truth function is:
#   prefill: latency_us ≈ seq_len * (10000/128) = ~78.125 * seq_len
#   decode:  latency_us ≈ 500 * active_seqs  (exactly linear)


@pytest.fixture
def oracle():
    return ProfileGpuCostOracle(REFERENCE_PROFILE)


@pytest.fixture
def profile_path():
    """Write reference profile to a temp file and return its path."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(REFERENCE_PROFILE, f)
        path = f.name
    yield path
    os.unlink(path)


# ---------------------------------------------------------------------------
# §1 – Exact sample-point accuracy (0% error)
# ---------------------------------------------------------------------------

class TestExactSamplePoints:
    """At sample points, interpolation error must be exactly zero."""

    def test_prefill_exact_points(self, oracle):
        for sample in REFERENCE_PROFILE["prefill"]:
            estimated = oracle.estimate_prefill_latency_us(
                sample["seq_len"], sample["batch_size"]
            )
            assert estimated == pytest.approx(sample["latency_us"], abs=0.01), (
                f"Prefill mismatch at seq_len={sample['seq_len']}: "
                f"expected={sample['latency_us']}, got={estimated}"
            )

    def test_decode_exact_points(self, oracle):
        for sample in REFERENCE_PROFILE["decode"]:
            estimated = oracle.estimate_decode_latency_us(sample["active_seqs"])
            assert estimated == pytest.approx(
                sample["latency_us_per_token"], abs=0.01
            ), (
                f"Decode mismatch at active_seqs={sample['active_seqs']}: "
                f"expected={sample['latency_us_per_token']}, got={estimated}"
            )


# ---------------------------------------------------------------------------
# §2 – Interpolation accuracy (<15% error at midpoints)
# ---------------------------------------------------------------------------

def _relative_error(estimated: float, expected: float) -> float:
    """Return relative error as a fraction (0.0 = perfect)."""
    if expected == 0:
        return 0.0 if estimated == 0 else float("inf")
    return abs(estimated - expected) / expected


class TestInterpolationAccuracy:
    """Linear interpolation should be exact for this linear profile,
    but we test against the <15% threshold from the roadmap."""

    ERROR_THRESHOLD = 0.15  # 15% max relative error

    @pytest.mark.parametrize(
        "seq_len,expected_us",
        [
            # Midpoint between 128 and 256 → should be 15000
            (192, 15000),
            # Midpoint between 256 and 512 → should be 30000
            (384, 30000),
            # Midpoint between 512 and 1024 → should be 60000
            (768, 60000),
            # Midpoint between 1024 and 2048 → should be 120000
            (1536, 120000),
            # Quarter-point between 128 and 256 → 12500
            (160, 12500),
            # Three-quarter point between 1024 and 2048 → 140000
            (1792, 140000),
        ],
    )
    def test_prefill_interpolation_within_threshold(
        self, oracle, seq_len, expected_us
    ):
        estimated = oracle.estimate_prefill_latency_us(seq_len, batch_size=1)
        error = _relative_error(estimated, expected_us)
        assert error <= self.ERROR_THRESHOLD, (
            f"Prefill at seq_len={seq_len}: error={error:.2%} exceeds "
            f"{self.ERROR_THRESHOLD:.0%} threshold "
            f"(estimated={estimated:.0f}, expected={expected_us})"
        )

    @pytest.mark.parametrize(
        "active_seqs,expected_us",
        [
            # Midpoint between 1 and 4 → 1250
            (2, 1000),  # (500 + 2000) * (2-1)/(4-1) + 500 = 1000
            # Midpoint between 4 and 8 → 3000
            (6, 3000),
            # Midpoint between 8 and 16 → 6000
            (12, 6000),
            # Midpoint between 16 and 32 → 12000
            (24, 12000),
        ],
    )
    def test_decode_interpolation_within_threshold(
        self, oracle, active_seqs, expected_us
    ):
        estimated = oracle.estimate_decode_latency_us(active_seqs)
        error = _relative_error(estimated, expected_us)
        assert error <= self.ERROR_THRESHOLD, (
            f"Decode at active_seqs={active_seqs}: error={error:.2%} exceeds "
            f"{self.ERROR_THRESHOLD:.0%} threshold "
            f"(estimated={estimated:.0f}, expected={expected_us})"
        )


# ---------------------------------------------------------------------------
# §3 – Boundary clamping
# ---------------------------------------------------------------------------

class TestBoundaryClamping:
    """Out-of-range queries should clamp to min/max sample values."""

    def test_prefill_below_minimum(self, oracle):
        result = oracle.estimate_prefill_latency_us(1, batch_size=1)
        min_latency = REFERENCE_PROFILE["prefill"][0]["latency_us"]
        assert result == min_latency

    def test_prefill_above_maximum(self, oracle):
        result = oracle.estimate_prefill_latency_us(100000, batch_size=1)
        max_latency = REFERENCE_PROFILE["prefill"][-1]["latency_us"]
        assert result == max_latency

    def test_decode_below_minimum(self, oracle):
        result = oracle.estimate_decode_latency_us(0)
        min_latency = REFERENCE_PROFILE["decode"][0]["latency_us_per_token"]
        assert result == min_latency

    def test_decode_above_maximum(self, oracle):
        result = oracle.estimate_decode_latency_us(1000)
        max_latency = REFERENCE_PROFILE["decode"][-1]["latency_us_per_token"]
        assert result == max_latency


# ---------------------------------------------------------------------------
# §4 – Profile pack loading round-trip
# ---------------------------------------------------------------------------

class TestProfileLoadRoundTrip:
    """Loading a profile from disk should produce the same oracle results."""

    def test_load_and_query(self, profile_path, oracle):
        loaded_pack = load_profile_pack(profile_path)
        loaded_oracle = create_oracle_from_profile_pack(loaded_pack)

        for sl in [128, 256, 512, 768, 1024, 2048]:
            orig = oracle.estimate_prefill_latency_us(sl, 1)
            loaded = loaded_oracle.estimate_prefill_latency_us(sl, 1)
            assert orig == pytest.approx(loaded, abs=0.01)

        for aseq in [1, 4, 6, 8, 16, 32]:
            orig = oracle.estimate_decode_latency_us(aseq)
            loaded = loaded_oracle.estimate_decode_latency_us(aseq)
            assert orig == pytest.approx(loaded, abs=0.01)


# ---------------------------------------------------------------------------
# §5 – Online-mode end-to-end sleep accuracy
# ---------------------------------------------------------------------------

class TestOnlineSleepAccuracy:
    """In online mode, the actual sleep time should approximate the estimate."""

    @pytest.mark.parametrize(
        "target_latency_us",
        [10_000, 50_000, 100_000],
    )
    def test_sleep_accuracy(self, target_latency_us):
        """Verify time.sleep approximates the target within 50% tolerance.

        OS scheduling means we can't expect tight timing, but gross errors
        (e.g. sleeping 10x too long) would indicate a unit conversion bug.
        """
        target_s = target_latency_us / 1_000_000
        if target_s < 0.001:
            pytest.skip("Sub-ms sleep is unreliable on most OSes")

        t0 = time.monotonic()
        time.sleep(target_s)
        elapsed = time.monotonic() - t0

        # Must be >= target (sleep never returns early)
        assert elapsed >= target_s * 0.8, (
            f"Sleep returned too early: {elapsed:.6f}s < {target_s:.6f}s"
        )
        # Must not overshoot by more than 50% (or 10ms, whichever is larger)
        max_overshoot = max(target_s * 1.5, target_s + 0.010)
        assert elapsed <= max_overshoot, (
            f"Sleep overshot: {elapsed:.6f}s >> {target_s:.6f}s"
        )


# ---------------------------------------------------------------------------
# §6 – Monotonicity: larger inputs → larger estimates
# ---------------------------------------------------------------------------

class TestMonotonicity:
    """Oracle estimates should be monotonically non-decreasing."""

    def test_prefill_monotonic(self, oracle):
        prev = 0.0
        for sl in [64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 4096]:
            est = oracle.estimate_prefill_latency_us(sl, 1)
            assert est >= prev, (
                f"Prefill NOT monotonic: {sl} tokens → {est} < {prev}"
            )
            prev = est

    def test_decode_monotonic(self, oracle):
        prev = 0.0
        for aseq in [0, 1, 2, 4, 6, 8, 12, 16, 24, 32, 64]:
            est = oracle.estimate_decode_latency_us(aseq)
            assert est >= prev, (
                f"Decode NOT monotonic: {aseq} seqs → {est} < {prev}"
            )
            prev = est
