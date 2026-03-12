"""Unit tests for GPU cost oracle."""

import pytest
from vllm_emulator.oracle import (
    BaseGpuCostOracle,
    ProfileGpuCostOracle,
    create_oracle_from_profile_pack,
)


# Sample profile pack for testing
SAMPLE_PROFILE_PACK = {
    "version": "1.0",
    "gpu_model": "A100",
    "prefill": [
        {"seq_len": 128, "batch_size": 1, "latency_us": 10000},
        {"seq_len": 512, "batch_size": 1, "latency_us": 35000},
        {"seq_len": 1024, "batch_size": 1, "latency_us": 65000},
        {"seq_len": 2048, "batch_size": 1, "latency_us": 125000},
    ],
    "decode": [
        {"active_seqs": 1, "latency_us_per_token": 500},
        {"active_seqs": 8, "latency_us_per_token": 3500},
        {"active_seqs": 16, "latency_us_per_token": 6500},
        {"active_seqs": 32, "latency_us_per_token": 12000},
    ],
}


def test_profile_oracle_creation():
    """Test creating oracle from profile pack."""
    oracle = ProfileGpuCostOracle(SAMPLE_PROFILE_PACK)
    assert oracle.gpu_model == "A100"


def test_profile_oracle_factory():
    """Test factory function."""
    oracle = create_oracle_from_profile_pack(SAMPLE_PROFILE_PACK)
    assert isinstance(oracle, ProfileGpuCostOracle)


def test_prefill_exact_match():
    """Test prefill lookup at exact sample points."""
    oracle = ProfileGpuCostOracle(SAMPLE_PROFILE_PACK)
    assert oracle.estimate_prefill_latency_us(128, 1) == 10000
    assert oracle.estimate_prefill_latency_us(512, 1) == 35000
    assert oracle.estimate_prefill_latency_us(1024, 1) == 65000


def test_prefill_interpolation():
    """Test prefill interpolation between samples."""
    oracle = ProfileGpuCostOracle(SAMPLE_PROFILE_PACK)
    # Between 128 and 512: should be ~22500 at 320 tokens
    result = oracle.estimate_prefill_latency_us(320, 1)
    assert 15000 < result < 30000  # Between 128 and 512 samples


def test_prefill_below_minimum():
    """Test prefill lookup below minimum sample."""
    oracle = ProfileGpuCostOracle(SAMPLE_PROFILE_PACK)
    result = oracle.estimate_prefill_latency_us(64, 1)
    assert result == 10000  # Should use minimum


def test_prefill_above_maximum():
    """Test prefill lookup above maximum sample."""
    oracle = ProfileGpuCostOracle(SAMPLE_PROFILE_PACK)
    result = oracle.estimate_prefill_latency_us(4096, 1)
    assert result == 125000  # Should use maximum


def test_decode_exact_match():
    """Test decode lookup at exact sample points."""
    oracle = ProfileGpuCostOracle(SAMPLE_PROFILE_PACK)
    assert oracle.estimate_decode_latency_us(1) == 500
    assert oracle.estimate_decode_latency_us(8) == 3500
    assert oracle.estimate_decode_latency_us(16) == 6500


def test_decode_interpolation():
    """Test decode interpolation between samples."""
    oracle = ProfileGpuCostOracle(SAMPLE_PROFILE_PACK)
    # Between 8 and 16: should be ~5000 at 12 seqs
    result = oracle.estimate_decode_latency_us(12)
    assert 3500 < result < 6500


def test_decode_below_minimum():
    """Test decode lookup below minimum sample."""
    oracle = ProfileGpuCostOracle(SAMPLE_PROFILE_PACK)
    result = oracle.estimate_decode_latency_us(0)
    assert result == 500


def test_decode_above_maximum():
    """Test decode lookup above maximum sample."""
    oracle = ProfileGpuCostOracle(SAMPLE_PROFILE_PACK)
    result = oracle.estimate_decode_latency_us(64)
    assert result == 12000


def test_base_oracle_is_abstract():
    """Test that BaseGpuCostOracle cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseGpuCostOracle()
