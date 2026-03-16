"""Unit tests for Network cost oracle."""

import pytest
from vllm_emulator.oracle import (
    BaseNetworkCostOracle,
    NetworkTopology,
    ProfileNetworkCostOracle,
    TransferDirection,
    create_network_oracle_from_profile_pack,
)


# Sample network profile pack for testing (correct format)
SAMPLE_NETWORK_PROFILE = {
    "version": "1.0",
    "all_reduce": {
        "nvlink": [
            {"bytes": 1024, "world_size": 2, "latency_us": 50},
            {"bytes": 10240, "world_size": 2, "latency_us": 120},
            {"bytes": 102400, "world_size": 2, "latency_us": 450},
            {"bytes": 1048576, "world_size": 2, "latency_us": 2500},
        ],
    },
    "send_recv": {
        "nvlink": [
            {"bytes": 1024, "latency_us": 30},
            {"bytes": 10240, "latency_us": 80},
            {"bytes": 102400, "latency_us": 350},
            {"bytes": 1048576, "latency_us": 2000},
        ],
    },
    "kv_transfer": {
        "nvlink": [
            {"bytes": 1024, "latency_us": 60},
            {"bytes": 10240, "latency_us": 150},
            {"bytes": 102400, "latency_us": 700},
            {"bytes": 1048576, "latency_us": 4000},
        ],
    },
}


def test_network_oracle_creation():
    """Test creating network oracle from profile pack."""
    oracle = ProfileNetworkCostOracle(SAMPLE_NETWORK_PROFILE)
    # Verify samples are loaded
    assert NetworkTopology.NVLINK in oracle._all_reduce_samples
    assert len(oracle._all_reduce_samples[NetworkTopology.NVLINK]) == 4


def test_network_oracle_factory():
    """Test factory function."""
    oracle = create_network_oracle_from_profile_pack(SAMPLE_NETWORK_PROFILE)
    assert isinstance(oracle, ProfileNetworkCostOracle)


def test_all_reduce_exact_match():
    """Test all_reduce lookup at exact sample points."""
    oracle = ProfileNetworkCostOracle(SAMPLE_NETWORK_PROFILE)
    # 1KB -> 50us, world_size=2
    assert oracle.get_all_reduce_latency_us(1024, 2, NetworkTopology.NVLINK) == 50
    # 10KB -> 120us
    assert oracle.get_all_reduce_latency_us(10240, 2, NetworkTopology.NVLINK) == 120


def test_all_reduce_interpolation():
    """Test all_reduce interpolation between samples."""
    oracle = ProfileNetworkCostOracle(SAMPLE_NETWORK_PROFILE)
    # Between 1KB (50us) and 10KB (120us) at ~5KB
    result = oracle.get_all_reduce_latency_us(5120, 2, NetworkTopology.NVLINK)
    assert 50 < result < 120


def test_send_latency_exact_match():
    """Test send lookup at exact sample points."""
    oracle = ProfileNetworkCostOracle(SAMPLE_NETWORK_PROFILE)
    # 1KB -> 30us
    assert oracle.get_send_latency_us(1024, NetworkTopology.NVLINK) == 30
    # 100KB -> 350us
    assert oracle.get_send_latency_us(102400, NetworkTopology.NVLINK) == 350


def test_recv_latency_exact_match():
    """Test recv lookup at exact sample points."""
    oracle = ProfileNetworkCostOracle(SAMPLE_NETWORK_PROFILE)
    # 1KB -> 30us (same as send in profile)
    assert oracle.get_recv_latency_us(1024, NetworkTopology.NVLINK) == 30


def test_send_below_minimum():
    """Test send lookup below minimum sample."""
    oracle = ProfileNetworkCostOracle(SAMPLE_NETWORK_PROFILE)
    # Below 1KB: should use minimum
    result = oracle.get_send_latency_us(512, NetworkTopology.NVLINK)
    assert result <= 30


def test_send_above_maximum():
    """Test send lookup above maximum sample."""
    oracle = ProfileNetworkCostOracle(SAMPLE_NETWORK_PROFILE)
    # Above 1MB: should extrapolate or cap at max
    result = oracle.get_send_latency_us(10 * 1024 * 1024, NetworkTopology.NVLINK)
    assert result >= 2000  # Should be at least 1MB case


def test_kv_transfer_latency():
    """Test KV transfer latency estimation."""
    oracle = ProfileNetworkCostOracle(SAMPLE_NETWORK_PROFILE)
    # KV transfer samples: 1KB -> 60us
    result = oracle.get_kv_transfer_latency_us(
        1024, TransferDirection.GPU_TO_GPU, NetworkTopology.NVLINK
    )
    assert result == 60


def test_base_class_interface():
    """Test that BaseNetworkCostOracle is abstract."""
    with pytest.raises(TypeError):
        BaseNetworkCostOracle()
