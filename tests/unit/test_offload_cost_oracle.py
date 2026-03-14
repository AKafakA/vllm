"""Unit tests for offload cost oracle."""

import unittest

from vllm_emulator.oracle import (
    ProfileOffloadCostOracle,
    TransferDirection,
    create_offload_oracle_from_profile_pack,
)


class TestProfileOffloadCostOracle(unittest.TestCase):
    """Tests for ProfileOffloadCostOracle."""

    def setUp(self):
        """Set up test profile pack."""
        self.profile_pack = {
            "gpu_model": "A100",
            "lookup": [
                {"num_blocks": 1, "latency_us": 10.0},
                {"num_blocks": 8, "latency_us": 50.0},
                {"num_blocks": 64, "latency_us": 400.0},
            ],
            "transfer": {
                "cpu_to_gpu": [
                    {"bytes": 4096, "latency_us": 100.0},
                    {"bytes": 1048576, "latency_us": 5000.0},  # 1MB
                    {"bytes": 16777216, "latency_us": 80000.0},  # 16MB
                ],
                "gpu_to_cpu": [
                    {"bytes": 4096, "latency_us": 120.0},
                    {"bytes": 1048576, "latency_us": 5500.0},
                ],
            },
            "evict": [
                {"num_blocks": 1, "latency_us": 5.0},
                {"num_blocks": 8, "latency_us": 30.0},
                {"num_blocks": 64, "latency_us": 200.0},
            ],
        }
        self.oracle = ProfileOffloadCostOracle(self.profile_pack)

    def test_lookup_below_minimum(self):
        """Test lookup with num_blocks below minimum sample."""
        latency = self.oracle.get_lookup_latency_us(0)
        self.assertEqual(latency, 10.0)  # Should use minimum

    def test_lookup_above_maximum(self):
        """Test lookup with num_blocks above maximum sample."""
        latency = self.oracle.get_lookup_latency_us(100)
        self.assertEqual(latency, 400.0)  # Should use maximum

    def test_lookup_interpolation(self):
        """Test lookup with num_blocks in middle (linear interpolation)."""
        # Between 1 and 8 blocks: 50% point
        latency = self.oracle.get_lookup_latency_us(4)
        # 10 + 0.5 * (50 - 10) = 10 + 20 = 30
        self.assertAlmostEqual(latency, 30.0, places=1)

    def test_lookup_exact_sample(self):
        """Test lookup with num_blocks exactly at a sample point."""
        latency = self.oracle.get_lookup_latency_us(8)
        self.assertEqual(latency, 50.0)

    def test_transfer_cpu_to_gpu_interpolation(self):
        """Test CPU->GPU transfer with interpolation."""
        # Between 1KB and 1MB: ~50% point (524800 bytes)
        latency = self.oracle.get_transfer_latency_us(
            524288,  # ~512KB
            TransferDirection.CPU_TO_GPU,
            concurrency=1
        )
        # Linear interpolation: 100 + 0.5 * (5000 - 100) = 100 + 1950 = 2050
        self.assertAlmostEqual(latency, 2050.0, delta=50)

    def test_transfer_gpu_to_cpu_different_profile(self):
        """Test GPU->CPU uses different profile than CPU->GPU."""
        cpu_to_gpu = self.oracle.get_transfer_latency_us(
            4096, TransferDirection.CPU_TO_GPU, concurrency=1
        )
        gpu_to_cpu = self.oracle.get_transfer_latency_us(
            4096, TransferDirection.GPU_TO_CPU, concurrency=1
        )
        # GPU->CPU is slightly slower in our test profile
        self.assertGreater(gpu_to_cpu, cpu_to_gpu)

    def test_transfer_concurrency_scaling(self):
        """Test that concurrency scales latency."""
        base_latency = self.oracle.get_transfer_latency_us(
            1048576, TransferDirection.CPU_TO_GPU, concurrency=1
        )
        concurrent_latency = self.oracle.get_transfer_latency_us(
            1048576, TransferDirection.CPU_TO_GPU, concurrency=4
        )
        # With concurrency=4, latency should increase (bandwidth contention)
        # sqrt(4) = 2x scaling
        self.assertGreater(concurrent_latency, base_latency)

    def test_evict_interpolation(self):
        """Test eviction latency interpolation."""
        # Between 1 and 8 blocks: ~50% point
        latency = self.oracle.get_evict_latency_us(4)
        # 5 + 0.5 * (30 - 5) = 5 + 12.5 = 17.5
        self.assertAlmostEqual(latency, 17.5, places=1)

    def test_empty_profile_defaults(self):
        """Test that empty profile returns default estimates."""
        oracle = ProfileOffloadCostOracle({})
        
        # Default: ~10us per block
        self.assertEqual(oracle.get_lookup_latency_us(10), 100.0)
        
        # Default: ~100us per MB
        self.assertEqual(oracle.get_transfer_latency_us(1048576, TransferDirection.CPU_TO_GPU), 100.0)
        
        # Default: ~5us per block
        self.assertEqual(oracle.get_evict_latency_us(10), 50.0)

    def test_factory_function(self):
        """Test factory function creates correct oracle."""
        oracle = create_offload_oracle_from_profile_pack(self.profile_pack)
        self.assertIsInstance(oracle, ProfileOffloadCostOracle)
        
        latency = oracle.get_lookup_latency_us(8)
        self.assertEqual(latency, 50.0)


if __name__ == "__main__":
    unittest.main()
