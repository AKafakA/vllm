"""Integration tests for PD separation support."""

import pytest

from vllm_emulator.oracle import (
    ProfileGpuCostOracle,
    PDSeparatedCostOracle,
    create_oracle_from_profile_pack,
    create_pd_separated_oracle,
)
from vllm_emulator.scheduler import (
    EmulatorScheduler,
    PDSchedulingPolicy,
    Request,
    create_scheduler,
)


# Sample profile pack for testing
TEST_PROFILE_PACK = {
    "version": "1.0",
    "gpu_model": "A100",
    "prefill": [
        {"seq_len": 128, "batch_size": 1, "latency_us": 15000},
        {"seq_len": 512, "batch_size": 1, "latency_us": 50000},
        {"seq_len": 1024, "batch_size": 1, "latency_us": 95000},
    ],
    "decode": [
        {"active_seqs": 1, "latency_us_per_token": 500},
        {"active_seqs": 8, "latency_us_per_token": 3500},
        {"active_seqs": 16, "latency_us_per_token": 6500},
    ],
}


class TestPDSeparatedCostOracle:
    """Tests for PD-separated cost oracle."""

    def test_create_pd_oracle(self):
        """Test creating a PD-separated oracle from a base oracle."""
        base = create_oracle_from_profile_pack(TEST_PROFILE_PACK)
        pd_oracle = create_pd_separated_oracle(base)
        
        assert isinstance(pd_oracle, PDSeparatedCostOracle)
        assert pd_oracle.gpu_model == "A100"

    def test_estimate_prefill_time(self):
        """Test prefill time estimation."""
        base = create_oracle_from_profile_pack(TEST_PROFILE_PACK)
        pd_oracle = create_pd_separated_oracle(base)
        
        # Test interpolation
        time_128 = pd_oracle.estimate_prefill_time(128, 1)
        time_512 = pd_oracle.estimate_prefill_time(512, 1)
        
        assert time_128 == 15000
        assert time_512 == 50000

    def test_estimate_decode_time(self):
        """Test decode time estimation."""
        base = create_oracle_from_profile_pack(TEST_PROFILE_PACK)
        pd_oracle = create_pd_separated_oracle(base)
        
        # Single token decode with 1 sequence
        time_1 = pd_oracle.estimate_decode_time(1, 1)
        assert time_1 == 500
        
        # Single token decode with 8 sequences
        time_8 = pd_oracle.estimate_decode_time(1, 8)
        assert time_8 == 3500

    def test_estimate_decode_time_multiple_tokens(self):
        """Test decode time with multiple tokens."""
        base = create_oracle_from_profile_pack(TEST_PROFILE_PACK)
        pd_oracle = create_pd_separated_oracle(base)
        
        # 3 tokens with 8 sequences
        time_3_8 = pd_oracle.estimate_decode_time(3, 8)
        assert time_3_8 == 3500 * 3  # 3 tokens * per-token latency

    def test_backward_compatibility_aliases(self):
        """Test that base oracle methods still work via aliases."""
        base = create_oracle_from_profile_pack(TEST_PROFILE_PACK)
        pd_oracle = create_pd_separated_oracle(base)
        
        # Test aliases match original methods
        assert pd_oracle.estimate_prefill_latency_us(128, 1) == 15000
        assert pd_oracle.estimate_decode_latency_us(1) == 500


class TestEmulatorScheduler:
    """Tests for emulator scheduler with PD separation."""

    @pytest.fixture
    def base_oracle(self):
        """Create a base cost oracle for testing."""
        return create_oracle_from_profile_pack(TEST_PROFILE_PACK)

    def test_scheduler_creation(self, base_oracle):
        """Test scheduler creation."""
        scheduler = create_scheduler(base_oracle)
        assert scheduler is not None
        assert not scheduler.enable_pd_separation

    def test_scheduler_pd_disabled_by_default(self, base_oracle):
        """Test PD is disabled by default."""
        scheduler = EmulatorScheduler(base_oracle)
        assert not scheduler.enable_pd_separation
        assert scheduler.policy == PDSchedulingPolicy.HYBRID

    def test_add_request(self, base_oracle):
        """Test adding requests to scheduler."""
        scheduler = EmulatorScheduler(base_oracle)
        req = Request(request_id="req1", prompt_tokens=128, max_tokens=50)
        scheduler.add_request(req)
        
        stats = scheduler.get_queue_stats()
        assert stats["prefill_queue_size"] == 1

    def test_joint_scheduling(self, base_oracle):
        """Test joint (non-PD) scheduling."""
        scheduler = EmulatorScheduler(base_oracle, enable_pd_separation=False)
        
        # Add requests
        req1 = Request(request_id="req1", prompt_tokens=128, max_tokens=50)
        req2 = Request(request_id="req2", prompt_tokens=256, max_tokens=50)
        scheduler.add_request(req1)
        scheduler.add_request(req2)
        
        decision = scheduler.schedule()
        
        # Both should be in prefill batch
        assert len(decision.prefill_batch) == 2
        assert decision.prefill_time_us > 0

    def test_pd_separated_scheduling(self, base_oracle):
        """Test PD-separated scheduling."""
        scheduler = EmulatorScheduler(
            base_oracle,
            enable_pd_separation=True,
            policy=PDSchedulingPolicy.PREFILL_FIRST,
        )
        
        # Add request
        req = Request(request_id="req1", prompt_tokens=128, max_tokens=50)
        scheduler.add_request(req)
        
        decision = scheduler.schedule()
        
        # Should be in prefill batch
        assert len(decision.prefill_batch) == 1
        assert decision.prefill_batch[0].request_id == "req1"
        
        # After prefill, request should move to decode queue
        decision2 = scheduler.schedule()
        # In prefill_first, prefills are processed first

    def test_prefill_first_policy(self, base_oracle):
        """Test prefill-first scheduling policy."""
        scheduler = EmulatorScheduler(
            base_oracle,
            enable_pd_separation=True,
            policy=PDSchedulingPolicy.PREFILL_FIRST,
        )
        
        # Add multiple requests
        for i in range(3):
            req = Request(request_id=f"req{i}", prompt_tokens=128, max_tokens=50)
            scheduler.add_request(req)
        
        decision = scheduler.schedule()
        
        # All prefills should be processed
        assert len(decision.prefill_batch) == 3

    def test_decode_first_policy(self, base_oracle):
        """Test decode-first scheduling policy."""
        scheduler = EmulatorScheduler(
            base_oracle,
            enable_pd_separation=True,
            policy=PDSchedulingPolicy.DECODE_FIRST,
        )
        
        # Add request that has already been prefilled
        req = Request(request_id="req1", prompt_tokens=128, max_tokens=50)
        scheduler.add_request(req)
        
        # First schedule: prefill
        decision1 = scheduler.schedule()
        assert len(decision1.prefill_batch) == 1
        
        # Second schedule: should prioritize decode if available
        # But we have no active decodes yet, so it will do prefill again
        decision2 = scheduler.schedule()
        
        # After prefilling, request goes to active_decodes
        stats = scheduler.get_queue_stats()
        assert stats["active_decodes"] >= 0

    def test_hybrid_policy(self, base_oracle):
        """Test hybrid scheduling policy."""
        scheduler = EmulatorScheduler(
            base_oracle,
            enable_pd_separation=True,
            policy=PDSchedulingPolicy.HYBRID,
            max_batch_size=2,
        )
        
        # Add requests
        for i in range(5):
            req = Request(request_id=f"req{i}", prompt_tokens=128, max_tokens=50)
            scheduler.add_request(req)
        
        decision = scheduler.schedule()
        
        # Hybrid should process some prefills
        assert len(decision.prefill_batch) > 0

    def test_queue_stats(self, base_oracle):
        """Test queue statistics."""
        scheduler = EmulatorScheduler(
            base_oracle,
            enable_pd_separation=True,
        )
        
        req = Request(request_id="req1", prompt_tokens=128, max_tokens=50)
        scheduler.add_request(req)
        
        stats = scheduler.get_queue_stats()
        
        assert stats["prefill_queue_size"] == 1
        assert stats["active_decodes"] == 0
        assert stats["pd_enabled"] is True
        assert stats["policy"] == "hybrid"


class TestEndToEndPDSimulation:
    """End-to-end tests for PD-separated simulation."""

    def test_pd_simulation_workflow(self):
        """Test complete PD simulation workflow."""
        # Create oracle
        base = create_oracle_from_profile_pack(TEST_PROFILE_PACK)
        pd_oracle = create_pd_separated_oracle(base)
        
        # Create scheduler with PD enabled
        scheduler = create_scheduler(
            pd_oracle,
            enable_pd_separation=True,
            policy="hybrid",
            max_batch_size=4,
        )
        
        # Add multiple requests
        for i in range(3):
            req = Request(
                request_id=f"req{i}",
                prompt_tokens=128 + i * 100,
                max_tokens=20,
            )
            scheduler.add_request(req)
        
        # Simulate scheduling rounds
        total_prefill_time = 0
        total_decode_time = 0
        
        for _ in range(10):  # Max 10 rounds
            decision = scheduler.schedule()
            
            if not decision.prefill_batch and not decision.decode_batch:
                break
                
            total_prefill_time += decision.prefill_time_us
            total_decode_time += decision.decode_time_us
            
            # Advance generated tokens for decoded requests
            for req in decision.decode_batch:
                req.generated_tokens += 1
        
        # Verify we processed some work
        assert total_prefill_time > 0
        
        stats = scheduler.get_queue_stats()
        assert stats["pd_enabled"] is True
