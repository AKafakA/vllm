"""Unit tests for GPU profiler schema conversion and aggregation."""

import json

import pytest

from vllm_emulator.profile.validator import ProfileValidationError, validate_profile_pack
from vllm_emulator.profiler.gpu_profiler import (
    AggregatedSample,
    GpuProfiler,
    GpuProfilingConfig,
    RawSample,
    aggregate_samples,
    samples_to_profile_pack,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_raw_samples(
    phase: str,
    params: dict[str, int],
    latencies: list[float],
) -> list[RawSample]:
    return [RawSample(phase=phase, params=params, latency_us=lat) for lat in latencies]


# ---------------------------------------------------------------------------
# aggregate_samples
# ---------------------------------------------------------------------------

class TestAggregateSamples:
    def test_median_single_group(self):
        samples = _make_raw_samples("prefill", {"seq_len": 128, "batch_size": 1}, [10, 20, 30])
        result = aggregate_samples(samples, method="median")
        assert len(result) == 1
        assert result[0].latency_us == 20.0
        assert result[0].num_samples == 3

    def test_mean_single_group(self):
        samples = _make_raw_samples("decode", {"active_seqs": 4}, [10, 20, 30])
        result = aggregate_samples(samples, method="mean")
        assert len(result) == 1
        assert result[0].latency_us == pytest.approx(20.0)
        assert result[0].num_samples == 3

    def test_multiple_groups_kept_separate(self):
        samples = (
            _make_raw_samples("prefill", {"seq_len": 128, "batch_size": 1}, [10, 20])
            + _make_raw_samples("prefill", {"seq_len": 256, "batch_size": 1}, [30, 40])
        )
        result = aggregate_samples(samples, method="median")
        assert len(result) == 2
        assert result[0].params["seq_len"] == 128
        assert result[1].params["seq_len"] == 256

    def test_phases_kept_separate(self):
        samples = (
            _make_raw_samples("prefill", {"seq_len": 128, "batch_size": 1}, [100])
            + _make_raw_samples("decode", {"active_seqs": 1}, [200])
        )
        result = aggregate_samples(samples)
        assert len(result) == 2
        phases = {r.phase for r in result}
        assert phases == {"prefill", "decode"}

    def test_unknown_aggregation_raises(self):
        with pytest.raises(ValueError, match="Unknown aggregation method"):
            aggregate_samples([], method="bogus")

    def test_empty_input(self):
        assert aggregate_samples([]) == []

    def test_even_number_median(self):
        samples = _make_raw_samples("decode", {"active_seqs": 2}, [10, 30])
        result = aggregate_samples(samples, method="median")
        assert result[0].latency_us == 20.0


# ---------------------------------------------------------------------------
# samples_to_profile_pack
# ---------------------------------------------------------------------------

class TestSamplesToProfilePack:
    def _default_config(self, **overrides):
        return GpuProfilingConfig(
            gpu_model="TestGPU",
            **overrides,
        )

    def test_basic_conversion(self):
        aggregated = [
            AggregatedSample("prefill", {"seq_len": 128, "batch_size": 1}, 15000, 10),
            AggregatedSample("decode", {"active_seqs": 1}, 500, 10),
        ]
        pack = samples_to_profile_pack(aggregated, self._default_config())

        assert pack["version"] == "1.0"
        assert pack["gpu_model"] == "TestGPU"
        assert len(pack["prefill"]) == 1
        assert len(pack["decode"]) == 1
        assert pack["prefill"][0]["seq_len"] == 128
        assert pack["decode"][0]["active_seqs"] == 1

    def test_output_validates_against_schema(self):
        aggregated = [
            AggregatedSample("prefill", {"seq_len": 256, "batch_size": 1}, 28000, 5),
            AggregatedSample("decode", {"active_seqs": 4}, 1800, 5),
        ]
        pack = samples_to_profile_pack(aggregated, self._default_config())
        # Should not raise
        validate_profile_pack(pack)

    def test_prefill_sorted_by_batch_then_seq(self):
        aggregated = [
            AggregatedSample("prefill", {"seq_len": 512, "batch_size": 1}, 55000, 5),
            AggregatedSample("prefill", {"seq_len": 128, "batch_size": 1}, 15000, 5),
            AggregatedSample("prefill", {"seq_len": 256, "batch_size": 2}, 30000, 5),
            AggregatedSample("decode", {"active_seqs": 1}, 500, 5),
        ]
        pack = samples_to_profile_pack(aggregated, self._default_config())
        seq_lens = [r["seq_len"] for r in pack["prefill"]]
        # batch_size=1 rows first (sorted by seq_len), then batch_size=2
        assert seq_lens == [128, 512, 256]

    def test_decode_sorted_by_active_seqs(self):
        aggregated = [
            AggregatedSample("prefill", {"seq_len": 128, "batch_size": 1}, 15000, 5),
            AggregatedSample("decode", {"active_seqs": 16}, 6800, 5),
            AggregatedSample("decode", {"active_seqs": 1}, 500, 5),
            AggregatedSample("decode", {"active_seqs": 8}, 3500, 5),
        ]
        pack = samples_to_profile_pack(aggregated, self._default_config())
        seqs = [r["active_seqs"] for r in pack["decode"]]
        assert seqs == [1, 8, 16]

    def test_empty_phase_raises_validation_error(self):
        # No decode samples → validation should reject
        aggregated = [
            AggregatedSample("prefill", {"seq_len": 128, "batch_size": 1}, 1, 1),
        ]
        with pytest.raises(ProfileValidationError, match="decode must include"):
            samples_to_profile_pack(aggregated, self._default_config())

    def test_roundtrip_json_serializable(self):
        aggregated = [
            AggregatedSample("prefill", {"seq_len": 128, "batch_size": 1}, 15000, 10),
            AggregatedSample("decode", {"active_seqs": 1}, 500, 10),
        ]
        pack = samples_to_profile_pack(aggregated, self._default_config())
        text = json.dumps(pack)
        loaded = json.loads(text)
        validate_profile_pack(loaded)


# ---------------------------------------------------------------------------
# GpuProfiler (with fake backend)
# ---------------------------------------------------------------------------

class TestGpuProfiler:
    def _fake_backend(self, phase: str, params: dict[str, int]) -> float:
        """Deterministic fake: latency = sum of param values."""
        return float(sum(params.values()))

    def test_run_produces_valid_pack(self):
        config = GpuProfilingConfig(
            gpu_model="FakeGPU",
            prefill_seq_lens=[128],
            prefill_batch_sizes=[1],
            decode_active_seqs=[1],
            warmup_iters=1,
            measure_iters=3,
        )
        profiler = GpuProfiler(config=config, backend=self._fake_backend)
        pack = profiler.run()
        validate_profile_pack(pack)
        assert pack["gpu_model"] == "FakeGPU"

    def test_run_and_save(self, tmp_path):
        config = GpuProfilingConfig(
            gpu_model="FakeGPU",
            prefill_seq_lens=[128],
            prefill_batch_sizes=[1],
            decode_active_seqs=[1],
            warmup_iters=0,
            measure_iters=2,
        )
        profiler = GpuProfiler(config=config, backend=self._fake_backend)
        out = profiler.run_and_save(tmp_path / "out.json")
        assert out.exists()
        loaded = json.loads(out.read_text())
        validate_profile_pack(loaded)

    def test_warmup_not_included_in_samples(self):
        call_count = {"n": 0}

        def counting_backend(phase, params):
            call_count["n"] += 1
            return 1.0

        config = GpuProfilingConfig(
            gpu_model="FakeGPU",
            prefill_seq_lens=[128],
            prefill_batch_sizes=[1],
            decode_active_seqs=[1],
            warmup_iters=5,
            measure_iters=3,
        )
        profiler = GpuProfiler(config=config, backend=counting_backend)
        pack = profiler.run()

        # 1 prefill combo * (5 warmup + 3 measure) + 1 decode combo * (5 + 3) = 16
        assert call_count["n"] == 16
        # But only 3 measurement samples per combo → 1 aggregated prefill + 1 decode
        assert len(pack["prefill"]) == 1
        assert len(pack["decode"]) == 1
