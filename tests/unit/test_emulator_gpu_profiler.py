from vllm_emulator.profiler.gpu_profiler import (
    ProfileSample,
    records_to_profile_pack,
)


def test_records_to_profile_pack_aggregates_by_median():
    prefill = [
        ProfileSample(phase="prefill", key=128, batch_size=1, latency_us=100.0),
        ProfileSample(phase="prefill", key=128, batch_size=1, latency_us=200.0),
        ProfileSample(phase="prefill", key=128, batch_size=1, latency_us=5000.0),
        ProfileSample(phase="prefill", key=256, batch_size=2, latency_us=300.0),
    ]
    decode = [
        ProfileSample(phase="decode", key=4, latency_us=10.0),
        ProfileSample(phase="decode", key=4, latency_us=20.0),
        ProfileSample(phase="decode", key=4, latency_us=1000.0),
    ]

    pack = records_to_profile_pack(
        gpu_model="A100",
        prefill_samples=prefill,
        decode_samples=decode,
    )

    assert pack["gpu_model"] == "A100"
    assert pack["prefill"][0] == {
        "seq_len": 128,
        "batch_size": 1,
        "latency_us": 200.0,
    }
    assert pack["decode"][0] == {
        "active_seqs": 4,
        "latency_us_per_token": 20.0,
    }


def test_records_to_profile_pack_rejects_empty_phase_samples():
    prefill = [ProfileSample(phase="prefill", key=128, batch_size=1, latency_us=100.0)]

    try:
        records_to_profile_pack(
            gpu_model="A100",
            prefill_samples=prefill,
            decode_samples=[],
        )
    except ValueError as exc:
        assert "No decode samples collected" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing decode samples")
