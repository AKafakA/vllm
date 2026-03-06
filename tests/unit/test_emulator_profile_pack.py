import pytest

from vllm_emulator.profile.loader import load_profile_pack, load_profile_pack_from_str
from vllm_emulator.profile.validator import ProfileValidationError, validate_profile_pack


VALID_PROFILE = {
    "version": "1.0",
    "gpu_model": "A100",
    "prefill": [
        {"seq_len": 128, "batch_size": 1, "latency_us": 15000},
    ],
    "decode": [
        {"active_seqs": 1, "latency_us_per_token": 500},
    ],
}


def test_validate_profile_pack_accepts_valid_schema():
    validate_profile_pack(VALID_PROFILE)


def test_validate_profile_pack_rejects_missing_required_field():
    with pytest.raises(ProfileValidationError, match="decode is required"):
        validate_profile_pack({k: v for k, v in VALID_PROFILE.items() if k != "decode"})


def test_validate_profile_pack_rejects_negative_values():
    invalid = {
        **VALID_PROFILE,
        "decode": [{"active_seqs": 1, "latency_us_per_token": -1}],
    }
    with pytest.raises(ProfileValidationError, match="must be >= 0"):
        validate_profile_pack(invalid)


def test_load_profile_pack_from_str():
    content = """{
      \"version\": \"1.0\",
      \"gpu_model\": \"A100\",
      \"prefill\": [{\"seq_len\": 128, \"batch_size\": 1, \"latency_us\": 1}],
      \"decode\": [{\"active_seqs\": 1, \"latency_us_per_token\": 1}]
    }"""

    loaded = load_profile_pack_from_str(content)
    assert loaded["gpu_model"] == "A100"


def test_load_profile_pack_from_file(tmp_path):
    profile_file = tmp_path / "profile.json"
    profile_file.write_text(
        """{
  \"version\": \"1.0\",
  \"gpu_model\": \"H100\",
  \"prefill\": [{\"seq_len\": 128, \"batch_size\": 1, \"latency_us\": 2}],
  \"decode\": [{\"active_seqs\": 1, \"latency_us_per_token\": 2}]
}""",
        encoding="utf-8",
    )

    loaded = load_profile_pack(profile_file)
    assert loaded["gpu_model"] == "H100"


def test_load_profile_pack_rejects_invalid_json():
    with pytest.raises(ProfileValidationError, match="Invalid JSON profile pack"):
        load_profile_pack_from_str("{not-json}")
