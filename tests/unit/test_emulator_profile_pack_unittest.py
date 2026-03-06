import tempfile
import unittest

from vllm_emulator.profile.loader import load_profile_pack, load_profile_pack_from_str
from vllm_emulator.profile.validator import ProfileValidationError, validate_profile_pack


class TestProfilePack(unittest.TestCase):
    def setUp(self):
        self.valid_profile = {
            "version": "1.0",
            "gpu_model": "A100",
            "prefill": [{"seq_len": 128, "batch_size": 1, "latency_us": 1}],
            "decode": [{"active_seqs": 1, "latency_us_per_token": 1}],
        }

    def test_validate_accepts_valid_profile(self):
        validate_profile_pack(self.valid_profile)

    def test_validate_rejects_missing_field(self):
        invalid = dict(self.valid_profile)
        invalid.pop("decode")
        with self.assertRaises(ProfileValidationError):
            validate_profile_pack(invalid)

    def test_load_from_str(self):
        loaded = load_profile_pack_from_str(
            '{"version":"1.0","gpu_model":"H100","prefill":[{"seq_len":128,"batch_size":1,"latency_us":2}],"decode":[{"active_seqs":1,"latency_us_per_token":2}]}'
        )
        self.assertEqual(loaded["gpu_model"], "H100")

    def test_load_from_file(self):
        payload = '{"version":"1.0","gpu_model":"H100","prefill":[{"seq_len":128,"batch_size":1,"latency_us":2}],"decode":[{"active_seqs":1,"latency_us_per_token":2}]}'
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".json") as f:
            f.write(payload)
            f.flush()
            loaded = load_profile_pack(f.name)
        self.assertEqual(loaded["gpu_model"], "H100")


if __name__ == "__main__":
    unittest.main()
