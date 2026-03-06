"""Validation helpers for emulator profile packs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


class ProfileValidationError(ValueError):
    """Raised when a profile pack does not match the expected schema."""


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProfileValidationError(f"{path} must be an object")
    return value


def _require_number(value: Any, path: str) -> float:
    if not isinstance(value, (int, float)):
        raise ProfileValidationError(f"{path} must be a number")
    if value < 0:
        raise ProfileValidationError(f"{path} must be >= 0")
    return float(value)


def _require_int(value: Any, path: str) -> int:
    if not isinstance(value, int):
        raise ProfileValidationError(f"{path} must be an integer")
    if value < 0:
        raise ProfileValidationError(f"{path} must be >= 0")
    return value


def _validate_prefill_rows(prefill: Sequence[Any]) -> None:
    for i, row in enumerate(prefill):
        obj = _require_mapping(row, f"prefill[{i}]")
        for key in ("seq_len", "batch_size", "latency_us"):
            if key not in obj:
                raise ProfileValidationError(f"prefill[{i}].{key} is required")
        _require_int(obj["seq_len"], f"prefill[{i}].seq_len")
        _require_int(obj["batch_size"], f"prefill[{i}].batch_size")
        _require_number(obj["latency_us"], f"prefill[{i}].latency_us")


def _validate_decode_rows(decode: Sequence[Any]) -> None:
    for i, row in enumerate(decode):
        obj = _require_mapping(row, f"decode[{i}]")
        for key in ("active_seqs", "latency_us_per_token"):
            if key not in obj:
                raise ProfileValidationError(f"decode[{i}].{key} is required")
        _require_int(obj["active_seqs"], f"decode[{i}].active_seqs")
        _require_number(
            obj["latency_us_per_token"],
            f"decode[{i}].latency_us_per_token",
        )


def validate_profile_pack(profile_pack: Mapping[str, Any]) -> None:
    """Validate a profile pack and raise :class:`ProfileValidationError` on error."""

    obj = _require_mapping(profile_pack, "profile_pack")

    required_top_level = ("version", "gpu_model", "prefill", "decode")
    for key in required_top_level:
        if key not in obj:
            raise ProfileValidationError(f"{key} is required")

    if not isinstance(obj["version"], str) or not obj["version"].strip():
        raise ProfileValidationError("version must be a non-empty string")

    if not isinstance(obj["gpu_model"], str) or not obj["gpu_model"].strip():
        raise ProfileValidationError("gpu_model must be a non-empty string")

    prefill = obj["prefill"]
    decode = obj["decode"]
    if not isinstance(prefill, Sequence) or isinstance(prefill, (str, bytes)):
        raise ProfileValidationError("prefill must be an array")
    if not isinstance(decode, Sequence) or isinstance(decode, (str, bytes)):
        raise ProfileValidationError("decode must be an array")

    if len(prefill) == 0:
        raise ProfileValidationError("prefill must include at least one sample")
    if len(decode) == 0:
        raise ProfileValidationError("decode must include at least one sample")

    _validate_prefill_rows(prefill)
    _validate_decode_rows(decode)
