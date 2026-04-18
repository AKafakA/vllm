#!/usr/bin/env python3
"""F1 byte-identity smoke test.

Verifies the merge-safety contract: with --outlier-filter=none (default),
the F1 builder produces output that is byte-identical to the parent-commit
(a102eed27) builder.

Strategy: check out parent-commit builder to a temp file, run both builders
on the same small synthetic trace, diff outputs.
"""
import json
import os
import subprocess
import sys
import tempfile


REPO = "/home/wd312/Code/llm/vllm-emulator"
PARENT = "a102eed27"


def make_synthetic_trace(path):
    """Write a small JSONL trace that hits the 2D bucketing code paths."""
    with open(path, "w") as f:
        # Header with metadata
        f.write(json.dumps({
            "_header": True,
            "gpu_name": "TestGPU",
            "model_name": "test/model",
            "num_hidden_layers": 12,
            "hidden_size": 768,
            "num_attention_heads": 12,
            "vocab_size": 50000,
            "max_model_len": 2048,
            "block_size": 16,
        }) + "\n")
        f.write(json.dumps({"__marker__": "profiling_start"}) + "\n")
        # Mix of prefill and decode records across (tt, conc) buckets.
        # Include some with extreme values that an outlier filter would remove.
        for i in range(50):
            f.write(json.dumps({
                "total_tokens": 10 + (i % 3),
                "num_new_reqs": 0,
                "num_decode_seqs": 1 + (i % 5),
                "step_cycle_us": 30000.0 + (i * 100),
            }) + "\n")
        for i in range(30):
            f.write(json.dumps({
                "total_tokens": 256 + (i % 4),
                "num_new_reqs": 1,
                "num_decode_seqs": i % 10,
                "step_cycle_us": 80000.0 + (i * 500),
            }) + "\n")
        # Heavy-tail outlier that only non-none filters would affect
        f.write(json.dumps({
            "total_tokens": 11,
            "num_new_reqs": 0,
            "num_decode_seqs": 3,
            "step_cycle_us": 900000.0,
        }) + "\n")


def run_builder(builder_path, trace, output, extra_args=()):
    result = subprocess.run(
        ["python3", builder_path, trace, output, *extra_args],
        capture_output=True, text=True, cwd=REPO,
    )
    return result


def main():
    with tempfile.TemporaryDirectory() as td:
        trace = os.path.join(td, "trace.jsonl")
        out_old = os.path.join(td, "out_parent.json")
        out_new = os.path.join(td, "out_new_default.json")
        out_explicit = os.path.join(td, "out_new_explicit_none.json")
        out_iqr = os.path.join(td, "out_new_iqr.json")
        make_synthetic_trace(trace)

        # Parent-commit version of builder
        old_builder_path = os.path.join(td, "build_parent.py")
        subprocess.check_call(
            ["git", "show",
             f"{PARENT}:vllm_emulator/profile/build_serving_profile_filtered.py"],
            stdout=open(old_builder_path, "w"), cwd=REPO,
        )

        current_builder = os.path.join(
            REPO, "vllm_emulator/profile/build_serving_profile_filtered.py")

        # 1. Parent-commit builder
        r1 = run_builder(old_builder_path, trace, out_old)
        if r1.returncode != 0:
            print("PARENT BUILDER FAILED:", r1.stderr); sys.exit(1)

        # 2. F1 builder with default (implicit none)
        r2 = run_builder(current_builder, trace, out_new)
        if r2.returncode != 0:
            print("F1 DEFAULT BUILDER FAILED:", r2.stderr); sys.exit(1)

        # 3. F1 builder with explicit --outlier-filter none
        r3 = run_builder(current_builder, trace, out_explicit,
                         ["--outlier-filter", "none"])
        if r3.returncode != 0:
            print("F1 EXPLICIT NONE BUILDER FAILED:", r3.stderr); sys.exit(1)

        # 4. F1 builder with --outlier-filter iqr (must differ from default)
        r4 = run_builder(current_builder, trace, out_iqr,
                         ["--outlier-filter", "iqr"])
        if r4.returncode != 0:
            print("F1 IQR BUILDER FAILED:", r4.stderr); sys.exit(1)

        # Byte-level comparisons
        with open(out_old, "rb") as f: b_old = f.read()
        with open(out_new, "rb") as f: b_new = f.read()
        with open(out_explicit, "rb") as f: b_explicit = f.read()
        with open(out_iqr, "rb") as f: b_iqr = f.read()

        print("=== F1 BYTE-IDENTITY RESULTS ===")
        print(f"parent vs F1-default-none       : "
              f"{'IDENTICAL' if b_old == b_new else 'DIFFER'}")
        print(f"parent vs F1-explicit-none      : "
              f"{'IDENTICAL' if b_old == b_explicit else 'DIFFER'}")
        print(f"parent vs F1-iqr (must DIFFER)  : "
              f"{'DIFFER (expected)' if b_old != b_iqr else 'IDENTICAL (BUG)'}")

        if b_old != b_new or b_old != b_explicit:
            print("\nBYTE-IDENTITY CONTRACT VIOLATED.")
            sys.exit(1)
        if b_old == b_iqr:
            print("\nIQR PATH DID NOT RUN.")
            sys.exit(1)
        print("\nALL SMOKE TESTS PASSED.")


if __name__ == "__main__":
    main()
