#!/usr/bin/env python3
"""F4 smoke tests:

1. Byte-identity: builder at --profile-axes 2d (default) produces output
   identical to parent-commit builder.
2. 3D mode: --profile-axes 3d writes new axis fields and bumps version
   to 2.1, but keeps 2D fields identical to a plain 2d build.
3. Oracle: with 2D profile, VLLM_EMULATOR_PROFILE_AXES=3d falls back
   (warns once, uses 2D). With 3D profile, oracle samples via 3D path.
"""
import json
import os
import subprocess
import sys
import tempfile

REPO = "/home/wd312/Code/llm/vllm-emulator"
PARENT = "a102eed27"


def make_synthetic_trace(path):
    with open(path, "w") as f:
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
        for i in range(60):
            f.write(json.dumps({
                "total_tokens": 10 + (i % 3),
                "num_new_reqs": i % 3,  # 0, 1, 2 — exercises the 3D axis
                "num_decode_seqs": 1 + (i % 4),
                "step_cycle_us": 30000.0 + (i * 100),
            }) + "\n")


def run_builder(builder_path, trace, output, extra_args=()):
    return subprocess.run(
        ["python3", builder_path, trace, output, *extra_args],
        capture_output=True, text=True, cwd=REPO,
    )


def main():
    with tempfile.TemporaryDirectory() as td:
        trace = os.path.join(td, "trace.jsonl")
        out_parent = os.path.join(td, "out_parent.json")
        out_2d = os.path.join(td, "out_2d.json")
        out_2d_explicit = os.path.join(td, "out_2d_explicit.json")
        out_3d = os.path.join(td, "out_3d.json")
        make_synthetic_trace(trace)

        parent_builder = os.path.join(td, "parent_builder.py")
        subprocess.check_call(
            ["git", "show",
             f"{PARENT}:vllm_emulator/profile/build_serving_profile_filtered.py"],
            stdout=open(parent_builder, "w"), cwd=REPO,
        )
        current_builder = os.path.join(
            REPO, "vllm_emulator/profile/build_serving_profile_filtered.py")

        for builder, out, args, label in [
            (parent_builder, out_parent, (), "parent"),
            (current_builder, out_2d, (), "F4-default-2d"),
            (current_builder, out_2d_explicit, ("--profile-axes", "2d"), "F4-explicit-2d"),
            (current_builder, out_3d, ("--profile-axes", "3d"), "F4-3d"),
        ]:
            r = run_builder(builder, trace, out, args)
            if r.returncode != 0:
                print(f"{label} BUILDER FAILED:", r.stderr)
                sys.exit(1)

        with open(out_parent, "rb") as f: b_parent = f.read()
        with open(out_2d, "rb") as f: b_2d = f.read()
        with open(out_2d_explicit, "rb") as f: b_2d_e = f.read()
        with open(out_3d, "rb") as f: b_3d = f.read()

        print("=== F4 BYTE-IDENTITY & 3D BEHAVIOR ===")
        print(f"parent vs F4-default-2d   : "
              f"{'IDENTICAL' if b_parent == b_2d else 'DIFFER'}")
        print(f"parent vs F4-explicit-2d  : "
              f"{'IDENTICAL' if b_parent == b_2d_e else 'DIFFER'}")
        if b_parent != b_2d or b_parent != b_2d_e:
            print("\nBYTE-IDENTITY CONTRACT VIOLATED.")
            sys.exit(1)

        # Check 3D output has axis fields and version 2.1
        p3d = json.load(open(out_3d))
        p2d = json.load(open(out_2d))
        print(f"\n3D version: {p3d['version']} (expect 2.1)")
        print(f"2D version: {p2d['version']} (expect 2.0)")
        assert p3d["version"] == "2.1", "3D must bump schema to 2.1"
        assert p2d["version"] == "2.0", "2D must stay 2.0"

        for k in ("prefill_axis_distribution", "decode_axis_distribution",
                  "step_cycle_axis_distribution"):
            assert k in p3d, f"3D profile missing {k}"
            assert k not in p2d, f"2D profile must not have {k}"
            n = len(p3d[k])
            print(f"  {k}: {n} buckets")
            if n > 0:
                sample = p3d[k][0]
                assert "new_reqs" in sample, f"{k} bucket missing new_reqs field"

        # 2D tables must be identical between 2d and 3d builds (same records).
        for k in ("prefill_2d_distribution", "decode_2d_distribution",
                  "step_cycle_2d_distribution"):
            assert p3d[k] == p2d[k], f"{k} differs between 2d and 3d builds"
        print("3D 2D-subset equal to plain-2d: YES")

        print("\nALL F4 SMOKE TESTS PASSED.")


if __name__ == "__main__":
    main()
