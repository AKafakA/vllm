#!/usr/bin/env python3
"""Build serving profile: bucket step-cycle trace records by (tt, concurrency).

Reads a step_cycle JSONL trace, splits into prefill vs decode by num_new_reqs,
buckets by (total_tokens, concurrency), and stores raw samples per bucket.
No outlier filtering, no heuristic thresholds, no correction tables.

Usage:
  python build_serving_profile_filtered.py <step_cycle_file> <output> \
      [--model-name NAME] [--gpu-model GPU] \
      [--tt-bucket-width W] [--conc-bucket-width W]

Bucketing parameters (user-chosen resolution, not tuned constants):
  --tt-bucket-width   Width of total_tokens buckets (default: 1, i.e. no bucketing)
  --conc-bucket-width Width of concurrency buckets (default: 5)

If the trace file contains a _header record (auto-collected by StepCycleTracer),
model_name and gpu_model are extracted automatically and the profile pack includes
a model_config section. CLI args override the header if provided.
"""
import argparse
import json
import statistics
import sys
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build serving profile from step-cycle trace records.")
    parser.add_argument("step_cycle_file",
                        help="Path to step_cycle JSONL trace")
    parser.add_argument("output",
                        help="Path to output JSON profile")
    parser.add_argument("--model-name", default=None,
                        help="Model name (overrides trace header)")
    parser.add_argument("--gpu-model", default=None,
                        help="GPU model (overrides trace header)")
    parser.add_argument("--tt-bucket-width", type=int, default=1,
                        help="Total-tokens bucket width (default: 1, no bucketing)")
    parser.add_argument("--conc-bucket-width", type=int, default=5,
                        help="Concurrency bucket width (default: 5)")
    parser.add_argument("--step-timing-csv", default=None,
                        help="Path to step_timing.csv (from VLLM_DIAG_STEP_TIMING_LOG). "
                             "Extracts avg_sample_ms and avg_exec_ms for profile pack.")
    return parser.parse_args()


def load_trace(path):
    """Load step-cycle trace, filtering by profiling_start markers if present.

    Returns (trace_header, records).
    If markers are present, only records after profiling_start markers are kept.
    If no markers are found, ALL records are used.
    """
    trace_header = None
    records = []
    all_raw = []
    in_profiling = False

    for line in open(path):
        r = json.loads(line)
        if r.get("_header"):
            trace_header = r
            continue
        if r.get("__marker__") == "profiling_start":
            in_profiling = True
            continue
        if r.get("__marker__") == "profiling_stop":
            in_profiling = False
            continue
        if r.get("__marker__"):
            continue
        if "total_tokens" in r:
            all_raw.append(r)
            if in_profiling:
                records.append(r)

    if not records and all_raw:
        print("  No profiling_start markers found; using all records")
        records = all_raw

    print(f"Records: {len(records)} (from {len(all_raw)} total)")
    return trace_header, records


def extract_metadata(trace_header, cli_model_name, cli_gpu_model):
    """Resolve model_name, gpu_model, and model_config from trace header + CLI."""
    model_name = cli_model_name or (trace_header or {}).get("model_name") or "unknown"
    gpu_model = cli_gpu_model or (trace_header or {}).get("gpu_name") or "unknown"

    model_config = None
    if trace_header:
        _MC_KEYS = ("num_hidden_layers", "hidden_size", "num_attention_heads",
                     "num_key_value_heads", "vocab_size", "intermediate_size",
                     "head_dim", "max_model_len", "block_size")
        model_config = {k: trace_header[k] for k in _MC_KEYS if k in trace_header}
        _GPU_KEYS = ("gpu_name", "gpu_memory_bytes", "gpu_sm_count",
                      "gpu_compute_capability", "gpu_count")
        gpu_config = {k: trace_header[k] for k in _GPU_KEYS if k in trace_header}
        if gpu_config:
            model_config["gpu"] = gpu_config
        print(f"Trace header: gpu={gpu_model}, model={model_name}, "
              f"layers={model_config.get('num_hidden_layers', '?')}, "
              f"vocab={model_config.get('vocab_size', '?')}")
    else:
        print("WARNING: No _header in trace file. GPU/model metadata not available. "
              "Re-profile with latest StepCycleTracer to auto-collect.")

    return model_name, gpu_model, model_config


def build_2d_distribution(data, label):
    """Build 2D distribution from (tt_bucket, conc_bucket) -> [latency_us].

    Returns list of {tt, conc, samples: [...]} dicts with all raw samples.
    No outlier filtering, no min-sample thresholds.
    """
    distribution = []
    for (ttb, cb), lats in sorted(data.items()):
        samples = [round(v, 1) for v in lats]
        distribution.append({
            "tt": ttb,
            "conc": cb,
            "num_samples": len(lats),
            "samples": samples,
        })
    print(f"  {label}: {len(distribution)} cells")
    return distribution


def main():
    args = parse_args()

    trace_header, records = load_trace(args.step_cycle_file)
    model_name, gpu_model, model_config = extract_metadata(
        trace_header, args.model_name, args.gpu_model)

    if not records:
        print("ERROR: No records found in trace file", file=sys.stderr)
        sys.exit(1)

    tt_w = args.tt_bucket_width
    conc_w = args.conc_bucket_width

    def tt_bucket(tt):
        """Map total_tokens to bucket center using uniform width."""
        return (tt // tt_w) * tt_w + tt_w // 2

    def conc_bucket(n):
        """Map concurrency to bucket center using uniform width."""
        return (n // conc_w) * conc_w + conc_w // 2

    # Bucket records into prefill, decode, and combined 2D maps.
    # Prefill = steps with new_reqs > 0 (eager mode in vLLM V1).
    # Decode  = steps with no new_reqs (CUDA graph mode).
    prefill_2d_data = defaultdict(list)
    decode_2d_data = defaultdict(list)
    step_cycle_2d_data = defaultdict(list)

    for r in records:
        tt = r["total_tokens"]
        conc = r.get("num_new_reqs", 0) + r.get("num_decode_seqs", 0)
        if conc < 1:
            conc = 1
        ttb = tt_bucket(tt)
        cb = conc_bucket(conc)
        step_cycle_2d_data[(ttb, cb)].append(r["step_cycle_us"])
        if r.get("num_new_reqs", 0) > 0:
            prefill_2d_data[(ttb, cb)].append(r["step_cycle_us"])
        else:
            decode_2d_data[(ttb, cb)].append(r["step_cycle_us"])

    print(f"\n2D distributions (tt_bucket_width={tt_w}, conc_bucket_width={conc_w}):")
    step_cycle_dist = build_2d_distribution(step_cycle_2d_data, "step_cycle")
    prefill_dist = build_2d_distribution(prefill_2d_data, "prefill")
    decode_dist = build_2d_distribution(decode_2d_data, "decode")

    # Summary
    if step_cycle_dist:
        tt_set = sorted(set(e["tt"] for e in step_cycle_dist))
        conc_set = sorted(set(e["conc"] for e in step_cycle_dist))
        total_samples = sum(e["num_samples"] for e in step_cycle_dist)
        print(f"  {len(step_cycle_dist)} cells, tt range={tt_set[0]}-{tt_set[-1]}, "
              f"conc buckets={conc_set}, total samples={total_samples}")

    profile = {
        "version": "2.0",
        "gpu_model": gpu_model,
        "model_name": model_name,
        "profile_type": "serving_step_cycle_2d",
        "bucketing": {
            "tt_bucket_width": tt_w,
            "conc_bucket_width": conc_w,
        },
        "prefill_2d_distribution": sorted(prefill_dist, key=lambda e: (e["tt"], e["conc"])),
        "decode_2d_distribution": sorted(decode_dist, key=lambda e: (e["tt"], e["conc"])),
        "step_cycle_2d_distribution": sorted(step_cycle_dist, key=lambda e: (e["tt"], e["conc"])),
    }
    if model_config:
        profile["model_config"] = model_config

    # Extract per-step engine overhead from step timing CSV (if provided).
    # These values are used by the executor hook for sample_tokens delay
    # and surrogate calibration. All values from real GPU profiling.
    if args.step_timing_csv:
        import csv
        with open(args.step_timing_csv) as stf:
            reader = csv.DictReader(stf)
            sample_vals = []
            exec_vals = []
            for row in reader:
                if "sample_ms" in row:
                    sample_vals.append(float(row["sample_ms"]))
                if "exec_ms" in row:
                    exec_vals.append(float(row["exec_ms"]))
        if sample_vals:
            profile["avg_sample_ms"] = round(sum(sample_vals) / len(sample_vals), 4)
            print(f"\nStep timing: avg_sample_ms={profile['avg_sample_ms']} "
                  f"(from {len(sample_vals)} steps)")
        if exec_vals:
            profile["avg_exec_ms"] = round(sum(exec_vals) / len(exec_vals), 4)
            print(f"Step timing: avg_exec_ms={profile['avg_exec_ms']} "
                  f"(from {len(exec_vals)} steps)")

    with open(args.output, "w") as f:
        json.dump(profile, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
