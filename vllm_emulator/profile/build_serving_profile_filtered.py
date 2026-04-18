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


_OUTLIER_FILTER_CHOICES = ("none", "iqr", "mad", "winsor")


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
    parser.add_argument("--outlier-filter", choices=_OUTLIER_FILTER_CHOICES,
                        default="none",
                        help="Per-bucket outlier filter applied to raw samples. "
                             "none (default): no change, byte-identical output. "
                             "iqr: Tukey 1.5x fence (1977). "
                             "mad: modified Z-score via MAD at 3.5 (Iglewicz-Hoaglin 1993). "
                             "winsor: clip to [p1, p99] (classical).")
    return parser.parse_args()


def _filter_outliers(samples, method):
    """Apply a named outlier filter to a list of bucket samples.

    Constants are all named textbook defaults, not tuned values:
    - 1.5 * IQR fence: Tukey, Exploratory Data Analysis (1977).
    - MAD consistency constant 0.6745 = inverse normal CDF at 0.75.
    - MAD outlier threshold 3.5: Iglewicz & Hoaglin (1993), "Volume 16:
      How to Detect and Handle Outliers".
    - Winsorize at 1/99 percentiles: classical default when no domain
      preference is specified.
    - Minimum-sample guards (4 for IQR/MAD, 100 for winsor) are
      definitional: the statistic is meaningless below them.

    Returns the filtered sample list. Never introduces fake samples;
    monotonically reduces (or, for winsor, equal-length with clipped
    values) the per-bucket distribution.
    """
    n = len(samples)
    if method == "none" or n < 4:
        return samples
    sorted_s = sorted(samples)

    if method == "iqr":
        q1 = sorted_s[n // 4]
        q3 = sorted_s[(3 * n) // 4]
        iqr = q3 - q1
        lo = q1 - 1.5 * iqr  # Tukey 1977
        hi = q3 + 1.5 * iqr
        return [x for x in samples if lo <= x <= hi]

    if method == "mad":
        median = sorted_s[n // 2]
        abs_dev = sorted(abs(x - median) for x in samples)
        mad = abs_dev[n // 2]
        if mad == 0:
            return samples
        # 0.6745 = Φ⁻¹(0.75); 3.5 per Iglewicz-Hoaglin 1993.
        return [x for x in samples
                if abs(0.6745 * (x - median) / mad) <= 3.5]

    if method == "winsor":
        if n < 100:
            return samples
        lo = sorted_s[n // 100]          # classical 1st percentile
        hi = sorted_s[(99 * n) // 100]   # classical 99th percentile
        return [min(max(x, lo), hi) for x in samples]

    return samples


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


def build_2d_distribution(data, label, outlier_filter="none"):
    """Build 2D distribution from (tt_bucket, conc_bucket) -> [latency_us].

    Returns list of {tt, conc, samples: [...]} dicts. When outlier_filter
    is "none" the output is byte-identical to the pre-F1 code. For the
    non-none methods every numeric constant is a named textbook default
    (see `_filter_outliers` docstring).
    """
    distribution = []
    total_in = 0
    total_out = 0
    buckets_filtered = 0
    for (ttb, cb), lats in sorted(data.items()):
        samples = [round(v, 1) for v in lats]
        total_in += len(samples)
        if outlier_filter != "none":
            before = len(samples)
            samples = _filter_outliers(samples, outlier_filter)
            if len(samples) != before:
                buckets_filtered += 1
        total_out += len(samples)
        distribution.append({
            "tt": ttb,
            "conc": cb,
            "num_samples": len(samples),
            "samples": samples,
        })
    print(f"  {label}: {len(distribution)} cells")
    if outlier_filter != "none" and total_in > 0:
        dropped = total_in - total_out
        pct = 100.0 * dropped / total_in
        print(f"    outlier_filter={outlier_filter}: "
              f"dropped {dropped} of {total_in} samples "
              f"({pct:.2f}%) across {buckets_filtered} buckets")
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

    print(f"\n2D distributions (tt_bucket_width={tt_w}, conc_bucket_width={conc_w}, "
          f"outlier_filter={args.outlier_filter}):")
    step_cycle_dist = build_2d_distribution(
        step_cycle_2d_data, "step_cycle", outlier_filter=args.outlier_filter)
    prefill_dist = build_2d_distribution(
        prefill_2d_data, "prefill", outlier_filter=args.outlier_filter)
    decode_dist = build_2d_distribution(
        decode_2d_data, "decode", outlier_filter=args.outlier_filter)

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
