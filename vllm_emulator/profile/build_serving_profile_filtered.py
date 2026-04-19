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
    parser.add_argument("--profile-axes", choices=("2d", "3d"), default="2d",
                        help="Bucket-key axes. 2d (default): (tt, conc) as today, "
                             "byte-identical output. 3d (F4): also emit axis fields "
                             "keyed by (tt, conc, num_new_reqs_bucket); schema v2.1.")
    parser.add_argument("--new-reqs-bucket-width", type=int, default=1,
                        help="Bucket width for num_new_reqs axis (F4). Default 1 "
                             "(no bucketing). Only used when --profile-axes 3d.")
    parser.add_argument("--alpha-kv", choices=("none", "model"), default="none",
                        help="KV-adjustment mode. none (default): key by (tt, conc), "
                             "byte-identical to pre-alpha output. model: compute "
                             "alpha from model_config (KV-bytes/FFN-bytes ratio at "
                             "decode) and bucket by (tt_eff=tt+alpha*sum_kv, conc). "
                             "Requires sum_kv in trace records.")
    parser.add_argument("--reservoir-size", type=int, default=0,
                        help="Experimental: per-bucket reservoir cap "
                             "(Vitter 1985 Algorithm R). 0 (default): no cap, "
                             "byte-identical output. N>0: randomly keep exactly "
                             "N samples per bucket when n>N, preserving the "
                             "distribution in expectation. Used to test whether "
                             "per-bucket sample-count dominance biases the "
                             "oracle's uniform random.choice.")
    return parser.parse_args()


def _reservoir_sample(samples, k, rng):
    """Vitter 1985 Algorithm R reservoir sample (preserves uniform probability).

    Given n samples and cap k, return a length-k random sub-sample.
    If n <= k, returns samples unchanged. Deterministic for a given rng state.
    """
    n = len(samples)
    if n <= k:
        return samples
    reservoir = list(samples[:k])
    for i in range(k, n):
        j = rng.randint(0, i)
        if j < k:
            reservoir[j] = samples[i]
    return reservoir


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


def build_2d_distribution(data, label, outlier_filter="none", reservoir_size=0):
    """Build 2D distribution from (tt_bucket, conc_bucket) -> [latency_us].

    Returns list of {tt, conc, samples: [...]} dicts. When outlier_filter
    is "none" and reservoir_size is 0 the output is byte-identical to
    the pre-F1 code.
    """
    import random as _random
    rng = _random.Random(42)
    distribution = []
    total_in = 0
    total_out = 0
    buckets_filtered = 0
    buckets_reservoired = 0
    for (ttb, cb), lats in sorted(data.items()):
        samples = [round(v, 1) for v in lats]
        total_in += len(samples)
        if outlier_filter != "none":
            before = len(samples)
            samples = _filter_outliers(samples, outlier_filter)
            if len(samples) != before:
                buckets_filtered += 1
        if reservoir_size > 0 and len(samples) > reservoir_size:
            samples = _reservoir_sample(samples, reservoir_size, rng)
            buckets_reservoired += 1
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
    if reservoir_size > 0:
        print(f"    reservoir_size={reservoir_size}: "
              f"capped {buckets_reservoired} buckets; "
              f"final total samples {total_out} "
              f"(from {total_in}, kept {100.0*total_out/total_in:.1f}%)")
    return distribution


def build_3d_distribution(data, label):
    """F4: Build 3D distribution from (tt_b, conc_b, new_reqs_b) -> [us].

    Same shape as build_2d_distribution but with a third axis for
    num_new_reqs. Written to profile v2.1 axis fields IN ADDITION to
    the existing 2D fields (which keep their usual shape from the same
    records).
    """
    distribution = []
    for (ttb, cb, nb), lats in sorted(data.items()):
        samples = [round(v, 1) for v in lats]
        distribution.append({
            "tt": ttb,
            "conc": cb,
            "new_reqs": nb,
            "num_samples": len(lats),
            "samples": samples,
        })
    print(f"  {label}: {len(distribution)} cells (3D)")
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
    new_reqs_w = args.new_reqs_bucket_width
    emit_3d = args.profile_axes == "3d"

    # Compute alpha_kv from model_config for decode-attention-work adjustment.
    # alpha = KV-cache bytes per token per layer / (FFN + projection bytes per
    # token per layer). Memory-bound decode assumption. All values from
    # model_config; no tunable constants.
    alpha_kv = 0.0
    if args.alpha_kv == "model":
        if not model_config:
            print("ERROR: --alpha-kv model requires model_config in trace header.",
                  file=sys.stderr)
            sys.exit(1)
        hidden = model_config.get("hidden_size")
        num_kv = model_config.get("num_key_value_heads") or model_config.get("num_kv_heads")
        head_dim = model_config.get("head_dim") or (
            hidden // model_config.get("num_attention_heads", 1) if hidden else None)
        intermediate = model_config.get("intermediate_size")
        dtype_bytes = 2  # bf16/fp16; standard for LLM inference
        if not all([hidden, num_kv, head_dim, intermediate]):
            print(f"ERROR: model_config missing required fields. "
                  f"hidden={hidden} num_kv={num_kv} head_dim={head_dim} "
                  f"intermediate={intermediate}", file=sys.stderr)
            sys.exit(1)
        # Per-token per-layer byte cost (memory-bound decode):
        #   kv_bytes_per_token_per_layer = 2 (K+V) * num_kv_heads * head_dim * dtype_bytes
        #   ffn_bytes_per_layer = 3 (gate+up+down) * hidden * intermediate * dtype_bytes
        #   proj_bytes_per_layer = 4 (Q,K,V,O) * hidden * (num_q_heads*head_dim) * dtype_bytes
        #     ≈ 4 * hidden^2 * dtype_bytes (assuming GQA projections collapse to hidden)
        kv_b = 2 * num_kv * head_dim * dtype_bytes
        ffn_b = 3 * hidden * intermediate * dtype_bytes
        proj_b = 4 * hidden * hidden * dtype_bytes
        alpha_kv = kv_b / (ffn_b + proj_b)
        print(f"\nalpha_kv from model_config: {alpha_kv:.6g}")
        print(f"  hidden={hidden} num_kv_heads={num_kv} head_dim={head_dim} "
              f"intermediate={intermediate}")
        print(f"  kv_bytes/token/layer={kv_b} "
              f"(ffn+proj)_bytes/token/layer={ffn_b + proj_b}")

    def tt_bucket(tt):
        """Map total_tokens (or tt_eff, which may be float) to integer bucket center."""
        return int(tt // tt_w) * tt_w + tt_w // 2

    def conc_bucket(n):
        """Map concurrency to bucket center using uniform width."""
        return (n // conc_w) * conc_w + conc_w // 2

    def new_reqs_bucket(n):
        """Map num_new_reqs to bucket center using uniform width (F4)."""
        return (n // new_reqs_w) * new_reqs_w + new_reqs_w // 2

    # Bucket records into prefill, decode, and combined 2D maps.
    # Prefill = steps with new_reqs > 0 (eager mode in vLLM V1).
    # Decode  = steps with no new_reqs (CUDA graph mode).
    prefill_2d_data = defaultdict(list)
    decode_2d_data = defaultdict(list)
    step_cycle_2d_data = defaultdict(list)
    # F4 3D tables (only populated when emit_3d).
    prefill_3d_data = defaultdict(list)
    decode_3d_data = defaultdict(list)
    step_cycle_3d_data = defaultdict(list)

    for r in records:
        tt = r["total_tokens"]
        nnr = r.get("num_new_reqs", 0)
        conc = nnr + r.get("num_decode_seqs", 0)
        if conc < 1:
            conc = 1
        # Apply KV-adjustment when enabled and sum_kv present in record.
        if alpha_kv > 0:
            sum_kv = r.get("sum_kv", 0)
            tt_eff = tt + alpha_kv * sum_kv
        else:
            tt_eff = tt
        ttb = tt_bucket(tt_eff)
        cb = conc_bucket(conc)
        step_cycle_2d_data[(ttb, cb)].append(r["step_cycle_us"])
        if nnr > 0:
            prefill_2d_data[(ttb, cb)].append(r["step_cycle_us"])
        else:
            decode_2d_data[(ttb, cb)].append(r["step_cycle_us"])
        if emit_3d:
            nb = new_reqs_bucket(nnr)
            step_cycle_3d_data[(ttb, cb, nb)].append(r["step_cycle_us"])
            if nnr > 0:
                prefill_3d_data[(ttb, cb, nb)].append(r["step_cycle_us"])
            else:
                decode_3d_data[(ttb, cb, nb)].append(r["step_cycle_us"])

    print(f"\n2D distributions (tt_bucket_width={tt_w}, conc_bucket_width={conc_w}, "
          f"outlier_filter={args.outlier_filter}):")
    step_cycle_dist = build_2d_distribution(
        step_cycle_2d_data, "step_cycle",
        outlier_filter=args.outlier_filter, reservoir_size=args.reservoir_size)
    prefill_dist = build_2d_distribution(
        prefill_2d_data, "prefill",
        outlier_filter=args.outlier_filter, reservoir_size=args.reservoir_size)
    decode_dist = build_2d_distribution(
        decode_2d_data, "decode",
        outlier_filter=args.outlier_filter, reservoir_size=args.reservoir_size)

    if emit_3d:
        print(f"\n3D distributions (new_reqs_bucket_width={new_reqs_w}):")
        step_cycle_3d_dist = build_3d_distribution(step_cycle_3d_data, "step_cycle_3d")
        prefill_3d_dist = build_3d_distribution(prefill_3d_data, "prefill_3d")
        decode_3d_dist = build_3d_distribution(decode_3d_data, "decode_3d")
    else:
        step_cycle_3d_dist = prefill_3d_dist = decode_3d_dist = None

    # Summary
    if step_cycle_dist:
        tt_set = sorted(set(e["tt"] for e in step_cycle_dist))
        conc_set = sorted(set(e["conc"] for e in step_cycle_dist))
        total_samples = sum(e["num_samples"] for e in step_cycle_dist)
        print(f"  {len(step_cycle_dist)} cells, tt range={tt_set[0]}-{tt_set[-1]}, "
              f"conc buckets={conc_set}, total samples={total_samples}")

    profile = {
        "version": "2.1" if emit_3d else "2.0",
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
    if emit_3d:
        profile["bucketing"]["new_reqs_bucket_width"] = new_reqs_w
        profile["prefill_axis_distribution"] = sorted(
            prefill_3d_dist, key=lambda e: (e["tt"], e["conc"], e["new_reqs"]))
        profile["decode_axis_distribution"] = sorted(
            decode_3d_dist, key=lambda e: (e["tt"], e["conc"], e["new_reqs"]))
        profile["step_cycle_axis_distribution"] = sorted(
            step_cycle_3d_dist, key=lambda e: (e["tt"], e["conc"], e["new_reqs"]))
    if model_config:
        profile["model_config"] = model_config

    # Store alpha_kv in the profile pack for oracle use at query time.
    if alpha_kv > 0:
        profile["alpha_kv"] = alpha_kv

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
