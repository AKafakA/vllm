#!/usr/bin/env python3
"""Bandwidth-slope calibration for the roofline correction.

Captures a single long-sequence decode trajectory and regresses
step_cycle_us against sum_kv. The slope is the per-token
kv-cache-movement cost (us/token) on this specific GPU.

Workflow:
  1. Assume a vLLM server is already running at --port.
  2. Send one synthetic request: long prompt (max-500 tokens),
     short decode (500 tokens). The server traces every step to
     --trace-path via VLLM_EMULATOR_TRACE_STEP_CYCLE=1.
  3. Parse decode-only steps from the trace (num_new_reqs==0,
     num_decode_seqs==1).
  4. Regress step_cycle_us ~ sum_kv. Output JSON:
       bw_slope_us_per_token
       bw_intercept_us
       bw_r_squared
       n_samples
       sum_kv_range

Usage:
  python3 tools/profile_bw_calibration.py \
      --model Qwen/Qwen3-8B \
      --base-url http://localhost:8100 \
      --trace-path /tmp/bw_calib_trace.jsonl \
      --prompt-len 3500 --output-len 500 \
      --out-json results/bw_calibration_a10_qwen3-8b.json

The caller (chain script) is responsible for starting and stopping
the server with VLLM_EMULATOR_TRACE_STEP_CYCLE=1 and the matching
VLLM_EMULATOR_STEP_TRACE_OUTPUT.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import requests


def send_one_long_request(base_url: str, model: str,
                          prompt_len: int, output_len: int,
                          timeout: int = 600) -> dict:
    """Send a single completion request with a long synthetic prompt."""
    # Build a long prompt by repeating a word.
    # We avoid tokenizer to keep the tool dependency-free; the server
    # tokenizes anyway and the exact token count is not critical —
    # calibration uses sum_kv values actually observed, not requested.
    word = "hello "
    # approx 1 word per token for gpt-style tokenizers → safe upper bound
    prompt = (word * (prompt_len * 2)).strip()[: prompt_len * 6]
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": output_len,
        "temperature": 0.0,
        "stream": False,
    }
    t0 = time.time()
    r = requests.post(
        f"{base_url}/v1/completions", json=payload, timeout=timeout,
    )
    r.raise_for_status()
    return {"elapsed_s": time.time() - t0, "response": r.json()}


def parse_decode_steps(trace_path: str, start_offset: int = 0):
    """Return (header, [(step_cycle_us, sum_kv)]) for decode-only steps.

    Header is parsed from the whole file (it's at the top).  Points
    are parsed only from lines written AFTER start_offset so we
    isolate this calibration request from warmup/profile records.
    """
    points = []
    header = None
    with open(trace_path) as f:
        # Header from top of file.
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("_header"):
                header = d
                break
        # Restart from start_offset for records.
        f.seek(start_offset)
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("_header") or d.get("__marker__"):
                continue
            if (
                d.get("num_new_reqs", 0) == 0
                and d.get("num_decode_seqs", 0) == 1
                and "sum_kv" in d
                and "step_cycle_us" in d
            ):
                points.append((d["step_cycle_us"], d["sum_kv"]))
    return header, points


def fit_linear(xs, ys):
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(slope), float(intercept), float(r2)


def compute_kv_per_token_bytes(header: dict) -> int:
    L = header.get("num_hidden_layers")
    KV_H = header.get("num_key_value_heads")
    D = header.get("head_dim")
    if not all([L, KV_H, D]):
        return 0
    return 2 * KV_H * D * 2 * L  # K+V * kv_heads * head_dim * bytes(fp16) * layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="http://localhost:8100")
    ap.add_argument("--trace-path", required=True,
                    help="Step-cycle trace file the server is writing to.")
    ap.add_argument("--prompt-len", type=int, default=3500)
    ap.add_argument("--output-len", type=int, default=500)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--min-samples", type=int, default=50,
                    help="Minimum decode steps required for a usable fit.")
    ap.add_argument("--hw-bw-gbs", type=float, default=None,
                    help="If provided, also emit a 'constant' slope "
                         "computed as kv_per_token_bytes / (hw_bw_gbs*1e9). "
                         "Lets downstream compare measured vs HW-spec BW. "
                         "E.g., 480 for A10 sustained, 600 for RTX 8000.")
    args = ap.parse_args()

    trace_path = Path(args.trace_path)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    # Record where the trace ends right now so we parse only lines
    # written after our request is sent. Do NOT truncate — the
    # server wrote its header at startup and we need it.
    pre_size = trace_path.stat().st_size if trace_path.exists() else 0
    print(f"[calib] trace pre-size = {pre_size} bytes "
          f"(will parse only records written after this point)", flush=True)

    print(f"[calib] sending long request: prompt_len≈{args.prompt_len} "
          f"output_len={args.output_len}", flush=True)
    r = send_one_long_request(args.base_url, args.model,
                              args.prompt_len, args.output_len)
    print(f"[calib] request done in {r['elapsed_s']:.1f}s", flush=True)

    # Small grace period for server to flush remaining trace lines.
    time.sleep(2)

    header, points = parse_decode_steps(str(trace_path), start_offset=pre_size)
    print(f"[calib] parsed {len(points)} decode-only steps "
          f"(from file offset {pre_size})", flush=True)

    if len(points) < args.min_samples:
        print(
            f"[calib] FAIL: only {len(points)} decode samples "
            f"(need >= {args.min_samples})", file=sys.stderr)
        out_json.write_text(json.dumps({
            "error": f"insufficient_samples:{len(points)}",
            "n_samples": len(points),
        }, indent=2))
        return 1

    ys = [p[0] for p in points]  # step_cycle_us
    xs = [p[1] for p in points]  # sum_kv
    slope, intercept, r2 = fit_linear(xs, ys)
    kv_per_tok = compute_kv_per_token_bytes(header or {})
    implied_bw_gbs = (
        (kv_per_tok / (slope * 1e-6)) / 1e9 if slope > 0 and kv_per_tok > 0 else 0.0
    )

    # Also emit a theoretical "constant" slope from HW-spec BW if provided.
    # Lets the oracle pick either "measured" or "constant" and compare.
    slope_constant = None
    if args.hw_bw_gbs and kv_per_tok > 0:
        # slope_us_per_token = kv_bytes / (BW_bytes_per_sec) * 1e6
        slope_constant = kv_per_tok / (args.hw_bw_gbs * 1e9) * 1e6

    result = {
        "gpu_name": (header or {}).get("gpu_name", "unknown"),
        "model_name": args.model,
        "kv_per_token_bytes": kv_per_tok,
        "bw_slope_measured_us_per_token": slope,
        "bw_slope_constant_us_per_token": slope_constant,  # None if no --hw-bw-gbs
        "hw_bw_gbs_input": args.hw_bw_gbs,
        "bw_intercept_us": intercept,
        "bw_r_squared": r2,
        "n_samples": len(points),
        "sum_kv_range": [min(xs), max(xs)],
        "implied_sustained_bw_gbs_from_measured": implied_bw_gbs,
        "calibration_prompt_len": args.prompt_len,
        "calibration_output_len": args.output_len,
    }
    out_json.write_text(json.dumps(result, indent=2))
    print(f"[calib] measured slope = {slope:.4f} us/token  "
          f"intercept = {intercept:.1f} us  "
          f"R² = {r2:.3f}  implied BW = {implied_bw_gbs:.1f} GB/s", flush=True)
    if slope_constant is not None:
        print(f"[calib] constant slope  = {slope_constant:.4f} us/token  "
              f"(from --hw-bw-gbs={args.hw_bw_gbs})", flush=True)
    print(f"[calib] wrote {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
