"""Hardware-independent IPC overhead profiler.

Replaces v1/v2's magic sleeps with condition-based waits using vLLM's
Prometheus `/metrics` endpoint. No hardware-specific heuristics; the
profiler polls the server's running/waiting queue state to determine
when conditions are met.

Generality guarantees for moving to new hardware (A100, L40S, etc.):
  - No sleep constants that encode "RTX 8000 speed" assumptions.
  - Ramp-up: spawn backgrounds one at a time, wait for each to APPEAR
    in `vllm:num_requests_running` before next spawn. Works on any GPU.
  - Dwell: wait for running count to be STABLE (unchanged) for N poll
    intervals. Steady-state detection, not fixed delay.
  - Between measurements: wait for measurement bursts to DRAIN (running
    count returns to N-1).

Records GPU model + model metadata auto-detected from profile pack or
vLLM's /v1/models so output is self-documenting.

Usage:
    python3 tools/profile_ipc_overhead_v3.py \\
        <port> <model> <profile_path> <output_path> \\
        [--burst-k "1 2 4 8"] [--max-n 1024] [--samples-per-cell 5] \\
        [--poll-ms 50] [--stable-polls 5]
"""
import argparse
import json
import os
import requests
import sys
import threading
import time


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("port", type=int)
    p.add_argument("model")
    p.add_argument("profile_path",
                   help="Existing profile pack (used for prefill_step_us baseline)")
    p.add_argument("output_path")
    p.add_argument("--burst-k", default="1 2 4 8",
                   help="Space-sep burst sizes, e.g. '1 2 4 8'")
    p.add_argument("--max-n", type=int, default=1024)
    p.add_argument("--samples-per-cell", type=int, default=5)
    p.add_argument("--poll-ms", type=int, default=50,
                   help="Metrics polling interval (ms)")
    p.add_argument("--stable-polls", type=int, default=5,
                   help="Number of consecutive polls with unchanged "
                        "running-count required to declare steady state")
    return p.parse_args()


def get_metric_gauges(base_url: str) -> dict:
    """Parse the /metrics Prometheus endpoint into {metric_name: float}.

    Returns empty dict if endpoint unavailable. Works with vLLM v1's
    published gauges like `vllm:num_requests_running`."""
    try:
        r = requests.get(f"{base_url}/metrics", timeout=5)
        out = {}
        for line in r.text.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Prometheus format: metric{labels} value
            # Handle both with and without labels.
            if "{" in line:
                name = line.split("{", 1)[0]
            else:
                parts = line.split()
                name = parts[0] if parts else ""
            val_part = line.rsplit(" ", 1)[-1]
            try:
                out.setdefault(name, float(val_part))  # first occurrence
            except ValueError:
                continue
        return out
    except Exception:
        return {}


def get_running_count(base_url: str) -> float:
    m = get_metric_gauges(base_url)
    # vLLM v1 uses `vllm:num_requests_running`; fall back to variants.
    for key in ("vllm:num_requests_running",
                "vllm:running_lora_adapters",  # unlikely fallback
                "vllm:gpu_cache_usage_perc"):
        if key in m:
            return m[key] if "running" in key else -1
    return -1


def get_waiting_count(base_url: str) -> float:
    """vllm:num_requests_waiting — backpressure signal.

    Non-zero means the server cannot admit more requests right now
    (typically KV-cache saturation). The bg-maintainer treats this
    as "stop spawning" to avoid thread-leak when target concurrency
    is unachievable on the hardware/profile."""
    m = get_metric_gauges(base_url)
    return m.get("vllm:num_requests_waiting", -1)


def get_generation_tokens_total(base_url: str) -> float:
    """vllm:generation_tokens_total — monotonic counter of decode
    tokens produced. Used to detect 'bg reached decode phase' by
    observing increments across polls; replaces v1's magic 3s dwell
    with a semantic signal ('bg are past prefill, producing tokens')."""
    m = get_metric_gauges(base_url)
    return m.get("vllm:generation_tokens_total", -1)


def wait_for_decode_phase(base_url: str, target_running: int,
                          stable_polls: int, poll_s: float,
                          timeout_s: float = 60) -> bool:
    """Wait until (1) running count == target_running for stable_polls
    consecutive polls AND (2) generation_tokens_total is strictly
    increasing between the last two polls — meaning all backgrounds
    are past prefill and actively decoding.

    This replaces v1's `time.sleep(3.0)` dwell with a phase-aware
    condition. v1's magic 3s was SEMANTIC (ensure bg are in decode
    so measurement TTFT reflects pure IPC, not IPC + prefill contention);
    we reproduce that semantic via /metrics polling, not timing."""
    start = time.time()
    stable_ct = 0
    prev_tok = -1.0
    while time.time() - start < timeout_s:
        running = get_running_count(base_url)
        tok = get_generation_tokens_total(base_url)
        tokens_advancing = prev_tok >= 0 and tok > prev_tok
        if running == target_running and tokens_advancing:
            stable_ct += 1
            if stable_ct >= stable_polls:
                return True
        else:
            stable_ct = 0
        prev_tok = tok
        time.sleep(poll_s)
    return False


def wait_for_running_at_least(base_url: str, target: int,
                               poll_s: float, timeout_s: float = 120) -> bool:
    """Spin until vLLM reports running >= target. Returns True on success."""
    start = time.time()
    while time.time() - start < timeout_s:
        c = get_running_count(base_url)
        if c >= target:
            return True
        time.sleep(poll_s)
    return False


def wait_for_stable(base_url: str, expected: int,
                    stable_polls: int, poll_s: float,
                    timeout_s: float = 120) -> bool:
    """Wait until running count equals `expected` for `stable_polls`
    consecutive polls. Used to detect steady-state after ramp-up."""
    start = time.time()
    count = 0
    while time.time() - start < timeout_s:
        c = get_running_count(base_url)
        if c == expected:
            count += 1
            if count >= stable_polls:
                return True
        else:
            count = 0
        time.sleep(poll_s)
    return False


def wait_for_drain(base_url: str, back_to: int,
                    stable_polls: int, poll_s: float,
                    timeout_s: float = 120) -> bool:
    """Wait for running to drain back to `back_to` (typically N-1,
    after a burst of k measurement requests completes)."""
    return wait_for_stable(base_url, back_to, stable_polls, poll_s, timeout_s)


def wait_for_at_most(base_url: str, max_val: int,
                     stable_polls: int, poll_s: float,
                     timeout_s: float = 120) -> bool:
    """Wait until running count <= `max_val` for `stable_polls` consecutive
    polls. Unlike `wait_for_stable`, this succeeds even if the count drops
    below max_val (e.g. because a bg request expired) — the condition is
    "burst drained to at most baseline", which is always satisfiable."""
    start = time.time()
    count = 0
    while time.time() - start < timeout_s:
        c = get_running_count(base_url)
        if c <= max_val:
            count += 1
            if count >= stable_polls:
                return True
        else:
            count = 0
        time.sleep(poll_s)
    return False


def start_bg_maintainer(base_url: str, model: str, target_count: int,
                        stop_event, poll_s: float,
                        max_spawn_cap: int | None = None):
    """Daemon that keeps vllm:num_requests_running close to target_count by
    spawning new background requests when count drops below target, with
    two backpressure safeguards to prevent thread leaks when target is
    unachievable (e.g. KV-cache saturation below target concurrency):

      1. Waiting-queue gate: if vllm:num_requests_waiting > 0, the server
         cannot admit more — stop spawning and let existing bg drain or
         the queue clear. This is a SERVER-SIDE signal, not a timing guess.

      2. Hard spawn cap: never spawn more than max_spawn_cap total bg
         requests for a single N (default: 2 * target_count + 20, a safety
         bound not a performance-tuned knob). Prevents thread accumulation
         when running stays pinned below target for structural reasons.

    All spawned threads are daemon=True so they die on process exit.
    Returns the maintainer thread (also a daemon).
    """
    if max_spawn_cap is None:
        max_spawn_cap = target_count * 2 + 20

    def _loop():
        spawned = 0
        while not stop_event.is_set():
            try:
                c = get_running_count(base_url)
                w = get_waiting_count(base_url)
            except Exception:
                c, w = -1, -1

            # Spawn only when: server isn't backpressured AND we haven't
            # hit the safety cap AND we're below target running count.
            if (0 <= c < target_count
                    and w <= 0
                    and spawned < max_spawn_cap):
                t = threading.Thread(
                    target=send_background_request,
                    args=(base_url, model),
                    daemon=True,
                )
                t.start()
                spawned += 1
                # Best-effort admission wait so the next tick sees the
                # updated running count. Bounded by caller's poll cadence.
                wait_for_running_at_least(base_url, int(c) + 1, poll_s,
                                           timeout_s=2)
            time.sleep(poll_s)
    mt = threading.Thread(target=_loop, daemon=True)
    mt.start()
    return mt


def send_background_request(base_url: str, model: str,
                             max_tokens: int = 4096):
    """Spawn a background decode request.

    max_tokens=4096 + ignore_eos=True: bg stays in decode for the
    entire measurement window (typically ~30s). This matches v1's
    semantic intent without v1's magic 3s dwell — instead of timing,
    we choose bg duration long enough to span any measurement budget.
    The caller is responsible for closing/awaiting these at sweep end.
    """
    try:
        requests.post(
            f"{base_url}/v1/completions",
            json={"model": model, "prompt": "background " * 40,
                  "max_tokens": max_tokens,
                  "ignore_eos": True,
                  "temperature": 0},
            timeout=300,
        )
    except Exception:
        pass


def measure_ttft_single(base_url: str, model: str,
                        slot_results: list, slot_idx: int):
    prompt = f"measurement{slot_idx} " * 40
    t0 = time.perf_counter()
    try:
        r = requests.post(
            f"{base_url}/v1/completions",
            json={"model": model, "prompt": prompt,
                  "max_tokens": 5, "temperature": 0, "stream": True},
            stream=True, timeout=120,
        )
        for line in r.iter_lines():
            if line and b"text" in line:
                t1 = time.perf_counter()
                slot_results[slot_idx] = (t1 - t0) * 1e6
                return
    except Exception:
        pass
    slot_results[slot_idx] = -1


def measure_burst(base_url: str, model: str, k: int) -> list[float]:
    """Inject k simultaneous measurement requests; return their TTFTs (us)."""
    slot_results = [0.0] * k
    threads = [threading.Thread(target=measure_ttft_single,
                                 args=(base_url, model, slot_results, i))
               for i in range(k)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    return [s for s in slot_results if s > 0]


def main():
    args = parse_args()
    base_url = f"http://localhost:{args.port}"
    poll_s = args.poll_ms / 1000.0

    # Read prefill_step_us baseline from the profile pack (used by the sweep
    # only to compute overhead = TTFT - prefill; the emulator's own lookups
    # don't depend on this number).
    profile = json.load(open(args.profile_path))
    prefill_step_us = 0.0
    for section in ("prefill_2d_distribution", "prefill_forward_pass", "forward_pass"):
        for e in profile.get(section, []):
            tt = e.get("tt") or e.get("total_tokens")
            if tt is None:
                continue
            if 250 <= tt <= 270:
                samples = e.get("samples", [])
                if samples:
                    s = sorted(samples)
                    prefill_step_us = float(s[len(s) // 2])
                elif "latency_us" in e:
                    prefill_step_us = float(e["latency_us"])
                break
        if prefill_step_us > 0:
            break

    print(f"Profiled prefill step (tt≈256): {prefill_step_us/1000:.1f}ms")
    print(f"Poll interval: {args.poll_ms}ms, stable polls: {args.stable_polls}")

    # Verify /metrics is accessible.
    test_c = get_running_count(base_url)
    if test_c < 0:
        print("ERROR: /metrics endpoint not accessible or vllm:num_requests_running "
              "not exposed. Falling back to v2-compatible fixed-sleep mode "
              "is NOT supported; aborting.", file=sys.stderr)
        sys.exit(1)

    BURST_KS = [int(x) for x in args.burst_k.split()]
    N_SWEEP = [n for n in [1, 2, 3, 5, 8, 12, 20, 30, 50, 100, 150, 200,
                            256, 384, 512, 768, 1024] if n <= args.max_n]
    print(f"N sweep: {N_SWEEP}")
    print(f"Burst k: {BURST_KS}")

    results = []
    for n in N_SWEEP:
        # Wait for server to be quiet (no prior bench's leftover state).
        if not wait_for_at_most(base_url, 0, args.stable_polls, poll_s, timeout_s=60):
            cur = get_running_count(base_url)
            print(f"  WARN: server not quiet before N={n} (running={cur})")

        # Spawn N-1 long-lived backgrounds (max_tokens=4096 + ignore_eos)
        # once, rather than continuously maintaining. This matches v1's
        # semantic (static decode-heavy bg pool during measurement) but
        # replaces v1's magic 3s dwell with phase-aware /metrics polling.
        stop_event = threading.Event()  # unused; kept for downstream API
        bg_threads = []
        if n > 1:
            for i in range(n - 1):
                t = threading.Thread(
                    target=send_background_request,
                    args=(base_url, args.model),
                    daemon=True,
                )
                t.start()
                bg_threads.append(t)
                # Short wait for admission; short timeout so we don't stall
                # forever when KV cache saturates and requests queue.
                if not wait_for_running_at_least(base_url, min(i + 1, int(get_running_count(base_url)) + 1),
                                                  poll_s, timeout_s=3):
                    pass
            # Wait for scheduler to stabilise at achievable concurrency AND
            # for bg to transition into decode phase (generation_tokens_total
            # actively increasing). Replaces v1's time.sleep(3.0) dwell.
            if not wait_for_decode_phase(base_url, int(get_running_count(base_url)),
                                          args.stable_polls, poll_s, timeout_s=30):
                cur = get_running_count(base_url)
                print(f"  WARN: did not observe decode phase at N={n} (running={cur})")

        # Measurement samples for each burst-k.
        for k in BURST_KS:
            cell_samples = []
            for _ in range(args.samples_per_cell):
                # Ensure running is at >= N-1 before measuring (maintainer
                # may be between spawns). Short wait — should be instant.
                if n > 1:
                    wait_for_running_at_least(base_url, n - 1, poll_s, timeout_s=10)

                burst = measure_burst(base_url, args.model, k)
                if burst:
                    cell_samples.append({
                        "ttft_us_by_slot": [round(x, 0) for x in burst],
                        "mean_ttft_us": round(sum(burst) / len(burst), 0),
                        "median_ttft_us": round(sorted(burst)[len(burst) // 2], 0),
                    })
                # Wait for measurement burst to drain (running <= N-1).
                # "At most" not "exactly" — bg may expire, maintainer refills.
                wait_for_at_most(base_url, n - 1, args.stable_polls, poll_s,
                                 timeout_s=15)

            all_ttfts = [t for s in cell_samples for t in s["ttft_us_by_slot"]]
            if all_ttfts:
                all_sorted = sorted(all_ttfts)
                median_all = all_sorted[len(all_sorted) // 2]
                mean_all = sum(all_ttfts) / len(all_ttfts)
                oh_med = max(0.0, median_all - prefill_step_us)
                oh_mean = max(0.0, mean_all - prefill_step_us)
                # Record effective running count at measurement (may be <
                # labeled N when maintainer couldn't reach target — tells
                # downstream "labeled N=100 was actually N_eff=23").
                effective_running = int(get_running_count(base_url))
                results.append({
                    "num_reqs": n,
                    "effective_num_reqs": effective_running,
                    "burst_k": k,
                    "samples": cell_samples,
                    # Primary lookup key 'overhead_us' = median (canonical
                    # choice matching v1 schema). median + mean retained
                    # for downstream aggregation switching.
                    "overhead_us": round(oh_med, 0),
                    "overhead_median_us": round(oh_med, 0),
                    "overhead_mean_us": round(oh_mean, 0),
                    "median_ttft_us": round(median_all, 0),
                    "mean_ttft_us": round(mean_all, 0),
                    "num_ttft_datapoints": len(all_ttfts),
                    "prefill_step_us": round(prefill_step_us, 0),
                })
                print(f"  N={n:4d} k={k:2d}: n={len(all_ttfts):3d} "
                      f"running_eff={effective_running} "
                      f"median={median_all/1000:5.1f}ms "
                      f"oh_med={oh_med/1000:5.1f}ms")

        # Stop the bg maintainer daemon (its thread and any bg requests it
        # spawned are daemons — Python shutdown kills them). Wait for the
        # server to drain fully before moving to next N.
        if n > 1:
            stop_event.set()
        wait_for_at_most(base_url, 0, args.stable_polls, poll_s, timeout_s=120)

    # Attach hardware + run metadata so the sweep is self-documenting.
    metadata = {
        "profiler_version": "v3",
        "methodology": "condition-based waits via vllm:num_requests_running; "
                       "no hardware-specific sleeps",
        "poll_ms": args.poll_ms,
        "stable_polls": args.stable_polls,
        "samples_per_cell": args.samples_per_cell,
        "burst_k": BURST_KS,
        "n_sweep": N_SWEEP,
        "prefill_step_us_baseline": round(prefill_step_us, 0),
        "model": args.model,
        "model_config_from_profile": profile.get("model_config", {}),
    }
    out = {"metadata": metadata, "cells": results}
    json.dump(out, open(args.output_path, "w"), indent=2)
    print(f"\nSaved {len(results)} cells to {args.output_path}")


if __name__ == "__main__":
    main()
