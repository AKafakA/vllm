#!/usr/bin/env python3
"""Analyze per-request latency from bench serve JSON output.

Compares real vs emulator at the request level to understand
where the TPOT gap comes from when step cycles match.

Usage: python analyze_request_latency.py <real.json> <emu.json>
"""
import json
import statistics
import sys


def main():
    real = json.load(open(sys.argv[1]))
    emu = json.load(open(sys.argv[2]))

    print("=== Request-level latency analysis ===")
    print(f"Real: {real.get('completed', '?')} completed, Emu: {emu.get('completed', '?')} completed")

    # Key metrics
    metrics = [
        ("mean_ttft_ms", "Mean TTFT"),
        ("mean_tpot_ms", "Mean TPOT"),
        ("mean_e2el_ms", "Mean E2E"),
        ("mean_itl_ms", "Mean ITL"),
        ("output_throughput", "Output tok/s"),
        ("request_throughput", "Request/s"),
    ]
    print(f"\n{'Metric':>20} {'Real':>10} {'Emu':>10} {'Gap':>10} {'Gap%':>8}")
    print("-" * 62)
    for key, label in metrics:
        rv = real.get(key, 0)
        ev = emu.get(key, 0)
        if rv == 0 and ev == 0:
            continue
        gap = ev - rv
        gap_pct = gap / rv * 100 if rv else 0
        print(f"{label:>20} {rv:>10.2f} {ev:>10.2f} {gap:>10.2f} {gap_pct:>7.1f}%")

    # Compute derived metrics
    r_out_len = real.get("total_output_tokens", 0) / max(real.get("completed", 1), 1)
    e_out_len = emu.get("total_output_tokens", 0) / max(emu.get("completed", 1), 1)

    print(f"\n=== Derived analysis ===")
    print(f"Avg output tokens: real={r_out_len:.0f}, emu={e_out_len:.0f}")

    r_tpot = real.get("mean_tpot_ms", 0)
    e_tpot = emu.get("mean_tpot_ms", 0)
    r_ttft = real.get("mean_ttft_ms", 0)
    e_ttft = emu.get("mean_ttft_ms", 0)

    # E2E = TTFT + (output_len - 1) * TPOT
    r_e2e_calc = r_ttft + (r_out_len - 1) * r_tpot
    e_e2e_calc = e_ttft + (e_out_len - 1) * e_tpot

    print(f"E2E (calculated): real={r_e2e_calc:.1f}ms, emu={e_e2e_calc:.1f}ms")

    # TPOT contribution to E2E gap
    tpot_contribution = (r_out_len - 1) * (e_tpot - r_tpot)
    ttft_contribution = e_ttft - r_ttft
    print(f"\nE2E gap breakdown:")
    print(f"  TTFT contribution: {ttft_contribution:+.1f}ms ({ttft_contribution/(r_e2e_calc)*100:+.1f}% of E2E)")
    print(f"  TPOT contribution: {tpot_contribution:+.1f}ms ({tpot_contribution/(r_e2e_calc)*100:+.1f}% of E2E)")

    # Per-token gap in absolute terms
    print(f"\nPer-token timing:")
    print(f"  Real TPOT: {r_tpot:.2f}ms")
    print(f"  Emu  TPOT: {e_tpot:.2f}ms")
    print(f"  Gap: {e_tpot - r_tpot:.2f}ms per token ({(e_tpot-r_tpot)/r_tpot*100:.1f}%)")

    # Request throughput vs step throughput comparison
    r_req_tp = real.get("request_throughput", 0)
    e_req_tp = emu.get("request_throughput", 0)
    r_tok_tp = real.get("output_throughput", 0)
    e_tok_tp = emu.get("output_throughput", 0)

    print(f"\n=== Throughput vs Latency paradox ===")
    print(f"Token throughput matches: {r_tok_tp:.1f} vs {e_tok_tp:.1f} ({(e_tok_tp-r_tok_tp)/r_tok_tp*100:+.1f}%)")
    print(f"But per-token latency diverges: {r_tpot:.2f} vs {e_tpot:.2f} ({(e_tpot-r_tpot)/r_tpot*100:+.1f}%)")

    if r_tpot > 0 and r_tok_tp > 0:
        r_avg_conc = r_tok_tp * r_tpot / 1000  # tok/s * s/tok = concurrent tokens
        e_avg_conc = e_tok_tp * e_tpot / 1000
        print(f"Implied avg concurrent tokens: real={r_avg_conc:.1f}, emu={e_avg_conc:.1f}")
        print(f"\nThis means: throughput is the same but the emulator processes")
        print(f"tokens with less queuing delay per request. The emulator's")
        print(f"engine loop runs faster between steps, reducing per-request latency.")
        print(f"Missing component: real GPU has output processing overhead")
        print(f"(IPC, detokenization, ZMQ) that adds ~{r_tpot - e_tpot:.1f}ms per token.")


if __name__ == "__main__":
    main()
