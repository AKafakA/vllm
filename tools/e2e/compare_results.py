"""Compare two bench serve result JSON files with full E2E metrics."""
import json
import sys


def main():
    if len(sys.argv) < 3:
        print("Usage: compare_results.py <real.json> <emu.json>")
        sys.exit(1)

    r = json.load(open(sys.argv[1]))
    e = json.load(open(sys.argv[2]))

    def fmt_err(rv, ev):
        err = (ev - rv) / rv * 100 if rv else 0
        status = "pass" if abs(err) <= 5 else ("~" if abs(err) <= 7 else "FAIL")
        return err, status

    print("=== Latency Metrics ===")
    metrics = [
        ("mean_ttft_ms", "Mean TTFT"),
        ("median_ttft_ms", "Median TTFT"),
        ("p99_ttft_ms", "P99 TTFT"),
        ("mean_tpot_ms", "Mean TPOT"),
        ("median_tpot_ms", "Median TPOT"),
        ("p99_tpot_ms", "P99 TPOT"),
        ("mean_e2el_ms", "Mean E2E (measured)"),
        ("median_e2el_ms", "Median E2E (measured)"),
        ("p99_e2el_ms", "P99 E2E (measured)"),
    ]
    for key, label in metrics:
        rv, ev = r.get(key, 0), e.get(key, 0)
        if rv == 0 and ev == 0:
            continue  # skip if not available
        err, status = fmt_err(rv, ev)
        print(f"  {label:25s} real={rv:8.1f}  emu={ev:8.1f}  err={err:+6.1f}% {status}")

    # Fallback: estimate E2E if measured not available
    if "mean_e2el_ms" not in r or "mean_e2el_ms" not in e:
        output_len = r.get("total_output_tokens", 0) / max(r.get("completed", 1), 1)
        r_e2e = r.get("mean_ttft_ms", 0) + (output_len - 1) * r.get("mean_tpot_ms", 0)
        e_e2e = e.get("mean_ttft_ms", 0) + (output_len - 1) * e.get("mean_tpot_ms", 0)
        err_s, status_s = fmt_err(r_e2e, e_e2e)
        print(f"\n=== E2E Latency (ESTIMATED, {output_len:.0f} output tokens) ===")
        print(f"  {'Mean E2E':25s} real={r_e2e:8.1f}  emu={e_e2e:8.1f}  err={err_s:+6.1f}% {status_s}")

    print("\n=== Throughput Metrics ===")
    tp_metrics = [
        ("output_throughput", "Output tok/s"),
        ("total_token_throughput", "Total tok/s"),
        ("request_throughput", "Request/s"),
        ("duration", "Duration (s)"),
    ]
    for key, label in tp_metrics:
        rv, ev = r.get(key, 0), e.get(key, 0)
        err, status = fmt_err(rv, ev)
        print(f"  {label:25s} real={rv:8.2f}  emu={ev:8.2f}  err={err:+6.1f}% {status}")

    # Flag if throughput is arrival-rate limited
    rate = r.get("request_rate", "inf")
    r_req_tp = r.get("request_throughput", 0)
    if rate != "inf" and r_req_tp > 0:
        expected_tp = float(rate)
        if abs(r_req_tp - expected_tp) / expected_tp < 0.05:
            print(f"\n  NOTE: Throughput ≈ arrival rate ({rate} req/s) — server NOT saturated.")
            print(f"  Throughput comparison is meaningless at this rate.")


if __name__ == "__main__":
    main()
