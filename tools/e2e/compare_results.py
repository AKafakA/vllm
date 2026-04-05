"""Compare two bench serve result JSON files with full E2E metrics."""
import json
import sys


def main():
    if len(sys.argv) < 3:
        print("Usage: compare_results.py <real.json> <emu.json>")
        sys.exit(1)

    r = json.load(open(sys.argv[1]))
    e = json.load(open(sys.argv[2]))

    # Per-request E2E latency estimate
    output_len = r.get("total_output_tokens", 0) / max(r.get("completed", 1), 1)
    r_e2e = r.get("mean_ttft_ms", 0) + (output_len - 1) * r.get("mean_tpot_ms", 0)
    e_e2e = e.get("mean_ttft_ms", 0) + (output_len - 1) * e.get("mean_tpot_ms", 0)
    e2e_err = (e_e2e - r_e2e) / r_e2e * 100 if r_e2e else 0

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
    ]
    for key, label in metrics:
        rv, ev = r.get(key, 0), e.get(key, 0)
        err, status = fmt_err(rv, ev)
        print(f"  {label:20s} real={rv:8.1f}  emu={ev:8.1f}  err={err:+6.1f}% {status}")

    print(f"\n=== E2E Latency (estimated, {output_len:.0f} output tokens) ===")
    err_s, status_s = fmt_err(r_e2e, e_e2e)
    print(f"  {'Mean E2E':20s} real={r_e2e:8.1f}  emu={e_e2e:8.1f}  err={err_s:+6.1f}% {status_s}")

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
        print(f"  {label:20s} real={rv:8.2f}  emu={ev:8.2f}  err={err:+6.1f}% {status}")


if __name__ == "__main__":
    main()
