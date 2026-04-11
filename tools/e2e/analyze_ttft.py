"""Analyze TTFT components from diagnostic traces."""
import json, statistics, os
from collections import defaultdict

ONLINE = "/workspace/eval_results/RTX-3060-12GB/online"
RESULT_DIR = "/workspace/eval_results/RTX-3060-12GB"

# Compare client TTFT
for rate in [4, 8]:
    print(f"\n=== R={rate} Client TTFT ===")
    for label, fname in [("Real", f"ttft_diag_real_r{rate}.json"),
                         ("Emu", f"ttft_diag_emu_r{rate}.json")]:
        try:
            d = json.load(open(f"{ONLINE}/{fname}"))
            print(f"  {label}: TTFT mean={d['mean_ttft_ms']:.1f}ms "
                  f"median={d['median_ttft_ms']:.1f}ms "
                  f"p99={d['p99_ttft_ms']:.1f}ms "
                  f"std={d['std_ttft_ms']:.1f}ms "
                  f"TPOT={d['mean_tpot_ms']:.1f}ms")
        except Exception as e:
            print(f"  {label}: {e}")
    try:
        r = json.load(open(f"{ONLINE}/ttft_diag_real_r{rate}.json"))
        e = json.load(open(f"{ONLINE}/ttft_diag_emu_r{rate}.json"))
        ttft_err = (e['mean_ttft_ms'] - r['mean_ttft_ms']) / r['mean_ttft_ms'] * 100
        tpot_err = (e['mean_tpot_ms'] - r['mean_tpot_ms']) / r['mean_tpot_ms'] * 100
        print(f"  Error: TTFT={ttft_err:+.1f}% TPOT={tpot_err:+.1f}%")
    except:
        pass

# Analyze prefill steps from traces
print(f"\n=== Prefill Step Analysis ===")
for label, trace_file in [
    ("Real", f"{RESULT_DIR}/diag_ttft_real.jsonl"),
    ("Emu", f"{RESULT_DIR}/diag_ttft_emu.jsonl"),
]:
    try:
        recs = []
        in_bench = False
        for line in open(trace_file):
            r = json.loads(line)
            if r.get("__marker__") == "benchmark_start":
                in_bench = True
                continue
            if in_bench and "step_cycle_us" in r:
                recs.append(r)

        prefill = [r for r in recs if r.get("num_new_reqs", 0) > 0]
        decode = [r for r in recs if r.get("num_new_reqs", 0) == 0]

        print(f"\n  {label}:")
        print(f"    Total steps: {len(recs)}, prefill: {len(prefill)}, decode: {len(decode)}")

        if prefill:
            pf_cycles = [r["step_cycle_us"]/1000 for r in prefill]
            pf_tts = [r["total_tokens"] for r in prefill]
            pf_concs = [r.get("num_decode_seqs", 0) + r.get("num_new_reqs", 0) for r in prefill]
            pf_new = [r.get("num_new_reqs", 0) for r in prefill]
            print(f"    Prefill step_cycle: median={statistics.median(pf_cycles):.1f}ms "
                  f"mean={statistics.mean(pf_cycles):.1f}ms "
                  f"p90={sorted(pf_cycles)[int(len(pf_cycles)*0.9)]:.1f}ms")
            print(f"    Prefill total_tokens: median={statistics.median(pf_tts):.0f} "
                  f"mean={statistics.mean(pf_tts):.0f} "
                  f"max={max(pf_tts)}")
            print(f"    Prefill concurrency: median={statistics.median(pf_concs):.0f} "
                  f"mean={statistics.mean(pf_concs):.1f}")
            print(f"    Prefill new_reqs: median={statistics.median(pf_new):.0f} "
                  f"mean={statistics.mean(pf_new):.1f}")

            # Breakdown by concurrency
            print(f"    Prefill step_cycle by concurrency:")
            by_conc = defaultdict(list)
            for r in prefill:
                c = r.get("num_decode_seqs", 0) + r.get("num_new_reqs", 0)
                by_conc[c].append(r["step_cycle_us"]/1000)
            for c in sorted(by_conc):
                if c in [1, 5, 10, 15, 20, 25, 30]:
                    nearby = []
                    for cc in range(max(1, c-2), c+3):
                        if cc in by_conc:
                            nearby.extend(by_conc[cc])
                    if nearby:
                        print(f"      conc~{c:>3}: n={len(nearby):>4} median={statistics.median(nearby):.1f}ms")

            # Breakdown if detailed timing available
            pf_detail = [r for r in prefill if "exec_ms" in r]
            if pf_detail:
                print(f"    Prefill detailed timing (n={len(pf_detail)}):")
                for field in ["sched_ms", "exec_ms", "sample_ms", "wait_ms", "update_ms"]:
                    vals = [r.get(field, 0) for r in pf_detail if field in r]
                    if vals:
                        print(f"      {field:>12}: median={statistics.median(vals):.2f}ms "
                              f"mean={statistics.mean(vals):.2f}ms")

        if decode:
            dc_cycles = [r["step_cycle_us"]/1000 for r in decode]
            print(f"    Decode step_cycle: median={statistics.median(dc_cycles):.1f}ms "
                  f"mean={statistics.mean(dc_cycles):.1f}ms")

    except Exception as e:
        print(f"  {label}: {e}")

# Step count comparison
print(f"\n=== Step Count Comparison ===")
for label, trace_file in [
    ("Real", f"{RESULT_DIR}/diag_ttft_real.jsonl"),
    ("Emu", f"{RESULT_DIR}/diag_ttft_emu.jsonl"),
]:
    try:
        total = 0
        pf = 0
        in_bench = False
        for line in open(trace_file):
            r = json.loads(line)
            if r.get("__marker__") == "benchmark_start":
                in_bench = True
                continue
            if in_bench and "step_cycle_us" in r:
                total += 1
                if r.get("num_new_reqs", 0) > 0:
                    pf += 1
        print(f"  {label}: total={total}, prefill={pf}, decode={total-pf}")
    except Exception as e:
        print(f"  {label}: {e}")
