#!/usr/bin/env python3
"""Test sleep accuracy at different durations to understand offline throughput gap."""
import time
import statistics

def test_sleep(target_us, n=1000):
    """Test time.sleep() accuracy at a given duration."""
    errors = []
    for _ in range(n):
        t0 = time.perf_counter()
        time.sleep(target_us / 1e6)
        actual_us = (time.perf_counter() - t0) * 1e6
        errors.append(actual_us - target_us)

    med = statistics.median(errors)
    mean = statistics.mean(errors)
    p99 = sorted(errors)[int(0.99 * len(errors))]
    return med, mean, p99

def test_busywait(target_us, n=1000):
    """Test busy-wait accuracy at a given duration."""
    errors = []
    for _ in range(n):
        t0 = time.perf_counter()
        end = t0 + target_us / 1e6
        while time.perf_counter() < end:
            pass
        actual_us = (time.perf_counter() - t0) * 1e6
        errors.append(actual_us - target_us)

    med = statistics.median(errors)
    mean = statistics.mean(errors)
    p99 = sorted(errors)[int(0.99 * len(errors))]
    return med, mean, p99

def test_hybrid(target_us, n=1000, busywait_threshold_us=1000):
    """Hybrid: sleep for most of the duration, then busy-wait for the last bit."""
    errors = []
    for _ in range(n):
        t0 = time.perf_counter()
        if target_us > busywait_threshold_us * 2:
            # Sleep for most of it, busywait for the last threshold
            time.sleep((target_us - busywait_threshold_us) / 1e6)
        end = t0 + target_us / 1e6
        while time.perf_counter() < end:
            pass
        actual_us = (time.perf_counter() - t0) * 1e6
        errors.append(actual_us - target_us)

    med = statistics.median(errors)
    mean = statistics.mean(errors)
    p99 = sorted(errors)[int(0.99 * len(errors))]
    return med, mean, p99

print("Sleep accuracy test (1000 iterations each)")
print(f"{'Target (us)':>12} {'Method':>10} {'Median err':>12} {'Mean err':>12} {'P99 err':>12}")
print("-" * 62)

for target_us in [100, 500, 1000, 5000, 10000, 13000, 20000, 50000]:
    med, mean, p99 = test_sleep(target_us)
    print(f"{target_us:>12} {'sleep':>10} {med:>12.1f} {mean:>12.1f} {p99:>12.1f}")

    med, mean, p99 = test_busywait(target_us, n=200)
    print(f"{target_us:>12} {'busywait':>10} {med:>12.1f} {mean:>12.1f} {p99:>12.1f}")

    med, mean, p99 = test_hybrid(target_us, n=200)
    print(f"{target_us:>12} {'hybrid':>10} {med:>12.1f} {mean:>12.1f} {p99:>12.1f}")
    print()

print("Key insight: if sleep overhead at 13000us (typical decode step) is >500us,")
print("that explains the 17% offline throughput error (500us * 12800 steps = 6.4s extra)")
