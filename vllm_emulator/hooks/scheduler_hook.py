"""Scheduler-level emulator hook: arrival-delay model.

When real vLLM receives a new request, per-request IPC setup (ZMQ socket,
detokenizer spin-up) adds latency before the client sees the first token.
The executor hook's step latency can't model this cleanly — adding it to
step latency delays all in-flight decode tokens and inflates TPOT.

This hook models it at the correct layer: the scheduler. New arrivals
are held in a pending list with an admission time = now + IPC_overhead(N).
The scheduler's schedule() is wrapped to drain ready arrivals (admission
time passed) into the real waiting queue before scheduling the next step.

Enable: VLLM_EMULATOR_SCHEDULER_HOOK=1 AND a profile with
sched_overhead_table (from tools/profile_ipc_overhead.py).
"""

from __future__ import annotations

import bisect
import json
import os
import random
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.request import Request


SCHEDULER_HOOK_ENV = "VLLM_EMULATOR_SCHEDULER_HOOK"
PROFILE_PACK_ENV = "VLLM_EMULATOR_PROFILE_PACK"
IPC_OVERHEAD_AGG_ENV = "VLLM_IPC_OVERHEAD_AGG"  # "median"|"mean"|"sample"|"2d-burst"


def _load_overhead_table() -> list[dict[str, Any]]:
    """Load sched_overhead_table from the profile pack.

    Aggregation via VLLM_IPC_OVERHEAD_AGG:
    - "median" (default): field `overhead_us` from the v1 (1D) table.
    - "mean": field `overhead_mean_us` from v1; falls back to `overhead_us`.
    - "sample": at each lookup, draw a random value from `raw_ttft_samples_us`
      (minus per-N prefill_step_us). Uses v1 table (k=1 samples only).
    - "2d-burst": load the v2 (2D) table `sched_overhead_table_v2` which has
      per-(N, k_burst) cells. Hook queries with (N_conc, k_burst) where k is
      the count of arrivals currently pending admission. Captures the ~3×
      overhead increase when real vLLM receives bursts.

    Returns a table list; for 1D modes each entry is
        {num_reqs, overhead_us, [samples_overhead_us]}
    For 2d-burst mode each entry is
        {num_reqs, burst_k, overhead_us, overhead_mean_us}
    """
    path = os.environ.get(PROFILE_PACK_ENV, "")
    if not path or not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            pack = json.load(f)
        agg = os.environ.get(IPC_OVERHEAD_AGG_ENV, "median").lower()
        if agg not in ("median", "mean", "sample", "2d-burst"):
            raise ValueError(
                f"{IPC_OVERHEAD_AGG_ENV} must be one of "
                f"median|mean|sample|2d-burst, got {agg!r}")

        if agg == "2d-burst":
            raw_v2 = pack.get("sched_overhead_table_v2") or []
            if not raw_v2:
                raise ValueError(
                    "2d-burst mode requires sched_overhead_table_v2 in profile")
            table = []
            for e in raw_v2:
                # Use mean overhead as default scalar for 2D cells (measured
                # across k requests within one burst sample).
                overhead_us = e.get("overhead_mean_us", e.get("overhead_median_us", 0))
                table.append({
                    "num_reqs": e.get("num_reqs", 0),
                    "burst_k": e.get("burst_k", 1),
                    "overhead_us": overhead_us,
                })
            return sorted(table, key=lambda e: (e["num_reqs"], e["burst_k"]))

        # 1D modes (median / mean / sample) use the v1 table.
        raw = pack.get("sched_overhead_table") or []
        table = []
        for e in raw:
            if agg == "mean" and "overhead_mean_us" in e:
                overhead_us = e["overhead_mean_us"]
            else:
                overhead_us = e.get("overhead_us", 0)
            entry = {
                "num_reqs": e.get("num_reqs", 0),
                "overhead_us": overhead_us,
            }
            if agg == "sample":
                ttft_samples = e.get("raw_ttft_samples_us") or []
                prefill_us = e.get("prefill_step_us", 0)
                if ttft_samples:
                    entry["samples_overhead_us"] = [
                        max(0.0, s - prefill_us) for s in ttft_samples
                    ]
            table.append(entry)
        return sorted(table, key=lambda e: e.get("num_reqs", 0))
    except Exception as exc:
        print(f"[SchedulerHook] failed to load overhead table: {exc}")
        return []


def _lookup_overhead_us_2d(
    table_2d: list[dict[str, Any]],
    num_reqs: int,
    burst_k: int,
) -> float:
    """Bilinear interpolation over the 2D (N, k) overhead grid.

    `table_2d` is a list of {num_reqs, burst_k, overhead_us} cells sorted
    by (num_reqs, burst_k). Clamps at grid edges; otherwise bilinearly
    interpolates the four neighbouring cells.
    """
    if not table_2d:
        return 0.0
    ns = sorted({e["num_reqs"] for e in table_2d})
    ks = sorted({e["burst_k"] for e in table_2d})
    if not ns or not ks:
        return 0.0

    # Clamp.
    n = max(ns[0], min(num_reqs, ns[-1]))
    k = max(ks[0], min(burst_k, ks[-1]))

    # Pick bounding (n_lo, n_hi) and (k_lo, k_hi).
    n_lo = max(x for x in ns if x <= n)
    n_hi = min(x for x in ns if x >= n)
    k_lo = max(x for x in ks if x <= k)
    k_hi = min(x for x in ks if x >= k)

    cell = {(e["num_reqs"], e["burst_k"]): float(e["overhead_us"]) for e in table_2d}
    v_ll = cell.get((n_lo, k_lo), 0.0)
    v_lh = cell.get((n_lo, k_hi), v_ll)
    v_hl = cell.get((n_hi, k_lo), v_ll)
    v_hh = cell.get((n_hi, k_hi), v_ll)

    n_ratio = 0.0 if n_hi == n_lo else (n - n_lo) / (n_hi - n_lo)
    k_ratio = 0.0 if k_hi == k_lo else (k - k_lo) / (k_hi - k_lo)
    v_lo = v_ll + k_ratio * (v_lh - v_ll)
    v_hi = v_hl + k_ratio * (v_hh - v_hl)
    return v_lo + n_ratio * (v_hi - v_lo)


def _lookup_overhead_us(
    table: list[dict[str, Any]],
    num_reqs: int,
    rng: random.Random | None = None,
) -> float:
    """Per-concurrency overhead lookup with optional per-call sampling.

    If the neighbouring table entry has `samples_overhead_us` (built by
    `_load_overhead_table` in 'sample' agg mode) AND an rng is supplied,
    draw a random overhead from that per-N sample pool. Otherwise return
    the interpolated scalar `overhead_us` value.
    """
    if not table:
        return 0.0
    nums = [e.get("num_reqs", 0) for e in table]
    overheads = [float(e.get("overhead_us", 0)) for e in table]

    # Find neighbour entry to pull samples from (clamp at edges, pick
    # nearest on interior — no cross-entry sample mixing).
    if num_reqs <= nums[0]:
        neighbour = table[0]
        scalar = overheads[0]
    elif num_reqs >= nums[-1]:
        neighbour = table[-1]
        scalar = overheads[-1]
    else:
        i = bisect.bisect_left(nums, num_reqs)
        if i < len(nums) and nums[i] == num_reqs:
            neighbour = table[i]
            scalar = overheads[i]
        else:
            n_lo, n_hi = nums[i - 1], nums[i]
            o_lo, o_hi = overheads[i - 1], overheads[i]
            ratio = (num_reqs - n_lo) / (n_hi - n_lo)
            scalar = o_lo + ratio * (o_hi - o_lo)
            # Pool samples from the CLOSER neighbour for 'sample' mode.
            neighbour = table[i - 1] if (num_reqs - n_lo) <= (n_hi - num_reqs) else table[i]

    if rng is not None:
        samples = neighbour.get("samples_overhead_us")
        if samples:
            return float(rng.choice(samples))
    return scalar


def install_arrival_delay(scheduler: "Scheduler") -> bool:
    """Patch the scheduler to delay new-request admission by IPC overhead.

    Returns True if the hook was installed, False otherwise (no env var or
    no overhead table in the profile pack).
    """
    if os.environ.get(SCHEDULER_HOOK_ENV, "").lower() not in ("1", "true", "yes"):
        return False

    table = _load_overhead_table()
    if not table:
        print(f"[SchedulerHook] {SCHEDULER_HOOK_ENV}=1 but profile pack has no "
              f"sched_overhead_table — no-op.")
        return False

    # Pending arrivals: list of (admission_time_s, Request).
    scheduler._emu_pending_arrivals: list[tuple[float, Any]] = []  # type: ignore[attr-defined]
    scheduler._emu_overhead_table = table  # type: ignore[attr-defined]
    agg = os.environ.get(IPC_OVERHEAD_AGG_ENV, "median").lower()
    scheduler._emu_overhead_agg = agg  # type: ignore[attr-defined]
    # Deterministic RNG seeded by env so runs are reproducible.
    scheduler._emu_overhead_rng = (  # type: ignore[attr-defined]
        random.Random(42) if agg == "sample" else None
    )

    orig_add_request = scheduler.add_request.__func__
    orig_schedule = scheduler.schedule.__func__
    orig_get_num_unfinished = scheduler.get_num_unfinished_requests.__func__

    def _patched_add_request(self, request: "Request") -> None:
        # Re-adds for same request_id (e.g. streaming updates) bypass the
        # arrival-delay path — only brand-new requests pay IPC setup.
        if request.request_id in self.requests:
            orig_add_request(self, request)
            return

        # Concurrency = current running + waiting + already-pending arrivals.
        # This matches what real vLLM's engine loop observes at the moment
        # the new request lands in the EngineCore.
        conc = (len(self.running)
                + len(self.waiting)
                + len(self.skipped_waiting)
                + len(self._emu_pending_arrivals))
        if self._emu_overhead_agg == "2d-burst":
            # burst_k = pending arrivals already waiting PLUS this new one.
            # Models the clustered-burst IPC contention effect that v1 sweep
            # could not measure (it only injected 1 new request at a time).
            k_burst = len(self._emu_pending_arrivals) + 1
            overhead_us = _lookup_overhead_us_2d(
                self._emu_overhead_table, max(conc, 1), max(k_burst, 1),
            )
        else:
            overhead_us = _lookup_overhead_us(
                self._emu_overhead_table, max(conc, 1),
                rng=self._emu_overhead_rng,
            )
        admission_time = time.perf_counter() + overhead_us / 1e6
        self._emu_pending_arrivals.append((admission_time, request))

    def _drain_pending(self) -> None:
        """Move requests whose admission time has passed into waiting."""
        if not self._emu_pending_arrivals:
            return
        now = time.perf_counter()
        remaining = []
        for admission_time, request in self._emu_pending_arrivals:
            if admission_time <= now:
                orig_add_request(self, request)
            else:
                remaining.append((admission_time, request))
        self._emu_pending_arrivals = remaining

    def _patched_schedule(self, *args, **kwargs):
        _drain_pending(self)
        return orig_schedule(self, *args, **kwargs)

    def _patched_get_num_unfinished(self) -> int:
        # Include pending arrivals so the engine loop keeps calling schedule()
        # until they reach admission time and get drained.
        return (orig_get_num_unfinished(self)
                + len(self._emu_pending_arrivals))

    # Bind as methods on the instance.
    import types
    scheduler.add_request = types.MethodType(_patched_add_request, scheduler)
    scheduler.schedule = types.MethodType(_patched_schedule, scheduler)
    scheduler.get_num_unfinished_requests = types.MethodType(
        _patched_get_num_unfinished, scheduler)


    agg = os.environ.get(IPC_OVERHEAD_AGG_ENV, "median").lower()
    print(f"[SchedulerHook] arrival-delay installed "
          f"({len(table)} overhead-table entries, agg={agg}, "
          f"min={table[0].get('overhead_us', 0)/1000:.1f}ms, "
          f"max={table[-1].get('overhead_us', 0)/1000:.1f}ms)")
    return True
