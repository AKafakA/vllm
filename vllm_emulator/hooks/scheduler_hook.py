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
IPC_OVERHEAD_AGG_ENV = "VLLM_IPC_OVERHEAD_AGG"  # "median" | "mean" | "sample"


def _load_overhead_table() -> list[dict[str, Any]]:
    """Load sched_overhead_table from the profile pack.

    Aggregation via VLLM_IPC_OVERHEAD_AGG:
    - "median" (default): field `overhead_us`.
    - "mean": field `overhead_mean_us` when present; falls back to
      `overhead_us` for older tables.
    - "sample": at each lookup, draw a random value from
      `raw_ttft_samples_us` (minus per-N prefill_step_us). Requires
      the profile sweep to have retained raw samples (profiler v2).
      This models the right-skewed IPC distribution per-request.

    Raw samples are retained in the table regardless, so the same
    profile pack supports all three aggregations via env var.
    """
    path = os.environ.get(PROFILE_PACK_ENV, "")
    if not path or not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            pack = json.load(f)
        raw = pack.get("sched_overhead_table") or []
        agg = os.environ.get(IPC_OVERHEAD_AGG_ENV, "median").lower()
        if agg not in ("median", "mean", "sample"):
            raise ValueError(
                f"{IPC_OVERHEAD_AGG_ENV} must be 'median', 'mean', or 'sample', "
                f"got {agg!r}")
        # Normalise per-N entry. Keep raw sample list in 'samples_overhead_us'
        # for 'sample' mode (TTFT samples minus that N's prefill baseline).
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
            # Precompute per-N raw overhead samples for 'sample' mode.
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
