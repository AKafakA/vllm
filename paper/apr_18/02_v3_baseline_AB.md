# v3 Baseline A/B (slot S0) — Status

Populated by the cron-driven auto-run against `results/RTX-8000-adaptive-v3/serving-full.json`.

## Status

_As of ~13:45 UTC Apr 18:_ Adaptive profiling v3 still in Round 5 (r=1/14 as of 13:35 BST). Expected completion ~14:30 BST. Auto A/B via cron `bebd21f5` fires next at :59 and will detect `.all_done`, rsync the profile, verify record/bucket counts (≥150k records, ≥500 buckets), then launch `validate_wiring_fix.sh` pointed at the v3 profile.

Wiring-fix harness runs two passes:
- `results/RTX-8000-v3-nosurr/` (surrogate off)
- `results/RTX-8000-v3-withsurr/` (surrogate on)

## Profile-choice fallback rule (per plan §"Profile choice + fallback rule")

After the auto A/B finishes:
1. Compare **withsurr** TPOT and TTFT at r=2/4/8/16 against the **archived 108k baseline** `results/RTX-8000-wiring-fix-withsurr/` locked earlier today:

| Rate | Archive TPOT | Archive TTFT |
|---|---|---|
|  2 | −0.6% | −32.5% |
|  4 | −1.6% | −30.7% |
|  8 | −6.0% | −28.5% |
| 16 | −5.6% | −25.0% |
| 32 | +4.9% | +1.9%  |

2. If v3 withsurr matches or improves on archive at every rate (each metric within ±2pp of the archive row) → **use v3 `serving-full.json`** as the baseline profile for all F1–F5 ablations.
3. If v3 withsurr is **worse** on any metric by more than 2pp → **use the archived `serving-dense.json`** as the baseline profile. Record v3's regression here + add "debug v3 profile regression" as tomorrow's top-priority task in `09_handoff_plan.md`.

## A/B results (to be filled)

Pass A (v3 + nosurr):

| Rate | TPOT% | TTFT% | E2E% | tok/s% | verdict |
|---|---|---|---|---|---|
| tbd | tbd | tbd | tbd | tbd | — |

Pass B (v3 + withsurr):

| Rate | TPOT% | TTFT% | E2E% | tok/s% | verdict |
|---|---|---|---|---|---|
| tbd | tbd | tbd | tbd | tbd | — |

## Selection decision

_Filled after A/B._

**Profile chosen for feature ablations:** _TBD (v3 or archive)_.

**Rationale:** _TBD._

**Tomorrow's v3-debug priority (only populated if archive chosen):** _n/a / TBD._
