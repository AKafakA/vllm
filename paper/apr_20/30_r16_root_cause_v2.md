# Apr 20 afternoon — r=16 TTFT root cause (revised)

Prior Apr 19/20 hypotheses for the r=16 TTFT gap (−14 to −16%):
1. IPC burst-scaling — turned out to be k=4-8 effect, but applying it broke TPOT elsewhere.
2. Arrival-delay hook changes batch composition — **refuted today by measurement** (see below).
3. v1 vs v2 k=1 sweep disagreement — **refuted** (v1/v2 agree within noise).

## Today's diagnostic

Ran `chain_r16_batch_diag.sh` at r=16 × 2000p × 2 passes (hook on, hook off). Line-buffered trace survived cleanup. Results in `results/r16-batch-diag-apr20pm/`.

**Batch composition comparison** (`paper/apr_20/01_batch_composition_r16.md`):
- hook_on pure-decode steps: 48.6% (814/1675)
- hook_off pure-decode steps: 48.9% (817/1671)
- Δ = 0.3pp

**Ruling**: hook DOES NOT meaningfully change batch composition. Hypothesis #2 refuted.

**TTFT comparison**:
- hook_on TTFT: 23,818 ms (−14.0% vs real 27,696 ms)
- hook_off TTFT: 24,336 ms (−12.1% vs real 27,696 ms)

Hook-off is 518ms closer to real. So enabling the hook actually makes emu MORE short of real, not less. That's counterintuitive for a hook that adds admission delay — but the effect is ~2%, within noise for r=16 at 2000p.

**Conclusion**: the r=16 gap exists WITH and WITHOUT the hook, with similar magnitude (~12-14%). The hook is a ~2pp bystander, not the cause.

## Root cause: profile coverage at saturation

Inspection of `archive-r2/serving-r2.json` decode_2d_distribution:

| conc bucket | sample count |
|---|---|
| 2   | 48,024 |
| 7   | 54,275 |
| 12  | ~50,000 |
| ... | ... |
| 202 | 33 |
| 207 | 20 |
| 212 | 26 |
| 222 | 68 |
| 227 | 17 |
| 232 | 255 |
| 237 | 10 |
| 242 | 107 |
| 247 | 19 |
| 252 | 132 |
| 257 | 1,173 |

**Max conc in profile: 257** (= vLLM's max_num_seqs cap during profile build).
**Observed max_conc at r=16 validation: 940.**

At r=16 saturation, the oracle queries happen at `scheduled_num_reqs ≤ 256` (clamped by running queue cap, same in emu and real). So oracle lookup hits conc=257 bucket. But:

1. **Sample density collapses** above conc=50. Buckets at conc=207-247 have only 10-255 samples each vs 48k at conc=2. Tail buckets are statistically thin.
2. **(tt, conc) joint coverage** may be sparse at the operating point. At r=16 saturation, typical step has many decode tokens → high tt. If profile's high-conc samples were collected under different tt, oracle's nearest-neighbor falls to a lower-tt, lower-latency bucket → underestimates step time.

**Net effect**: emu thinks each step at high conc is faster than real actually executes it. Queue drains faster in emu. TTFT for later requests is lower. Consistent with the observed −14% gap.

## What this means for the fix

Previous candidates (burst-aware 2D hook, output-side delay hook, architectural admit-at-t0) were all **hook-level fixes**. They can't help because the root cause is upstream — profile data coverage.

**Correct fix path**:
1. **Saturation-oriented reprofile**: run archive's recipe but extend benches that drive conc to 200+ AND tt to 400+ (long decode tails). Explicitly target the (tt=400-500, conc=200-257) region with enough prompts to get ≥1000 samples per bucket.
2. **Verify oracle coverage**: after reprofile, check bucket density across the operating points r=16 actually visits, before validating.
3. **If (1) still shows gap**: likely means the oracle's nearest-neighbor rule is structurally wrong at edges. Consider bilinear interpolation across (tt, conc) cells or kNN.

## Priority

The r=16 gap has persisted across multiple v3/v4/v5 hook variants. All variants kept it around −14 to −22% because they were attacking the wrong layer. Shifting effort to profile coverage is the next step.

For tonight's overnight:
- **Saturation reprofile** (separate recipe, targets conc=200-250 × tt=400-500 density).
- Compare validation r=16 with the new profile vs archive-r2.
- If saturation reprofile closes to ≤10% at r=16, problem solved.

## What's refuted, what stays

- REFUTED: hook changes batch composition at r=16 (batch-comp diag).
- REFUTED: v1/v2 k=1 sweep disagreement is cause (v1/v2 agree within noise).
- REFUTED: burst-aware hook closes r=16 (v5-2d-burst variants tried, residual still large).
- **LIKELY CAUSE**: profile has thin sample coverage at high-conc × high-tt buckets.
- **NEXT ACTION**: saturation-focused reprofile.

## Today's TTFT deliverable

No code committed. Investigation-only. Three hypotheses refuted, one strong hypothesis (profile coverage) confirmed with direct evidence (bucket density inspection). Tomorrow's overnight can act on this.
