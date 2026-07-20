# Experiment 046: Self-Consistency, Properly Built (E045 redo)

Date: 2026-07-20
Status: Complete
Source: E045 next-steps (that run was invalidated by a ceiling artifact plus a harness bug)

## What was rebuilt

E045 failed for three reasons, each fixed here:

| E045 flaw | E046 fix |
|---|---|
| Answer key written by hand (E042 shipped 2 wrong ones) | **All answers brute-forced in code** (scripts/e046-build-hard.py); the key is machine-derived by construction |
| 700-token cap truncated working; regex fallback then invented answers from cut-off text | 6000-token budget; `finish_reason == "length"` invalidates a sample, never guesses |
| Problem set far too easy (ceiling) | Hard set: derangements, non-attacking rooks, strict-inequality triples, taxicab number, base-12 trailing zeros, Euler-polynomial failures |
| No difficulty calibration | **Pre-check gate run before committing** the full run |

## The near-miss worth recording

The first hard-set pre-check reported **single 58.3%, vote 83.3%, +25.0 points**: a headline result. It was false. Truncated samples were being counted as WRONG in the single-sample metric but silently DROPPED from the vote, so voting "won" by dodging failures that single-sampling was charged for. Fixed by excluding truncated trials from **both** metrics. Publishing that +25 would have been a fabricated breakthrough, and it looked entirely plausible.

## Actual Result

12 hard problems x 4 samples, Qwen3-30B-A3B-Instruct-2507, `-np 4`, vendor samplers (temp 0.7 / top-p 0.8 / top-k 20), 6000-token budget, 2083 s wall.

| Metric | Value |
|---|---|
| Valid samples | 44 (4 truncated on q11, excluded from both metrics) |
| Problems scored | 11 of 12 |
| Single-sample accuracy | 97.7% (43/44) |
| Majority-vote accuracy | 100.0% (11/11) |
| **Gain from voting** | **+2.3 points** |
| Cost (E044) | 2.32x wall clock |

## Verdict: self-consistency does NOT clear the bar (+2.3 vs +5 required)

**Recommendation: do not pay 2.32x for majority voting on this model for verifiable reasoning work.**

The honest caveat, stated plainly: single-sample accuracy was 97.7%, still above the ~90% ceiling threshold pre-registered in E045. So the correct claim is **"voting has no room to help because this model almost never errs on constructible verifiable problems"**, NOT "voting fails when the model errs." Exactly one problem (q4) produced disagreement, `['82','82','82','86']`, and there voting selected correctly. The mechanism works; the opportunity does not exist.

## The finding across three experiments

E045 (24 ordinary problems), E046 v1 (18 moderate), E046 v2 (12 deliberately hard) all landed at 97-100% single-sample accuracy with near-zero answer variance: 4/4 identical samples on 10 of 11 scored problems here. **I could not construct a machine-verifiable integer-math problem set this model gets wrong often enough for self-consistency to matter.** For this task class the technique is inapplicable on this rig, regardless of its affordability.

A plausible contributing factor, untested: the vendor samplers shipped in E042 (top-k 20, top-p 0.8) deliberately narrow the token distribution, which suppresses exactly the sampling diversity self-consistency feeds on. Our own quality fix may be closing the door this experiment tried to walk through. Testable later by running self-consistency at wider samplers, though it is unclear why one would want a deliberately worse-calibrated model just to give voting something to repair.

## Lessons Learned

1. **The pre-check gate paid for itself immediately.** It killed a 32-minute invalid run on the first hard set and caught the phantom +25 on the second. Cost: ~9 minutes each.
2. **Asymmetric exclusion is a silent fabricator.** Any metric where an invalid trial is charged to one arm and dropped from the other will manufacture an effect. Check both arms handle failure identically.
3. **Three harness bugs across E042-E046 all pushed the same direction:** toward reporting a result that did not exist (wrong keys, guessed answers, asymmetric exclusion). Measurement error is not random here; it flatters the hypothesis.
4. Brute-forcing answer keys in code eliminated an entire class of error and cost about twenty minutes to build.

## Next Steps

- Self-consistency is **closed** for verifiable math on this model. If revisited, it needs a task class with genuine model uncertainty (ambiguous, creative, or far harder domains), not harder arithmetic.
- E044's affordability result (K=4 costs 2.32x, aggregate scales K^0.38) stands independently and remains the reusable finding from this lane.
- Track A's remaining untested item is the verifier/critic lane, where the survey's warning about self-grading traps applies and a small model would act as router rather than arbiter.
