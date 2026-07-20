# Experiment 045: Self-Consistency Quality-per-Second (Queue B2)

Date: 2026-07-20
Status: In progress
Source: QUEUE.md B2, gated on E044 which PASSED (K=4 costs 2.32x, not 4x)

## Goal

Now that E044 proved 4 samples are affordable, measure whether majority voting over those 4 samples actually buys accuracy worth the 2.32x wall clock.

## Design (one run, both numbers)

24 multi-step reasoning problems with verified integer answers (datasets/e045-reasoning.jsonl; every answer recomputed by hand, per the E042 lesson that a wrong key corrupts the whole experiment). Model: Qwen3-30B-A3B-Instruct-2507 (non-reasoning, so it sits nearer the edge of its ability where the literature says self-consistency helps most). Server `-np 4`, vendor samplers (temp 0.7 / top-p 0.8 / top-k 20), 4 samples per problem fired concurrently, distinct seeds.

From the same 4 samples:
- **Single-sample accuracy** = mean correctness across all 4 x 24 = 96 individual samples
- **Majority-vote accuracy** = correctness of the modal answer per problem

Answers are forced into a parseable `ANSWER: <number>` final line (same format for both conditions, so the comparison is fair).

## Hypothesis (pre-registered)

1. **Success bar: majority vote beats single-sample by >= 5 accuracy points.** That would justify the 2.32x cost for hard reasoning work.
2. **A null is a real possibility and gets published either way.** arXiv 2511.00751 reports self-consistency gains have largely collapsed on modern models, and can go negative on easy items. If the gain is < 5 points, the honest conclusion is that self-consistency is not worth it for this model on this rig.
3. Single-sample accuracy lands in the 45-80% band. If it is above ~90% the set is too easy to show any voting effect and the result is inconclusive rather than negative (a ceiling artifact I will report as such).
4. Cost is the E044-measured 2.32x, not re-derived here.

## Actual Result

| Metric | Value |
|---|---|
| Single-sample accuracy | 91.7% (88/96 samples) |
| Majority-vote accuracy | 91.7% (22/24 problems) |
| Gain from voting | **0.0 points** |
| Wall clock | 689 s for 24 problems x 4 samples |

## Verdict: INCONCLUSIVE (ceiling artifact plus a harness bug), NOT a negative result

Pre-registered H3 said: if single-sample accuracy exceeds ~90%, the set is too easy and the result is a ceiling artifact to be reported as such. Measured 91.7%, so that clause fires. But the real story is worse than a ceiling, and it is my fault.

**22 of 24 problems returned four identical correct samples.** Only q2 and q6 "failed", with scattered answers like `['0','6400','644','2']`. A follow-up diagnostic at a 2500-token budget showed the model answers BOTH correctly (10 and 17), needing 1234 and 734 tokens against my 700-token cap. **The failures were my token cap truncating the working, and my regex fallback ("last integer anywhere") then invented an answer from the cut-off text and scored it wrong.** True single-sample accuracy on this set is effectively 100%.

So this run measured nothing about self-consistency. Voting cannot help a model that already answers every item correctly on every sample.

**The one real finding:** at vendor sampling (temp 0.7), this model's outputs on ordinary multi-step problems have almost **zero answer variance**: 22/24 problems produced 4/4 identical answers. Self-consistency needs disagreement to exploit, so on this class of everyday task there is nothing for it to fix, independent of cost. That is a genuine, if partial, practical conclusion.

## Lessons Learned

1. **A fallback parser that guesses is worse than one that fails.** Returning "the last integer anywhere" converted truncations into confident wrong answers, manufacturing a 2-problem failure rate out of nothing. Fixed: `finish_reason == "length"` now invalidates a sample instead of guessing.
2. **Always check finish_reason.** A token cap is a silent experimental confound; it looks exactly like a reasoning failure in the scored output.
3. **Calibrate difficulty before spending the run.** I should have measured single-sample accuracy on 5 problems first and only built the full run once it landed in the 40-70% band. 11 minutes of compute answered a question I had accidentally made unanswerable.
4. E042 taught me to verify the answer key; E045 adds: verify the *extraction path* too. Both failure modes silently corrupt accuracy numbers in the same direction.

## Next Steps

- **E046 (the real test):** rebuild with (a) genuinely hard items calibrated to 40-70% single-sample accuracy, (b) the fixed harness, (c) a 5-problem difficulty pre-check before committing the full run. Only then does the +5 point bar mean anything.
- Until then, the honest status of self-consistency on this rig is **untested**, not rejected. E044's affordability result (2.32x for K=4) still stands and is unaffected by this harness bug.
