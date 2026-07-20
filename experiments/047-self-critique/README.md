# Experiment 047: Does Self-Critique Help or Hurt?

Date: 2026-07-20
Status: In progress
Source: QUEUE.md Track A verifier/critic lane

## Goal

Test the most commonly recommended "quality" prompt pattern: make the model review and revise its own answer. The sweep's verifier survey flagged this as the well-documented trap: loops where a model grades itself do not reliably help and often degrade (Huang 2310.01798, Stechly/Kambhampati 2402.08115, Kamoi 2406.01297). Gains require injecting a signal the generator lacks: a tool, a genuinely different model, or cross-sample agreement.

## Why this design works at the ceiling that killed E045/E046

Our model sits at 97.7% single-sample accuracy on the E046 hard set. Self-consistency needed variance to exploit and found none. Self-critique is different: at 97.7% there is only +2.3 to gain but 97.7 to lose, so **degradation is the measurable effect**. The headline metric is therefore not net accuracy but the flip counts:

- **repairs** = wrong on pass 1, right after critique
- **damages** = right on pass 1, wrong after critique

Those two numbers answer the practical question directly, regardless of where the baseline sits.

## Method

E046 hard set (12 problems, all answers brute-forced in code). Qwen3-30B-A3B-Instruct-2507, `-np 4`, vendor samplers, 6000-token budget. Per trial:

1. Pass 1: solve normally, end with `ANSWER: <n>`.
2. Pass 2: same conversation plus the model's own answer, then "Review your solution above critically... if you find a mistake, correct it," ending with `ANSWER: <n>`.

4 trials per problem (48 trials). **Truncation discipline (E046 lesson): a trial counts only if BOTH passes yield a parseable answer**; invalid trials are excluded from both arms, never charged to one. This is exactly the asymmetry that manufactured a phantom +25 in E046.

## Hypothesis (pre-registered)

1. **Self-critique does not improve accuracy.** Predicted net change between -10 and +2 points.
2. **Damages exceed repairs.** This is the mechanism claim and the reason to advise against the pattern.
3. Cost is roughly 2x tokens and wall clock for that non-benefit.
4. **Falsification:** net accuracy improves by >= +3 points AND repairs > damages. Then self-critique earns its place and gets recommended.

Ceiling caveat stated up front: with a 97.7% baseline the upside is arithmetically capped at +2.3, so H1's "improvement" range is narrow by construction. The flip counts carry the real information.

## Actual Result

12 problems x 4 trials x 2 passes, 5286 s wall (88 min).

| Metric | Value |
|---|---|
| Valid trials | 43 (5 invalid, truncation, almost all q11) |
| Pass 1 accuracy | 100.0% |
| Pass 2 accuracy (after self-critique) | 100.0% |
| Net change | **0.0 points** |
| **Repairs** (wrong -> right) | **0** |
| **Damages** (right -> wrong) | **0** |
| Both wrong | 0 |

Every single pass-2 answer was identical to its pass-1 answer. Self-critique changed nothing, in either direction, in 43 attempts.

## Verdict: self-critique is a no-op here. Do not pay 2x for it.

**H1 confirmed** (no improvement; net 0.0, inside the predicted -10 to +2 band). **H3 confirmed** (cost ~2.5x: 88 min versus ~35 min for the single-pass E046 run over the same problems).

**H2 FALSIFIED, and this is the interesting part.** I predicted damages would exceed repairs, citing the literature's well-documented finding that self-correction degrades reasoning (Huang 2310.01798, Stechly/Kambhampati 2402.08115). It did not reproduce. Given 43 opportunities to talk itself out of a correct answer, this model did so **zero** times. Asked to review correct work, it correctly left it alone every time. Self-critique here is harmless, not harmful.

**The honest limit, stated plainly:** pass-1 accuracy was 100% on valid trials and `both_wrong = 0`, meaning there were **no wrong answers available to repair**. The repair side of the ledger is untested by construction, for the fourth experiment running. What this experiment genuinely measured is the damage side, and the answer there is a clean zero.

So the practical claim is narrow and defensible: **on work this model already gets right, self-critique costs 2x wall clock and changes nothing.** Whether it repairs genuine errors remains unmeasured.

## The strategic finding (E045 - E047)

Four consecutive experiments have now been blocked by the same wall: **Qwen3-30B-A3B-Instruct answers essentially every machine-verifiable math problem I can construct and verify, correctly, on every sample, and does not revise those answers when challenged.**

- E045: 24 ordinary problems, ~100% after correcting for the truncation bug
- E046 v1: 18 moderate, 100%
- E046 v2: 12 deliberately hard, 97.7%, one disagreement in 44 samples
- E047: 12 hard, 100% pass 1, zero flips in 43 trials

Every quality technique in Track A (self-consistency, best-of-N, self-critique) operates on model *uncertainty*. This model exhibits almost none on this task class, so the entire lane is untestable here, regardless of technique. **The binding constraint on the quality-research programme is not the techniques; it is the absence of errors to fix.**

The productive path forward is to change the subject rather than the method: run the same harness against **Qwen3-8B** (our fast model, which errs far more often) to get a population with real variance, or move to a domain where correctness is checkable but the model is genuinely weak. Building harder arithmetic has failed three times and should not be attempted a fourth.

## Lessons Learned

1. A pre-registered mechanism claim can be wrong in the *favourable* direction. I predicted self-critique would cause harm; it caused nothing. Recording that plainly matters more than the headline no-op.
2. `both_wrong = 0` is the diagnostic that separates "the technique did not help" from "the technique had no opportunity". Log the confusion-matrix cells, not just the net.
3. The truncation discipline held: 5 invalid trials excluded symmetrically, no phantom effect (contrast E046's near-miss +25).
4. Four experiments to learn that the model, not the method, was the wrong variable. Cheaper to have measured baseline error rate first and picked the subject accordingly.

## Next Steps

- Track A verifier/critic lane is **closed for this model**. Its practical output: do not use self-critique loops on the 30B for verifiable reasoning, they cost 2x for nothing.
- If the lane is reopened, run it against Qwen3-8B where errors are plentiful. That is a genuine test of whether critique repairs mistakes, which is the question all four experiments failed to reach.
- Track A's shipped win remains E042 (vendor samplers + top-n-sigma): free, measurable, already live in the hub.
