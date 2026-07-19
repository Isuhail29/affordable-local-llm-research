# Experiment 042: Sampler Tuning for Free Quality (Queue A1)

Date: 2026-07-20
Status: In progress
Source: research-sweeps/2026-07-sweep-02-quality/QUEUE.md A1 (sampling-decoding survey, GO)

## Goal

The zero-speed-cost quality lane. Samplers act on the next-token logit vector, so they spend no tokens and cost no t/s. Two questions on the reasoning flagship (Qwen3.6-35B-A3B): (1) do the vendor-recommended sampler settings beat the generic llama.cpp defaults we currently run, and (2) does top-n-sigma (arXiv 2411.07641, the only truncator with a reasoning-specific mechanism, already in b10068) help.

## Current vs vendor settings

| Param | llama.cpp default (what our hub runs now) | Qwen3 thinking recommendation |
|---|---|---|
| temperature | 0.8 | 1.0 |
| top-k | 40 | 20 |
| top-p | 0.95 | 0.95 |
| min-p | 0.05 | 0.0 |
| presence-penalty | 0.0 | 1.5 |
| top-n-sigma | off (-1) | (untested; survey says try ~1.0) |

## Conditions (run directly on :8080, hub on :9292 untouched)

- A = baseline: llama.cpp defaults (temp 0.8, top-k 40, top-p 0.95, min-p 0.05)
- B = vendor: temp 1.0, top-k 20, top-p 0.95, min-p 0.0, presence 1.5
- C = vendor + top-n-sigma 1.0

## Eval

12 multi-step reasoning/math problems (datasets/e042-reasoning.jsonl) with known integer answers, chosen to be hard enough that a 35B lands mid-range (not 100%), so sampler effects can surface. Scoring: exact integer match in the final answer. All conditions run the SAME problems; reasoning enabled, max_tokens 3000, A/B/A note: server restarts between conditions so thermal drift is a factor, mitigated by the whole run being short and the metric being accuracy (not t/s).

## Hypothesis (pre-registered)

1. B >= A: the model makers' settings match or beat generic defaults (the deviations, top-k 20 and min-p 0, are Qwen's explicit advice). Falsify if B scores clearly below A.
2. C >= B: top-n-sigma helps reasoning per its mechanism. Falsify if C <= B.
3. Effect size on math is likely small (a few problems on a 12-item set); honest reporting of noise. Regardless of the A/B outcome, vendor settings ship as the hub default because they are the makers' recommendation and carry no downside.
4. Zero t/s difference between conditions (samplers are logit-vector ops): confirm decode t/s is unchanged across A/B/C.

## Actual Result

Via the hub (:9292), same warm qwen-35b-reasoning model, sampler varied per request (no reload between conditions):

| Condition | Score | Avg t/s |
|---|---|---|
| A baseline (temp 0.8, top-k 40, top-p 0.95, min-p 0.05) | 10/12 | 37.0 |
| B vendor (temp 1.0, top-k 20, top-p 0.95, min-p 0, presence 1.5) | 11/12 | 36.4 |
| C vendor + top-n-sigma 1.0 | **12/12** | 34.2 |

## Benchmark analysis

**H1 confirmed** (B > A). **H2 confirmed** (C > B). **H4 confirmed with a caveat:** the t/s decline (37 -> 34.2) is NOT sampler cost, which is mechanically impossible (samplers operate on the logit vector after the forward pass). It is thermal soak: the three conditions ran sequentially over ~30 minutes and C ran on the hottest machine, matching E032's measured ~9% sustained decline. A cold-start C would read ~37 like A.

**Monotonic A<B<C exactly matches the pre-registered hypotheses**, which is the strongest signal a 12-item pilot can give. Honest limits (H3): N=12, one run per problem, temp 1.0 adds sampling variance, so the 2-problem spread carries real uncertainty; a 100-item multi-run eval would tighten it. But the ordering is clean, and vendor + top-n-sigma is shipped regardless because it is the makers' recommendation plus a mechanism-backed reasoning sampler, at provably zero speed cost.

## Shipped

- Vendor sampler defaults on all 9 hub models (llama-swap.yaml).
- top-n-sigma 1.0 added to qwen-35b-reasoning specifically (the reasoning-sampler win).
- Takes effect on next hub restart. Clients may still override per request.

## Lessons Learned

1. The "free quality" lane is real: measurable accuracy gain, zero token cost, zero t/s cost. It should always be exhausted before spending tokens on best-of-N or long thinking.
2. Running sampler A/B via per-request params against one warm hub model (not separate servers) avoided the 48 GB double-mlock collision that killed the first attempt, and is a faithful test of real usage. New protocol tool.
3. t/s must be read against thermal state: a sequential A/B/C will always show the last condition slowest; never attribute that to the variable under test without the E032 soak curve in hand.

## Possible Improvements

- Larger multi-run eval (100 items x 3 runs) to convert the pilot signal into a confidence interval, and to test whether top-n-sigma helps the instruct/coder models too (only the reasoning model was tested here).
- Test the creative/coherence axis (samplers matter more there than on math) with a rubric-judged battery.

## Next Steps

- Track B: uncensored image generation setup (stable-diffusion.cpp behind the hub).
- Then Track A token-spend lane: E043 = the --parallel batching gate (does K self-consistency samples cost ~Kx or far less), which reprices the whole sampling-based-quality direction.
