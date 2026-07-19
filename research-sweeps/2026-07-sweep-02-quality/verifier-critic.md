# Verification and critique loops: what raises quality per wall-clock-second, and what is illusory

Survey date: 2026-07-20. Sweep: 2026-07-sweep-02-quality. Scope: the family of inference-time techniques that generate a candidate answer and then check, critique, rank, or debate it before committing, covering LLM-as-judge, self-critique and iterative reflection (Reflexion, Self-Refine), generator-verifier splits (process/outcome reward models, generative verifiers, best-of-N reranking), multi-agent debate, and specifically using our small fast model (Qwen3-8B at ~50 t/s) as a verifier or reranker for a big-model generator. The question is not "does this improve a benchmark somewhere" but "does it improve quality per wall-clock-second, or quality at a fixed token budget, on our rig, with no retraining, orchestrated by scripts we write against our hub." The skeptical literature is treated as first-class: several of the most-cited loops degrade quality when the feedback signal is the model grading itself.

Rig constraints applied throughout: i7-14650HX (AVX2 only, no AVX-512/AMX), 48 GB DDR5-5600 (~60 GB/s ceiling, ~37-42 GB/s effective), RTX 5060 Laptop 8 GB, Windows 11, llama-swap hub at :9292 (models auto-swap: Qwen3-30B-A3B-Instruct-2507 ~33-42 t/s, Qwen3.6-35B-A3B reasoning 37-40 t/s, GLM-4.7-Flash coding, Qwen3-8B ~50 t/s, abliterated 30B general + coder, Qwen3-VL-30B), ~3 MB/s internet. No fine-tuning, no cloud.

## 0. The framing that changes every verdict

Throughput is fixed. Memory bandwidth pins decode at ~30-42 t/s and none of these techniques raise it. So every loop below spends its budget in exactly one of two currencies:

1. **Extra tokens** (best-of-N samples, long critiques, debate rounds). These do NOT lower t/s. They lower time-to-answer, because the wall clock is (tokens emitted / throughput) plus any model-swap latency on the hub. A best-of-8 loop that samples 8 answers plus a verify pass emits roughly 9x the tokens of a single greedy answer, so it costs ~9x the wall clock. It is worth it only if the quality gain beats what the same token budget buys elsewhere.

2. **Model swaps.** Every time the orchestration hands off between the 30B generator and a different verifier model, llama-swap tears down and loads weights. On our rig that is seconds of dead time per swap, dwarfing token cost for short answers. A verifier that lives on the *same* loaded model (a second prompt, not a second model) has near-zero swap cost; a genuinely different verifier model pays the swap tax on every round-trip unless we can hold both resident (Qwen3-8B at ~5 GB + a 30B with experts on CPU can coexist, but two large models cannot).

The correct control for any verify/critique loop is therefore **self-consistency at a matched token budget** (section 5). If a critique loop that emits N tokens does not beat plain majority-vote-over-samples using the same N tokens, the loop is illusory, no matter how good its benchmark number looks in isolation.

## 1. Intrinsic self-correction: the skeptical core

The single most important result for this sweep is negative. [Huang et al., "Large Language Models Cannot Self-Correct Reasoning Yet" (arXiv 2310.01798, ICLR 2024)](https://arxiv.org/abs/2310.01798) showed that when a model is asked to critique and revise its own reasoning **using only its own feedback and no external signal**, accuracy does not improve and often *drops*. The apparent gains in earlier self-correction papers came from oracle stopping (using the ground-truth answer to decide when to stop revising), which is unavailable at inference. Without the oracle, the model changes correct answers to wrong ones about as often as the reverse.

[Stechly, Valmeekam, and Kambhampati, "On the Self-Verification Limitations of LLMs on Reasoning and Planning Tasks" (arXiv 2402.08115, ICLR 2025)](https://arxiv.org/abs/2402.08115) sharpened this on Game-of-24, graph coloring, and STRIPS planning: self-critique caused a **performance collapse**, while a *sound external verifier* produced large gains, and simply re-prompting the generator with that external verifier's pass/fail signal captured most of the benefit of fancier setups. Their framing is the one to internalize: the widespread belief that self-correction works rests on the assumption that verification is easier than generation, but an LLM grading itself is doing approximate retrieval, not sound verification, so the asymmetry it needs does not exist when the checker is the same distribution as the generator.

The authoritative synthesis is [Kamoi et al., "When Can LLMs Actually Correct Their Own Mistakes? A Critical Survey" (arXiv 2406.01297, TACL 2024)](https://arxiv.org/abs/2406.01297). Its conclusions are directly load-bearing for our design:

- No prior work demonstrates successful self-correction with feedback from *prompted same-model* LLMs, except on tasks unusually suited to it (e.g. tasks where the response can be decomposed and checked piecewise).
- Self-correction is "**bottlenecked by the verifier, not the refiner.**" The model can usually *fix* an error once told where it is; it just cannot reliably *find* the error itself.
- Many headline results used impractical frameworks (oracle stopping, weak baselines, or feedback that leaks the answer).

**Takeaway for us:** an intrinsic "generate then ask the same model to check itself" loop is the one thing the literature most consistently says is a trap. Any loop we build must inject a signal the generator does not already have: a *different* model, a *tool* (code execution, retrieval, a calculator), or *diversity/agreement* across independent samples. This is the design constraint, not a side note.

A caveat on scope: intrinsic self-correction *can* be made to work, but only by **training** the model for it, e.g. [SCoRe, "Training Language Models to Self-Correct via RL" (DeepMind, ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/file/871ac99fdc5282d0301934d23945ebaa-Paper-Conference.pdf). That is multi-turn RL and out of scope for us (no fine-tuning). It is worth knowing the ceiling exists, and that reaching it is a training problem, not a prompting one.

## 2. LLM-as-judge: useful, but a biased instrument

LLM-as-judge (introduced at scale by Zheng et al.'s MT-Bench / Chatbot Arena, 2023) is the substrate under every critique loop: a model reads a candidate and emits a score or a preference. It correlates with human judgment well enough to be useful, but it is a *biased* instrument, and the biases matter because they systematically corrupt the verify signal we would build on. The survey [Gu et al., "From Generation to Judgment: Opportunities and Challenges of LLM-as-a-Judge" (arXiv 2411.16594)](https://arxiv.org/abs/2411.16594) catalogs the field; the bias-specific work is what constrains us:

- **Self-preference / self-recognition.** [Panickssery, Bowman, Feng, "LLM Evaluators Recognize and Favor Their Own Generations" (arXiv 2404.13076, NeurIPS 2024)](https://arxiv.org/abs/2404.13076) showed models can recognize their own outputs and score them higher than humans rate them, with a linear link between self-recognition strength and self-preference. **Direct consequence: a model should not be the final judge of its own generations.** Cross-model judging (35B reasoning judging 30B instruct's output, or vice versa) is not a nicety, it is the mitigation.
- **Position, verbosity, and format biases.** [Ye et al., "Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge" (arXiv 2410.02736)](https://arxiv.org/abs/2410.02736) and [a systematic position-bias study (ACL/IJCNLP 2025)](https://aclanthology.org/2025.ijcnlp-long.18.pdf) show judges favor the first-presented option, longer answers, and better-formatted ones regardless of substance. Mitigations we can script: randomize/swap order and average both directions, strip or normalize formatting before judging, cap length in the judge prompt.
- **Reliability without validity.** [A large-scale 2026 evaluation (arXiv 2606.19544)](https://arxiv.org/pdf/2606.19544) found judges can be *consistent* (reliable) while still *disagreeing with ground truth* (invalid). Consistency is not correctness. Any judge we deploy must be validated against a small held-out labeled set of our own before we trust its verdicts.

**Takeaway:** LLM-as-judge is a usable reranking signal on our rig if and only if we (a) use a different model than the generator, (b) de-bias order and length in the prompt harness, and (c) calibrate it once against a small labeled set. Treat its score as noisy evidence to aggregate, never as an oracle.

## 3. Generator-verifier splits: the part that actually works

The robust, repeatedly-confirmed finding is that a *separate verifier selecting among multiple candidate answers* beats a single greedy answer. This is where the token budget earns its keep.

- **Verifier-reranked best-of-N.** [Cobbe et al. (GSM8K, 2021)](https://arxiv.org/abs/2110.14168) and [Lightman et al., "Let's Verify Step by Step" (arXiv 2305.20050)](https://arxiv.org/abs/2305.20050) established that training a verifier to score sampled solutions and picking the best (best-of-N) substantially beats majority vote and single-sample. Lightman's headline: process supervision (scoring each reasoning step, a PRM) beats outcome supervision (scoring only the final answer, an ORM), reaching 78% on a MATH subset via best-of-N. **The catch for us: those verifiers are trained.** We cannot train one.
- **Generative verifiers (GenRM).** [Zhang et al., "Generative Verifiers: Reward Modeling as Next-Token Prediction" (arXiv 2408.15240)](https://arxiv.org/abs/2408.15240) reframed the verifier as an LLM that *generates a chain-of-thought critique and then says correct/incorrect*, usable with plain prompting and majority-voted over its own reasoning. It beat discriminative verifiers, DPO verifiers, and vanilla LLM-as-judge, with best-of-N gains like 73% to 93.4% on GSM8K. **This is the runnable-today path for us:** GenRM is a prompt pattern, not a trained head. We can implement a generative verifier as a scripted prompt on any of our models. Its quality is bounded by the base model's judging ability (section 2's biases apply), but it needs no training.
- **The generation-verification asymmetry.** [Jason Wei, "Asymmetry of verification and verifier's law" (jasonwei.net, July 2025)](https://www.jasonwei.net/blog/asymmetry-of-verification-and-verifiers-law) is the clean conceptual frame: some tasks are far cheaper to *check* than to *solve* (sudoku, unit-tested code, factual claims with a lookup). The size of the achievable win from any verify loop is proportional to how large this asymmetry is *for the specific task*. Coding with executable tests: huge asymmetry, verify loops win big. Open-ended reasoning with no ground truth: small or negative asymmetry, verify loops are fragile. **This is the single best predictor of which of our workloads a verifier will help.**

**Takeaway:** the generator-verifier split is the winning structure, best-of-N reranked by a generative verifier is the runnable instance, and the expected win is large exactly where the task is cheap to check (code with tests, math with a checker, claims with retrieval) and small where it is not.

## 4. Small model as verifier for a big generator: the direct question for our rig

This is the crux of our specific hardware bet: can Qwen3-8B at 50 t/s cheaply verify or rerank the 30B/35B flagship's output, buying quality for little wall-clock? The literature is split in a way that tells us exactly how to build it.

**The discouraging direction.** [Zhang et al., "Small Language Models Need Strong Verifiers to Self-Correct Reasoning" (arXiv 2404.17140, ACL 2024 Findings)](https://arxiv.org/abs/2404.17140) found a weak self-verifier is worse than useless; only a *strong* verifier (GPT-4-class) maintained the right correction frequency and actually improved accuracy. Combined with the self-verification-limitations result (section 1), the naive hope, "use the cheap small model as the final arbiter over the big model," is exactly the configuration the literature says fails: a weaker judge cannot reliably catch a stronger generator's errors, and will flip correct answers.

**The encouraging direction.** Weakness is fixable by *aggregation* and *routing*, not by trusting one weak judge:

- [Saad-Falcon et al., "Shrinking the Generation-Verification Gap with Weak Verifiers" / Weaver (arXiv 2506.18203)](https://arxiv.org/abs/2506.18203) showed an *ensemble of weak verifiers* combined under weak-supervision (Weaver) approximates a strong verifier: with Llama-3.3-70B as generator and only 70B-or-smaller judges/reward models, they reached o3-mini-level selection accuracy. Crucially for us, they then distilled the ensemble into a **400M cross-encoder that kept 98.7% of the accuracy while cutting verification compute by up to 99.97%.** The principle (many cheap independent checks beat one) is scriptable without their training step.
- [Saad-Falcon et al., "Variation in Verification" (arXiv 2509.17995)](https://arxiv.org/html/2509.17995v1) analyzes when aggregating weak verifiers works, i.e. when their errors are decorrelated.
- [Chen et al., verification granularity (arXiv 2505.11730)](https://arxiv.org/pdf/2505.11730) found strong generators need only *sparse* verification (check every 2-4 steps) while weak generators need frequent checks. Our generator is strong, so sparse, cheap verification is the right regime, which favors the small verifier being invoked selectively rather than on every token.
- [Reward-model code verification (arXiv 2506.10056)](https://arxiv.org/html/2506.10056v1) frames verification as trading accuracy for throughput: a cheaper, less accurate verifier run *more times* over more candidates can beat one expensive verifier, which is precisely the small-model-many-samples tradeoff our rig is built for.

**Synthesis for our rig.** The small model should be a **filter/router and an ensemble member, not the final arbiter.** Two robust patterns emerge:

1. **Cheap-first cascade.** Qwen3-8B does a fast, cheap pass (e.g. sanity checks, obvious-error detection, agreement scoring across the generator's own samples) and only *routes the uncertain cases* to an expensive verify (a second flagship pass, or the reasoning model). Most answers never pay the expensive tax. This respects "weak verifiers can't be the last word" while still cashing the 50 t/s speed.
2. **Diverse weak ensemble.** Several cheap independent verify signals (Qwen3-8B judge + a tool check + cross-sample agreement) aggregated, per Weaver, rather than one small model's verdict.

**What to avoid:** Qwen3-8B as the sole final judge that can overrule the 30B. That is the 2404.17140 failure mode.

## 5. Self-consistency: the baseline every loop must beat

[Wang et al., "Self-Consistency Improves Chain-of-Thought Reasoning" (arXiv 2203.11171)](https://arxiv.org/abs/2203.11171) is the cheapest generator-side quality lever: sample N reasoning chains, majority-vote the final answers. No verifier, no second model, no swap cost, trivially parallel in spirit (though serialized by our single-stream decode). It is the correct *control* for every fancier loop because it spends the same currency (extra tokens) with zero orchestration.

Two facts make it the referee:
- It reliably beats greedy single-sample on tasks with a canonical answer (math, multiple-choice, extraction).
- On our rig its cost is purely tokens, no model swap, so its quality-per-wall-clock-second is the number to beat.

It has real limits: it needs a *comparable* final answer to vote on (fails on open-ended generation), and it cannot exceed the generator's own support (if the model never samples the right answer, voting cannot find it). But any critique/debate loop that emits more tokens than a budget-matched self-consistency run and does not beat it is not earning its keep. **We must run self-consistency as a matched-budget control in every candidate below.** [Universal Self-Consistency (arXiv 2311.17311)](https://arxiv.org/pdf/2311.17311) extends voting to free-form outputs by having the model pick the most consistent response, a cheap way to get a self-consistency control even for open-ended tasks.

## 6. Multi-agent debate: promising claim, deflated by fair baselines

[Du et al., "Improving Factuality and Reasoning through Multiagent Debate" (arXiv 2305.14325, ICML 2024)](https://arxiv.org/abs/2305.14325) had multiple model instances propose answers and critique each other over rounds, reporting gains on reasoning and factuality. The skeptical follow-ups matter more for us:

- [Wang et al., "Rethinking the Bounds of LLM Reasoning: Are Multi-Agent Discussions the Key?" (arXiv 2402.18272, ACL 2024)](https://aclanthology.org/2024.acl-long.331/) found a **single agent with a strong prompt matches the best multi-agent discussion** across many reasoning tasks and backbones; debate only helped when the prompt lacked demonstrations.
- [Smit et al., "If Multi-Agent Debate is the Answer, What is the Question?" (arXiv 2502.08788)](https://arxiv.org/html/2502.08788) found debate only *slightly* beats a single agent at equal agent count, and **significantly underperforms plain self-consistency at an equal number of sampled responses.** In other words, at a fixed token budget, majority vote beats debate.
- Du's own paper flags the failure mode: agents can *converge confidently on a wrong answer*. Debate manufactures agreement, not correctness.

**The escape hatch is diversity.** [Chen et al., ReConcile (arXiv 2309.13007, ACL 2024)](https://arxiv.org/abs/2309.13007) got real gains (up to 11.4%) specifically by using *different* base models in the round table, not clones. Homogeneous debate (one model talking to copies of itself) inherits correlated errors and self-preference; heterogeneous debate injects genuinely independent signal. We have unusual assets here: Qwen3-30B instruct, Qwen3.6-35B reasoning, GLM-4.7-Flash, and abliterated variants are *architecturally and behaviorally diverse* models on one hub. But every cross-model round pays the swap tax.

**Takeaway:** homogeneous debate is dominated by budget-matched self-consistency and is likely a net loss on our rig once swap cost is counted. Heterogeneous debate (our diverse models) is the only version with a research-backed reason to win, and even that must beat self-consistency at matched budget to count.

## 7. Tool-grounded critique: the one intrinsic loop that reliably wins

The cleanest positive result in the whole area: when the feedback comes from a *tool* rather than the model's opinion, critique loops work. [Gou et al., "CRITIC: LLMs Can Self-Correct with Tool-Interactive Critiquing" (arXiv 2305.11738, ICLR 2024)](https://arxiv.org/abs/2305.11738) had the model call a code interpreter, search engine, or calculator, read the *real* result, and revise. This consistently improved factuality, code, and math, because the feedback is grounded truth, not self-opinion, which is exactly the "inject an external signal" requirement section 1 demands.

This maps perfectly onto the generation-verification asymmetry (section 3): tools are cheap sound verifiers precisely where the asymmetry is largest.
- **Code:** run the code, run unit tests, read the traceback. GLM-4.7-Flash (coding) or the abliterated coder generates, a sandbox executes, failures feed back. This is the highest-expected-value loop we can build.
- **Math/logic:** a Python/calculator check of the final computation.
- **Factual claims:** retrieval against a local corpus or the model's tool.

**Takeaway:** every dollar of orchestration effort spent on *tool-grounded* verification is better spent than on *opinion-based* self-critique. For coding especially, this is not a research bet, it is the known-good pattern, and our stack (coding models + a sandbox we script) can run it today.

## 8. The budget-accounting summary

| Loop | External signal? | Runnable on our rig today? | Beats budget-matched self-consistency? | Main cost on our rig |
|---|---|---|---|---|
| Same-model self-critique (Reflexion/Self-Refine style, no tool) | No | Yes | Literature says no, often worse | Tokens wasted; risk of flipping correct answers |
| LLM-as-judge, same model | No (self-preference) | Yes | Unreliable | Biased signal |
| LLM-as-judge, cross-model | Yes (diversity) | Yes, with swap cost | Sometimes, task-dependent | Model swap per round-trip |
| Best-of-N + trained PRM/ORM | Yes | No (needs training) | Yes (published) | We can't train it |
| Best-of-N + GenRM (prompted) | Partial (needs cross-model to be safe) | **Yes** | Task-dependent, likely yes with cross-model judge | N samples + verify tokens |
| Self-consistency (majority vote) | Diversity across samples | **Yes** | It IS the baseline | N samples, no swap |
| Homogeneous multi-agent debate | Weak (correlated) | Yes | Published result: no | Tokens + rounds, loses to self-consistency |
| Heterogeneous debate (our diverse models) | Yes (real diversity) | Yes, with swap cost | Plausibly, must be tested | Swap tax per round |
| Small model as sole final verifier | Yes but too weak | Yes | No (2404.17140 failure) | Flips correct answers |
| Small model as cheap-first filter/router | Yes (routes to strong verify) | **Yes** | Plausibly, saves wall clock | Cheap 8B pass + occasional strong verify |
| **Tool-grounded critique (CRITIC)** | **Yes (ground truth)** | **Yes** | **Yes, where task is checkable** | Sandbox/retrieval plumbing |

The pattern is stark: **the loops that win all inject a signal the generator lacks (a tool, a genuinely different model, or cross-sample agreement); the loops that lose all rely on a model grading itself.**

## Candidate experiments for our rig

Ordered by expected quality-per-wall-clock-second. Each is scriptable against the :9292 hub with no fine-tuning. The non-negotiable control in all of them is **budget-matched self-consistency** (section 5): report quality per token and per wall-clock second against it, or the result is not interpretable.

### C1. Tool-grounded critique loop for coding (CRITIC pattern) with an execution sandbox
Highest expected value because coding has the largest generation-verification asymmetry (section 3, 7). Generator: GLM-4.7-Flash or abliterated coder. Loop: generate to a scratch file, run it plus any unit tests in a sandboxed subprocess (we already write PowerShell/Python orchestration), feed real stdout/tracebacks back for a bounded number of revisions (cap at 2-3), stop on green. Controls: single greedy answer, and best-of-N with the *same total token budget* selected by pass/fail. Metric: pass rate and pass@(equal wall clock). Expect a clear win, this is the known-good pattern, and it is a clean public datapoint for a fixed-throughput laptop.
- Needs: a sandbox runner + a small held-out task set (HumanEval-style or our own). No download. Effort: hours to a day. Class: high-confidence win.

### C2. Cross-model generative verifier for best-of-N reranking (30B generator, 35B-reasoning judge)
Directly tests the runnable generator-verifier split (section 3) while dodging self-preference (section 2) by making the judge a *different* model. Generator: Qwen3-30B-A3B-Instruct samples N answers (N in {4, 8}). Verifier: Qwen3.6-35B-A3B reasoning acts as a GenRM (arXiv 2408.15240 prompt pattern): emit a short critique then a correct/incorrect-with-confidence verdict for each candidate, de-biased by randomizing candidate order and averaging. Pick the argmax. Controls: budget-matched self-consistency over the same N samples; same-model self-judge (to measure the self-preference penalty). Metric: accuracy and accuracy-per-wall-clock (must include swap cost between generator and judge). Run on a task with checkable answers (math word problems, extraction) where the asymmetry is real. Falsifiable prediction from the literature: cross-model GenRM > self-judge, and > self-consistency only where verification is genuinely easier than generation.
- Needs: eval set with ground truth, prompt harness, swap-cost measurement. No download. Effort: days. Class: core test of the split.

### C3. Cheap-first cascade: Qwen3-8B as filter/router, flagship as fallback verifier
Tests the honest role for our small fast model (section 4): NOT final arbiter, but a 50 t/s triage that routes only low-confidence cases to expensive verification. Pipeline: 30B generates; Qwen3-8B scores confidence / flags likely errors cheaply; only flagged cases trigger an expensive step (a second 30B sample + GenRM, or a tool check). Measure what fraction of the expensive-verify quality we retain at what fraction of its wall clock. Controls: always-expensive-verify (quality ceiling, cost ceiling), never-verify (quality floor), and self-consistency. Prediction: recovers most of the quality at a large wall-clock saving *if* the 8B's error-flagging correlates with real errors, which itself is the thing to measure. If the 8B cannot flag errors better than chance, this fails cleanly and we learn our small verifier's true ceiling (a publishable negative, given 2404.17140).
- Needs: Qwen3-8B (on disk) + 30B (on disk), a routing threshold swept on a labeled set. No download. Effort: days. Class: the defining test of the small-verifier bet.

### C4. Heterogeneous vs homogeneous debate, both against self-consistency at matched budget
Settles for our rig whether debate earns its swap tax. Three arms at equal total token budget: (a) self-consistency over Qwen3-30B (control), (b) homogeneous debate: 3 Qwen3-30B instances, (c) heterogeneous debate: Qwen3-30B + Qwen3.6-35B-reasoning + GLM-4.7-Flash (the ReConcile diversity condition, arXiv 2309.13007). Metric: accuracy and accuracy-per-wall-clock including per-round swap cost. Literature predictions to falsify: (b) <= (a) (Smit et al., arXiv 2502.08788), and (c) is the only arm with a chance of beating (a), and even then may lose once swap cost is charged. High information value precisely because a null result here saves us from a whole class of expensive orchestration.
- Needs: multi-model prompt harness, swap-cost accounting. No download. Effort: days. Class: mostly falsification, guards against wasted effort.

### C5. Diverse weak-verifier ensemble (Weaver-lite, no training)
Tests whether aggregating several cheap checks approximates a strong verifier (section 4, arXiv 2506.18203) without their distillation step. For each candidate answer, gather 3-5 cheap independent signals: Qwen3-8B judge verdict, cross-sample agreement (self-consistency vote share), a tool/sanity check where applicable, and one flagship GenRM verdict. Aggregate by simple weighted vote (or unsupervised agreement weighting). Compare selection accuracy to (a) any single signal alone and (b) a single flagship verifier. Prediction: the aggregate beats each weak signal and approaches the flagship verifier at lower cost, *only if* the signals' errors are decorrelated (arXiv 2509.17995), which the experiment measures directly via inter-verifier error correlation.
- Needs: the C1-C3 signal harnesses reused; no download. Effort: days (build on C1-C3). Class: extension, run after C2/C3 land.

## Feasibility notes and pitfalls

- **Everything here is runnable today.** No candidate needs fine-tuning, a download, or cloud. The trained-verifier results (PRM in "Let's Verify Step by Step", SCoRe's RL) are cited as the ceiling and are explicitly out of scope; our runnable analog is the prompted GenRM.
- **Swap cost is the hidden tax.** Any cross-model candidate (C2, C4, partly C5) pays llama-swap teardown/load latency per model change. Measure it once and include it in every wall-clock number; batch all judgments for one model before swapping, do not interleave.
- **Self-preference is real; never let a model be its own final judge** (section 2, arXiv 2404.13076). C2 and C4 are built around cross-model judging for this reason.
- **The small verifier is weak by default** (arXiv 2404.17140). C3 uses it as a router, not an arbiter; if used as an arbiter it will flip correct answers, which is the specific trap to avoid.
- **De-bias the judge harness:** randomize candidate order and average both directions, normalize formatting, cap length in the judge prompt (section 2). A judge that looks good but is just consistently wrong is the "reliability without validity" failure (arXiv 2606.19544); validate against a small labeled set first.
- **Report the honest headline.** Every candidate's result is quality *per wall-clock-second* against budget-matched self-consistency, not raw accuracy. A loop that raises accuracy but costs 9x the wall clock must be compared to what 9x more sampling-plus-voting would have bought, or the claim is illusory in exactly the way section 1's literature warns about.

## Feasibility verdicts

Adversarial pass (2026-07-20). Test applied to each: fits 8 GB VRAM / 48 GB RAM / AVX2 / Windows / single-machine hub; runnable TODAY (not paper-only); honestly costs zero per-token speed (flag hidden concurrency or retraining); effort honest. Structural fact that drives most verdicts: **llama-swap serializes models** (it is a swap router, not a concurrent multi-model server), so no candidate actually runs two large models at once. The real taxes are (a) swap latency on every cross-model handoff and (b) any design that needs two models co-resident, which pushes more of the 30B's experts onto CPU and *does* lower its per-token decode. The doc is honest about (a) throughout; (b) is the one place a candidate's premise is at risk.

- **C1 (tool-grounded coding critique / CRITIC): GO.** Single resident model, sandbox is a sequential subprocess (LLM idle during execution, so zero bandwidth contention and zero per-token hit), no swap, no download if the task set is home-grown. This is the known-good pattern with the largest verification asymmetry. Effort "hours to a day" is honest for a timeout-guarded runner plus a small task set. Only caveats: true OS-level sandboxing is weaker on Windows than Linux, so run the abliterated-coder output under a timeout/resource-capped subprocess on a throwaway task set, not arbitrary generated code with network/file access; if HumanEval is used it is a trivial download, not "no download."

- **C2 (cross-model GenRM best-of-N, 30B gen + 35B-reasoning judge): GO.** Two large models cannot co-reside, so this is necessarily swap-serialized: generate all N with the 30B, swap once, judge all N with the 35B. The doc mandates exactly this batching and charges swap cost, so no hidden concurrency and no secret per-token hit (each model runs at native speed while resident). Runnable today (GenRM is a prompt, not a trained head). Effort "days" is honest. Caveat to enforce: the 35B is reasoning-native and may not honor "short critique," emitting long CoT before each verdict and inflating verify tokens several-fold; this is not a blocker because the mandated per-wall-clock accounting will expose it, but budget for it.

- **C3 (cheap-first cascade, Qwen3-8B router + flagship fallback): MAYBE.** Runnable and a legitimate test (a clean negative is publishable per 2404.17140), but its entire reason for existing, wall-clock *savings*, rests on an unresolved premise. Invoking the 8B is not free on this hub: either you swap 30B->8B->30B per batch (the swap tax can dwarf the 8B's 50 t/s advantage and the cascade goes net-negative on wall clock), or you co-reside 8B (~5 GB) with the 30B, which leaves the 30B only ~3 GB of VRAM, forces more experts to CPU, and *lowers the 30B's per-token decode* (the exact hidden speed cost we screen for). The doc's generic swap warning does not resolve which path C3 takes. Verdict stays MAYBE until the 8B-invocation path is pinned down and measured; as a quality/negative experiment it is sound, as a wall-clock-saver it is unproven.

- **C4 (heterogeneous vs homogeneous debate vs self-consistency): GO.** No candidate here needs concurrency: the homogeneous "3 Qwen3-30B instances" are operationally three sequential prompts/histories against one resident model (no extra VRAM, no swap), and self-consistency is one model. Only the heterogeneous arm (30B + 35B + GLM) pays a swap per round, which the doc flags honestly and treats as the thing under test. No secret per-token hit. Effort "days" honest. High value precisely because a null result (Smit et al. predicts homogeneous <= self-consistency) retires a whole class of orchestration.

- **C5 (Weaver-lite diverse weak-verifier ensemble): GO, contingent.** Honestly scoped: the training/distillation half of Weaver (the 400M cross-encoder) is explicitly dropped, leaving a scriptable aggregation of signals, so no retraining. Reuses C1-C3 harnesses, batched per model to amortize swaps, no co-residence, no download. Effort "days, build on C1-C3" is honest and it is correctly ordered last. It inherits C3's unresolved 8B-invocation swap caveat for the 8B-judge signal, so its cost accounting is only trustworthy once C3's path is settled; the science (measuring inter-verifier error decorrelation) is valid regardless.
