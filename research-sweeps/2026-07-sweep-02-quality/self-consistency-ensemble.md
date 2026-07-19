# Sampling-based quality: self-consistency, best-of-N, and the wall-clock economics on our rig

Survey date: 2026-07-19. Scope: the family of test-time methods that spend extra tokens on the same model to raise answer quality without any retraining. Self-consistency (majority vote over N sampled chains), best-of-N with a scorer, universal self-consistency (LLM-as-selector for free-form text), sample-set-then-vote / weighted voting, execution-based selection for code, and the temperature/sample-count tradeoffs that govern all of them. Everything here is pure orchestration on top of our OpenAI-compatible hub (llama-swap at :9292); none of it needs a new llama.cpp feature, and all of it is runnable today with scripts we write.

The critical reframing for this project: our decode throughput is pinned by memory bandwidth at ~30-42 t/s and these methods cannot raise it. What they change is the number of tokens spent per answered query. So the only honest metric is **quality gained per extra wall-clock second** (and, for a fixed deadline, quality at a fixed token budget). Sampling N chains does not slow the model down; it multiplies the time-to-answer, and the multiplier is NOT simply N once continuous batching enters the picture (section 10). That batching correction is the single most important rig-specific finding in this survey, and it is gated by the same expert-scattering parameter `s` that sweep-01's R3 already measures.

Rig constraints applied throughout: i7-14650HX (AVX2 only), 48 GB DDR5-5600 (~37-42 GB/s measured effective per E021-E028), RTX 5060 Laptop 8 GB, Windows 11, llama.cpp b10064/b10068 via llama-swap. Live stack: Qwen3-30B-A3B-Instruct-2507 (~33-42 t/s), Qwen3.6-35B-A3B reasoning (37-40 t/s), Qwen3-8B (~50 t/s), plus abliterated/vision variants. All hub models currently launch with `-np 1` (single slot), which section 10 shows is the first thing to change.

---

## 1. The method family and the one distinction that matters

All of these methods draw K independent samples from one model at temperature T > 0, then reduce the K candidates to one answer. They differ only in the reducer:

| Method | Reducer | Needs | Works on free-form? |
|---|---|---|---|
| Self-consistency (SC) | majority vote over extracted answers | answer extractor | no (fixed-answer only) |
| Weighted SC / weighted vote | vote weighted by a score | scorer/verifier | no |
| Best-of-N (BoN) | argmax of a scorer | reward model or proxy | yes |
| Universal SC (USC) | ask the LLM to pick the most consistent | one extra LLM call | yes |
| Execution agreement (CodeT) | consensus over generated tests | a way to run code | code only |
| Coverage / pass@k | any sample is correct | an oracle verifier | verifiable only |

The load-bearing distinction is **verifier vs no verifier**. With a cheap correct-answer oracle (unit tests, a formal checker, a math grader), drawing more samples is nearly free money: quality tracks *coverage*, the fraction of problems some sample solves, which keeps rising for thousands of samples ([Large Language Monkeys, Brown et al. 2024, arXiv 2407.21787](https://arxiv.org/abs/2407.21787)). Without a verifier you can only *select*, and selection quality saturates fast (sections 3-4). Our rig has real verifiers for exactly one domain (code, via execution) and cheap proxies elsewhere (self-consistency vote, self-certainty). That gap decides which candidate below is a likely win versus a likely wash.

---

## 2. Self-consistency: the canonical result and its actual settings

[Self-Consistency Improves Chain of Thought Reasoning (Wang et al. 2022, arXiv 2203.11171)](https://arxiv.org/abs/2203.11171) is the origin. Sample a diverse set of reasoning chains at T > 0, extract each chain's final answer, take the plurality. Reported gains over greedy CoT (PaLM-540B unless noted):

- **GSM8K +17.9%** (to ~74.4% at 40 paths)
- **SVAMP +11.0%**, **AQuA +12.2%**, **StrategyQA +6.4%**, **ARC-challenge +3.9%**

Settings that matter for reproduction: temperature **T=0.5-0.7 with top-k=40** (T=0.5 for the smaller LaMDA/UL2 models, T=0.7 for PaLM/GPT-3), robust to nucleus (top-p) sampling. Default **N=40 paths**; the ablation swept N ∈ {1, 5, 10, 20, 40}. The curve rises steeply from 1 to ~20 paths and flattens after; the bulk of the gain is present by N=10-20. The paper documents **no case where SC hurt** on these fixed-answer reasoning tasks, and the smallest gains were on symbolic tasks (coinflip +0.5%) where a single chain is already near-deterministic. That "no harm" claim is specific to weak-to-mid 2022 models on hard-for-them tasks; section 4 shows it breaks on strong 2025-2026 models.

Temperature is the hidden control. At T=0 all samples are identical and voting does nothing; too high and each chain is individually incoherent so you vote among bad reasoners. The consensus operating band for math SC is **T≈0.5-0.8**. This is a knob we set per-request through the hub API, so it costs nothing to sweep.

---

## 3. How N trades against accuracy: the scaling curves

Two different quantities scale very differently with N, and conflating them is the classic error.

**Coverage (with an oracle) keeps climbing.** Brown et al. 2024 (arXiv 2407.21787) show coverage/pass@k is log-linear in N over *four orders of magnitude*, fit by an exponentiated power law. Concrete: DeepSeek-Coder-V2 on SWE-bench Lite goes **15.9% (1 sample) -> 56% (250 samples)**; Llama-3 coverage on GSM8K/MATH exceeds **95% at 10,000 samples**. But they are explicit that turning coverage into accuracy without a verifier "remains an important direction for future research." Coverage is the ceiling BoN could reach with a perfect selector, not what majority vote delivers.

**Selection accuracy (majority vote, no oracle) saturates early.** The consistent finding across the literature: most of the gain lands by **N=5-10**, the curve is nearly flat by **N=20-40**, and per-sample accuracy improvement drops below ~0.5% well before N=20 on tasks where the model is already competent. [Inference Scaling Laws (Wu et al. 2024, arXiv 2408.00724)](https://arxiv.org/abs/2408.00724) prove why: in the infinite-compute limit, majority voting converges to a fixed point set by the *generation policy itself*, so it cannot exceed the model's own most-probable answer distribution. More votes cannot fix a model that is confidently wrong on a problem.

Practical read for us: on fixed-answer reasoning, budget **N≈8-16** as the sweet spot; N=32+ is almost always wasted wall clock unless a verifier is turning coverage into accuracy.

---

## 4. The 2025-2026 correction: SC is losing its edge on strong models

This is the most important recent shift and it directly governs whether SC is worth running on our current-generation stack. [Self-Consistency Is Losing Its Edge (arXiv 2511.00751, Nov 2025)](https://arxiv.org/html/2511.00751) re-runs SC on frontier models and finds the returns have collapsed and can go *negative*:

- **HotpotQA** (Gemini-2.5-Flash-Lite): 1 -> 20 samples buys **+0.4%**, and accuracy fluctuates irregularly rather than rising. That is a 20x token cost for 0.4%.
- **MATH-500** (same model): accuracy **peaks near N=10 then declines beyond N=15** - genuine degradation, not just flat, because extra samples inject spurious errors the aggregator cannot suppress on problems already solved in one pass.
- **MATH-500** (Gemini-2.5-Pro): **98% -> 99.6% at N=15** - a 15x cost for 1.6% off an already-excellent baseline.

Their recommendation: SC should be "a targeted tool rather than an automatic one," reserved for problems that *demonstrably exceed the model's single-pass reliability*. A complementary failure mode: [Self-Consistency Falls Short on long-context (arXiv 2411.01101)](https://arxiv.org/html/2411.01101) shows SC not only fails to help but *degrades* long-context tasks (positional bias), with 8 -> 16 samples buying <1%.

The counter-evidence keeps SC alive for genuinely hard problems: DeepSeek-R1-Zero on AIME 2024 improves from single-pass to **86.7% with 16-vote majority** (a large, real gain), and SC remains a lightweight scaling lever for both weak and strong models on tasks at the edge of their ability. The synthesis: **the harder the problem is for the specific model, the more SC pays; the easier it is, the more likely SC is a wash or a net loss.** This is a per-task, per-model gate, not a universal on/off, and it is exactly what a candidate experiment should map for *our* models.

---

## 5. Beyond majority vote: best-of-N, weighted voting, verifiers, self-certainty

Once you have a scorer, you can beat plain majority vote. The ranking that emerges from the literature:

- **Weighted majority vote > vanilla best-of-N.** Averaged across tasks, picking the single highest-scored trace (BoN) *lags* verifier-weighted voting by **>10 points at equal decode cost**, because a single high-confidence trace is brittle; weighting every vote by its score is more robust and also breaks ties BoN and plain voting cannot ([Inference Scaling Laws, arXiv 2408.00724](https://arxiv.org/abs/2408.00724); confirmed in the weak-verifier line, [arXiv 2506.18203](https://arxiv.org/html/2506.18203v1)). As long as the scorer is "better than random," weighted voting's ceiling is strictly above plain voting's.
- **Compute-optimal beats naive BoN by ~4x.** [Scaling LLM Test-Time Compute Optimally (Snell et al. 2024, arXiv 2408.03314)](https://arxiv.org/abs/2408.03314) shows a per-prompt strategy that routes easy prompts to sequential revision and hard prompts to parallel sampling is **>4x more token-efficient than a best-of-N baseline** at equal accuracy, and can let a small model + test-time compute beat a 14x-larger model on easy/medium problems. Difficulty-adaptive allocation is the lever, not raw N.
- **REBASE tree search is Pareto-optimal** over weighted voting and MCTS, hitting equal accuracy with up to **7x less compute** on a 7B model (arXiv 2408.00724). This is a reward-model-guided search; it is *paper/serving-stack* territory for us because it needs a process reward model we do not have.

The catch for our rig: **we have no trained reward/verifier model** and cannot fine-tune one. So the actionable variants are the *verifier-free* ones:

- **Self-certainty / confidence-based BoN** ([Scalable Best-of-N via Self-Certainty, arXiv 2502.18581](https://arxiv.org/html/2502.18581v1)) selects using the model's own token-level confidence, no reward model, and reportedly scales like BoN. We get logprobs from the hub, so this is measurable.
- **Weighted SC using self-certainty as the weight** is the natural verifier-free upgrade to plain majority vote and is a clean A/B against it.

---

## 6. Universal self-consistency and free-form / open-ended tasks

Plain SC needs an answer extractor, so it dies on free-form output (summaries, explanations, open QA, most chat). [Universal Self-Consistency (USC, Chen et al. 2023, arXiv 2311.17311)](https://arxiv.org/abs/2311.17311) fixes this by concatenating all K sampled responses into one prompt and asking the *same LLM* to pick the most consistent one. No answer-format constraint, no separate scorer, one extra generation call.

Reported behavior: USC **matches standard SC on math** (without needing matching answer formats) and **extends the benefit to code generation, long-context summarization, and open-ended QA** where SC is not even applicable. It is the most directly useful ensemble method for our chat/assistant workloads because it needs nothing but the API.

Costs and cautions specific to us:
- The selector call must fit all K candidates in context. K=8 answers of ~600 tokens is ~5k tokens of context plus the prompt - fine at our 8k-16k configs, but it grows fast and collides with long candidates. This is a real ceiling on our 8 GB KV budget.
- USC inherits the [positional-bias problem (arXiv 2411.01101)](https://arxiv.org/html/2411.01101): the model over-weights candidates by position in the concatenation. Shuffling candidate order across a couple of selector calls is a cheap mitigation worth testing.
- Related 2025-2026 work (soft/representation-space consistency, MBR-style selection, [arXiv 2410.02902](https://arxiv.org/pdf/2410.02902)) generalizes the idea but adds machinery; USC is the minimal runnable version and the right first probe.

---

## 7. Code generation: the one domain where coverage becomes accuracy for free

Code is special because *we can run it*. Execution turns the unbounded coverage of section 3 into real selected accuracy without any trained verifier. [CodeT (Chen et al. 2022, arXiv 2207.10397)](https://arxiv.org/abs/2207.10397) has the model generate both candidate solutions *and* test cases, executes every solution against every test, and uses "dual execution agreement" (a RANSAC-style consensus: solutions that pass the same set of tests form a consensus set, scored by solutions x tests). It beats AlphaCode-style output clustering consistently on HumanEval/MBPP and lifts pass@1 substantially over single-sample generation.

Why this is the strongest candidate domain for us:
- The scorer is *ground truth-ish* (actual execution), not a proxy that saturates.
- We already run a coding stack (GLM-4.7-Flash, Qwen3-Coder-30B) and can execute generated Python/JS locally in a sandbox we write.
- It maps cleanly onto the section-10 batching win: generate K solutions concurrently, execute them (near-free CPU), select by test agreement.

The honest limit: it only works where generated tests are meaningful (self-contained functions, scripts with checkable I/O), not for whole-repo agentic edits where "does it run" is not a tight signal. And test generation can itself be wrong, so consensus over tests (CodeT) beats trusting any single generated test.

---

## 8. Adaptive sample counts: stop early, spend where it helps

The fixed-N schedule is wasteful because easy queries are decided after 2-3 agreeing samples while hard ones never converge. The adaptive line fixes this and is directly implementable as hub orchestration:

- **Adaptive-Consistency ([arXiv 2305.11860](https://arxiv.org/abs/2305.11860))**: after each sample, evaluate a lightweight stopping rule on the running answer distribution (a Dirichlet/beta posterior on whether the current plurality will hold). Stop when confident. Across **17 reasoning + code datasets and 3 LLMs**, this cut the sample budget by up to **7.9x with <0.1% average accuracy drop**. This is the highest-leverage, lowest-risk technique in the whole survey for us.
- **Early-Stopping SC / Difficulty-Adaptive SC ([arXiv 2408.13457](https://arxiv.org/abs/2408.13457))**: window-based early stop, plus using a cheap difficulty estimate to set the initial budget so easy questions get one shot.
- **Reasoning-Aware / Reliability-Aware SC ([arXiv 2408.17017](https://arxiv.org/html/2408.17017v1), [arXiv 2601.02970](https://arxiv.org/abs/2601.02970))**: gate additional samples on reasoning-path quality / evidence sufficiency rather than raw count.

For our fixed-throughput rig this is the difference between "SC costs 8x wall clock always" and "SC costs 8x only on the ~20% of queries that actually need it, ~1.5-2x on average." Early stopping is what makes SC economically defensible here at all.

---

## 9. Which tasks benefit, which don't

Synthesizing the above into a decision table for our workloads:

| Task type | Verifier available? | SC/ensemble payoff | Note |
|---|---|---|---|
| Grade-school / competition math (GSM8K, AIME) hard-for-model | grader (exact match) | high | Wang 2022; R1 AIME +to 86.7% at 16 votes |
| Math the model already aces | grader | low / negative | arXiv 2511.00751: peaks then declines |
| Code (self-contained functions) | execution | high | CodeT; coverage -> accuracy for free |
| Agentic multi-file code edits | weak (runs?) | low-medium | test signal too loose |
| Multiple-choice / classification | exact match | medium | vote helps; cheap |
| Open-ended QA / explanation / chat | none (USC only) | low-medium | USC extends reach; positional bias |
| Long-context QA / summarization | none | negative | arXiv 2411.01101: SC degrades it |
| Factual single-hop lookup | none | ~zero | one confident pass is the answer |
| Creative / subjective writing | none | ~zero | no "correct" answer to converge on |

The pattern: **payoff is high exactly where (a) the model is near the edge of its ability AND (b) a real or execution-based verifier exists.** It fades to zero or negative on easy tasks, long context, and open-ended generation. This is the filter every candidate below is designed to measure on *our specific models*, because the 2511.00751 result means we cannot assume the 2022 gains transfer.

---

## 10. The wall-clock economics on our rig: the batching escape hatch

This is the decision engine, the analogue of sweep-01's expert-union section. Every method above spends K generations. The question is what K costs in wall-clock seconds on *this* rig.

**Naive (sequential, one slot).** All our hub models launch with `-np 1`. Firing K samples back-to-back is K full generations. A ~600-token math CoT at 37 t/s is ~16 s; K=8 is ~130 s per query versus ~16 s single-pass. At `-np 1`, SC's cost multiplier is exactly K, and the quality-per-second math is brutal: on a task where SC buys +5% for K=8, that is +5% for 8x the wait.

**The escape hatch: continuous batching (`-np K`).** llama.cpp continuous batching merges the decode step of multiple slots into a single forward pass. During decode each sequence contributes one token, so a batch of B sequences turns B matrix-vector products into one matrix-matrix product - the weights are read from memory *once* and reused across all B tokens. For memory-bandwidth-bound decode this raises *aggregate* throughput sharply (community numbers show ~3-3.7x aggregate for small models at B=16), with the gain tapering as bandwidth saturates. K self-consistency samples are B=K identical-prompt sequences: the perfect batching case (shared prefix prefill too).

**How much do we actually gain? The same `s` that governs speculative decoding.** For our MoE with routed experts on CPU, batching B tokens at one position touches a *union* of experts across the batch: `U(B) = E_tok * (1 + (B-1)*s)`, capped at `E_total` (128 for the 30B, 256 for the 35B), where `s` is the expert-scattering fraction sweep-01's R3 measures. The dense/shared parts (attention, shared expert, router, embeddings, all on GPU) amortize perfectly across the batch. So:
- Low `s` (experts overlap across the K samples of one prompt - plausible, since same prompt = similar routing): expert bytes barely grow with K, and aggregate throughput scales nearly linearly. SC's real cost multiplier drops toward ~1-2x instead of K.
- High `s` (disjoint experts): expert term grows ~linearly with K, so only the dense/shared term amortizes; aggregate gain is modest but still > 1x.

Either way the true cost of K samples is **substantially less than K x** once `-np` is raised, and the exact number is a direct, cheap measurement that reuses R3. **This single measurement reprices every technique in this survey.** If low-`s` batching gives, say, 3x aggregate throughput at K=8, then SC's effective cost is ~2.7x wall clock, not 8x, and the quality-per-second verdict flips for a whole class of tasks.

**The VRAM cost of parallelism.** Each slot needs its own KV cache. With a fixed `-c`, context is split across slots (`-c 8192 -np 8` -> 1024 tokens/slot). On our 8 GB card this caps K x context. For short-answer math SC (a few hundred tokens/sample) this is a non-issue; for long CoT or USC over long candidates it is the binding constraint and forces a K-vs-context tradeoff we must measure. q8_0 KV (E040/R1) doubles the room.

**Prefill is shared, and that helps.** K samples share one prompt, so continuous batching prefills the shared prefix once. On long prompts (our expensive CPU-bound prefill) this is a real saving that pure sequential sampling throws away.

Bottom line: the honest headline for any sampling-quality result on this rig must be **quality gained per extra wall-clock second at the best measured `-np`**, not per extra sample. Reporting SC as "8x cost" when batching makes it ~2.7x would understate it by 3x.

---

## Candidate experiments for our rig

Ordered by expected value per unit effort. All are zero-download orchestration on the existing hub unless noted.

### C1. Parallel-sampling throughput law: the `-np` sweep that reprices everything (gate for C2-C5)
Measure aggregate decode t/s on the flagship Qwen3-30B-A3B at `-np` ∈ {1, 2, 4, 8, 16} firing that many identical-prompt completions concurrently, at context splits that keep each slot viable (start `-c 16384`, short 256-token answers). Report aggregate t/s, per-stream t/s, and VRAM. Cross-reference R3's expert-scattering `s`: predict aggregate scaling from `U(B)` before measuring, then falsify. This converts "K samples = K x wall clock" into the real cost multiplier and is the referee for every candidate here. Also a first public MoE-experts-on-CPU batching-scaling datapoint.
- **First step:** relaunch `qwen-30b` with `-np 8 -c 16384`, script 8 concurrent `/v1/completions` with the same prompt, log aggregate vs single-stream t/s.
- **Success:** a measured cost-multiplier curve; low-`s` linear scaling green-lights C2-C5 as cheap, high-`s` caps expectations. **Effort:** hours. **Class:** incremental (but gates the lane).

### C2. Self-consistency quality-per-second frontier on math/reasoning, with adaptive early stop
On Qwen3-30B-A3B and the Qwen3.6-35B reasoning model, run SC at T=0.6, K ∈ {1,4,8,16,32} on a hard-for-our-model set (a GSM8K-hard slice, a MATH-500 subset, an AIME slice), majority vote, plot accuracy vs K AND accuracy-gain per wall-clock second using C1's real cost. Then add the Adaptive-Consistency stopping rule ([arXiv 2305.11860](https://arxiv.org/abs/2305.11860)) and measure the average-K reduction at matched accuracy. This is the direct test of whether the 2022 SC gains survive on our 2026-generation models (the arXiv 2511.00751 warning) and where the per-second frontier actually peaks here.
- **First step:** build the graders + a K-sample vote harness against the hub; run K=8 on 100 GSM8K-hard items, both models.
- **Success:** a per-model, per-difficulty map of where SC clears +X% at acceptable seconds/query, plus the adaptive-K schedule that gets it for ~1.5-2x average cost. **Kill:** no task slice clears +3% even at K=16. **Effort:** days. **Class:** quality-per-second frontier.

### C3. Execution-based selection for code (CodeT-style), the highest-ceiling verifier we actually have
On GLM-4.7-Flash and Qwen3-Coder-30B, generate K solutions + K generated tests for self-contained function tasks (HumanEval/MBPP-style and a few of our own), execute all pairs in a local sandbox we write, select by dual-execution agreement ([CodeT, arXiv 2207.10397](https://arxiv.org/abs/2207.10397)). Compare pass@1-selected vs single-sample and vs plain vote. This is the one domain where coverage (section 3) converts to real accuracy for free, and it rides C1's batching win directly (K solutions in parallel, execution is near-free CPU).
- **First step:** sandbox runner for generated Python; K=8 solutions + tests on 50 HumanEval problems via `qwen-coder`.
- **Success:** selected pass@1 clearly beats single-sample at a wall-clock cost C1 shows is < K x. **Kill:** agreement selection no better than first-sample. **Effort:** days. **Class:** quality-per-second frontier (coding).

### C4. Universal self-consistency for free-form + verifier-free best-of-N via self-certainty
Two verifier-free selectors on open-ended and chat-style tasks where majority vote does not apply. (a) USC ([arXiv 2311.17311](https://arxiv.org/abs/2311.17311)): concatenate K candidates, ask the model to pick the most consistent, with candidate-order shuffling to test the positional-bias trap ([arXiv 2411.01101](https://arxiv.org/html/2411.01101)). (b) Self-certainty BoN / self-certainty-weighted vote ([arXiv 2502.18581](https://arxiv.org/html/2502.18581v1)) using hub logprobs, since we have no trained reward model. Judge on a held-out win-rate (LLM-judge or task-specific check) vs single-pass.
- **First step:** USC selector prompt + a K=6 harness on 50 open-ended QA / explanation items; log win-rate and position sensitivity.
- **Success:** USC or self-certainty beats single-pass win-rate by a clear margin at C1-priced cost. **Kill:** no selector beats single-pass, or positional bias dominates. **Effort:** days. **Class:** capability (quality on free-form, where SC cannot reach).

### C5. Difficulty-adaptive compute allocation across the stack (stretch, gated on C2)
Combine C2's adaptive-K with a cheap first-pass difficulty/confidence estimate to route per query: one shot for easy, K-sample SC for hard, USC/execution for the right task type. This is the local, verifier-free shadow of Snell et al.'s compute-optimal allocation ([arXiv 2408.03314](https://arxiv.org/abs/2408.03314)) - not the process-reward-model version (paper-only for us), but a difficulty-gated budget we can actually build. Payoff is the whole-hub quality-per-second improvement at a bounded average cost multiplier.
- **First step:** define the difficulty proxy (single-pass self-certainty or vote-entropy on a K=3 probe), wire the router, replay C2/C3 sets.
- **Success:** matched or better quality than fixed-K SC at materially lower average seconds/query across a mixed workload. **Effort:** week+. **Class:** quality-per-second frontier (system-level).

### Dead ends and paper-only, for the record
- **Trained process/outcome reward models and REBASE-style verifier search** ([arXiv 2408.00724](https://arxiv.org/abs/2408.00724)): the strongest scaling results (7x compute efficiency, Pareto-dominant search) need a reward model we cannot train or fit. Verifier-free proxies (self-certainty, execution) are our ceiling.
- **Cross-model ensembling via the hub:** llama-swap holds one model resident at a time; a true multi-model vote means serial model reloads (tens of seconds each) that dwarf any quality gain. Diverse *sampling from one model* is the only practical ensemble here. Re-open only if two small models ever fit resident simultaneously.
- **SC on easy tasks, long-context, and factual lookup:** arXiv 2511.00751 and 2411.01101 predict wash-to-negative; do not spend the batch there. C2/C5's difficulty gate exists precisely to route these to a single pass.
- **Massive-N coverage chasing (N in the hundreds/thousands):** Brown et al.'s coverage curves need an oracle to cash in; without one, selection saturates by N≈16 (section 3). Our N ceiling is set by selection quality, not by what we can afford to sample.

---

## Feasibility verdicts

Adversarial pass against: fits 8 GB VRAM / 48 GB RAM / AVX2 / Windows / single-machine-hub; runnable today (not paper-only); honestly zero per-token speed cost (no secret concurrent models, no retraining); effort honest.

- **C1 (`-np` throughput sweep) — GO.** Pure measurement on standard llama.cpp continuous batching; `--parallel` composes with CPU expert offload (`--n-cpu-moe`/`-ot`), both already in use. VRAM is a non-issue: `-c 16384` KV is fixed regardless of `-np` (context splits across slots), and E040/R1 already runs 64k with q8 KV. No per-token cost: it reports per-stream AND aggregate t/s, so it does not pretend batching raises single-request speed. Honest caveat, not a blocker: the low-`s` linear-scaling win is genuinely uncertain for CPU-offloaded MoE (CPU expert GEMM may be compute- not bandwidth-bound, so batching may amortize less than GPU-resident decode) — but falsifying exactly that is the experiment. Effort "hours" is honest. This is the correct gate for the whole lane.
- **C2 (SC quality-per-second + adaptive stop) — GO.** Pure hub orchestration; exact-match math graders and a vote harness are trivial; GSM8K/MATH-500/AIME are small downloads (fine at ~3 MB/s); Adaptive-Consistency (2305.11860) is a lightweight posterior stopping rule, not a trained model. Both models tested serially, no concurrency, no retraining. Metric is honestly quality-per-wall-clock-second at C1's real multiplier. Effort "days" honest (building the hard-for-our-model slice adds a first-pass filtering step, still within days).
- **C3 (CodeT execution selection) — GO.** The one real verifier we own; HumanEval/MBPP are small; models reached via serial llama-swap, not concurrently. Execution is local CPU, near-free, zero per-token cost. Real effort/risk the doc underplays: a SAFE sandbox for model-generated code on Windows is non-trivial (no seccomp/firejail) — do it via WSL or a container with hard timeouts and no network/FS access, which nudges effort to the high end of "days." Minor: "Qwen3-Coder-30B" naming — confirm it maps to a resident coder in the stack (abliterated coder / GLM-4.7-Flash). Still GO.
- **C4 (USC + self-certainty BoN) — GO, with one dependency to verify first.** USC is pure API (concatenate-and-select), solidly runnable and the right free-form probe. The self-certainty half hinges on the hub returning usable per-token logprobs: llama-server supports `logprobs`/`n_probs` (top-k only), and self-certainty (2502.18581) ideally wants the full-vocab distribution, so top-k is an approximation — verify logprobs actually pass through llama-swap's proxy before trusting that half. LLM-judge win-rate is subjective/noisy (bias if the judge is the same model), a methodology caveat not a feasibility one. No concurrency, no retraining. Effort "days" honest.
- **C5 (difficulty-adaptive allocation) — MAYBE.** Buildable orchestration, honestly labeled stretch / "week+", but doubly contingent: (1) its payoff evaporates if C1 shows high-`s` (near-Kx cost) or C2/C3 wash — there is nothing to route to; and (2) "USC/execution for the right task type" routing across DIFFERENT models on a mixed stream triggers exactly the serial llama-swap reloads (tens of seconds each) the doc's own dead-ends section says dwarf any quality gain. It is only safely feasible as difficulty-adaptive K WITHIN one resident model; cross-model task routing reintroduces the swap tax. Feasible but conditional — hence MAYBE, not GO.

No candidate rates KILL: the genuine kills (trained reward models/REBASE, simultaneously-resident cross-model voting, SC on easy/long-context/factual, massive-N without an oracle) are already correctly quarantined in the dead-ends list above.
