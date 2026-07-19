# Inference-time reasoning-scaling techniques: quality per wall-clock second

Survey date: 2026-07-20. Sweep 2026-07-sweep-02-quality. Scope: every inference-time technique that raises answer QUALITY with NO retraining and NO fine-tuning, implementable by orchestrating our OpenAI-compatible hub (:9292) with PowerShell/Python we write. Techniques covered: extended/iterative chain-of-thought, budget forcing (s1-style "Wait"), self-consistency / majority vote, best-of-N + verifier and compute-optimal allocation, self-refine / reflexion loops, plan-then-execute / plan-and-solve / least-to-most, tree-of-thoughts / graph-of-thoughts, and the 2025-2026 test-time-compute frontier (overthinking / inverse-U, adaptive budgets, parallel thinking).

## The framing that governs every number below

Our decode throughput is pinned by memory bandwidth at ~30-42 t/s and CANNOT be raised by any technique in this survey. So the reported paper metric (accuracy at pass@k, human-preference win rate) is only half the story. The other half is **time**. Every technique here spends extra tokens, and extra tokens cost wall-clock seconds at a fixed t/s. The objective is therefore:

- **maximize quality per extra wall-clock second**, and
- **maximize quality at a fixed token budget** (equivalently, at a fixed time-to-answer).

A technique that lifts accuracy 5 points but costs 8x the tokens is a worse buy than one that lifts 3 points at 2x, unless the task genuinely justifies the wait. Rankings at the end are by quality-gained-per-extra-second on OUR rig, not by headline accuracy.

## The single most important rig-specific fact: "parallel" is not free here, but it is sublinear

Most of the literature assumes a datacenter where N independent samples run on N GPUs at once, so best-of-N and self-consistency are "free latency, N x cost." We have ONE memory-bound GPU with `--cpu-moe`, so N samples do NOT run at zero added latency. But they are also NOT a flat Nx wall-clock penalty, and this is the crux of everything below:

- llama.cpp server exposes `--parallel N` slots with continuous batching. Decode on our MoE is bound by streaming weights/experts from the ~60 GB/s RAM bus, and a batched step amortizes each expert read across all sequences that touch that expert. So 4 sampled reasoning traces in 4 slots should finish in FAR less than 4x the wall clock of one trace.
- The counter-force is the **expert-union penalty we already quantified in E014/E023**: different sampled sequences route to different experts, so a batched step may fetch the union of their top-k experts, not the intersection. The amortization is real but partial, and the exact factor is an empirical question for THIS model, not a number to assume.

Net: on our rig, parallel-sample methods (self-consistency, best-of-N) may be dramatically cheaper per unit quality than the naive Nx model predicts, and the size of that discount is itself a publishable measurement (candidate C1 below). Sequential methods (longer CoT, budget forcing, refine loops) get no batching discount at all: their token cost is their time cost, one-for-one.

---

## 1. Extended / iterative chain-of-thought and budget forcing (s1)

**What it is.** Force the model to keep thinking past its natural stopping point. The s1 recipe ([Muennighoff et al 2025, arXiv 2501.19393](https://arxiv.org/abs/2501.19393)) does this with **budget forcing**: suppress the end-of-thinking token and append the literal string "Wait" so the model second-guesses and continues, or hard-cap the think block to truncate it.

**Reported gain.** s1-32B (a fine-tune of Qwen2.5-32B on 1,000 traces) exceeds o1-preview by up to 27% on competition math (MATH500 / AIME24), and budget forcing alone extrapolates the model **from 50% to 57% on AIME24** without any other change. Their ablation reports sequential budget forcing beating parallel majority voting in their setup, up to a saturation point.

**Token/time multiplier.** Directly proportional to Waits appended: each "Wait" adds another full think segment (often 500-2,000 tokens = 15-60 s on our rig). No batching discount (it is one long sequence).

**The critical caveat for us.** Two independent results say this is the WEAKEST buy on a reasoning-native model:

1. **It saturates and oscillates.** The follow-up analysis ["It's Not That Simple" (Wu 2025, arXiv 2507.14419)](https://arxiv.org/abs/2507.14419) finds the apparent scaling "is largely attributed to scaling down by enforcing a maximum length," that appending "Wait" makes the model "oscillate between solutions," and that budget forcing "progressively imposes a lower upper limit on model performance as it scales." It mimics the shape of scaling without unlocking capability the model did not already have.
2. **We already own the real thing.** Our Qwen3.6-35B-A3B is RL-trained to produce long CoT natively. Budget forcing is a bolt-on approximation of exactly what our reasoning model does by construction. Bolting "Wait" onto the non-reasoning 30B-Instruct is the only version worth testing, and the prior says it underperforms the native reasoner.

**Verdict vs. "just let the 35B think longer":** Budget forcing loses. The honest lever on a reasoning-native model is not "more Waits," it is **tuning the thinking budget to the difficulty** (section 7). Forcing MORE thinking on our model risks the overthinking regime, not gains.

---

## 2. Self-consistency / majority vote (parallel sampling)

**What it is.** Sample K independent CoT traces at temperature > 0, extract the final answer from each, return the majority ([Wang et al 2022, arXiv 2203.11171](https://arxiv.org/abs/2203.11171)). No verifier, no training, trivially orchestratable against our hub.

**Reported gain.** +17.9% GSM8K, +11.0% SVAMP, +12.2% AQuA, +6.4% StrategyQA, +3.9% ARC-challenge over single-path CoT. These are large, robust, and among the most reproduced numbers in the field.

**Token/time multiplier.** K x tokens nominally, BUT this is the one method that collects the llama.cpp batching discount (section top). Realistic on our rig: K=5 traces in 5 parallel slots may cost ~2-3x wall clock, not 5x. That is what makes its quality-per-second potentially the best in the survey.

**Caveats.** (a) Needs a machine-extractable answer to vote on: math results, multiple-choice letters, code that passes a test, JSON fields. It does nothing for open-ended prose. (b) On a strong reasoning model the base single-sample accuracy is already high, so the absolute headroom is smaller than the GPT-3-era numbers; the gain concentrates on genuinely hard items (AIME-class), which is exactly where we want it. (c) The 2025 survey ["The Art of Scaling Test-Time Compute" (arXiv 2512.02008)](https://arxiv.org/html/2512.02008v1) reports majority voting is the optimal strategy specifically in the HIGH-compute regime, with shortest-answer selection winning at low budget and beam search in the middle: "no free lunch," the winner depends on budget.

**Verdict vs. "just let the 35B think longer":** For any task with a checkable answer, self-consistency is very likely the best quality-per-second buy we have, precisely because it is the only technique that turns our idle GPU batching capacity into free-ish parallel compute. This is the headline candidate.

---

## 3. Best-of-N + verifier, and compute-optimal allocation

**What it is.** Sample N, then SELECT the best with a verifier (a reward model / process reward model / an LLM-as-judge) rather than by vote. [Snell et al 2024, arXiv 2408.03314](https://arxiv.org/abs/2408.03314) formalizes "compute-optimal" allocation: spend the token budget differently by difficulty.

**Reported gain.** Up to ~21.6% on MATH; compute-optimal selection is >4x more token-efficient than vanilla best-of-N; a small model + optimal test-time compute beat a model **14x larger** with no test-time compute on some problems. The difficulty split is the actionable insight: **easy/medium problems benefit most from sequential revision, hard problems from parallel search/best-of-N.**

**Repeated-sampling coverage.** ["Large Language Monkeys" (Brown et al 2024, arXiv 2407.21787)](https://arxiv.org/abs/2407.21787) shows coverage (fraction solvable by ANY of k samples) scales log-linearly over four orders of magnitude: SWE-bench Lite went 15.9% (1 sample) to 56% (250 samples) with DeepSeek-V2-Coder. Crucial caveat: coverage is an oracle upper bound (pass@k). It only converts to real accuracy if you have a verifier good enough to PICK the right sample. Without one you fall back to majority vote (section 2).

**Token/time multiplier.** N samples (batchable, like section 2) PLUS the verifier pass. The verifier is the problem for us: we have no trained PRM/ORM loaded, and using a second model as judge forces a model swap on the hub (reload cost) unless we judge with the same model in a fresh call. For code, the verifier is free and perfect: run the tests.

**Verdict vs. "just let the 35B think longer":** Best-of-N wins decisively ONLY where a cheap real verifier exists (code with tests, math with a checker, tool-use with a success signal). Elsewhere it degenerates to self-consistency. Compute-optimal difficulty routing is the sophisticated prize but needs a difficulty estimator first.

---

## 4. Self-Refine and Reflexion (iterative self-correction)

**What it is.** Generate, then have the model critique its own output and rewrite, looping ([Self-Refine, Madaan et al 2023, arXiv 2303.17651](https://arxiv.org/abs/2303.17651); [Reflexion, Shinn et al 2023, arXiv 2303.11366](https://arxiv.org/abs/2303.11366)).

**Reported gain.** Self-Refine improves outputs by ~20% on average across 7 tasks (dialogue, code readability, sentiment rewriting) judged by humans/auto-metrics on GPT-3.5/4. Reflexion, using EXTERNAL feedback (test results, environment signals), reached 91% pass@1 on HumanEval with GPT-4.

**The decisive caveat.** ["Large Language Models Cannot Self-Correct Reasoning Yet" (Huang et al, ICLR 2024, arXiv 2310.01798)](https://arxiv.org/abs/2310.01798) shows that INTRINSIC self-correction (no external signal) does not improve reasoning and frequently DEGRADES it: the model second-guesses correct answers into wrong ones. Self-Refine's real wins are on open-ended, subjective tasks; its reasoning wins evaporate without a grounded feedback source.

**Token/time multiplier.** ~2-4x per refine round (generate + critique + rewrite), sequential, no batching discount.

**Verdict vs. "just let the 35B think longer":** Split decision. For CODE (feedback = test/lint output) and open-ended PROSE (quality is genuinely subjective), a single grounded refine pass is a good buy. For MATH/logic with no external checker, it is at best neutral and at worst negative, and letting the 35B think once, longer, is safer. Route by whether real external feedback exists.

---

## 5. Plan-then-execute: plan-and-solve and least-to-most

**What it is.** Decompose before solving. **Plan-and-Solve** ([Wang et al 2023, arXiv 2305.04091](https://arxiv.org/abs/2305.04091)) prepends "devise a plan, then carry it out" to a zero-shot prompt. **Least-to-most** ([Zhou et al 2022, arXiv 2205.10625](https://arxiv.org/abs/2205.10625)) explicitly breaks a problem into ordered sub-questions and feeds each answer into the next.

**Reported gain.** Plan-and-Solve beats zero-shot-CoT across all tested datasets and rivals 8-shot CoT on math. Least-to-most hits **99.7% on SCAN** (compositional generalization) with 14 exemplars, vs much lower plain CoT, and generalizes to harder-than-seen problems.

**Token/time multiplier.** Small: a planning preamble plus structured sub-steps, roughly 1.2-1.6x. Cheap.

**The caveat for us.** These techniques were designed to coax planning out of models that did not plan on their own. A reasoning-native model already emits an implicit plan inside its CoT, so the marginal gain on our 35B is likely small on standard math. Where they still earn their keep: **very long, multi-constraint tasks** (respect N requirements, multi-file refactors, structured extraction) where an explicit externally-tracked plan reduces dropped constraints, and as **scaffolding for tool/agent loops**.

**Verdict vs. "just let the 35B think longer":** Roughly a tie on ordinary reasoning; a modest win on long multi-constraint work where explicit decomposition curbs constraint-dropping. Cheap enough that it costs almost nothing to keep in the toolbox for the right task.

---

## 6. Tree-of-Thoughts and Graph-of-Thoughts

**What it is.** Search over a structured space of partial reasoning states with an evaluator scoring nodes and explicit backtracking. **ToT** ([Yao et al 2023, arXiv 2305.10601](https://arxiv.org/abs/2305.10601)) explores a tree; **GoT** ([Besta et al 2023, arXiv 2308.09687](https://arxiv.org/abs/2308.09687)) allows arbitrary graphs with merge/aggregate operations.

**Reported gain.** The most dramatic headlines in the survey: ToT solved **74% of Game-of-24** vs 4% for CoT and 9% for CoT-self-consistency (b=1 already gives 45%). GoT raises sorting quality **62% over ToT while cutting cost >31%**.

**Token/time multiplier.** Brutal. Each node is one or more LLM calls plus evaluator calls; Game-of-24 ToT runs on the order of ~100 model calls per problem. Effective multiplier is 50-100x+, almost entirely sequential (the search frontier is narrow), so it maps to minutes-to-tens-of-minutes per problem on our rig.

**The caveat for us.** The gains are real only on problems that are genuinely SEARCH-shaped (a large combinatorial space with a cheap, reliable state evaluator: puzzles, constraint satisfaction, some planning). On ordinary reasoning or open-ended tasks there is no good per-node evaluator and the structure adds cost without payoff. Each new task needs a bespoke state representation and scorer, i.e. real engineering, not a prompt wrapper.

**Verdict vs. "just let the 35B think longer":** Loses badly on quality-per-second for general use: enormous, mostly-sequential token cost for benefits confined to a narrow class of search problems. Keep on the shelf; only build if we hit a specific combinatorial task with a cheap verifier.

---

## 7. The 2025-2026 frontier: overthinking, adaptive budgets, parallel thinking

This is the material that most changes how we should spend compute on a reasoning-native model.

**Overthinking is real and it is an inverted-U.** ["When More Thinking Hurts" (arXiv 2604.10739)](https://arxiv.org/abs/2604.10739) shows accuracy rises, peaks, then DECLINES as CoT length grows, and that extended reasoning is associated with abandoning previously-correct answers. Critically, the peak is difficulty-dependent: easy problems overthink past ~2K thinking tokens, hard problems keep paying off to ~8K. Uniform compute allocation is provably suboptimal. See also ["Do NOT Think That Much for 2+3=?" (arXiv 2412.21187)](https://arxiv.org/abs/2412.21187) on o1-like overthinking of trivial inputs.

**Implication:** on our 35B, the highest-ROI move is not "think more," it is **matching the thinking budget to difficulty** so we stop wasting seconds (and accuracy) on the flat/declining side of the curve. This reclaims wall-clock time, the exact currency we care about.

**Parallel thinking beats sequential at equal budget.** ParaThinker ([MarkTechPost writeup](https://www.marktechpost.com/2025/09/08/parathinker-scaling-llm-test-time-compute-with-native-parallel-thinking-to-overcome-tunnel-vision-in-sequential-reasoning/)) argues sequential CoT suffers "tunnel vision," where an early bad step poisons the whole trace, and shows native parallel thinking gives +12.3% over sequential and +4.3% over majority voting at matched budget on a 1.5B model. The training-based version is not runnable as-is, but the mechanism is the same one self-consistency exploits: independent traces escape a single trace's early commitment. It is direct theoretical support for spending our batching capacity on parallel samples (section 2) rather than one longer sequential trace.

**Consensus of the frontier survey.** [arXiv 2512.02008](https://arxiv.org/html/2512.02008v1): optimal strategy is budget-dependent (shortest / beam / majority-vote as budget rises); there is no universally best method. This argues for a small ROUTER, not a single fixed technique.

---

## Ranking: quality gained per extra wall-clock second, on OUR rig

Ordered best buy first. "Batch discount" = benefits from llama.cpp parallel slots so wall-clock cost is sublinear in samples.

| Rank | Technique | Where it wins | Token cost | Batch discount | Verdict vs. 35B-thinking-longer |
|---|---|---|---|---|---|
| 1 | **Adaptive thinking-budget** (cap by difficulty) | ALL reasoning tasks | NEGATIVE (saves tokens) | n/a | Strictly better: reclaims time AND avoids overthinking |
| 2 | **Self-consistency / majority vote** | any checkable answer (math, MCQ, code, JSON) | Kx, sublinear here | YES | Beats it on hard items; the flagship buy |
| 3 | **Best-of-N + verifier** | code (tests), math (checker), tool-use | Nx + verify | YES (samples) | Beats it where a cheap verifier exists |
| 4 | **Plan-and-solve / least-to-most** | long multi-constraint / agentic tasks | ~1.2-1.6x | no | Roughly ties; modest win on constraint-heavy work |
| 5 | **Self-refine (grounded)** | code + open-ended prose ONLY | 2-4x | no | Wins with real feedback; neutral/negative on pure reasoning |
| 6 | **Budget forcing ("Wait")** | non-reasoning base models | +1 think block per Wait | no | Loses: our 35B already does native long CoT |
| 7 | **Tree/Graph-of-Thoughts** | narrow combinatorial search w/ cheap evaluator | 50-100x+ | no | Loses for general use; niche only |

The through-line: the two techniques that beat "just think longer" for general use are (1) spending LESS on the flat part of the overthinking curve, and (2) spending parallel samples that our idle GPU batching capacity makes cheap. Everything sequential-and-token-heavy (budget forcing, deep refine, ToT/GoT) is a poor buy against a model that is already a native long-CoT reasoner.

---

## Candidate experiments for our rig

Ranked by (quality-per-second upside x breadth) / effort. Naming CX to feed QUEUE.md.

### C1. Measure the batching discount for parallel sampling (Phase-0 referee for all parallel methods)
**Class:** incremental, but it GATES C2/C3 and is publishable alone. **Effort:** hours. **Verdict:** GO.
Zero download. Launch the 35B reasoning model with `--parallel 4` (then 8) and continuous batching; fire 1, 2, 4, 8 identical reasoning requests and measure aggregate tokens/s and per-request wall clock vs the single-slot baseline. This directly quantifies how far below Nx the true cost of self-consistency and best-of-N is on our expert-union-limited MoE. Log per-request latency and total GPU busy time.
- **First step:** `llama-server -m <35B reasoning GGUF> -ngl 99 --cpu-moe -fa on --parallel 4 -c <ctx>`; then a PowerShell/Python driver posting K concurrent identical chat completions to :9292, timing each.
- **Success:** K=4 finishes in < 2.5x single-request wall clock (i.e. real batch discount exists). **Kill:** ~Kx scaling with no amortization (expert-union dominates), which pre-kills C2's economics and reframes it as a pure quality-vs-time tradeoff.

### C2. Self-consistency on the flagship, gated by C1
**Class:** capability/quality (higher accuracy on hard items at bounded added time). **Effort:** hours-to-days. **Verdict:** GO.
Zero download. Take a hard, machine-checkable set (AIME-style math + a GPQA-diamond subset + a MCQ slice). Run single-sample baseline vs K in {3,5,8} majority-vote, using the C1 parallel-slot config so K traces batch. Report accuracy delta AND wall-clock delta, i.e. accuracy points per added second. Compare the vote against shortest-trace selection and (where a checker exists) best-of-N, per the "no free lunch" survey.
- **First step:** driver that samples K traces at temp ~0.6, regex-extracts the final answer, majority-votes; A/B/A vs single sample on the frozen set.
- **Success:** >= +5 accuracy points on the hard set at <= 2.5x wall clock (clears the bar to become a hub preset for "hard mode"). **Kill:** < +2 points or the time cost blows past 3x with no accuracy payoff.

### C3. Adaptive thinking-budget router (the overthinking lever)
**Class:** speed AND quality (reclaims seconds by not overthinking; avoids the declining side of the inverted-U). **Effort:** days. **Verdict:** GO.
Zero download. Use the fast Qwen3-8B (~50 t/s) as a cheap difficulty classifier that sets a thinking-token cap (e.g. 1K / 4K / 12K) before dispatching to the 35B via `/no_think` gating or a max-think stop condition. Sweep caps on a mixed easy/medium/hard set to map our own inverted-U, then evaluate the router end-to-end. This is the one technique that is quality-positive AND time-negative, so it can only help.
- **First step:** budget sweep first: run the 35B at fixed think caps {1K,2K,4K,8K,16K} across difficulty tiers, plot accuracy and time to locate our peak per tier.
- **Success:** router matches or beats fixed-long-thinking accuracy while cutting mean time-to-answer >= 20% (overthinking waste reclaimed). **Kill:** flat accuracy-vs-length curves on our model (no overthinking regime to exploit), in which case document the null and drop the router.

### C4. Grounded self-refine for code, with a reasoning control
**Class:** quality on a real daily workload (code). **Effort:** hours-to-days. **Verdict:** GO.
Zero download. For coding tasks via the hub: generate → execute tests/lint → feed failures back for one refine pass (external feedback loop, the Reflexion-style setup that actually works). Include a MATH control with NO external checker to reproduce the Huang result that intrinsic refine is neutral/negative, so we prove to ourselves where refine pays and where it does not.
- **First step:** driver: generate solution, run the provided test suite, on failure append the traceback and request a fix (cap 2 rounds); measure pass@1 → pass-after-refine on a held-out code set.
- **Success:** >= +10 points solve rate on code from grounded refine, with the math control flat-or-negative (confirming the route-by-feedback rule). **Kill:** code gain < +3 points, meaning our model already one-shots the set.

### C5. Plan-and-solve wrapper for long multi-constraint tasks (low-priority, cheap)
**Class:** quality on constraint-heavy prose/refactor work. **Effort:** hours. **Verdict:** MAYBE.
Zero download. Add an explicit "list every constraint, plan, then execute and self-check against the list" preamble for tasks with many simultaneous requirements (structured extraction, multi-file edits, spec-driven writing). Measure constraint-satisfaction rate vs vanilla 35B. Cheap (~1.3x tokens), so it only needs a small win to justify keeping.
- **First step:** build a 20-item set of tasks each carrying 5-10 explicit constraints; A/B vanilla vs plan-and-solve wrapper, score constraints met.
- **Success:** >= +15% constraint-satisfaction at ~1.3x tokens. **Kill:** < +5%, i.e. the native CoT already tracks constraints, then close.

---

## What NOT to build

- **Standalone budget forcing ("Wait") on the 35B.** It approximates what our RL-trained reasoner already does, saturates, and can oscillate into wrong answers ([arXiv 2507.14419](https://arxiv.org/abs/2507.14419)). Only meaningful as a probe on the non-reasoning 30B-Instruct, and even there the reasoning model is the better answer.
- **Tree/Graph-of-Thoughts as general infrastructure.** 50-100x mostly-sequential token cost for gains confined to combinatorial-search tasks with cheap evaluators. Revisit only if a specific such task appears.
- **Intrinsic self-refine on math/logic.** Documented to degrade reasoning ([arXiv 2310.01798](https://arxiv.org/abs/2310.01798)); build refine only where external feedback (tests, checker, subjective rubric) grounds it.

---

## Feasibility verdicts

Adversarial pass against our constraints (8 GB VRAM / 48 GB RAM / AVX2 / Windows / single auto-swap hub). Checks: fits the rig, runnable TODAY (not paper-only), costs zero PER-TOKEN speed (no hidden concurrent-model or retraining requirement), effort honest.

- **C1 (batching-discount measurement) — GO.** Pure measurement; `--parallel N` + continuous batching is a live llama.cpp feature, single model, zero download, no retraining. It does not claim to raise t/s, it quantifies the sublinear wall-clock of parallel samples, and the explicit kill criterion makes the result publishable whichever way it lands. Only watch item: `-c` is split across slots, so size per-slot context for long reasoning traces (they already run 64K with q8-KV per E040).
- **C2 (self-consistency on the flagship) — GO.** Orchestration-only against one model in parallel slots; needs machine-checkable answer sets (AIME/GPQA/MCQ, small downloads). Spends tokens and time, not t/s, exactly as the framing states. Gated on C1 but runs regardless; if the batch discount is absent it degrades to a pure quality-vs-time tradeoff, not to infeasible.
- **C3 (adaptive thinking-budget router) — MAYBE.** The first step (35B-only think-cap sweep across difficulty tiers) is a clean GO and needs no second model. The full router is the problem: it leans on Qwen3-8B "at ~50 t/s" as a per-request difficulty classifier, and 50 t/s implies the 8B is on the GPU, where it CANNOT co-reside with the `--cpu-moe` 35B in 8 GB VRAM. Live per-request routing would therefore thrash the auto-swap hub (each request = unload/reload a ~20 GB GGUF, tens of seconds), which negates the "reclaim 20% of wall-clock" premise. Feasible only as offline eval (pre-classify the frozen set with the 8B, swap to the 35B once) or by pinning the 8B CPU-resident at much lower speed. No retraining needed and the budget-control mechanism (stop at cap, inject `</think>`) is runnable, so it is not a KILL, but the concurrent-model assumption is unstated and materially changes the effort and the payoff.
- **C4 (grounded self-refine for code + math control) — GO.** Reflexion-style external-feedback loop: single model, pure orchestration plus running the provided tests. The no-checker math control is an honest falsifier for the intrinsic-refine null result. Costs 2-4x tokens (time, not t/s). Only operational note: it executes model-generated code, so run the test harness in a throwaway/sandboxed workspace.
- **C5 (plan-and-solve wrapper) — MAYBE.** Trivially feasible (prompt wrapper, ~1.3x tokens, single model, zero download) with no per-token-speed or concurrency cost. The MAYBE is on expected value, not feasibility: a reasoning-native 35B already emits an implicit plan inside its CoT, so the marginal constraint-satisfaction gain is likely small except on long multi-constraint work. Cheap enough to keep, low enough priority to defer behind C1-C4.
