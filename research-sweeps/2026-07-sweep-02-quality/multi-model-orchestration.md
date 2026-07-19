# Multi-Model Orchestration on a Single-Machine Swap Hub

Survey date: 2026-07-20. Scope: orchestration patterns that spend more inference to raise answer quality on a hub where models run **one at a time** and switching costs real seconds. Mixture-of-Agents (layered model collaboration), model routing / cascades (cheap model triages, escalates hard queries to the flagship or the 35B reasoner), speculative-style drafting for quality, and best-of-N / self-consistency / verifier selection. The question throughout: which patterns pay off on OUR hub, which need concurrent models we cannot afford, and which are net losses once the swap tax is priced in.

Framing law inherited from the project: token throughput is pinned at ~30-42 t/s by memory bandwidth and no orchestration technique raises it. So "improve quality without reducing speed" reduces to two measurable quantities: **quality per wall-clock second** and **quality at a fixed token budget**. Best-of-N and long thinking do NOT lower t/s; they raise time-to-answer, and are judged on quality gained per extra second. Everything here must run through our llama-swap hub at `:9292` with orchestration WE write on top of the OpenAI-compatible API. No retraining, no cloud.

Rig constraints applied throughout: i7-14650HX (AVX2 only, no AVX-512/AMX), 48 GB DDR5-5600 (~37-42 GB/s effective per E021-E028), RTX 5060 Laptop 8 GB, Windows 11, llama.cpp b10064/b10068, ~3 MB/s internet.

---

## 0. The governing constraint: the 30-90 second swap tax

Our hub (`llama-swap.yaml`) defines eight models, **every one with a `ttl` and no `groups` block**, so it is a pure single-slot switchboard: a request for a model not currently loaded stops the running `llama-server`, starts the new one, `mlock`s the whole GGUF into RAM, runs a health check, then proxies. Our own [HOW-TO-USE.md](../../HOW-TO-USE.md) states the measured cost plainly: **"First message to a model takes 30-90 seconds while it loads; after that it is fast. Switching models triggers one reload."**

That single number reorganizes the entire orchestration literature for our rig. Split every pattern into two families:

- **Single-model patterns** (Self-MoA, self-consistency, best-of-N, verifier selection, self-refine): all passes hit the model already resident. Swap cost = 0. The only cost is extra tokens, which at fixed t/s is extra wall-clock and nothing else. This is the family that fits us.
- **Cross-model patterns** (mixed Mixture-of-Agents, cheap->strong routing/cascades, cross-model speculative decoding): each transition to a not-yet-loaded model pays 30-90 s. For a single interactive query a 2000-token answer at 35 t/s is ~57 s, so **one swap roughly doubles time-to-answer, and a bounce (8B -> 30B -> 8B) triples or worse.** Cross-model orchestration is only viable if the swap is either amortized over a batch or eliminated by co-residency (section 6).

llama-swap does support concurrency: a `groups` block with `swap: false` keeps multiple members loaded together ([config docs](https://github.com/mostlygeek/llama-swap/wiki/Configuration)), and the newer "custom DSL swap matrix" ([#643](https://github.com/mostlygeek/llama-swap/issues/643)) plus startup `hooks` preloading ([#235](https://github.com/mostlygeek/llama-swap/pull/235)) exist. But co-residency spends the one resource we are shortest on: **8 GB VRAM.** Whether any two useful models fit at once is the gating question for the entire cross-model family (section 6).

---

## 1. Mixture-of-Agents: real, but the mixed version is the wrong shape for us

**Original MoA** ([arXiv 2406.04692](https://arxiv.org/abs/2406.04692), ICLR 2025, [code](https://github.com/togethercomputer/moa)) layers LLMs: several "proposer" models answer, an "aggregator" model synthesizes their outputs, repeated across layers. The default config is **3 layers x 6 proposers**, and it hit **65.1% LC win rate on AlpacaEval 2.0 vs GPT-4 Omni's 57.5%** using only open models. The mechanism they name is "collaborativeness": a model writes a better answer when shown other models' drafts, even weaker ones.

Two facts make classic MoA a poor fit for a swap hub:

1. **It is maximally cross-model.** 6 distinct proposers across 3 layers is up to 18 model invocations spanning many distinct checkpoints. On our hub that is a swap storm: dozens of 30-90 s reloads for one answer. Even collapsing proposers to our 3-4 resident-capable models, each layer transition that changes the loaded model pays the tax.
2. **Mixing often lowers quality anyway.** [Rethinking Mixture-of-Agents / Self-MoA (arXiv 2502.00674)](https://arxiv.org/abs/2502.00674) showed that **Self-MoA — sampling many outputs from the single best model and aggregating those — beats mixed MoA by 6.6% on AlpacaEval 2.0 and 3.8% on average across MMLU, CRUX, MATH.** Their diagnosis: MoA quality is dominated by the average quality of proposers, and folding weaker models in dilutes it. Diversity helps only when the models are of comparable strength and genuinely complementary.

The 2026 follow-ups keep pushing on aggregation quality, not model mixing: [Attention-MoA (arXiv 2601.16596)](https://arxiv.org/html/2601.16596v1) adds inter-agent semantic attention and residual synthesis, and [ReM-MoA (arXiv 2606.24437)](https://arxiv.org/pdf/2606.24437) adds reasoning-memory to sustain gains as you add rounds. Both are single- or few-model friendly.

**The version that fits us is Self-MoA, specifically Self-MoA-Seq.** The sequential variant slides a window over candidate outputs, repeatedly folding "best-so-far" with new samples, so it **aggregates an arbitrary number of samples at constant memory footprint and constant context length** and is reported "as effective as aggregating all outputs at once." That is exactly a single-model, zero-swap, bounded-context loop we can script against one hub endpoint. Cost is N+1 passes over one resident model = (N+1)x tokens = (N+1)x wall-clock, t/s untouched.

---

## 2. Model routing and cascades: right idea, reshaped by the swap tax

The routing/cascade literature is built for API cost, where "cheap" means dollars and switching models is free. On our hub the currency is wall-clock and switching is the expensive part, which inverts several conclusions.

**RouteLLM** ([arXiv 2406.18665](https://arxiv.org/abs/2406.18665), [code](https://github.com/lm-sys/RouteLLM)) trains a router (BERT / matrix-factorization / LLM classifiers) on preference data to send easy queries to a weak model and hard ones to a strong model, reporting **>2x cost reduction at ~95% of strong-model quality.** **FrugalGPT**-style cascades (Chen et al., 2023) run models cheapest-first and stop when a scored answer clears a confidence threshold, claiming up to ~98% cost savings at GPT-4-level accuracy. 2026 work formalizes the machinery: [Cluster, Route, Escalate (arXiv 2606.27457)](https://arxiv.org/html/2606.27457) instantiates pre-router + quality estimator + escalation policy; [Arch-Router (arXiv 2506.16655)](https://arxiv.org/html/2506.16655v1) aligns routing to human-labeled task categories; and a full survey exists in [Dynamic Model Routing and Cascading (arXiv 2603.04445)](https://arxiv.org/html/2603.04445v2).

The load-bearing 2026 result for us is [Is Escalation Worth It? A Decision-Theoretic Characterization of LLM Cascades (arXiv 2605.06350)](https://arxiv.org/pdf/2605.06350). Its frontier geometry says cascades create value only when **(a) the cheap model is already reasonably accurate, (b) its failures are identifiable via a confidence/uncertainty signal, and (c) the cost gap is large relative to the quality gap.** Cascades fail when the cheap model is too weak to emit a trustworthy confidence signal, or when routing cannot predict which queries actually benefit.

Two distinct routing modes matter on our hub, and they have opposite economics:

- **Capability routing (swap-reducing, a clear win):** route each query to the single best-suited resident model — coding to `qwen-coder-uncensored`, vision to `qwen-vision`, hard reasoning to `qwen-35b-reasoning`, chat to `qwen-30b`, trivial/latency-sensitive to `qwen-8b-fast`. This does not add passes; it picks the right model once, so it *avoids* wasted swaps and mis-routed low-quality answers. This is the highest-value, lowest-risk orchestration on the hub and is mostly a scripting/product task.
- **Quality cascade (swap-adding, gated):** cheap model answers, a confidence check escalates hard queries to the flagship or 35B reasoner. On our hub every escalation is a 30-90 s swap. If the cheap model handles most queries and the answer is short, average latency still drops; if the escalation rate is high, the swap tax dominates and the cascade is slower than just always using the strong model. This only pays under one of two conditions: **batching** (sort a queue of queries by their routed target and process each model's block together, so T_swap amortizes over the block) or **co-residency** (section 6).

---

## 3. Speculative-style drafting for quality: mostly cross-reference, one open door

"Small drafts, big verifies" splits into a token-level technique and a semantic-level one.

**Token-level (standard speculative decoding)** is lossless: the big model's exact distribution is reproduced faster. That is a *speed* technique, it needs both draft and target resident and interleaved token-by-token, and it is thoroughly covered by our sibling survey [speculative-decoding-frontier.md](../2026-07-sweep-01/speculative-decoding-frontier.md) and by E014. The key finding there transfers directly: on our A3B MoE flagship, verifying k drafted tokens touches the *union* of their experts (the expert-union penalty), which erases most of the gain, and a separate draft model also competes for TDP. For a *quality* sweep it is out of scope except to note that cross-model draft+verify is a cross-model pattern that demands co-residency (section 6).

**Semantic-level "speculative cascades"** ([Faster Cascades via Speculative Decoding, arXiv 2405.19261](https://arxiv.org/abs/2405.19261), ICLR 2025; [Google Research blog](https://research.google/blog/speculative-cascades-a-hybrid-approach-for-smarter-faster-llm-inference/)) is the interesting one. It replaces speculative decoding's strict token-match with a flexible **deferral rule** that decides per token whether to accept the small model's draft or defer to the large model, using confidence checks or comparative confidence. Because it is a cascade at heart, it can *exceed* the large model's quality (the cheap model's confident tokens are sometimes better), while keeping speculative decoding's parallel-verify latency profile. Reported as consistently better quality-latency trade-offs than plain speculative decoding across translation, summarization, and QA.

The catch for us is the same as token-level spec decoding: it requires **both models resident and interleaved**, plus a token-level deferral hook that **no llama.cpp release implements** (llama.cpp offers strict-verify spec types only). So speculative cascades are a watch-item / paper-only for our runtime today. The one door they leave open is conceptual: the deferral idea also works at the *answer* granularity (draft a full answer with the 8B, have the 30B/35B accept, edit, or regenerate based on its own confidence), which is scriptable on our hub and folds into the verifier candidate in section 4.

---

## 4. Best-of-N, self-consistency, and verifiers: the single-model test-time-compute core

This is the family that costs only tokens, and 2024-2026 work has made the token spend far more efficient.

- **Self-consistency** (sample N chain-of-thoughts, majority-vote the extractable answer) remains the strong, dumb baseline for tasks with a checkable answer (math, MCQ, code-with-tests). Zero swap, trivial to script.
- **Optimal test-time compute** ([Snell et al., arXiv 2408.03314](https://arxiv.org/html/2408.03314v1)) showed compute-optimal allocation of test-time sampling can beat scaling parameters on some tasks. [Scaling Test-Time Compute Without Verification or RL is Suboptimal (arXiv 2502.12118)](https://arxiv.org/pdf/2502.12118) sharpens it: **verifier-based selection dominates verifier-free (majority vote) at matched compute budgets.**
- **Verifier / best-of-N selection** is the token-efficiency win. [Calibrated Reasoning: An Explanatory Verifier (arXiv 2509.19681)](https://arxiv.org/pdf/2509.19681) reports a verifier reaching **0.77 on AIME 2025 with Qwen3-32B generations while using only 75% of the tokens of self-consistency, and 1-3x fewer tokens overall.** [Multi-Agent Verification / BoN-MAV (arXiv 2502.20379)](https://arxiv.org/html/2502.20379) combines best-of-N with multiple lightweight verifiers and scales better than self-consistency or a single reward model. [Generative Verifiers (arXiv 2408.15240)](https://arxiv.org/html/2408.15240v1) frames verification as next-token prediction, so **the same model can act as its own verifier** (LLM-as-judge over its own N samples) with no extra checkpoint and no swap. [Efficient Test-Time Scaling via Self-Calibration (arXiv 2503.00031)](https://arxiv.org/html/2503.00031) gets a usable confidence signal from ~10 extra tokens, cheap enough to gate a cascade or a stopping rule.

The load-bearing theory under all of this is the **generator-verifier gap** ([Mind the Gap, arXiv 2412.02674](https://arxiv.org/pdf/2412.02674)): for reasoning tasks a model verifies more reliably than it generates, so a same-model verifier can lift best-of-N even though it is no smarter than the generator. Important honesty caveat from the same literature: the gap **shrinks or vanishes on pure factual/trivia tasks**, where verifying requires the same missing knowledge as generating. So verifier-selection is a reasoning/coding/structured-output play, not a hallucinated-fact fixer.

Table of what is real and single-model:

| Pattern | Extra passes | Swap | Quality signal | Best on |
|---|---|---|---|---|
| Self-consistency (majority vote) | N | 0 | none (vote) | math, MCQ, code+tests |
| Best-of-N + same-model verifier | N + judging | 0 | generator-verifier gap | reasoning, code, structured |
| Self-MoA-Seq (aggregate samples) | N + 1 | 0 | aggregator synthesis | open-ended writing, QA |
| Long thinking (already on 35B) | 1 (long) | 0 | internal | hard reasoning |

---

## 5. Self-refinement and debate: mostly negative, and worth knowing why

The pattern people reach for first — "have the model critique and fix its own answer" — is the weakest, and the 2023-2026 evidence is consistent:

- [Large Language Models Cannot Self-Correct Reasoning Yet (arXiv 2310.01798)](https://arxiv.org/pdf/2310.01798): **without an external signal, intrinsic self-correction often makes reasoning answers worse**, because the model has no independent way to know it was wrong.
- Multi-agent debate underperforms plain self-consistency at equal sample budget ([ICLR 2025 blogpost analysis](https://d2jud02ci9yv69.cloudfront.net/2025-04-28-mad-159/blog/mad/), [Stay Focused: Problem Drift in Multi-Agent Debate, arXiv 2502.19559](https://arxiv.org/pdf/2502.19559)). It behaves more like an expensive consistency mechanism than genuine critique.
- [The Cost of Consensus (arXiv 2605.00914)](https://arxiv.org/pdf/2605.00914): **isolated self-correction beats unguided homogeneous multi-agent debate** — running one model with independent correction passes outperforms staging several instances arguing, which is doubly good news for a single-model hub (the better method is also the cheaper, swap-free one).

The exception that flips self-refinement from loss to win is a **grounded external verifier**: run the drafted code against a test suite / interpreter, feed the concrete failure back, and revise. Here the verifier is the runtime, not the model's own belief, so the generator-verifier gap is maximized (judges with tool access verify far better, per section 4). This is a real, scriptable loop for our coder models and the only self-refine variant with a solid prior.

---

## 6. Co-residency: the one lever that unlocks the cross-model family

Every cross-model pattern above dies on the 30-90 s swap unless two models can be loaded at once. So the pivotal empirical question for this whole sweep is: **can our 8 GB VRAM + 48 GB RAM hold two useful models simultaneously via a `swap: false` group?**

Rough budget from `llama-swap.yaml`:
- `qwen-8b-fast`: `-ngl 34` (partial offload), roughly 4-5 GB VRAM as configured, or near-zero VRAM if forced CPU-only (slower, but as a router/drafter it emits few tokens).
- `qwen-30b` / flagship: `-ngl 99 --n-cpu-moe 40` puts ~17 GB of experts in RAM and only the non-expert tensors + KV in VRAM, empirically ~4 GB.

RAM is fine (17 + 5 = ~22 GB of 48 GB). **VRAM is the binding constraint**: 4 GB (30B non-experts) + 4-5 GB (8B at `-ngl 34`) is at or just over the 8 GB ceiling once Windows desktop overhead is counted. The realistic co-resident configs to test are (i) 30B flagship in VRAM as the verifier/strong model plus an 8B router/drafter pushed mostly or fully to CPU, or (ii) two A3B MoE models both in `--n-cpu-moe` mode sharing the ~4 GB non-expert VRAM slice with reduced context. If either fits, the swap tax on capability routing and quality cascades goes to zero after warmup, and even answer-level speculative-cascade drafting becomes scriptable. If neither fits, the cross-model family stays batch-only.

Heterogeneity note in our favor (same as the DuoDecoding observation, [arXiv 2503.00784](https://arxiv.org/pdf/2503.00784)): our drafting/routing resource (a small model on GPU or spare CPU) and our flagship's bottleneck (CPU RAM bandwidth for experts) are different silicon, so a co-resident small router need not steal much from the flagship's decode — modulo the E014 TDP-sharing lesson, which is milder for an 8B router emitting a few tokens than for a full draft model.

---

## 7. Orchestration cost model: quality per wall-clock second

Decision engine for every candidate. Let `T` = time for one normal answer on the resident model (tokens / t/s), `T_swap` = 30-90 s measured, `S` = number of not-yet-loaded model transitions, `N` = extra full passes.

```
time_single-model(N)   = (N + 1) * T                      # Self-MoA, best-of-N, verifier
time_cross-model(S, B) = sum(T_i) + S * T_swap / B         # routing/cascade, B = queries per swap-amortized batch
value                  = delta_Quality / (time - T)        # quality gained per extra second over a 1-shot answer
```

Readings that decide the candidates:

- **Single-model best-of-N with N=4 costs ~4x wall-clock, no swap.** At 35 t/s a 1500-token answer is ~43 s; N=4 with same-model verifier judging is ~3-4 min for a measurable quality lift on reasoning/code. Worth it for hard queries, absurd for trivial ones — so it should be *gated* by a cheap difficulty/confidence check, not applied blanket.
- **Cross-model, interactive (B=1): S=1 adds 30-90 s, i.e. it roughly doubles a typical answer's wall-clock before any tokens are generated.** A cascade that escalates >~40% of queries is slower on average than always running the strong model. This is the quantitative reason quality cascades are a batch or co-residency play, not an interactive one, on our hub.
- **Cross-model, batched (B large): S*T_swap/B -> 0.** Sort an offline queue by routed target model, process each model's block in one residency. This makes the full RouteLLM / cascade playbook usable for overnight/offline jobs at its published cost-quality numbers, with the swap tax amortized to noise.
- **Co-residency (S_effective = 0 after warmup):** collapses cross-model to single-model economics at the price of the VRAM headroom measured in section 6.

---

## Candidate experiments for our rig

Ordered by expected value per unit effort. All run against the hub at `:9292` with orchestration scripts we write (PowerShell/Python), no retraining, no cloud.

### C1. Self-MoA-Seq on the flagship: single-model quality lift at zero swap
Script the sequential Self-MoA loop ([arXiv 2502.00674](https://arxiv.org/abs/2502.00674)) against one resident endpoint (`qwen-30b` or `qwen-35b-reasoning`): sample N candidates at temperature, slide the fold-in window aggregating best-so-far with new samples at constant context. Sweep N = 2/4/8. Measure quality (a fixed rubric set: open-ended writing, QA, a few reasoning items) and exact wall-clock, report quality-per-extra-second. Baseline is the 1-shot answer and a plain self-consistency majority vote at the same N. Zero download, zero swap, pure token spend.
- Needs: an orchestration script only. Effort: hours. Class: quality-per-second (the cleanest fit to the hub).

### C2. Gated best-of-N with a same-model generative verifier
On one resident model, sample N answers, then have the **same model** score/rank them (generative-verifier style, [arXiv 2408.15240](https://arxiv.org/html/2408.15240v1)) and return the top pick. Gate the whole thing behind a cheap ~10-token self-calibration confidence check ([arXiv 2503.00031](https://arxiv.org/html/2503.00031)) so only low-confidence/hard queries pay the N-pass cost. Compare token efficiency and quality against C1's self-consistency baseline; expect the 1-3x token savings the verifier literature reports ([arXiv 2509.19681](https://arxiv.org/pdf/2509.19681)) to hold on reasoning/code and to shrink on factual queries (the generator-verifier gap caveat). Zero swap.
- Needs: orchestration script; a small tagged eval set spanning reasoning/code/factual to expose the gap boundary. Effort: hours to a day. Class: quality-at-fixed-token-budget.

### C3. Capability router in front of the hub (swap-avoiding, not swap-adding)
A thin classifier (rules + an 8B-fast zero-shot label, or a small embedding match a la [Arch-Router, arXiv 2506.16655](https://arxiv.org/html/2506.16655v1)) that maps each incoming request to the single best resident model: code -> `qwen-coder-uncensored`, vision -> `qwen-vision`, hard reasoning -> `qwen-35b-reasoning`, chat -> `qwen-30b`, trivial/latency-critical -> `qwen-8b-fast`. This does not add passes; it prevents wasted swaps and mis-routed low-quality answers, so its quality-per-second is strictly positive. Measure routing accuracy and the swap-count reduction versus naive per-request model choice.
- Needs: router script in front of `:9292`; a labeled request sample. Effort: days. Class: quality-per-second + swap reduction (highest practical value, lowest risk).

### C4. Co-residency probe: can two models share 8 GB VRAM, and does a swap-free quality cascade then beat always-strong?
Gate experiment for the entire cross-model family. Define a llama-swap `groups` block with `swap: false` and test the two configs from section 6: (i) 30B flagship in VRAM + 8B router/drafter pushed to CPU, (ii) two `--n-cpu-moe` A3B models sharing the non-expert VRAM slice at reduced context. Confirm both load and stay stable under the E032 thermal protocol. If a config fits, build the quality cascade (8B answers, ~10-token confidence gate escalates to 30B/35B, [decision-theoretic thresholds from arXiv 2605.06350](https://arxiv.org/pdf/2605.06350)) and measure average wall-clock + quality against the always-strong baseline; escalation must stay well under the ~40% break-even the section 7 model predicts.
- Needs: config edits + orchestration; no download. Effort: days. Class: enabler (unlocks C5 and interactive cross-model routing; may simply refute co-residency, which is itself a publishable rig fact).

### C5. Grounded self-refine loop for code (the one self-correction variant with a real prior)
On `qwen-coder-uncensored`: generate code, execute it against a test harness we supply, feed concrete failures back, revise, cap at k iterations. This is the tool-grounded verifier case where the generator-verifier gap is maximized and self-correction actually works, versus the ungrounded self-critique that the literature shows hurts ([arXiv 2310.01798](https://arxiv.org/pdf/2310.01798), [arXiv 2605.00914](https://arxiv.org/pdf/2605.00914)). Zero swap (single model), cost is k extra passes plus test-run time. Measure pass@1-after-refine versus 1-shot on a small coding set.
- Needs: sandboxed execution harness + orchestration; no download. Effort: days. Class: quality-per-second (bounded, well-priored).

Dead ends / watch-items this sweep, for the record: **classic mixed Mixture-of-Agents** (swap storm plus the Self-MoA finding that mixing dilutes quality); **multi-agent debate** (underperforms self-consistency at equal budget, and cross-model on our hub); **token-level cross-model speculative decoding for quality** (a speed technique, covered by the sibling spec-decoding sweep, killed on MoE by the expert-union penalty); **speculative cascades** (best-in-class idea but no llama.cpp deferral-rule implementation exists — watch upstream). Interactive single-query quality cascades are a dead end until C4 proves co-residency; until then they are batch-only.

---

## Feasibility verdicts

Adversarial review, 2026-07-20. Each candidate checked for: 8 GB VRAM + 48 GB RAM + AVX2-only + Windows fit; runnable on our b10064/b10068 hub with scripts we write (not paper-only); non-duplication of prior experiments; honest swap/token/wall-clock math. Grounded against the live `llama-swap.yaml` (single-slot, no groups) and the HOW-TO-USE 30-90 s swap figure.

- **C1 Self-MoA-Seq: GO.** Single resident model, zero swap, pure token spend that does not touch t/s; Self-MoA-Seq's constant-memory sliding window ([arXiv 2502.00674](https://arxiv.org/abs/2502.00674)) is explicitly designed for exactly this bounded-context, on-the-fly aggregation, so it needs no VRAM headroom we lack. Cleanest possible fit; the only risk is that the quality lift on our single-family samples is smaller than the cross-model gains the paper's diversity provides, which is precisely what the experiment measures.
- **C2 gated best-of-N + same-model verifier: GO.** Zero swap, zero download; generative-verifier and self-calibration are prompt-level techniques ([arXiv 2408.15240](https://arxiv.org/html/2408.15240v1), [arXiv 2503.00031](https://arxiv.org/html/2503.00031)) with no runtime dependency. Honest scope limit baked into the design: the token-efficiency win is a reasoning/code phenomenon and is expected to shrink on factual queries per the generator-verifier gap, so the eval set must be tagged by task type or the headline will be misleading.
- **C3 capability router: GO.** Pure orchestration in front of `:9292`, adds no passes, and its whole point is to *reduce* swaps rather than add them, so it cannot lose on quality-per-second if routing accuracy is decent. Lowest-risk, highest-practical-value item; the only real work is the labeled request sample and honest routing-accuracy measurement. Overlaps with product work already implied by the hub, so frame the research contribution as the measured swap-reduction and mis-route-avoidance numbers.
- **C4 co-residency probe: MAYBE.** The `groups`/`swap: false` mechanism is confirmed real in llama-swap ([config docs](https://github.com/mostlygeek/llama-swap/wiki/Configuration)), and RAM easily holds two A3B models. The live risk is the 8 GB VRAM ceiling: 4 GB (30B non-experts) + 4-5 GB (8B at `-ngl 34`) is at or over budget with Windows overhead, so config (i) likely needs the 8B forced to CPU and config (ii) needs reduced context on both. Kept MAYBE because it may simply refute co-residency on this rig — which is a clean, publishable negative that also justifies the "cross-model = batch-only" conclusion. Gate everything cross-model on this result.
- **C5 grounded self-refine for code: MAYBE.** Single model, zero swap, and it is the one self-correction variant with a positive prior because the verifier is the interpreter, not the model's own belief. Downgraded from GO because it needs a sandboxed execution harness (real engineering + a safety boundary on running model-generated code on the host) and because the honest baseline is strong: modern coder models already 1-shot much of a small test set, so the refine delta may be thin. Run after C1-C3; scope the coding eval set so the refine gain is measurable rather than swamped by 1-shot passes.

## Feasibility verdicts (independent adversarial pass)

Second, adversarial reviewer, 2026-07-20. Re-checked each candidate independently against the four gates: (1) fits 8 GB VRAM + 48 GB RAM + AVX2 + Windows + single-slot hub; (2) runnable TODAY on b10064/b10068 with scripts we write, not paper-only; (3) honestly costs zero PER-TOKEN speed and does not secretly need concurrent models or retraining; (4) effort estimate honest. I concur with three verdicts, sharpen a caveat on a fourth, and split the fifth. No candidate is an outright KILL because the true swap-storm / paper-only patterns were already relegated to the dead-ends paragraph.

- **C1 Self-MoA-Seq: GO (concur).** All four gates pass cleanly. One resident model, N+1 sequential passes, constant-context sliding window so no VRAM/context headroom we lack, no retraining, no swap. The N+1x wall-clock cost is stated honestly and does not touch t/s. "Hours" of scripting is realistic. Only substantive risk is scientific, not feasibility: single-family aggregation may under-deliver versus the paper's cross-model diversity, which is exactly what the experiment measures.

- **C2 gated best-of-N + same-model verifier: GO (concur, with one honesty note).** Zero swap, zero download, single model, no retraining. Adversarial catch on the confidence gate: the cited self-calibration method ([arXiv 2503.00031](https://arxiv.org/html/2503.00031)) trains a calibrated checkpoint, so on our stock hub models the "~10-token confidence" gate degrades to a prompt/logprob heuristic (coarser, but still zero-swap and adequate to gate). Verdict is robust to this. The generator-verifier-gap scope limit (win shrinks on factual queries) is correctly baked in; keep the eval set tagged by task type or the headline misleads. "Hours to a day" is fair.

- **C3 capability router: GO (concur, with a swap-trap caveat the draft understates).** Pure orchestration in front of `:9292`, adds no passes, reduces swaps, no retraining. The load-bearing caveat: the router must NOT classify by calling a non-resident model. If it invokes `qwen-8b-fast` for a zero-shot label and the 8B is not already loaded, you pay one swap to classify plus one to serve, so a "swap-reducing" router becomes swap-adding on the interactive path. It only stays swap-free if the classifier is rules/embedding-based on CPU (or a tiny embedding model, ~hundreds of MB, kept CPU-resident), OR the 8B is pinned resident. Build the rules/embedding variant; measure swap-count reduction versus naive per-request choice as the actual contribution.

- **C4 co-residency probe: MAYBE (concur it is a probe; sharpen that it likely fails, and that "swap-free" is not "speed-free").** The `groups`/`swap: false` mechanism is real, and the probe is runnable today, so it clears gates 1-2 as an experiment whose negative result is itself publishable. Two adversarial corrections to the framing: (a) VRAM/RAM math is worse than the draft's summary admits. Config (i) (30B non-experts ~4 GB + 8B at `-ngl 34` ~4-5 GB) is at or over 8 GB with Windows overhead, so the 8B almost certainly must drop to `-ngl 0`; config (ii) (two 30B-class A3B models) puts ~34 GB of experts in RAM (the "17+5=22 GB" reassurance covers only config (i), not two 30Bs) AND ~8 GB of non-experts in VRAM before KV/desktop, so config (ii) is effectively dead on BOTH VRAM and RAM and only config (i) is worth probing. (b) Gate 3 partial fail: once the 8B is forced to CPU to fit VRAM, the DuoDecoding "different silicon" argument collapses, because a CPU-side router contends for the same ~37-42 GB/s RAM bandwidth that IS the flagship's decode bottleneck. Contention is transient in a sequential cascade (classify then decode, not simultaneous), so it is survivable, but co-residency is NOT free on t/s whenever both models are active at once. Keep MAYBE; expect a likely refutation on config (ii) and a marginal pass at best on config (i).

- **C5 grounded self-refine for code: MAYBE (concur).** Single model, zero swap, no retraining, and the only self-correction variant with a real prior because the verifier is the interpreter. The two downgrade reasons are honest and load-bearing: a sandboxed harness for running model-generated code on a Windows host is real engineering plus a genuine safety boundary (do not exec untrusted code on the bare host), and the 1-shot baseline on modern coders is strong enough to swamp a thin refine delta. Run after C1-C3; size the coding set so the refine gain is measurable.

Concurrence on the dead-ends paragraph: classic mixed MoA (swap storm + Self-MoA dilution), multi-agent debate (loses to self-consistency at equal budget, cross-model here), token-level cross-model spec decoding (speed technique, killed on MoE by expert-union), and speculative cascades (no llama.cpp deferral-rule impl) are all correctly KILLED/watch-listed and stay there.
