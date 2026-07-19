# Decoding-level quality knobs in llama.cpp (the zero-speed-cost lane)

Sweep: 2026-07-sweep-02-quality | Written: 2026-07-20 | Method: llama.cpp source docs and merged PRs (verified against our b10064/b10068 tree), primary sampler papers (arXiv 2024-2026), adversarial rebuttals, and per-model vendor cards for the exact models we run. Everything below is runnable-today on our hub unless flagged paper-only.

## Why this is the highest-value lane

Every knob in this file operates on the **logit/probability vector for the next token**, not on the matmuls. Decode is memory-bandwidth-bound at ~30-42 t/s on our rig (the MEASURED LAW). A sampler touches a ~150k-entry vector once per token; that is nanoseconds against a forward pass that streams gigabytes of weights from DDR5. So **the entire sampler stack is free: zero tokens spent, zero t/s lost.** The only cost is engineering time to A/B them.

Framing consequence: unlike best-of-N or long thinking (which spend tokens and only trade quality for wall-clock), sampler tuning is a pure quality gain at fixed speed AND fixed token budget. This is the one lane where "improve quality without reducing speed" is literally, not approximately, true.

Two honest caveats up front, both load-bearing for how we should test:
1. **Samplers are model-conditioned.** A setting that helps a base model can hurt an RLHF/reasoning-tuned model whose logit distribution is already sharp. Vendor defaults exist for a reason (Section 6).
2. **The literature here is thin and partly contested.** Min-p's headline paper was substantially rebutted (Section 3.3). We should treat every "creativity" claim as a hypothesis to falsify on our own eval, not a settled result.

## 1. The llama.cpp sampler chain: order is a real variable

llama.cpp applies samplers as an ordered pipeline. The default chain (from [`tools/completion/README.md`](https://github.com/ggml-org/llama.cpp/blob/master/tools/completion/README.md), verified in our tree) is:

```
--samplers "penalties;dry;top_n_sigma;top_k;typ_p;top_p;min_p;xtc;temperature"
```

with the short alias form `--sampling-seq edskypmxt` (e=penalties, d=dry, s=top_n_sigma, k=top_k, y=typ_p, p=top_p, m=min_p, x=xtc, t=temperature).

Order matters and is itself tunable. Two facts that bite:
- **top_n_sigma expects raw logits and must run before temperature scaling** in effect, yet in the default string it sits early (good) while temperature sits last. If you hand-roll a chain, keep temperature after top_n_sigma or the temperature-invariance property breaks. See [PR #11223](https://github.com/ggml-org/llama.cpp/pull/11223).
- **XTC sits just before temperature**, so it prunes the high-probability head after all other truncation. That is deliberate: XTC is meant to be the last creative perturbation.

### Full knob table with exact defaults (llama.cpp master / b10064)

| Knob | Flag | Default | Disabled at | One-line role |
|---|---|---|---|---|
| Temperature | `--temp` | 0.80 | (n/a; 0 = greedy) | Global sharpness of the distribution |
| Top-k | `--top-k` | 40 | 0 | Keep k highest-prob tokens |
| Top-p (nucleus) | `--top-p` | 0.95 | 1.0 | Keep smallest set summing to p |
| Min-p | `--min-p` | 0.05 | 0.0 | Keep tokens >= p x P(top token) |
| Top-n-sigma | `--top-nsigma` | -1.0 (off) | -1.0 | Keep logits within n sigma of max logit |
| Typical-p | `--typical` | 1.0 (off) | 1.0 | Keep locally-typical (entropy-matched) tokens |
| Repeat penalty | `--repeat-penalty` | 1.0 (off) | 1.0 | Divide logits of recent tokens |
| Repeat last-n | `--repeat-last-n` | 64 | 0 | Window for the penalties |
| Presence penalty | `--presence-penalty` | 0.0 (off) | 0.0 | Flat penalty for any seen token |
| Frequency penalty | `--frequency-penalty` | 0.0 (off) | 0.0 | Penalty scaled by count |
| DRY multiplier | `--dry-multiplier` | 0.0 (off) | 0.0 | Strength of n-gram repetition penalty |
| DRY base | `--dry-base` | 1.75 | | Exponential base per extra repeated token |
| DRY allowed length | `--dry-allowed-length` | 2 | | Free repetition before penalty kicks in |
| DRY penalty last-n | `--dry-penalty-last-n` | -1 (ctx) | 0 | Lookback window |
| XTC probability | `--xtc-probability` | 0.0 (off) | 0.0 | Chance to apply XTC this step |
| XTC threshold | `--xtc-threshold` | 0.10 | 1.0 | Prob floor for tokens XTC may drop |
| Mirostat | `--mirostat` | 0 (off) | 0 | 1 or 2 = target-entropy feedback control |
| Mirostat LR | `--mirostat-lr` | 0.10 | | Feedback learning rate |
| Mirostat target | `--mirostat-ent` | 5.00 | | Target surprise (tau) |
| Grammar | `--grammar` / `--grammar-file` | none | | GBNF constraint on output |
| JSON schema | `-j` / `--json-schema` | none | | Schema -> GBNF constraint |

Note the **llama.cpp out-of-box default is already a truncation stack** (top-k 40 + top-p 0.95 + min-p 0.05 + temp 0.8). Our hub launchers should be setting per-model values explicitly rather than inheriting these, because the Qwen3 and GLM cards disagree with every one of these numbers (Section 6).

## 2. Temperature: the master control, and greedy is a trap for reasoning models

Temperature divides logits before softmax. Low = sharp/deterministic, high = flat/diverse. Nothing new, except two rig-relevant points:

- **Greedy (temp 0) is actively discouraged for our reasoning models.** The Qwen3.6 card warns that greedy decoding causes repetition and degradation on thinking models, and DeepSeek-R1 carries the same warning. Reasoning-tuned models are calibrated to be *sampled*, not argmaxed. So the intuition "temp 0 for max accuracy" is wrong for Qwen3.6-35B reasoning; use their temp 0.6-1.0 band.
- **Temperature interacts with truncation order.** Applied last (llama.cpp default), it reshapes an already-truncated set, so a high temp is safe. Applied first, high temp drags noise into top-p/min-p. This is exactly the failure top-n-sigma was built to kill (Section 3.4).

## 3. The truncation samplers (the core of the lane)

### 3.1 Top-k and top-p (nucleus): the baseline both papers try to beat

Top-k keeps a fixed count; top-p keeps a dynamic set summing to p. Both are the 2019-era baseline. The knock on top-p (from the min-p paper, [arXiv 2407.01082](https://arxiv.org/abs/2407.01082)) is that at high temperature the nucleus swells to include incoherent tokens. Top-k is blunt: it keeps k tokens whether the model is confident or not. Vendors still ship top-k 20 (Qwen3) as a cheap guardrail. Keep top-k around as a safety cap; do not rely on top-p alone at temp > 0.8.

### 3.2 Min-p: confidence-scaled truncation

Min-p keeps tokens whose probability is at least `p x P(top token)`. When the model is confident (sharp top token), the bar is high and few tokens survive; when uncertain, the bar drops and more survive. This adaptivity is the appeal. llama.cpp default is **0.05**; creative-writing lore uses 0.02-0.03, reasoning lore uses 0.1. Paper: [Nguyen et al., "Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs", arXiv 2407.01082](https://arxiv.org/abs/2407.01082) (ICLR 2025 oral).

### 3.3 Falsification datapoint: min-p's claims were substantially rebutted

Per our "rigorously falsified" rule, this belongs front and center. [Ahmed et al., "Min-p, Max Exaggeration: A Critical Analysis of Min-p Sampling", arXiv 2506.13681](https://arxiv.org/abs/2506.13681) reran the original paper and found:
- the human eval "omitted data, conducted statistical tests incorrectly, and described qualitative feedback inaccurately";
- on a proper hyperparameter-matched benchmark sweep, **min-p did not beat baselines** on quality, diversity, or the tradeoff;
- the headline adoption stats were "unsubstantiated" and later removed.

Their conclusion: the original evidence "fails to support claims that min-p improves quality, diversity, or a trade-off." **Takeaway for us:** min-p is a fine, cheap, adaptive truncator and worth keeping enabled, but the "it unlocks high-temp creativity" story is unproven. Do not adopt aggressive min-p + high-temp on faith; measure it.

### 3.4 Top-n-sigma: the strongest reasoning candidate in this file

Keeps only tokens whose **pre-softmax logit** is within `n x sigma` of the max logit, where sigma is the std-dev of the logits. Paper: [Tang et al., "Top-nσ: Not All Logits Are You Need", arXiv 2411.07641](https://arxiv.org/abs/2411.07641). llama.cpp: [PR #11223](https://github.com/ggml-org/llama.cpp/pull/11223) (VJHack), server support [PR #11896](https://github.com/ggml-org/llama.cpp/pull/11896).

Why it is interesting for reasoning specifically:
- The key empirical claim is that logits split into a **Gaussian noise region and a distinct informative region**, and the n-sigma cut isolates the informative region.
- Unlike top-p/min-p, the retained set is **temperature-invariant**: you can crank temperature for exploration without dragging in noise, because the cut is on raw-logit geometry, not post-softmax mass. The paper reports reasoning benchmarks (e.g. GSM8K-style) holding up at temperatures where top-p collapses.
- Recommended n: **~1.0** focuses on the informative region; higher (e.g. 5) admits more noise. Off by default (-1.0) in llama.cpp.

This is the one truncator with a mechanism-level reason to help *reasoning* rather than creativity, and it is already in our binary. Prime A/B candidate for Qwen3.6-35B reasoning.

### 3.5 Typical-p (locally typical sampling)

Keeps tokens whose surprisal is close to the conditional entropy (the "typical set"). Off by default (1.0). In practice it has been largely superseded by min-p and top-n-sigma for our model class and rarely appears in modern vendor cards. Low priority; note it exists, do not spend a slot on it.

## 4. The anti-repetition and creativity samplers

### 4.1 DRY (Don't Repeat Yourself): the good repetition penalty

DRY penalizes *extended repeated n-grams*, not individual tokens, applying an exponential penalty `multiplier x base^(matchlen - allowed_length)`. It kills verbatim loops and boilerplate repetition without the collateral damage of flat repeat-penalty (which punishes legitimately frequent tokens like "the" or code syntax). Origin: p-e-w's concept in oobabooga, ported from koboldcpp ([PR #982](https://github.com/LostRuins/koboldcpp/pull/982)) into llama.cpp by wwoodsTM/pi6am ([PR #9702](https://github.com/ggml-org/llama.cpp/pull/9702)).

Defaults when enabled: base **1.75**, allowed-length **2**, sequence breakers `['\n', ':', '"', '*']`. Off by default (multiplier 0.0). Typical enable value: multiplier **0.8**. **Rig relevance:** reasoning models sometimes loop mid-chain-of-thought; DRY at ~0.8 is a safer loop-breaker than repeat-penalty, and it is exactly the kind of thing that shows up on long-context 64k runs.

### 4.2 XTC (Exclude Top Choices): creativity by deleting the obvious

XTC finds all tokens above `xtc-threshold` and, with probability `xtc-probability`, **removes all of them except the least-probable one above the threshold**. It deliberately throws away the model's safest choices to force less clichéd continuations while staying coherent (the kept token is still above threshold). Origin p-e-w; llama.cpp [PR #9742](https://github.com/ggml-org/llama.cpp/pull/9742) (MaggotHATE). Defaults: threshold **0.10**, probability **0.0** (off). Recommended creative combo from the docs: `--sampling-seq mx --min-p 0.02 --xtc-probability 0.5`.

**Warning:** XTC is a creativity tool and is *actively harmful to reasoning and code*, because "the obvious next token" is usually the correct one in a proof or a function signature. Never enable XTC on the reasoning or coder endpoints. Creative endpoint only.

### 4.3 Mirostat: mostly deprecated, and it fights the modern chain

Mirostat 1/2 is a feedback controller that adjusts truncation each step to hold output perplexity at a target (`mirostat-ent`, default tau 5.0). Paper: [Basu et al., "Mirostat", arXiv 2007.14966](https://arxiv.org/abs/2007.14966) (ICLR 2021). It predates min-p/top-n-sigma and **overrides top-k/top-p/min-p** when active, so it does not compose with them. Practitioner guidance is to disable the other truncators when using it (top_p=1, top_k=0, min_p=0). Given that top-n-sigma achieves the "stable sampling space" goal more simply and composably, mirostat is low priority for us. Note it, skip it, unless a specific model regresses on everything else.

### 4.4 Penalties (repeat / presence / frequency)

- **repeat-penalty** (default off, 1.0): divides logits of tokens seen in the last `repeat-last-n` (64). Blunt; can damage code and reasoning. Vendors set it to **1.0 (off)**.
- **presence-penalty** (default 0.0): flat penalty for any previously-seen token. **This is the one Qwen actually recommends** (see 6.1), at **1.5** for thinking-general to curb repetition. The Qwen3.6 card explicitly warns values too high cause "language mixing and a slight decrease in performance," so 1.5 is a ceiling, not a floor.
- **frequency-penalty** (default 0.0): count-scaled version; rarely used on these models.

## 5. Constrained decoding: GBNF grammar and JSON schema (structured reliability, with a real tradeoff)

This is the other free lane: **guaranteed-valid structured output at zero speed cost** and near-zero overhead, because the grammar just masks illegal tokens in the logit vector during sampling. Docs: [`grammars/README.md`](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md).

What we get:
- **`--grammar` / `--grammar-file`**: GBNF (GGML BNF) constrains output to a formal grammar. Force valid JSON, enums, regex-like patterns, "emoji only," a fixed report skeleton, etc.
- **`-j` / `--json-schema`**: llama.cpp compiles a JSON Schema to GBNF automatically. Available on `llama-server` and the OpenAI-compatible endpoint (so it flows through our hub).
- `additionalProperties` defaults to **false**, which the docs say "produces faster grammars + reduces hallucinations."

Two caveats that matter for how we deploy it:
1. **The schema is NOT injected into the prompt.** "The model has no visibility into the schema." The grammar only masks tokens; the model does not *know* the target shape. So you still describe the format in the prompt AND constrain with the grammar. Grammar alone with no prompt hint produces valid-but-dumb output.
2. **Falsification datapoint: hard format constraints can degrade reasoning.** [Tam et al., "Let Me Speak Freely?", arXiv 2408.02442](https://arxiv.org/abs/2408.02442) (EMNLP 2024) found constrained JSON-mode decoding measurably *lowers* reasoning accuracy versus free-form, while it *helps* pure classification/extraction. The fix they validate is **reason-then-format**: let the model think in natural language, then constrain only the final answer field (or do a second constrained pass). For our reasoning endpoint, never wrap the whole chain-of-thought in a grammar; constrain only the extraction step.

Performance footgun from the docs: avoid `x? x? x?...` (N optionals); write `x{0,N}` or sampling gets "extremely slow."

## 6. Best-practice settings for the models we actually run

These are the **vendor-recommended** numbers for our exact stack, which should override llama.cpp defaults in every launcher. Sources are the model cards / vendor docs.

### 6.1 Qwen3.6-35B-A3B (our reasoning flagship)

From the [official model card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (values re-verified after the card was corrected, per [discussion #23](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/discussions/23)):

| Mode | temp | top_p | top_k | min_p | presence_pen | repeat_pen |
|---|---|---|---|---|---|---|
| Thinking, general | 1.0 | 0.95 | 20 | 0.0 | 1.5 | 1.0 |
| Thinking, precise code/webdev | 0.6 | 0.95 | 20 | 0.0 | 0.0 | 1.0 |
| Non-thinking (instruct) | 0.7 | 0.80 | 20 | 0.0 | 1.5 | 1.0 |

Plus: greedy discouraged; keep context >= 128K "to preserve thinking capabilities"; output budget 32,768 tokens typical, up to 81,920 for hard math/code.

### 6.2 Qwen3-30B-A3B-Instruct-2507 (our throughput flagship)

Instruct/non-thinking preset (Qwen3 family, [vendor quick-reference](https://muxup.com/2025q2/recommended-llm-parameter-quick-reference)): **temp 0.7, top_p 0.80, top_k 20, min_p 0.0**. Add presence_penalty ~0.5-1.5 only if you see repetition. Note this is far from the llama.cpp default (temp 0.8 / top_p 0.95 / min_p 0.05), so an unconfigured launcher is running the wrong sampler.

### 6.3 GLM-4.7-Flash / GLM-4.x (our coding model)

Vendor guidance ([Z.AI docs](https://docs.z.ai/guides/overview/migrate-to-glm-4.6); GLM-4.7-Flash defers to GLM-4.7 per [discussion #6](https://huggingface.co/zai-org/GLM-4.7-Flash/discussions/6)): **temp 1.0, top_p 0.95, and tune only ONE of the two, not both.** For deterministic coding, the community setting is temp ~0.6, top_p 0.95, top_k 20. GLM-Z1 reasoning uses temp 0.6, top_p 0.95, top_k 40.

### 6.4 Distilled task presets (synthesis of vendor cards + practitioner guides)

For any Qwen3/GLM-class model on our hub, starting points to A/B:

- **Reasoning / math / factual:** temp 0.6, top_p 0.95, top_k 20, min_p 0.0, presence_pen 0-1.0. Candidate upgrade: add **top-n-sigma ~1.0** and consider raising temp toward 1.0 (top-n-sigma should hold accuracy). No XTC. DRY 0.8 only if looping.
- **Creative writing:** temp 0.9-1.1, min_p 0.02-0.03, top_p 0.95, top_k 0. Add **XTC prob 0.5 / threshold 0.1** and **DRY 0.8**. This is the only preset where XTC belongs.
- **Code:** temp 0.2-0.6, top_p 0.95, top_k 20, min_p 0.0, repeat_pen 1.0, presence_pen 0.0. `repeat-last-n` up to 256-512 (code repeats tokens legitimately). No XTC, no DRY on tight code (can break valid repetition); GBNF/JSON-schema for tool-call and structured outputs, but reason-then-constrain.

## 7. Candidate experiments for our rig

All run through the hub at :9292 with orchestration we write; all are zero-token, zero-t/s-cost by construction, so the only metric is quality delta on a fixed eval set. Suggested eval: a frozen 40-60 item set spanning GSM8K-style math, a few code tasks (HumanEval-ish), and 5-10 creative prompts, scored by our own rubric or an LLM-judge on a second model to avoid self-preference.

1. **E-SAMP-1: top-n-sigma vs baseline on Qwen3.6-35B reasoning.** Hold temp at the vendor 1.0, top_p 0.95, top_k 20. Arm A = vendor baseline. Arm B = add `--top-nsigma 1.0`. Arm C = `--top-nsigma 1.0` with temp pushed to 1.3 (test the temperature-invariance claim). Measure math/reasoning accuracy and answer coherence. Hypothesis: B >= A, and C stays near A where a top-p-only C would collapse. This is the flagship experiment: mechanism-backed, reasoning-targeted, already in b10064.

2. **E-SAMP-2: falsify min-p creativity on our creative endpoint.** Given the 2506.13681 rebuttal, directly test it. Arm A = min_p 0.05 temp 0.8 (llama default). Arm B = min_p 0.02 temp 1.1 (the "unlock creativity" recipe). Arm C = B + XTC 0.5/0.1 + DRY 0.8. Score diversity (distinct-n, self-BLEU) AND coherence (judge). Hypothesis to try to break: that B beats A on the quality/diversity tradeoff. If we cannot reproduce the gain, we report the null, consistent with the rebuttal.

3. **E-SAMP-3: DRY as a long-context loop-breaker on 64k runs.** We already ship a 64k flagship config (E040/R1). Reasoning models sometimes loop deep in long contexts. Arm A = no anti-repetition. Arm B = DRY multiplier 0.8 (base 1.75, allowed-len 2). Arm C = presence_penalty 1.5 (Qwen's own recommendation). Measure loop/repetition incidence on long-generation prompts and any accuracy cost. Cheap, directly ties to a shipped launcher.

4. **E-SAMP-4: reason-then-constrain vs full-grammar for structured output.** Motivated by 2408.02442. Task = produce a strict JSON object requiring a computed/reasoned field. Arm A = `--json-schema` over the whole response. Arm B = free-form reasoning, then a second constrained pass (or constrain only the final field). Measure both JSON validity (should be ~100% for both) and correctness of the reasoned field. Hypothesis: B matches A on validity but beats it on correctness. Directly informs how our hub should expose structured output to opencode/Open WebUI.

5. **E-SAMP-5 (optional, low priority): sampler-order ablation.** Move XTC before vs after min-p, and top-n-sigma before vs after top-k, on the creative and reasoning sets respectively, to see whether chain order is a real free knob or noise on our models. Run only if E-SAMP-1/2 show sampler sensitivity worth chasing.

Priority order: **E-SAMP-1 first** (biggest mechanism-backed upside, reasoning flagship), then **E-SAMP-3** (cheap, ties to shipped 64k config), then **E-SAMP-4** (unblocks structured output design), then **E-SAMP-2** (falsification, publishable either way), E-SAMP-5 last.

## Feasibility verdicts

Adversarial review against the rig (8 GB VRAM / 48 GB RAM / AVX2 / Windows / single-machine hub), runnable-today (not paper-only), and honest zero-per-token-speed cost. Verified `--top-nsigma` is server-exposed (PR #11896, in default chain) and that `json_schema` over the OpenAI `/v1/chat/completions` endpoint has documented reliability bugs (llama.cpp issues #10732, #11847).

- **E-SAMP-1 (top-n-sigma vs baseline, Qwen3.6-35B) — GO.** Pure logit-vector sampler, already in b10064 and server-exposed; zero tokens, zero t/s cost. Model already runs on the rig. Judge model runs sequentially via llama-swap (no concurrency). Flagship experiment is sound as written.
- **E-SAMP-2 (falsify min-p creativity) — GO.** min-p/XTC/DRY all merged and present; samplers only, zero t/s. distinct-n and self-BLEU are trivial offline scripts; judge is a sequential swap. Falsification framing is honest and publishable either way.
- **E-SAMP-3 (DRY as 64k loop-breaker) — GO.** Reuses the shipped E040/R1 64k config; DRY is a sampler, no t/s cost. Only nuance: `dry-penalty-last-n -1` scans the full 64k lookback per token, so this is the one sampler with non-zero CPU cost — but it is tens of microseconds against a ~25 ms/token forward pass, i.e. still negligible. Cheap as claimed.
- **E-SAMP-4 (reason-then-constrain vs full-grammar) — MAYBE.** Constrained decoding is real and free, but two honest caveats: (1) Arm B runs **two passes**, so it is NOT "zero-token" — it spends extra tokens and wall-clock (t/s and single-machine constraints still hold; no concurrency/retraining). (2) The `json_schema` type over the OpenAI endpoint has documented silent-failure bugs; route via the `--json-schema` flag, the `response_format.schema` sub-field, or raw GBNF and verify on the actual b10068 build before trusting Arm A's ~100% validity assumption.
- **E-SAMP-5 (sampler-order ablation) — GO.** `--sampling-seq` / `samplers` array is a real free knob; reordering is genuinely low effort. Correctly deprioritized.
- **Umbrella "all zero-token, zero-t/s by construction" (line 170) — CORRECTION.** True for E-SAMP-1/2/3/5. False for E-SAMP-4 Arm B (two passes = extra tokens + wall-clock). Also, every arm that uses the second-model judge adds swap + judge wall-clock at eval time (not generation cost, and not a concurrency violation since llama-swap serializes). Recommend narrowing the claim to "zero per-token speed cost on the measured generation."
