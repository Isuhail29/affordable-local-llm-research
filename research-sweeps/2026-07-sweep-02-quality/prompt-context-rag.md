# Domain survey: prompt-level and context-level quality, no per-token speed cost

Sweep 2026-07-sweep-02-quality. Surveyed 2026-07-19/20. Runtime baseline: llama-swap hub at `:9292`, OpenAI-compatible API, models auto-swap. Live stack: Qwen3-30B-A3B-Instruct-2507 (~33-42 t/s), Qwen3.6-35B-A3B reasoning (37-40 t/s, reasoning-native), GLM-4.7-Flash (coding), Qwen3-8B (~50 t/s), abliterated 30B general + coder, Qwen3-VL-30B. Open WebUI + opencode on top. GUI RAG lives in Open WebUI.

Guiding question: token throughput is fixed by memory bandwidth at ~30-42 t/s and cannot be raised by anything in this survey. So the target is **quality per wall-clock second** and **quality at a fixed token budget**. The levers here (system prompt, exemplars, retrieved context, context layout) mostly change the *prefill*, not the decode rate, so per-decoded-token speed is untouched. The cost we actually pay is prefill wall-clock (CPU-expert-bound on our MoE flagship) plus one-time embedding/rerank passes. Every recommendation is judged on quality gained versus that added wall-clock, not on t/s.

---

## 0. Why these levers are (almost) free on our rig

Decode t/s is set by how many bytes of weights + KV must be read per token. None of prompt engineering, exemplar choice, or retrieval changes the per-token read volume of the *generation* phase, so **decode t/s is invariant** under everything below. What they change:

- **System prompt / instructions**: a few hundred extra prefill tokens, processed once. Effectively free.
- **Few-shot exemplars**: N extra prefill tokens per call, processed once per call. Costs prefill wall-clock, and (critically) eats into the context budget and can trigger lost-in-the-middle (section 4).
- **RAG**: an embedding pass over the query (tens of ms on CPU for a small model), a vector lookup (sub-ms), an optional cross-encoder rerank pass, then the retrieved chunks become extra prefill tokens. The whole point is that retrieving 2k relevant tokens is far cheaper to prefill than stuffing a 32k document, *and* it dodges context rot.
- **Context layout**: reordering the same tokens. Literally free; pure orchestration.

The one real, measurable cost is prefill. On the flagship, prefill runs the same CPU-expert path as decode, so a 32k-token document costs on the order of minutes to process the first time (established in [sweep-01 kv-cache doc](../2026-07-sweep-01/kv-cache-long-context.md)). This is exactly why RAG (retrieve 2k tokens, not 32k) and prompt caching (`--cache-ram`, slot save/restore) are disproportionately valuable *for us specifically*, more than for a GPU-rich setup.

---

## 1. System-prompt engineering for our specific models

The single highest-leverage, zero-download, zero-latency change. Our models are not generic; they have documented, model-specific requirements that we are probably not all honoring through the hub yet.

### 1.1 Sampling settings are part of the "prompt" and are model-specific

Qwen publishes different sampling settings for Instruct vs Thinking, and using the wrong ones measurably degrades quality:

- **Qwen3-30B-A3B-Instruct-2507 (our general flagship, non-thinking)**: `temperature 0.7, top_p 0.8, top_k 20, min_p 0`, plus a small `presence_penalty` (0 to ~1.5) to curb repetition ([Unsloth Qwen3-2507 guide](https://unsloth.ai/docs/models/qwen3-how-to-run-and-fine-tune/qwen3-2507), [Jan.ai Qwen3 settings](https://www.jan.ai/post/qwen3-settings)).
- **Qwen3 Thinking / Qwen3.6-35B-A3B reasoning (our reasoning driver)**: `temperature 0.6, top_p 0.95, top_k 20`, `max_tokens` high (32k class) so it can finish its thinking ([Jan.ai](https://www.jan.ai/post/qwen3-settings)).
- **Greedy decoding is a trap**: with temperature 0 the Qwen3 models repeat or break, and thinking mode in particular gets stuck ([Jan.ai](https://www.jan.ai/post/qwen3-settings), [Local AI Master setup guide](https://localaimaster.com/blog/qwen-3-local-setup-guide)). If any hub route or client is defaulting to `temperature 0` for "determinism", that is actively hurting quality at zero speed benefit.

Action item that costs nothing: audit what sampling defaults each llama-swap route and each client (Open WebUI, opencode) actually sends, and pin the correct per-model values in the hub config. This is a quality change with no token cost at all.

### 1.2 The system message is the strongest lever

Qwen reads the system message first and uses it as the lens for every user turn; it carries strong weight for the whole conversation ([Qwen prompt guide](https://qwen3lm.com/prompt/)). Practical, model-agnostic structure that works well for instruct models:

1. **Role + scope** in one or two lines (who the model is, what it does, what it must not do).
2. **Output contract**: format, length ceiling, language, whether to show reasoning. For non-thinking Instruct, explicitly asking for a brief plan then the answer recovers much of the reasoning benefit without a think block.
3. **Rules as a short bulleted list**, not prose; models follow enumerated constraints better than paragraphs.
4. **Refusal / uncertainty policy**: tell it to say "I don't know" rather than fabricate. This is the cheapest hallucination reducer and it pairs directly with RAG (section 3).

### 1.3 Thinking vs non-thinking hygiene

- **Do not feed the think block back into history.** Logging the `<think>...</think>` into the running chat bloats context and drags later answers off-topic ([Jan.ai](https://www.jan.ai/post/qwen3-settings)). Confirm Open WebUI and opencode strip reasoning from the persisted transcript for our Qwen3.6 reasoning route.
- **Match model to task through the hub, not through prompt hacks.** We already have a non-thinking flagship and a reasoning driver as separate routes; that is the right design. Forcing reasoning behavior on the Instruct model via elaborate "think step by step" system prompts is weaker than just routing to the reasoning model, and it spends tokens.

### 1.4 Per-model notes

- **GLM-4.7-Flash (coding)**: GLM-family models are sensitive to chat-template correctness and tool-call formatting; keep system instructions terse and put the actual task/spec in the user turn. Verify the template the hub applies matches the official one before blaming quality.
- **Qwen3-8B (fast)**: benefits most from few-shot and tight output contracts because it has the least internal capacity; it is also our best "cheap judge" and draft candidate.
- **Abliterated 30B**: alignment is stripped, so the system prompt is the *only* remaining behavioral guardrail; be explicit about scope and format or output drifts.

Bottom line: section 1 is where the fastest, cheapest quality wins live. Nothing here downloads anything or slows decode.

---

## 2. Few-shot exemplar selection

Few-shot is the classic quality lever, but 2024-2026 research says its value now depends heavily on the model class, and it is not free (each exemplar is prefill tokens and context-budget pressure).

### 2.1 It still helps weaker / format-sensitive tasks, and selection matters

- LLMs are **highly sensitive** to which exemplars are chosen; the wrong set produces large performance variance ([Revisiting Demonstration Selection, arXiv 2401.12087](https://arxiv.org/pdf/2401.12087)).
- The workhorse method is **kNN-ICL**: embed the query, retrieve the most semantically similar labeled examples, use those as the shots. Test-similar demonstrations typically beat random or fixed ones ([Revisiting Demonstration Selection, arXiv 2401.12087](https://arxiv.org/pdf/2401.12087)).
- kNN alone tends to grab **redundant, easy** examples near the query; refinements target the decision boundary. **Delta-KNN** scores each candidate by its marginal contribution and retrieves "representatives", beating plain kNN ([Delta-KNN, ACL 2025](https://aclanthology.org/2025.acl-long.1253.pdf)); **MarginSel** picks max-margin (hard) examples ([arXiv 2506.06699](https://arxiv.org/pdf/2506.06699)).
- Note the caveat that even "correct" demonstrations can hurt in some regimes ([When Correct Demonstrations Hurt, arXiv 2605.26350](https://arxiv.org/pdf/2605.26350)), which is why exemplar choice must be measured per task, not assumed.

### 2.2 For strong reasoning models, few-shot often does nothing or backfires

This is the key 2025 finding and it directly shapes how we should use our two-tier stack:

- Augmenting zero-shot chain-of-thought with few-shot exemplars **rarely improves and can reduce** performance in strong, instruction-tuned models; the marginal benefit of external scaffolding shrinks as model capacity rises ([Zero-shot CoT overview, EmergentMind](https://www.emergentmind.com/topics/zero-shot-chain-of-thought-cot-b757956d-2c2f-449a-821c-a61b63eed6c7)).
- Implication for us: on the **Qwen3.6-35B-A3B reasoning** route, prefer a clean zero-shot instruction and let it think; do not burn context and prefill on exemplars. On the **Qwen3-8B** and **Instruct-2507** routes for structured/extraction tasks, kNN-selected few-shot is more likely to pay off.

### 2.3 Ordering and recency

Position matters: models weight the end of the prompt more (section 4), so the **most relevant exemplar should sit last**, closest to the query. This is a free ordering change once you already have the exemplars.

### 2.4 Runnable-now shape on our rig

We can build a tiny local exemplar store: a JSONL of (input, ideal output) pairs, embedded once with the same small model we pick for RAG (section 3), and a kNN lookup in our own orchestration script that injects the top few into the prompt before it hits the hub. No new infrastructure beyond what RAG already needs.

---

## 3. Lightweight local RAG on our rig

RAG is the lever with the best quality-per-prefill-second story *because* prefill is our bottleneck: retrieving ~2k relevant tokens instead of prefilling a 32k document is both faster and higher quality (it sidesteps context rot, section 4). Open WebUI ships a full RAG pipeline, so most of this is configuration, not code.

### 3.1 What Open WebUI gives us out of the box

Open WebUI has built-in document RAG with a pluggable embedding engine, hybrid search, and reranking ([Open WebUI RAG docs](https://docs.openwebui.com/features/chat-conversations/rag/), [Retrieval and Reranking, DeepWiki](https://deepwiki.com/open-webui/open-webui/7.6-retrieval-and-reranking)):

- **Embedding engine** (`RAG_EMBEDDING_ENGINE`): default `""` = local SentenceTransformers (runs in the Open WebUI backend, CPU by default on our box). Alternatives: `ollama`, `openai`, `azure_openai`.
- **Default embedding model** (`RAG_EMBEDDING_MODEL`): `sentence-transformers/all-MiniLM-L6-v2`. This is the important gotcha: MiniLM-L6 is **384-dim and truncates input at 256 word-piece tokens**. If your chunk is larger than ~256 tokens, the tail is silently dropped before embedding. It is fast and tiny (~22M params, ~90 MB) but it is the weakest link in the default pipeline.
- **Chunking**: `CHUNK_SIZE` / `CHUNK_OVERLAP`, configured in Admin > Settings > Documents. Defaults have drifted across versions (commonly ~1000 / 100 in char-ish units, some builds ~500); treat them as unset and pin them explicitly.
- **Retrieval**: `RAG_TOP_K` (small by default, ~3), plus `RAG_TOP_K_RERANKER` and `RAG_RELEVANCE_THRESHOLD` (`r`).
- **Hybrid search** (`ENABLE_RAG_HYBRID_SEARCH`): **off by default.** When on, it adds BM25 keyword matching to vector search and a CrossEncoder rerank stage ([Advanced RAG in Open WebUI, Crazy Alpaca](https://thecrazyalpaca.com/blog/advanced-rag-strategies-re-ranking-and-hybrid-search-in-open-webui), [Multi-Source RAG, ProductivAI](https://productiv-ai.guide/start/multi-source-rag-openwebui/)). Common reranker: `BAAI/bge-reranker-v2-m3`. Note a known bug where reranking can return empty results in some configs ([discussion #13221](https://github.com/open-webui/open-webui/discussions/13221)), so verify retrieval actually returns chunks.

### 3.2 Which embedding model fits our rig, and is worth the swap

We have 48 GB RAM and 8 GB VRAM, but the VRAM is spoken for by the LLM. The embedder should run on **CPU** (or share GPU only when no LLM is loaded) so it never competes with the flagship for VRAM. Candidates, all local, ranked by fit:

| Model | Params | Dim | Max ctx | Notes | Footprint |
|---|---|---|---|---|---|
| all-MiniLM-L6-v2 (default) | ~22M | 384 | **256 tok** | fast, weak, truncates chunks | ~90 MB |
| **nomic-embed-text-v1.5** | 137M | 768 (MRL 256/128) | **8192** | strong small model, long ctx, needs `search_query:`/`search_document:` prefixes | ~280 MB |
| **EmbeddingGemma-300M** | 308M | 768 (MRL 512/256/128) | 2048 | best-in-class under 500M on MTEB, runs in <200 MB quantized, needs task prefixes | <200 MB q |
| **BGE-M3** | 568M | 1024 | 8192 | dense+sparse+multivector in one model, 100+ languages, heavy production default | ~1.2 GB |
| **Qwen3-Embedding-0.6B** | 600M | 1024 (MRL 32-1024) | **32768** | same family as our LLMs, instruction-aware, 32k ctx, MTEB-multi 64.3 | ~1.2 GB |

Sources: [EmbeddingGemma blog](https://huggingface.co/blog/embeddinggemma) and [arXiv 2509.20354](https://arxiv.org/abs/2509.20354); [Qwen3-Embedding-0.6B card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B); [Milvus 2026 embedding guide](https://milvus.io/blog/choose-embedding-model-rag-2026.md); [Best open-weight embedding models 2026, Presenc](https://presenc.ai/research/best-open-weight-embedding-models-2026).

Reading of the field: **Qwen3-Embedding-8B tops the open MTEB leaderboard (~75)**, but the 8B/4B variants are overkill for a laptop CPU embedder. **BGE-M3 is the most-deployed production model** (dense+sparse+ColBERT-style multi-vector in one pass). For our rig the sweet spot is one of **nomic-embed-text-v1.5, EmbeddingGemma-300M, or Qwen3-Embedding-0.6B**: all clear the 256-token wall of the default, all run comfortably on CPU, and all sharply beat MiniLM on real retrieval. Qwen3-Embedding-0.6B is the intriguing one because its 32k context means a whole chunk (even a large one) embeds without truncation, and it is instruction-aware like our LLMs. Two caveats: nomic and EmbeddingGemma need the correct query/document **prefixes** or they quietly lose 1-5%+ of their quality, and Open WebUI must be told to apply them.

### 3.3 Reranking: usually worth it, and it does not touch decode t/s

A cross-encoder reranker (`bge-reranker-v2-m3`) re-scores the top vector hits and reorders them, which reliably lifts precision, especially for technical text where exact terms matter ([Advanced RAG in Open WebUI](https://thecrazyalpaca.com/blog/advanced-rag-strategies-re-ranking-and-hybrid-search-in-open-webui)). Its cost is a single extra forward pass over (query, chunk) pairs on CPU, on the order of tens to low hundreds of ms for a handful of candidates. Because it lets you send **fewer, better** chunks to the LLM, it can *reduce* net prefill wall-clock while raising quality. This is the clean quality-per-second win.

### 3.4 Chunking: pin it, do not accept defaults

2026 consensus ([Firecrawl chunking guide](https://www.firecrawl.dev/blog/best-chunking-strategies-rag), [Denser.ai](https://denser.ai/blog/rag-chunking-strategies/), [Digital Applied](https://www.digitalapplied.com/blog/rag-chunking-strategies-2026-retrieval-quality-playbook)):

- **~512 tokens is the best default chunk size**; 256 for short Q&A, 1024 for long technical docs. Below ~128 chunks fragment; above ~1000 the embedding signal dilutes.
- **Recursive character splitting** is the best default strategy (~69% end-to-end accuracy in benchmarks); **semantic chunking** can add a couple points (~71%) at higher indexing cost.
- **Overlap**: 10-20% is the traditional advice, but a Jan-2026 systematic analysis found overlap gave **no measurable benefit** and only raised indexing cost. Start at 10% or zero and only raise if recall is poor.
- Chunking matters **as much as embedding-model choice** for retrieval quality (NAACL 2025), and the wrong strategy can cost up to 9% recall. So chunk config is a first-class experiment variable, not an afterthought.

Critical interaction: if you keep the default 256-token MiniLM embedder but set 512-token chunks, half of every chunk is discarded before embedding. Chunk size and embedder max-context must be chosen together.

### 3.5 Does retrieval quality justify the prefill cost? (the core question)

On our rig the arithmetic favors RAG strongly:

- **Stuffing a 32k document**: minutes of CPU-expert prefill every session, and it drives the model into the context-rot / lost-in-the-middle regime (section 4), so quality can actually *drop* versus a focused prompt.
- **RAG top-5 at 512 tokens**: ~2.5k prefill tokens, seconds of prefill, plus a fast CPU embed of the query and a cheap rerank. Same decode t/s. Higher answer precision when the answer is a localized fact.
- The honest boundary: RAG wins when the useful information is **localized** (facts, snippets, specific passages). It loses when the task needs **global** understanding of the whole document (summarize this 50-page contract, find cross-references), where you genuinely need the long context and should instead lean on prompt caching to amortize the prefill. So the right rig-level design is: RAG for lookup-style questions, long-context-plus-prompt-cache for whole-document reasoning. Both avoid re-prefilling from scratch.

---

## 4. Context management given our 64k capability

Having 64k does not mean filling 64k. The 2025-2026 literature is blunt: every model gets worse as input grows, so context is a budget to spend carefully, not a bucket to fill.

### 4.1 The two failure modes

- **Lost in the middle**: accuracy is a U-shaped function of position; information at the very start or very end is used well, and mid-context material degrades by 30%+. Replicated across many model families ([Lost in the Middle explainer, Morph](https://www.morphllm.com/lost-in-the-middle-llm), [Atlan](https://atlan.com/know/llm/lost-in-the-middle-problem/)). Root cause is partly RoPE long-range decay, so it is architectural, not a bug that training fixes.
- **Context rot**: Chroma tested 18 frontier models (including Qwen3) and found *all* degrade with input length, even on trivial tasks, and even well below the advertised window ([Chroma context rot report](https://www.trychroma.com/research/context-rot), [Morph summary](https://www.morphllm.com/context-rot)). Three findings we can exploit: (a) lower query-to-answer semantic similarity accelerates the decay, so good retrieval that raises similarity directly buys length-robustness; (b) even a single distractor lowers accuracy, so a tight, reranked context beats a big loose one; (c) models did *better on shuffled haystacks than coherent ones*, hinting that how context is arranged matters as much as what is in it.

### 4.2 Runnable-now tactics (all zero per-token cost)

1. **Keep the working context small and relevant.** Retrieve-then-rerank to a handful of chunks rather than dumping documents. This is the same move as section 3, motivated from the context side.
2. **Put the query/instruction last.** With the answer-relevant material and the actual question near the end of the prompt, you sit in the strong end-of-context region. Free reordering.
3. **Bracket, do not bury.** If you must include a long block, put the most important content at the top and the instruction at the bottom; avoid stranding the key fact in the middle.
4. **Prune history.** Strip think blocks (section 1.3) and stale turns; summarize old context into a short running state rather than carrying raw transcript. Multi-turn chats rot precisely because old turns accumulate.
5. **Amortize prefill instead of re-paying it.** From [sweep-01 kv-cache](../2026-07-sweep-01/kv-cache-long-context.md): `--cache-ram` (host-RAM prompt cache, default 8192 MiB, may already be silently active) and `--slot-save-path` + `POST /slots/0?action=save|restore` turn "re-read the 32k doc every morning" into a seconds-long restore. This does not fight context rot but it removes the prefill penalty that otherwise makes long context impractical on our CPU-expert box.
6. **Hybrid-model caveat.** On the Qwen3.6 reasoning route (Gated DeltaNet hybrid), context shift and KV shifting are disabled ([sweep-01, discussion #24944](https://github.com/ggml-org/llama.cpp/discussions/24944)); rely on `--ctx-checkpoints` there and prefer keeping its context lean, since you cannot cheaply roll its window.

### 4.3 The synthesis

For our stack the optimal policy is: **small, high-relevance, well-ordered context, assembled by retrieval + reranking, with prefill amortized by the KV cache**, and long-context-dump reserved only for genuine whole-document reasoning. That maximizes quality per wall-clock second because it minimizes prefill while staying out of the rot regime, and it never touches decode t/s.

---

## Candidate experiments for our rig

All follow project protocol: A/B/A flanking where a config is toggled, warm cache, quality judged at a **fixed output-token budget** (so decode t/s is held constant and only quality moves), extra wall-clock reported separately as the real cost. Everything runs through the hub with orchestration we write. No retraining, no cloud.

### C1. Sampling + system-prompt audit and fix (flagship + reasoning routes)

- **What**: enumerate the sampling params each hub route and each client (Open WebUI, opencode) actually sends. A/B the documented per-model settings (Instruct-2507: temp 0.7 / top_p 0.8 / top_k 20 / min_p 0; Qwen3.6 reasoning: temp 0.6 / top_p 0.95 / top_k 20) against whatever is live now, plus a structured system prompt (role + output contract + rules + refusal policy) vs the current one. Small fixed task battery, fixed output length, blind-scored (Qwen3-8B as cheap judge, spot-checked).
- **Why**: zero download, zero decode cost, likely the biggest quality-per-effort win in the sweep. If any route is running greedy or wrong-mode sampling, this is pure lost quality we recover for free.
- **Needs**: nothing. Hours.
- **Class**: free quality (no token cost at all).

### C2. Local RAG embedding bake-off, and retrieval-vs-long-context-dump

- **What**: build a small representative corpus (our own docs/notes). Compare embedders in Open WebUI: default all-MiniLM-L6-v2 vs nomic-embed-text-v1.5 vs EmbeddingGemma-300M vs Qwen3-Embedding-0.6B (correct prefixes applied), at 512-token recursive chunks. Measure retrieval quality (recall@k on a hand-labeled query set) and end-to-end answer quality of RAG-top-k versus stuffing the full document into 32k/64k context. Report prefill wall-clock and query-embed time for each path.
- **Why**: directly answers the charter question "does retrieval quality justify the prefill cost." Establishes our house embedder and proves (or refutes) that focused retrieval beats long-context dump for lookup tasks on our specific models.
- **Needs**: small downloads (embedders, ~90 MB to ~1.2 GB each). Days.
- **Class**: capability + quality-per-second.

### C3. Reranking value test (hybrid BM25 + bge-reranker-v2-m3)

- **What**: with the C2 winner as embedder, A/B pure vector top-k vs hybrid (BM25 + vector) + `bge-reranker-v2-m3`. Measure answer quality and, crucially, net prefill tokens sent to the LLM (reranking should let us send fewer, better chunks). Report the reranker's CPU pass latency.
- **Why**: tests whether reranking raises quality while *lowering* net prefill (a rare win-win), and whether hybrid search matters for our technical/code-heavy corpus. Guards against the known empty-result rerank bug.
- **Needs**: reranker download (~1 GB). Hours-to-days, gated on C2.
- **Class**: quality-per-second.

### C4. Few-shot exemplar selection: kNN vs zero-shot, instruct vs reasoning

- **What**: build a small labeled exemplar store, kNN-retrieve top-k similar shots via the C2 embedder, and A/B {zero-shot, fixed few-shot, kNN few-shot} on a structured-output task, run on Qwen3-8B, Instruct-2507, and the Qwen3.6 reasoning route, all at a fixed output-token budget. Order the best exemplar last.
- **Why**: tests the 2025 claim that few-shot backfires on strong reasoning models but still helps smaller/instruct models, on *our* models. Tells us a routing rule: exemplars for 8B/Instruct extraction, clean zero-shot for the reasoning driver.
- **Needs**: nothing beyond the C2 embedder. Hours.
- **Class**: quality-at-fixed-budget.

### C5. Context-position and context-rot probe on our stack

- **What**: needle-in-haystack plus distractor and shuffled-vs-coherent variants at 4k/8k/16k/32k/64k on the flagship (and a short run on the reasoning route). Vary query position (start vs end) and needle-question similarity. Chart accuracy vs depth to find our models' usable-context ceiling.
- **Why**: confirms whether lost-in-the-middle and context rot bite our specific models and *where*, which sets the maximum useful retrieval context for C2/C3 and validates the "query-last, keep-it-small" tactics. Zero download, pure orchestration, reuses our long-context harness.
- **Needs**: nothing. Hours.
- **Class**: incremental (calibrates C2-C4 and section 4 tactics).

Suggested order by risk-adjusted value per hour: **C1 -> C5 -> C2 -> C3 -> C4.** C1 is free and immediate; C5 is free and sets the context ceiling everything else depends on; C2 is the flagship experiment that answers the prefill-cost question; C3 and C4 refine the winner.

---

## Watch list

- **Qwen3-Embedding-4B/8B**: top of MTEB but heavy for a CPU embedder; revisit if we ever want a GPU-resident embedder during no-LLM windows.
- **EmbeddingGemma finetune path**: it is designed for cheap on-device finetuning; a domain-adapted embedder is a future lever if generic retrieval underperforms on our corpus.
- **Delta-KNN / MarginSel exemplar selection** ([ACL 2025](https://aclanthology.org/2025.acl-long.1253.pdf), [arXiv 2506.06699](https://arxiv.org/pdf/2506.06699)): if plain kNN few-shot (C4) shows value but noisy variance, these are the next refinement.

## Dead ends / non-starters for this sweep

- Anything claiming to raise **decode t/s** via prompting: impossible, throughput is bandwidth-bound.
- Treating 64k as free capacity to fill: context rot makes big loose contexts a quality *loss*, not a gain.
- Greedy decoding for "determinism" on Qwen3: documented to break the models.

---

## Feasibility verdicts

Adversarial review 2026-07-20. Checked each candidate against: fits 8 GB VRAM / 48 GB RAM / AVX2 / Windows / single-machine hub; runnable TODAY (not paper-only); honestly zero per-decoded-token speed cost (no secret concurrent VRAM-competing model, no retraining); effort honest. All five survive; none are paper-only and none reduce decode t/s. Caveats are about getting full quality, not about whether the experiment runs.

- **C1 (sampling + system-prompt audit) — GO.** Pure config, zero download, zero decode cost, correct that sampling has negligible per-token cost. Only nuance: client-sent params (Open WebUI/opencode) override llama-server launch defaults, so "pin in hub config" alone is insufficient; C1 already audits per-client sends, which covers it. Judge runs (Qwen3-8B) are sequential swaps, not concurrent. Highest value-per-hour; do first.
- **C2 (embedding bake-off + RAG vs long-context dump) — GO, with the one real rig trap flagged.** Runnable today via Open WebUI `RAG_EMBEDDING_MODEL`; downloads trivial at ~3 MB/s (1.2 GB embedder is ~7 min). Decode t/s untouched (embedding is one-time CPU indexing + per-query embed). TRAP: SentenceTransformers/torch auto-selects CUDA if it sees the GPU, so the embedder can silently load onto the 8 GB card and evict/OOM the flagship. Must force the embedder to CPU (the doc asserts "CPU by default on our box" — verify, do not assume). Second caveat: nomic / EmbeddingGemma / Qwen3-Embedding need task prefixes (and nomic needs `trust_remote_code`); if Open WebUI does not apply them the non-default embedders are under-measured. Neither kills the experiment. Effort "Days" honest.
- **C3 (reranking / hybrid BM25 + bge-reranker-v2-m3) — GO.** bge-reranker-v2-m3 (~568M) fits RAM; one-time CPU forward pass, zero decode cost, and it can lower net prefill (real win-win). Same CPU-pinning trap as C2 (reranker must not grab the flagship's VRAM). Known empty-result rerank bug is acknowledged and guarded. Gated on C2. Effort honest.
- **C4 (few-shot kNN vs zero-shot across routes) — GO.** Pure orchestration + the C2 embedder, no new infra, runnable today. At a fixed output-token budget decode t/s is held constant; extra exemplars are prefill only (honest cost). Routes run sequentially via hub auto-swap, no concurrency. Gated on C2 embedder. "Hours" is fair for a small battery.
- **C5 (context-position / context-rot probe) — GO, effort optimistic on machine wall-clock.** Zero download, reuses the confirmed 64k harness (`Start-30B-AI-64K.bat`, sweep-01). Pure measurement, decode untouched. Human effort is hours, but the full 5-length x position x distractor x shuffle matrix with repeated 64k CPU-expert prefills is many hours of machine time; run in background/overnight. Reasoning-route arm is correctly limited to a "short run" given disabled KV/context-shift on the Gated DeltaNet hybrid (sweep-01). Sets the context ceiling for C2-C4; do early.
