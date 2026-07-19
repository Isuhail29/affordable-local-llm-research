# QUEUE: quality-extraction research phase (2026-07-sweep-02-quality)

Compiled 2026-07-20 from the eight verified surveys in this folder (test-time-compute, self-consistency-ensemble, verifier-critic, sampling-decoding, prompt-context-rag, multi-model-orchestration, imagegen-models, imagegen-runtime), reading every Feasibility verdict. Only GO and strong-MAYBE candidates survive. Duplicate proposals across surveys are merged into one ranked entry.

## Framing (governs every rank)

Decode throughput is pinned by memory bandwidth at ~30-42 t/s and nothing here raises it. So "improve quality without reducing speed" means two things: (1) quality at fixed speed AND fixed token budget (the zero-cost lane), and (2) quality gained per extra wall-clock second when we spend more tokens (the token-budget lane). Everything runs through the llama-swap hub at :9292 with orchestration we write. No retraining, no cloud. Track A is ranked by quality-gained-per-second per unit effort, not by headline accuracy.

Two rig facts do the heavy lifting:
1. Parallel samples are NOT free but are sublinear. llama.cpp continuous batching (`--parallel N`) amortizes weight/expert reads across concurrent sequences, so K samples of one prompt can cost far less than Kx wall clock. The exact discount is the single measurement that reprices the whole token-budget lane (entry B1).
2. The hub is a single-slot switchboard with a 30-90 s swap tax. Single-model loops (self-consistency, best-of-N, Self-MoA, grounded refine) pay zero swap. Every cross-model round pays the tax, so cross-model patterns are batch-only or dead until co-residency is proven (which the surveys expect to fail on 8 GB).

---

# TRACK A: LLM quality-per-wall-clock

## Group 1: Zero-speed-cost wins (do first)

These spend no extra tokens and cost no t/s. Pure quality at fixed speed and fixed token budget. Strictly dominant, so they lead the queue.

### A1. Sampling-param and system-prompt audit and fix, plus top-n-sigma on the reasoning flagship
Merges prompt-context C1 and sampling-decoding E-SAMP-1, the two surveys that independently flagged this as the biggest quality-per-effort lever. Enumerate what sampling params each hub route AND each client (Open WebUI, opencode) actually sends, because client-sent values override launch flags. Pin the vendor-correct per-model presets and kill any greedy default (documented to break Qwen3 thinking models): Instruct-2507 at temp 0.7 / top_p 0.8 / top_k 20 / min_p 0; Qwen3.6-35B reasoning at temp 0.6-1.0 / top_p 0.95 / top_k 20 / presence 1.5. Add a structured system prompt (role, output contract, enumerated rules, refusal policy). Then A/B `--top-nsigma 1.0` on the reasoning flagship: it is the one truncator with a mechanism-level reason to help reasoning (isolates the informative-logit region, temperature-invariant), and it is already server-exposed in b10068.
- First step: post a request through :9292 and log the effective sampler chain; diff against `llama-swap.yaml` launch flags and the Open WebUI/opencode client sends.
- Effort: hours.
- Success: measurable quality lift on a frozen 40-60 item battery (math, code, factual, a few creative) at identical output-token budget, blind-scored by Qwen3-8B and spot-checked; top-n-sigma arm B >= baseline A on reasoning accuracy, and a temp-1.3 arm C holds near A where a top-p-only run would collapse. Kill: no route was mis-sampled and top-n-sigma is flat.

### A2. Context-rot and lost-in-the-middle probe (sets the context ceiling for everything else)
prompt-context C5. Free, pure orchestration, reuses the shipped 64k harness. Needle-in-haystack plus a distractor and a shuffled-vs-coherent variant at 4k/8k/16k/32k/64k on the flagship, varying query position (start vs end) and needle-question similarity. This calibrates the maximum useful retrieval context for A4 and validates the query-last, keep-it-small tactics. Reasoning-route arm stays short because KV and context-shift are disabled on the Gated DeltaNet hybrid.
- First step: run `Start-30B-AI-64K.bat`, drive a needle set at 5 depths x {query-first, query-last} through :9292.
- Effort: hours of human time, many hours of machine time (run overnight).
- Success: a depth-vs-accuracy curve locating our usable-context ceiling and confirming query-last beats query-first and that a single distractor measurably hurts. Kill: no degradation anywhere across 64k (would be a surprising null; still publishable).

### A3. DRY long-context loop-breaker and reason-then-constrain structured output
sampling-decoding E-SAMP-3 (GO) plus E-SAMP-4 (strong-MAYBE). Two cheap, high-certainty sampler wins tied to shipped configs. DRY at multiplier 0.8 is a safer loop-breaker than flat repeat-penalty for reasoning models that loop deep in 64k contexts (it penalizes repeated n-grams, not legitimate frequent tokens). Reason-then-constrain tests the constrained-decoding-degrades-reasoning finding: let the model reason free-form, then constrain only the final answer field, versus wrapping the whole response in a JSON-schema grammar.
- First step: add `--dry-multiplier 0.8` (base 1.75, allowed-length 2) to the 64k launcher and count repetition incidence on long-generation prompts.
- Effort: hours.
- Success: DRY reduces loop/repetition incidence with no accuracy cost; reason-then-constrain matches full-grammar on JSON validity (~100%) while beating it on the reasoned field's correctness. Kill: DRY changes nothing (no looping present) and full-grammar shows no reasoning penalty on our models.

### A4. Local RAG embedding bake-off plus reranking (retrieval vs long-context dump)
prompt-context C2 and C3. Higher effort than A1-A3 (downloads, days) but disproportionately valuable on our rig because prefill is CPU-expert-bound: retrieving ~2.5k relevant tokens beats prefilling a 32k document on both wall-clock AND quality (it dodges context rot). Bake off the MiniLM default against nomic-embed-text-v1.5, EmbeddingGemma-300M, and Qwen3-Embedding-0.6B (correct task prefixes applied, embedder forced to CPU) at 512-token recursive chunks, then add hybrid BM25 plus bge-reranker-v2-m3. Establishes the house embedder and answers the charter question of whether retrieval quality justifies the prefill cost.
- First step: build a small labeled corpus and query set, set `RAG_EMBEDDING_MODEL` in Open WebUI and force the embedder to CPU (verify torch does not silently grab the 8 GB card and evict the flagship).
- Effort: days.
- Success: a non-default embedder beats MiniLM on recall@k, reranking raises answer quality while lowering net prefill tokens sent to the LLM, and RAG-top-k beats a 32k dump on lookup tasks. Kill: MiniLM already saturates recall on our corpus and reranking adds no precision.

### A5. Capability router in front of the hub (swap-avoiding, not swap-adding)
multi-model C3. Not a test-time-compute technique, but a strictly-positive quality-per-wall-clock win: a thin rules/embedding classifier (kept CPU-resident, never a call to a non-loaded model) maps each request to the single best resident model (code to the coder, vision to qwen-vision, hard reasoning to qwen-35b-reasoning, chat to qwen-30b, trivial to qwen-8b-fast). It adds no passes; it prevents mis-routed low-quality answers and wasted swaps. Product-flavored, so ranked last in this group despite low risk.
- First step: build the rules/embedding router as a proxy in front of :9292 on a labeled request sample; the classifier must not invoke a non-resident model.
- Effort: days.
- Success: measured swap-count reduction and mis-route-avoidance versus naive per-request model choice, with routing accuracy reported. Kill: routing accuracy too low to beat always-flagship.

## Group 2: Token-budget-spending (worth the wall-clock)

These spend extra tokens, which at fixed t/s is extra time-to-answer (no t/s loss). Judged on quality gained per extra second. B1 gates the economics of the rest.

### B1. The gate: `--parallel N` batching-discount measurement
test-time-compute C1 and self-consistency C1, the same experiment proposed by two surveys. This is the referee that reprices the entire token-budget lane. Launch the flagship with `--parallel {2,4,8}` and continuous batching, fire K identical requests concurrently, and measure aggregate vs per-stream t/s and per-request wall clock against the single-slot baseline. Predict the scaling from the expert-union model U(B) first, then falsify. Determines whether K self-consistency or best-of-N samples cost ~Kx or far less. Publishable either way as a first MoE-experts-on-CPU batching datapoint. Watch item: `-c` splits across slots, so size per-slot context for long reasoning traces (q8-KV per E040 doubles the room).
- First step: relaunch qwen-30b (then qwen-35b-reasoning) with `-np 8 -c 16384`, script 8 concurrent `/v1/completions` on one shared prompt, log aggregate vs single-stream t/s and VRAM.
- Effort: hours.
- Success: K=4 finishes in under 2.5x single-request wall clock (real batch discount, green-lights B2-B6 as cheap). Kill: near-Kx scaling with no amortization (expert-union dominates), which reframes the lane as a pure quality-vs-time tradeoff rather than a cheap win.

### B2. Self-consistency quality-per-second frontier on hard-for-our-model tasks, with adaptive early stop
test-time-compute C2 and self-consistency C2, the flagship token-spend buy. Gated on B1. On both flagships, run K in {1,4,8,16} majority vote at T~0.6 over a HARD, machine-checkable slice (AIME-class math, GPQA-diamond subset, MATH-500-hard, an MCQ slice), plotting accuracy AND accuracy-gain per wall-clock second at B1's real cost. Then add the Adaptive-Consistency stopping rule to cut average K. This directly tests whether the 2022 self-consistency gains survive on our 2026 models (the "SC is losing its edge on strong models" correction), so the eval set must be genuinely hard for our models or the result is a wash by construction.
- First step: build exact-match graders plus a K-sample vote harness against the hub; run K=8 on 100 hard math/AIME items on both models.
- Effort: days.
- Success: >= +5 accuracy points on the hard set at <= 2.5x wall clock (clears the bar to ship as a "hard mode" hub preset), with adaptive-K delivering it at ~1.5-2x average cost. Kill: no task slice clears +3% even at K=16.

### B3. Grounded tool-critique and execution-selection loop for code (CRITIC plus CodeT)
The single highest-expected-value loop across four surveys (test-time-compute C4, verifier-critic C1, self-consistency C3, multi-model C5), because code has the largest generation-verification asymmetry we own: the verifier is real execution, not model opinion. Generate K solutions (plus K generated tests for CodeT dual-execution selection), run them in a sandboxed subprocess, feed real tracebacks back for a bounded refine (cap 2-3), select by test agreement or stop-on-green. Include a no-checker MATH control to reproduce the documented intrinsic-refine null, proving to ourselves where refine pays and where it degrades. Single model, zero swap; the real work and risk is a safe sandbox for model-generated code on Windows.
- First step: build a timeout-guarded, no-network WSL/container sandbox runner; K=8 solutions plus tests on 50 HumanEval-style problems via the coder route.
- Effort: days (sandbox plus safety boundary is most of it).
- Success: >= +10 points solve rate from grounded refine/selection at a wall-clock cost B1 shows is under Kx, with the math control flat-or-negative. Kill: code gain < +3 points (the coder already one-shots the set).

### B4. Adaptive thinking-budget sweep (map our overthinking inverted-U)
test-time-compute C3, taking the clean-GO first step and leaving the live router as a deferred MAYBE. The one technique that is quality-positive AND time-negative: it reclaims seconds by not overthinking. Sweep fixed think caps {1K,2K,4K,8K,16K} across easy/medium/hard tiers on the reasoning flagship, injecting `</think>` at the cap, and plot accuracy and time to locate our peak per difficulty tier. Keep difficulty classification offline (pre-classify the frozen set once); a live per-request 8B classifier is infeasible because it cannot co-reside with the `--cpu-moe` flagship in 8 GB and would thrash the swap hub.
- First step: run the 35B at fixed max-think caps across the difficulty tiers with a stop-at-cap condition.
- Effort: days.
- Success: a per-tier accuracy-vs-length curve with an exploitable peak that cuts mean time-to-answer >= 20% at matched accuracy. Kill: flat accuracy-vs-length curves (no overthinking regime; document the null and drop the router).

### B5. Gated best-of-N with a generative verifier (same-model zero-swap; cross-model arm measures self-preference)
multi-model C2, verifier-critic C2, self-consistency C4. On one resident model, sample N, have the same model GenRM-score and rank them, gated behind a cheap confidence check so only hard queries pay the N-pass cost. Compare token efficiency against B2's self-consistency baseline (literature reports 1-3x token savings on reasoning/code, shrinking on factual per the generator-verifier gap, so the eval set must be tagged by task type). Add a swap-serialized cross-model arm (30B generates, 35B-reasoning judges) to quantify the self-preference penalty and whether cross-model beats same-model by more than the swap cost. Fold in USC plus self-certainty for free-form tasks where majority vote does not apply.
- First step: build an N=8 sample plus same-model GenRM-rank harness on a task-tagged reasoning/code/factual set; measure accuracy-per-token vs SC.
- Effort: days.
- Success: verifier selection beats SC at matched tokens on reasoning/code, and the cross-model judge beats the same-model judge by more than its swap cost. Kill: no selector beats SC, or USC positional bias dominates.

### B6. Self-MoA-Seq for open-ended writing and QA (single-model, zero swap)
multi-model C1. The version of Mixture-of-Agents that fits a swap hub, and the answer for open-ended tasks where B2's majority vote cannot apply. Sample N candidates from the single best model, then slide a fold-in window aggregating best-so-far with new samples at constant context (Self-MoA-Seq), which the paper reports as effective as aggregating all at once. Self-MoA is documented to beat mixed MoA, so single-family aggregation is the right bet on our hub.
- First step: script the sequential fold-in loop against one endpoint; sweep N=2/4/8 on a writing/QA rubric set with 1-shot and plain-SC-at-same-N baselines.
- Effort: hours.
- Success: Self-MoA-Seq beats 1-shot and matches or beats plain SC on the rubric at a quantified quality-per-extra-second. Kill: single-family aggregation gives no lift over 1-shot.

---

# TRACK B: uncensored image generation

Diffusion physics is the opposite of our LLM law: denoising is GPU-compute-bound, not host-bandwidth-bound, so if the denoiser fits 8 GB it runs at full Blackwell speed. The 8 GB is a capacity wall, not a speed wall. Cost axis is seconds-per-1024px-image and peak VRAM, not t/s. Image and LLM backends never co-reside; llama-swap unloads the LLM before loading the diffusion model (image gen is just another swap target).

## Recommended setup (the single pick): stable-diffusion.cpp `sd-server` behind the hub

The on-brand, lowest-friction path (imagegen-runtime TL;DR and E-IMG-1, both GO). sd.cpp is the literal llama.cpp sibling: pure C/C++ ggml, GGUF weights, builds exactly like b10068 with `-DSD_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120`, and has NO PyTorch dependency, so the entire sm_120 headache disappears. Its `sd-server` exposes a native OpenAI `/v1/images/generations` endpoint, so it drops behind llama-swap with zero translation glue and Open WebUI's image button just works. `--vae-on-cpu` and `--clip-on-cpu` offload encoders to our 48 GB RAM, keeping the 8 GB for the denoiser. Keep ComfyUI-GGUF portable (also GO, E-IMG-2) as the quality-ceiling fallback if sd.cpp's images disappoint, since it needs only a ~40-line OpenAI-images shim.

### Short path to first image
1. Build sd.cpp from source (`cmake .. -DSD_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120`; ignore any snippet saying `=90`, that is Hopper) or grab the prebuilt Windows CUDA binary from releases.
2. Download one SDXL uncensored finetune (~6.9 GB, ~40 min): Juggernaut XL (photoreal) or NoobAI-XL V-Pred (anime). SDXL is CLIP-only and fully resident in 8 GB with no T5 offload, the fastest thing to first image.
3. Register `sd-server` as a llama-swap entry, generate one 1024x1024 image, then drive it from Open WebUI via `/v1/images/generations`.
- Success (E-IMG-1 falsification): the C++/ggml path runs acceptably on sm_120 without PyTorch and the image quality is good enough not to need the PyTorch stack. Kill: sm_120 build fails or quality is unusable, in which case fall back to ComfyUI portable (ships cu128, day-1 Blackwell).

## Quality ladder (climb after first image, GO models only)
1. SDXL uncensored finetune (~6.9 GB, fully resident, no T5): the fast, unrestricted floor and deepest LoRA/ControlNet ecosystem. NoobAI V-Pred needs v-pred settings plus Euler, CFG 4-5, 28+ steps.
2. Z-Image Turbo (6B, Apache 2.0, ~6 GB GGUF, 8 NFEs, small Qwen-class encoder so no T5 gotcha): the sleeper. Best quality-per-second and per-GB on the rig if the leaderboard claim survives our own falsification.
3. Chroma1-HD Q5_K_S (~6.5 GB, Apache 2.0, uncensored by design, Flux-class): the flagship uncensored pick. Fits 8 GB with the T5 encoder CPU-offloaded (expect occasional `--lowvram`). Most likely to become the default.
4. Flux.1-dev Q4_K_S (~6.8 GB plus T5 offload): coherence and in-image-text ceiling, kept as an internal reference only (non-commercial license), used to benchmark the three above.

Then run the step/quant sweep (E-IMG-4, GO): find the lowest steps and smallest quant that clears the quality bar, the MEASURED-LAW analogue for images, giving defensible hub launcher defaults.

---

# Skip list: seductive but doomed on our rig

- Budget forcing ("Wait") on the 35B reasoning model. Approximates what our RL-trained reasoner already does natively; saturates and oscillates correct answers into wrong ones (arXiv 2507.14419). Only meaningful as a probe on the non-reasoning Instruct model, where the reasoning model is still the better answer.
- Tree-of-Thoughts / Graph-of-Thoughts as general infrastructure. 50-100x mostly-sequential token cost for gains confined to combinatorial-search tasks with a cheap evaluator. Shelf it until a specific such task appears.
- Intrinsic self-refine on math/logic (no external signal). Documented to degrade reasoning (arXiv 2310.01798, 2402.08115): the model flips correct answers. Build refine only where a tool grounds it (code, math checker).
- Small model (Qwen3-8B) as the sole final judge that can overrule the 30B. The weak-verifier failure mode (arXiv 2404.17140); it flips correct answers. The 8B is a router/filter and an ensemble member, never the arbiter.
- Homogeneous multi-agent debate. Loses to budget-matched self-consistency (arXiv 2502.08788), and manufactures confident agreement, not correctness. Only heterogeneous debate across our diverse models has a research-backed reason to win, and even that must beat SC after the swap tax.
- Classic mixed Mixture-of-Agents. A swap storm of 18 invocations on a single-slot hub, and Self-MoA shows mixing weaker proposers dilutes quality anyway. Use Self-MoA-Seq (B6).
- Trained process/outcome reward models and REBASE-style verifier search. The strongest scaling results need a reward model we cannot train. Our runnable ceiling is prompted GenRM plus execution.
- Cross-model token-level speculative decoding for quality. A speed technique, killed on our A3B MoE by the expert-union penalty; covered by the sweep-01 spec-decoding survey.
- Speculative cascades (token-level deferral). Best-in-class idea but no llama.cpp deferral-rule implementation exists. Watch upstream only.
- Co-residency config with two 30B-class MoE models. Dead on both VRAM (~8 GB non-experts before KV) and RAM (~34 GB experts). Only a 30B-flagship-plus-CPU-8B config is even worth probing, and even it is a marginal pass at best.
- Interactive (B=1) cross-model quality cascades before co-residency is proven. One swap roughly doubles time-to-answer; a cascade escalating over ~40% of queries is slower than always-flagship. Batch-only until then.
- Mirostat and typical-p samplers. Mirostat overrides and fights the modern sampler chain; typical-p is superseded by min-p and top-n-sigma. Note and skip.
- Aggressive min-p plus high-temp "creativity unlock" adopted on faith. The headline min-p paper was substantially rebutted (arXiv 2506.13681). Keep min-p enabled as a cheap truncator, but measure the creativity claim, do not assume it.
- XTC on reasoning or code endpoints. It deletes the obvious next token, which is usually the correct one in a proof or a function signature. Creative endpoint only.
- Full-grammar constraint over an entire reasoning chain. Degrades reasoning (arXiv 2408.02442). Constrain only the final answer field (reason-then-constrain, A3).
- Self-consistency on easy, long-context, or factual-lookup tasks. Wash to negative (arXiv 2511.00751, 2411.01101). The whole point of B2's hard-slice gate is to route these to a single pass.
- Massive-N coverage chasing (hundreds/thousands of samples). Coverage needs an oracle verifier to cash in; without one, selection saturates by N~16.
- Treating 64k context as free capacity to fill. Context rot makes big loose contexts a quality loss. Retrieve small, order query-last, amortize prefill with the KV cache.
- Flux.2-dev and HunyuanImage for images. Flux.2 shipped mandatory safety filtering in license and pipeline and does not fit 8 GB; Hunyuan is a 12 GB-plus tier that loses its zero-config-uncensored edge at aggressive 8 GB quant.
- Forge / reForge / Fooocus / Automatic1111 as the image runtime. All need manual torch surgery for Blackwell and are stale or unmaintained; ComfyUI offload matches their low-VRAM edge without the surgery.
