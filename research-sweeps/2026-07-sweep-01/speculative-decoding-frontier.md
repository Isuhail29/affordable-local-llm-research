# The speculative decoding frontier beyond E014

Survey date: 2026-07-19. Scope: every speculative decoding path in llama.cpp b10064 we have NOT tested (draft-eagle3, draft-mtp, draft-dflash, ngram-map-k, ngram-map-k4v, ngram-mod, ngram-cache), plus self-speculation and 2026 research on memory-bound and MoE setups. E014 established that draft-simple and ngram-simple are net losses on our rig because verifying k tokens touches the union of their experts (the expert-union penalty) and because a separate draft model shares the laptop TDP budget. The question here: which untested paths change those economics, and by how much.

Rig constraints applied throughout: i7-14650HX (AVX2 only), 48 GB DDR5-5600 (~60 GB/s ceiling, ~37-42 GB/s measured effective per E021-E028), RTX 5060 Laptop 8 GB, Windows 11, llama.cpp b10064 (official + instrumented source build), ~3 MB/s internet (~10.5 GB/hour), 416 GB free disk.

---

## 1. The b10064 spec-type inventory

Per the current [docs/speculative.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md), the spec types are:

| Type | Needs | Extra download | E-status |
|---|---|---|---|
| draft-simple | separate draft GGUF | draft model | TESTED, net loss (E014) |
| ngram-simple | nothing | 0 | TESTED, net loss (E014) |
| draft-eagle3 | EAGLE-3 head GGUF trained for the target | ~1-2 GB head | untested |
| draft-mtp | MTP head baked into the target GGUF | new target GGUF | untested |
| draft-dflash | block-diffusion draft GGUF trained for the target | ~1-3 GB draft | untested |
| ngram-map-k | nothing | 0 | untested |
| ngram-map-k4v | nothing | 0 | untested |
| ngram-mod | nothing (~16 MB hash pool) | 0 | untested |
| ngram-cache | nothing (optional stats files) | 0 | untested |

Two facts from the docs that matter to us:

- **Spec types can be combined**, e.g. `--spec-type ngram-mod,ngram-map-k4v`, and draftless types take precedence over the draft model when both fire. The [MTP clean-up PR #23269](https://github.com/ggml-org/llama.cpp/pull/23269) confirms MTP + ngram-mod + ngram-map-k4v run simultaneously. We have never tested a combined stack.
- The docs contain the exact sentence that describes our E014 finding from the other side: for ngram-mod, "MoEs require long drafts." See section 7 for why long drafts flip the expert-union math instead of losing to it.

There is also a bundled benchmark harness (SPEED-Bench, `tools/server/bench/speed-bench/`) for end-to-end spec-decoding measurement, worth checking against our own protocol.

---

## 2. EAGLE-3 (`--spec-type draft-eagle3`)

**Status: merged and runnable.** [PR #18039](https://github.com/ggml-org/llama.cpp/pull/18039) (see also [discussion #15902](https://github.com/ggml-org/llama.cpp/discussions/15902) and [issue #15305](https://github.com/ggml-org/llama.cpp/issues/15305)) added EAGLE-3: a one-layer draft that reads the target's hidden states from three layers through a fusion layer, so it reaches much higher acceptance than a standalone draft model. Both SpecForge and vLLM/AngelSlim checkpoint formats are supported.

**Numbers from the PR** (RTX A6000 48 GB):
- LLaMA3.1-8B-Instruct BF16: 3.28x speedup at 80.6% acceptance; Q4_K_M target: 2.26x.
- LLaMA3.3-70B-Instruct: 2.41x.
- **GPT-OSS-20B (MoE): 1.06x**, with the PR explicitly blaming "more experts invoked during verification." This is our E014 expert-union penalty, reproduced independently on a full-VRAM GPU rig.

**Compatible drafts for our flagship class:**
- [Tengyunw/qwen3_30b_moe_eagle3](https://huggingface.co/Tengyunw/qwen3_30b_moe_eagle3), trained for Qwen/Qwen3-30B-A3B, listed as the official pairing in the [EAGLE repo model list](https://github.com/HuYunhai-Alex/EAGLE-Qwen3). Head is roughly a single decoder layer plus fusion, order 1-2 GB in BF16.
- AngelSlim publishes EAGLE-3 heads across the Qwen3 line ([AngelSlim/Qwen3-4B_eagle3](https://huggingface.co/AngelSlim/Qwen3-4B_eagle3), [Qwen3-32B_eagle3](https://huggingface.co/AngelSlim/Qwen3-32B_eagle3), a 30B-A3B variant referenced in the PR, plus [Qwen3-VL-30B-A3B](https://huggingface.co/AngelSlim/Qwen3-VL-30B-A3B-Instruct_eagle3)). AngelSlim reports accept lengths of 1.8-3.5 and speedups of 1.4-1.9x in their own (GPU serving) harness.

**Practical catch for us:** no widely mirrored pre-converted EAGLE-3 GGUFs were found. Conversion is `convert_hf_to_gguf.py <eagle3_hf> --target-model-dir <target_hf> --outtype bf16`. The `--target-model-dir` flag exists "so it inherits the target's tokenizer and layer indices" per [issue #15305](https://github.com/ggml-org/llama.cpp/issues/15305). Whether the script reads target weights (61 GB BF16 download for Qwen3-30B-A3B, ~6 h at our line) or only config/tokenizer (KB) must be verified by reading the converter source before committing to this path.

**Known limitations:** batch size 1 only, crashes reported with `-sm tensor` (irrelevant to us), and third-party heads with attention-output gates (Baichuan style) unsupported.

**Verdict for our rig:** the 1.06x MoE datapoint is a strong prior against, BUT that measurement is on a homogeneous GPU where draft and verify contend for the same bandwidth. On our rig the draft head would run on the otherwise-idle GPU while experts stream from CPU RAM, so the draft cost term is closer to free (modulo the E014 TDP-sharing lesson, which is much milder for a ~1 GB head than for the 0.6-1.7B draft models we tried). It lives or dies on the expert-union math in section 7.

---

## 3. MTP heads (`--spec-type draft-mtp`) and the Qwen3.6 window

**Status: merged, and the single most important development for us since E013.** [PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673) added multi-token-prediction speculative decoding: the target model's own auxiliary MTP head(s) draft the next tokens during the same forward pass infrastructure, verified in bulk. No separate draft model, no separate KV, no second backbone fighting for TDP. Cleaned up in [PR #23269](https://github.com/ggml-org/llama.cpp/pull/23269).

**Flags:** `--spec-type draft-mtp --spec-draft-n-max 2` (or 3), `-fa on`. The MTP head must be baked into the GGUF; you need an MTP-specific GGUF variant.

**Reported numbers (PR #22673):**
- Qwen3.6-27B (dense): aggregate acceptance **0.8258** at n-max 2, **0.7218** at n-max 3, "more than 2x speed-up" at n=3.
- RTX 3090: **22.97 -> 42.45 t/s (~1.85x)**.
- Tested hardware includes RTX 3060 12 GB and Strix Halo, so it is not a datacenter-only path.
- Known costs: ~10% extra memory, prompt processing takes a hit, n_parallel=1 only.

**The model that matters: Qwen3.6-35B-A3B.** Released 2026-04-16, Apache 2.0 ([Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B), [release blog](https://qwen.ai/blog?id=qwen3.6-35b-a3b)). It is the direct successor of our flagship class: 35B total, **3B active**, 256 experts with 8 routed + 1 shared per token, hybrid layout "10 x (3 x (Gated DeltaNet -> MoE) -> 1 x (Gated Attention -> MoE))", 262k native context. Reported **73.4% SWE-bench Verified** ([review](https://www.buildfastwithai.com/blogs/qwen3-6-35b-a3b-review)), beating dense Qwen3.5-27B on coding benchmarks with 3B active. That is a generational quality jump over our Qwen3-30B-A3B at essentially the same active-parameter budget.

**MTP GGUFs already exist**, head included, no conversion needed:
- [ggml-org/Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/ggml-org/Qwen3.6-35B-A3B-MTP-GGUF) (reference)
- [unsloth/Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF): UD-Q4_K_XL **22.9 GB**, UD-IQ4_XS **18.2 GB**, Q3_K_XL **17.2 GB**
- [localweights/Qwen3.6-35B-A3B-MTP-IQ4_XS-GGUF](https://huggingface.co/localweights/Qwen3.6-35B-A3B-MTP-IQ4_XS-GGUF)

**Fit on our rig (estimates, verify at load):** with `--n-cpu-moe`, routed experts (~17-19 GB of the UD-Q4_K_XL file) sit in RAM (fine in 48 GB), while DeltaNet/attention/shared-expert/router/embeddings sit in 8 GB VRAM. The hybrid layout is a hidden bonus: only 10 of 40 layers carry KV cache and the 30 DeltaNet layers keep constant-size state, so long context costs a fraction of what Qwen3-30B-A3B pays. Download: 18.2-22.9 GB, ~1.7-2.2 h.

**Why MTP beats the E014 failure mode on paper:** (a) acceptance 0.72-0.83 is far above anything draft-simple achieved; (b) draft cost is one extra shallow pass on the GPU, not a second model; (c) the always-on shared expert amortizes perfectly across verified tokens; (d) the RTX 3090 datapoint is itself a batch-1, bandwidth-bound MoE paying the expert-union tax, and it still cleared 1.85x. Whether that survives 37-42 GB/s CPU expert streaming instead of ~900 GB/s GDDR6X is exactly the section 7 calculation, and no public datapoint exists for MTP + `--n-cpu-moe` (searched, nothing found), so we would be producing a novel result either way.

**Risks:** b10064 must actually load the qwen3.6 hybrid arch (verify from logs before anything else); DeltaNet recurrent state makes rejection-replay more expensive than pure attention (state checkpointing, flagged in the DFlash PR for hybrids); prompt-processing regression needs measuring; MTP-head quant quality varies across the community GGUFs (use ggml-org or unsloth).

---

## 4. DFlash (`--spec-type draft-dflash`)

**Status: merged.** [PR #22105](https://github.com/ggml-org/llama.cpp/pull/22105) added DFlash: a block-diffusion draft model that emits an entire draft block in one forward pass while cross-attending to the target's recent hidden states (ring buffer, `--spec-dflash-cross-ctx`). Output is verified, so text is byte-identical to the target alone.

**Reported numbers:** Qwen3-8B up to **8.08x on code generation at 93.3% acceptance**; Qwen3.6-27B Q4_K_M overall **2.69x**; hybrid Qwen3.5-9B only 1.90x (state checkpointing cost). Independent writeups ([Medium](https://xhinker.medium.com/dflash-just-landed-in-llama-cpp-worth-to-upgrade-to-get-speed-boost-a20db434e8f7), [DataCamp](https://www.datacamp.com/tutorial/how-to-speed-up-local-llms-with-dflash-speculative-decoding), [InventiveHQ](https://inventivehq.com/blog/llama-cpp-speculative-decoding-consumer-gpu)) converge on "tens of percent to ~2x on consumer hardware, can be a net loss on already-fast small models."

**Available drafts** (z-lab format, per the PR): [z-lab/Qwen3-8B-DFlash](https://huggingface.co/z-lab/Qwen3-8B-DFlash) (~1.7B, 3.2 GiB BF16), Qwen3-4B-DFlash, gpt-oss-20b-DFlash, Qwen3.5-4B/9B-DFlash, community GGUF [Alittlehammmer/Qwen3.6-27B-DFlash-GGUF](https://huggingface.co/Alittlehammmer/Qwen3.6-27B-DFlash-GGUF-llama.cpp).

**Verdict for our rig:** the PR states MoE targets show minimal speedup (same expert-union reason), and **no DFlash draft exists for any A3B MoE target**, so DFlash cannot touch our flagship config. It is only actionable on our dense Qwen3-8B baseline, which fits entirely in 8 GB VRAM (Q4_K_M ~5.0 GB + quantized draft ~1-2 GB + KV at reduced context). The 8.08x code-gen headline is the best-case ceiling; a 2-3x realistic gain on the 8B would be a nice demo but is not a flagship-config breakthrough.

---

## 5. The draftless ngram family beyond ngram-simple

[docs/speculative.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md) documents four untested draftless types. All cost zero download and are combinable.

- **ngram-mod** ([PR #19164](https://github.com/ggml-org/llama.cpp/pull/19164), ggerganov): rolling LCG hash of the last n tokens -> next-token prediction, iterated to produce variable-length drafts. ~16 MB constant memory, shared across server slots. Defaults `--spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64`. Reported: 0.70 acceptance in the PR example; a user reported >0.90 acceptance on GPT-OSS-120B repeating source code, eval time ~240 s -> ~18 s. Fails on non-repetitive text and resets on CRLF/LF mismatches (a real trap on Windows, worth checking in our harness). Docs explicitly note **"MoEs require long drafts"**, which is the saturation effect in section 7.
- **ngram-map-k**: exact-match n-gram -> m-token continuation with a hit-count threshold (`--spec-ngram-map-k-min-hits`), stricter than ngram-simple.
- **ngram-map-k4v**: experimental, tracks up to 4 continuations per key, defaults n=8 m=8 min-hits 2, recommended for longer repetitions.
- **ngram-cache**: statistics-based drafting, optionally seeded from external n-gram stats files.

E014 tested ngram-simple only. ngram-mod is a different algorithm operated in a different regime (48-64-token drafts vs short ones), and the combined stack `draft-mtp,ngram-mod` has never been measured by anyone we can find.

---

## 6. Self-speculation: papers only, nothing runnable

Layer-skip/self-drafting methods (SWIFT, [arXiv 2410.06916](https://arxiv.org/abs/2410.06916); LayerSkip, [arXiv 2404.16710](https://arxiv.org/abs/2404.16710); Kangaroo, [arXiv 2404.18911](https://arxiv.org/abs/2404.18911)) have no llama.cpp implementation. The b10064 spec-type list in section 1 is exhaustive and contains no self-speculation mode. MTP is the spiritual successor here anyway (the model drafts for itself, but with a trained head instead of skipped layers), with better acceptance than any training-free layer-skip result. Dead end for this sweep.

---

## 7. Expert-union economics: what acceptance rate beats our penalty

This is the decision engine for every candidate. Model of one speculation cycle on our rig, k draft tokens, verify batch of k+1:

```
S(k, a) = L(a, k) / [ B * min(U(k+1), E_total) / E_tok  +  (1 - B)  +  k * d ]

L(a, k)  = (1 - a^(k+1)) / (1 - a)        expected tokens emitted per cycle
U(m)     = E_tok * (1 + (m - 1) * s)      distinct experts touched by m tokens (per layer, capped at E_total)
B        = expert-stream share of baseline step time (~0.85 on our rig: ~1.05 GB expert reads/token
           at 37-42 GB/s is ~25-28 ms of a ~26-32 ms step at our measured 31-42 t/s)
d        = draft cost per drafted token / baseline step time
s        = expert scattering: fraction of a new token's experts NOT already in the union
```

For Qwen3-30B-A3B: E_tok = 8, E_total = 128. For Qwen3.6-35B-A3B: 8 routed of 256 finer-grained experts + 1 shared (the shared expert's bytes amortize across the whole verify batch, effectively lowering B for the routed term).

**Worked break-evens** (d = 0.07 for MTP/EAGLE3 heads on GPU, d = 0 for ngram):

| k | a | s | cost multiple | L | S |
|---|-----|-----|------|------|------|
| 2 | 0.75 | 0.7 | 2.31 | 2.31 | 1.00 wash |
| 2 | 0.83 (MTP n=2 measured) | 0.7 | 2.31 | 2.52 | 1.09 |
| 2 | 0.83 | 0.5 | 1.97 | 2.52 | 1.28 win |
| 3 | 0.72 (MTP n=3 measured) | 0.7 | 2.94 | 2.61 | 0.89 loss |
| 3 | 0.72 | 0.35 | 2.04 | 2.61 | 1.28 win |

Reading: **at MTP's measured acceptance, everything hinges on s, the expert overlap between adjacent decode positions.** s >= ~0.7 (near-disjoint experts) kills even MTP, s <= ~0.5 makes n-max 2 a >=20% breakthrough. This is measurable on our instrumented build in an afternoon, before downloading anything: log router top-k IDs per layer for consecutive tokens and compute U(2), U(3), U(4) directly. It also retroactively explains E014 quantitatively, which is publishable on its own.

**The saturation escape hatch:** U(m) caps at E_total. Verifying 64 tokens costs at most 128/8 = 16 token-equivalents of expert reads on Qwen3-30B-A3B. A 64-token draft with high acceptance (verbatim repetition, ngram-mod's home turf) yields S up to ~4x. Short drafts pay the union penalty at its steepest; very long correct drafts amortize past it. This is precisely why the docs say MoEs require long drafts, and it is the regime E014 never entered.

**2026 literature confirms the frame.** [EcoSpec / "Less Experts, Faster Decoding" (arXiv 2607.12696)](https://arxiv.org/abs/2607.12696) names the phenomenon "expert scattering" and selects draft paths that reuse experts already in the verification set; [MoE-Spec (arXiv 2602.16052)](https://arxiv.org/pdf/2602.16052) budgets experts during verification; [Utility-Driven Speculative Decoding for MoE (arXiv 2506.20675)](https://arxiv.org/abs/2506.20675) and [Adaptive Verification (arXiv 2605.00342)](https://arxiv.org/pdf/2605.00342) tune k dynamically against expert cost; [MoE-SpeQ (arXiv 2511.14102)](https://arxiv.org/abs/2511.14102) uses speculation to prefetch experts under offloading, which is our exact topology. All are papers or vendor codebases, none is in llama.cpp; their value to us is validated cost models plus a possible instrumented-build patch idea (draft-token gating by predicted marginal expert cost, EcoSpec-style, would be a first for llama.cpp).

One structural advantage nobody's numbers include: on our rig the drafting resource (GPU) and the verify bottleneck (CPU RAM bandwidth) are different pieces of silicon. [DuoDecoding (arXiv 2503.00784)](https://arxiv.org/pdf/2503.00784) exploits exactly this heterogeneity (they draft on CPU under a GPU target; ours is inverted). The E014 TDP-sharing penalty is the counterweight, but a 1-2 GB head is a far smaller power draw than the full draft models E014 ran.

---

## Candidate experiments for our rig

Ordered by expected value per unit effort.

### C1. Phase-0: measure expert scattering s and publish the union-cost law (gate for C2-C4)
Instrumented build, Qwen3-30B-A3B (already on disk): log per-layer router top-k IDs across decode, compute U(2..8) and s on 3 workload types (chat, code gen, code rewrite). Plug into the section 7 model, pre-register predicted speedups for MTP/EAGLE3/ngram-mod, then falsify with the runs below. Zero download. Also quantitatively explains E014 post hoc.
- Needs: instrumented build patch (~50 lines of router logging). Effort: hours. Class: incremental (but it is the referee for everything else).

### C2. Qwen3.6-35B-A3B-MTP with experts on CPU: model upgrade and speed upgrade in one shot
Download [unsloth/Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF) UD-IQ4_XS (18.2 GB, ~1.8 h) or UD-Q4_K_XL (22.9 GB, ~2.2 h). Step 1: verify b10064 loads the hybrid arch (else this candidate also decides whether we bump builds). Step 2: baseline A/B/A without MTP vs our Qwen3-30B-A3B flagship (expect ~similar decode speed at 3B active with tiny KV, at generationally better quality: 73.4% SWE-bench V). Step 3: `--spec-type draft-mtp --spec-draft-n-max 2 -fa on` vs 3, measure acceptance and sustained decode under the E032 thermal protocol. Even if MTP nets zero, step 2 alone can deliver breakthrough (a); if C1's s <= 0.5, MTP should add 20-30% for breakthrough (b) as well. First public MTP + `--n-cpu-moe` datapoint anywhere.
- Needs: 18-23 GB download, ~20 GB RAM + ~4 GB VRAM at load. Effort: days. Class: model-upgrade.

### C3. ngram-mod in the long-draft saturation regime, alone and stacked with MTP
Zero download. On Qwen3-30B-A3B (and C2's model if landed): `--spec-type ngram-mod --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64` on repetition-heavy workloads (code rewrite/refactor, agentic edit loops, summarize-with-quotes), where union saturation caps verify cost at 16 token-equivalents and the PR shows 0.7-0.9 acceptance. Then the untested combined stack `draft-mtp,ngram-mod`. Watch the CRLF reset trap on Windows. Distinct from E014's ngram-simple in both algorithm and regime; frame result as workload-conditional speedup law, not a universal win.
- Needs: nothing. Effort: hours. Class: speed-upgrade (workload-dependent; honest headline reports the mix).

### C4. EAGLE-3 head for Qwen3-30B-A3B: cheap falsification of the "MoE 1.06x" prior on heterogeneous hardware
[Tengyunw/qwen3_30b_moe_eagle3](https://huggingface.co/Tengyunw/qwen3_30b_moe_eagle3) (~1-2 GB) converted per [PR #18039](https://github.com/ggml-org/llama.cpp/pull/18039). Gate A: read convert_hf_to_gguf.py to confirm `--target-model-dir` needs only config/tokenizer, not the 61 GB HF target; if it needs weights, kill or defer. Gate B: C1 must show s <= ~0.55, else pre-registered prediction is a loss and we run it only to confirm the model. Value: PR's own MoE datapoint says 1.06x on homogeneous GPU; our GPU-draft/CPU-verify split is the one topology where that number could move meaningfully.
- Needs: ~1-2 GB download + conversion, C1 result. Effort: days. Class: speed-upgrade (long shot, high falsification value).

### C5. DFlash on the dense Qwen3-8B baseline, fully in VRAM
[z-lab/Qwen3-8B-DFlash](https://huggingface.co/z-lab/Qwen3-8B-DFlash) quantized (~1-2 GB) + our existing Qwen3-8B Q4_K_M (5 GB), both resident in 8 GB VRAM at reduced context. PR claims up to 8.08x on code at 93.3% acceptance; realistic consumer reports say 1.5-2.5x. No MoE relevance (no A3B drafts exist), so this cannot touch the flagship; it is a bounded-effort demo of the strongest per-token technique in the tree and a useful public datapoint for 8 GB Blackwell laptops.
- Needs: ~1-2 GB download, VRAM budget check with KV. Effort: hours to a day. Class: incremental.

Dead ends this sweep, for the record: self-speculation (no implementation, section 6), DFlash on MoE (no drafts, PR reports minimal gain), ngram-cache (subsumed by ngram-mod for our workloads, run only if C3 surprises), dense Qwen3.6-27B via any spec type (16-17 GB of dense weights through a 40 GB/s pipe is ~3 t/s baseline; no 2-3x multiplier reaches 25 t/s).

---

## Feasibility verdicts

Adversarial review 2026-07-19. Verified against the local b10064 source tree (`llama.cpp-master`), on-disk models, and live HF/GitHub fetches. On-disk facts confirmed: Qwen3-30B-A3B-Instruct-2507 Q4_K_M (17.28 GB) and Qwen3-8B Q4_K_M (4.68 GB) present in `models/`; all seven untested spec types present in `common/speculative.cpp` and `common/arg.cpp`.

- **C1 (measure expert scattering s): GO.** Zero download, both models and the instrumented build already on disk, ~50-line router-logging patch, no overlap with E021-E028 (those measured bandwidth, not router overlap), and it is the pre-registration gate for C2-C4.
- **C2 (Qwen3.6-35B-A3B-MTP + experts on CPU): GO.** Verified end to end: unsloth MTP GGUF repo exists at exactly the claimed sizes (18.2/22.9 GB, ~1.7-2.2 h at 3 MB/s); its GGUF arch string is `qwen35moe`, which b10064 has in `llama-arch.cpp` including the hybrid-memory list AND `llm_arch_supports_rs_rollback` (the recurrent-state rollback spec verification needs); `draft-mtp` is implemented in local `common/speculative.cpp` with an explicit qwen35moe single-head mode. Fits: ~17-19 GB experts in 48 GB RAM, ~4 GB VRAM. Not a duplicate (E014 never touched MTP). Keep the load-from-logs check, but the arch-support risk flagged in section 3 is now largely retired at source level.
- **C3 (ngram-mod long-draft regime, alone and stacked with MTP): GO.** Zero download; `--spec-ngram-mod-n-min/max/match` all confirmed in local `arg.cpp`; different algorithm and operating regime from E014's ngram-simple, so not a duplicate; workload-conditional framing is honest. Test the CRLF reset trap explicitly, it is real on Windows.
- **C4 (EAGLE-3 head for Qwen3-30B-A3B): MAYBE.** Gate A already resolved in its favor by reading local `conversion/llama.py`: `--target-model-dir` reads only config.json and tokenizer files, never target weights, so the feared 61 GB download is dead (a few MB of config/tokenizer instead). Tengyunw head repo confirmed live (PyTorch safetensors, no GGUF, conversion required). Stays MAYBE because it is correctly gated on C1 showing s <= ~0.55 against the PR's own 1.06x MoE prior, and the Tengyunw checkpoint's exact format compatibility with the auto-detection in conversion/llama.py (draft_vocab_size + 1 layer) is unverified until we open its config.
- **C5 (DFlash on dense Qwen3-8B in VRAM): MAYBE.** Implementation fully present locally (DFLASH arch, `draft-dflash`, DFlashModel converter class); z-lab draft confirmed live, listed as ~1B params BF16 (doc says ~1.7B, correct this at conversion time). The squeeze: 4.68 GB target + ~1 GB quantized draft + KV + cross-ctx ring buffer + Windows desktop overhead on 8 GB VRAM forces reduced context and possibly quantized KV, and conversion needs a working HF Python env plus target config/tokenizer download. Feasible but incremental class by our own definition; run last if at all.

**Correction to the dead-ends list:** "DFlash on MoE (no drafts)" is stale as of this review. z-lab/Qwen3.6-35B-A3B-DFlash exists, plus a community GGUF test conversion (lym00/Qwen3.6-35B-A3B-DFlash-GGUF-Test). The PR's minimal-MoE-gain finding still argues against it, but if C2 lands and C1 shows low s, a DFlash-vs-MTP head-to-head on the same Qwen3.6-35B-A3B target becomes a cheap optional extension rather than an impossibility.
