# Domain survey: KV-cache and long-context advances

Sweep 2026-07-sweep-01. Surveyed 2026-07-19. Runtime baseline: llama.cpp b10064 (built Jul 17-18, 2026; feature flags verified against our local `llama-server.exe --help`).

Guiding question: what lets our 8 GB RTX 5060 Laptop hold 64k+ context on the flagship config (Qwen3-30B-A3B-Instruct-2507, experts on CPU, ~31-40 t/s sustained)?

---

## 0. Our starting point: the KV math

Qwen3-30B-A3B-Instruct-2507 is 48 layers, GQA 32 Q / 4 KV heads, head dim 128, native context 262,144 ([model card](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507)). So no RoPE hacks are needed for 64k+; the only obstacle is memory.

KV bytes per token = 48 layers x 4 KV heads x 128 dim x 2 (K+V) x bytes/elem:

| Context | f16 (2 B) | q8_0 (1.0625 B) | q4_0 (0.5625 B) |
|---|---|---|---|
| per token | 96 KiB | 51 KiB | 27 KiB |
| 16k | 1.5 GiB | 0.80 GiB | 0.42 GiB |
| 32k | 3.0 GiB | 1.60 GiB | 0.84 GiB |
| 64k | 6.0 GiB | 3.19 GiB | 1.69 GiB |
| 128k | 12.0 GiB | 6.38 GiB | 3.38 GiB |

With all experts on CPU (`--cpu-moe`), the GPU holds only non-expert weights (~2-2.5 GB). Budget on the 8 GB card: roughly 5-5.5 GB free for KV + compute buffers. Conclusion from arithmetic alone: **64k fits at q8_0 KV; 128k does not fit on GPU for this model; f16 KV at 64k is marginal-to-impossible.**

Depth cost prediction (falsifiable): at 64k full depth, attention must read ~3.2 GB of q8_0 KV per token. At our measured 312.7 GB/s VRAM bandwidth (E001) that is ~10 ms/token for KV reads alone, on top of ~25-32 ms/token for CPU experts. Expected decode at full 64k depth: ~15-22 t/s. Still usable; the experiment should measure the full depth curve, not one point.

---

## 1. KV-cache quantization

### What b10064 supports (verified locally)

`-ctk / -ctv` accept: f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1 (default f16), independently for K and V, plus separate draft-model cache types. Quantized V requires FlashAttention; our E014 protocol note applies: on this rig `-fa auto` plus KV quant halved llama-server speed once, so always set `-fa on` explicitly and A/B it.

### Quality evidence

- q8_0 KV is repeatedly measured as near-lossless: under 0.1% perplexity delta across models, halving cache size ([TechPlained roundup](https://www.techplained.com/kv-cache-quantization), [smcleod's Ollama port of llama.cpp KV quant](https://smcleod.net/2024/12/bringing-k/v-context-quantisation-to-ollama/)).
- 4-bit is contested, and the K-vs-V asymmetry claims **conflict across sources**. The TechPlained data says q4_0 K costs ~0.4% PPL and q4_0 V ~1.4% (V more fragile). The KIVI paper argues the opposite sensitivity: keys have per-channel outliers, so per-token block quant (which is what llama.cpp q4_0 does) should hurt K more, while V tolerates per-token quant ([KIVI, arXiv 2402.02750 via its successors](https://arxiv.org/abs/2402.02750)). Nobody has published this ladder for Qwen3-30B-A3B on consumer hardware. That is a gap we can fill with our paired-PPL protocol.
- Speed caveat: at long context, q4_0 KV dequant overhead can dominate; one measurement shows q4_0 KV up to 92% slower than f16 at 64k ([TechPlained](https://www.techplained.com/kv-cache-quantization)). Unverified on Blackwell; needs local measurement.
- Hybrid-model caveat: Unsloth's Qwen3.5 guide explicitly suggests `--cache-type-k bf16 --cache-type-v bf16` if output degrades ([Unsloth Qwen3.5 docs](https://unsloth.ai/docs/models/qwen3.5)); with only a few full-attention layers, hybrids concentrate all retrieval in a small KV, so per-layer sensitivity may be higher.

### Frontier (watch, not testable yet)

- Per-head adaptive KV quantization proposal in llama.cpp, claiming lossless q4_0 on hybrid models ([issue #21385](https://github.com/ggml-org/llama.cpp/issues/21385)).
- TurboQuant extreme KV quantization discussion ([discussion #20969](https://github.com/ggml-org/llama.cpp/discussions/20969)).

---

## 2. Eviction and compression (SnapKV-class)

The research is rich; the runnable-on-our-stack implementations are nearly nonexistent.

- **SnapKV** ([arXiv 2404.14469](https://arxiv.org/abs/2404.14469)) and **H2O** ([arXiv 2306.14048](https://arxiv.org/abs/2306.14048)): attention-score-based eviction. Python/HF implementations only.
- **KVzip** ([arXiv 2505.23416](https://arxiv.org/abs/2505.23416), NeurIPS 2025 oral, [code](https://github.com/snu-mllab/kvzip)): query-agnostic eviction via context reconstruction, 3-4x KV reduction and ~2x decode latency cut on Qwen3/Gemma3/Llama3 at up to 170k context. The strongest paper of the class because compressed caches stay reusable across different queries. Implementation is PyTorch/HF, not llama.cpp. A follow-up, Fast KVzip, exists ([arXiv 2601.17668](https://www.arxiv.org/pdf/2601.17668)).
- **Ada-KV** ([arXiv 2407.11550](https://arxiv.org/pdf/2407.11550)), **EvicPress** ([arXiv 2512.14946](https://arxiv.org/html/2512.14946)): adaptive per-head budgets, joint compression+eviction. Same story: serving-framework research code.
- **llama.cpp status: no eviction in mainline.** The standing feature request is [discussion #13986](https://github.com/ggml-org/llama.cpp/discussions/13986). The only llama.cpp-adjacent implementation found is **ElasticKV** ([repo](https://github.com/infolake/elastickv)), a post-attention CUDA hook patch: 1.48-1.57x compression, <0.31% PPL, but Linux/WSL2 only, requires a from-source CUDA build (blocked for us until the CUDA toolkit is installed), zero stars, single author. Watch item, not a candidate.

Verdict: paper exists, runnable implementation does not, for our Windows + llama.cpp stack. The practical "eviction" available to us is architectural (SWA/hybrid models, section 5) or positional (context shift, section 3).

---

## 3. Streaming attention and context shift

- **StreamingLLM / attention sinks** ([arXiv 2309.17453](https://arxiv.org/pdf/2309.17453)): keep the first few tokens plus a rolling window; enables unbounded generation, does NOT extend effective memory. llama.cpp implements this as `--context-shift` with `--keep N` (verified in b10064; **default disabled now**, must be enabled explicitly).
- **gpt-oss models bake sinks in as learned parameters**, paired with 128-token sliding-window layers ([ik_llama.cpp analysis](https://github.com/ikawrakow/ik_llama.cpp/discussions/758), [llama.cpp gpt-oss guide](https://github.com/ggml-org/llama.cpp/discussions/15396)).
- **Hard limitation on the newest models**: context shift and KV shifting are disabled for hybrid-attention models (Qwen3.5/3.6, Gemma 4) because linear-attention state cannot be rolled or rotated; hitting the context limit forces stop or full recompute ([discussion #24944](https://github.com/ggml-org/llama.cpp/discussions/24944)). b10064's mitigation is context checkpoints: `--ctx-checkpoints N` and `--checkpoint-min-step` (verified locally), which snapshot recurrent/SWA state so partial rollback avoids full reprocessing.
- Trade-off to internalize: **hybrid models slash KV size but give up cache flexibility** (shift, reuse-with-shift). For chat-style append-only use (our use), this costs little; for edit-heavy RAG prompt reshuffling it matters.

---

## 4. Prompt caching and context reuse across sessions

This is the most immediately valuable, zero-download area for us. Three mechanisms, all in b10064 (verified):

1. **Host-memory prompt cache, `--cache-ram` (default 8192 MiB)**: llama-server automatically saves idle slots' KV state to system RAM and restores on prefix match; up to 93% TTFT reduction; works with dense, MoE, SWA, SSM ([PR #16391](https://github.com/ggml-org/llama.cpp/pull/16391), [tutorial discussion #20574](https://github.com/ggml-org/llama.cpp/discussions/20574), [--cache-ram explainer](https://jessequinn.info/blog/llama-cpp-cache-ram-prompt-caching)). Our launchers already run llama-server, so we may be getting some of this for free WITHIN a session; nobody has verified it engages on our MoE hybrid config (protocol law: verify feature engagement from logs).
2. **`--cache-reuse N`**: chunk-level KV reuse via shifting for prompts that share content but not exact prefixes ([KV reuse tutorial, discussion #13606](https://github.com/ggml-org/llama.cpp/discussions/13606)). Full-attention models only (shifting; see section 3).
3. **Disk persistence: `--slot-save-path` + `POST /slots/0?action=save|restore`**: survives server restarts ([discussion #13606](https://github.com/ggml-org/llama.cpp/discussions/13606), [per-session hooks tutorial, discussion #20572](https://github.com/ggml-org/llama.cpp/discussions/20572)). Known gap: broken for vision-enabled models ([issue #19466](https://github.com/ggml-org/llama.cpp/issues/19466)). Community measured ~7x faster restore vs re-prefill even through a naive proxy ([writeup](https://ai-muninn.com/en/blog/kv-cache-disk-restore-7x)).

Why this is disproportionately valuable on OUR rig: prefill on the flagship is CPU-expert-bound, so a 32k-token document costs minutes of prompt processing every session. The saved slot file for 32k tokens is ~3.0 GiB at f16 KV (~1.6 GiB at q8_0); our NVMe reads ~5 GB/s, so restore is seconds. This converts "re-read the whole document every morning" into "instant resume". No downloads, pure measurement.

---

## 5. Hybrid-attention and hybrid-SSM models (the architectural fix)

The 2025-2026 model generation attacks KV size at the architecture level. Status for models that fit this rig:

| Model | Architecture | KV per token (f16) | Fit on our rig | llama.cpp status |
|---|---|---|---|---|
| Qwen3-30B-A3B-2507 (flagship) | full attention, 48L, 4 KV heads | 96 KiB | running today | supported |
| **Qwen3.6-35B-A3B** / Qwen3.5-35B-A3B | 30 Gated DeltaNet layers + 10 full-attention (2 KV heads, dim 256), 256 experts, A3B, native 262k | **20 KiB** (4.8x less) | Q4_K_M 21.4 GB; 4-bit needs ~22 GB total | supported, incl. MTP GGUFs ([Unsloth 3.5](https://unsloth.ai/docs/models/qwen3.5), [Unsloth 3.6](https://unsloth.ai/docs/models/qwen3.6), [family overview](https://enclaveai.app/blog/2026/03/08/qwen-3-5-complete-model-family-local-ai/), [arch details](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)) |
| **gpt-oss-20b** | 24L alternating dense / SWA-128 + learned sinks, MXFP4 MoE, 32 experts top-4, 3.6B active, 131k ctx | ~24 KiB growing (dense half) + fixed SWA windows | ~12 GB GGUF; runs in ~6 GB VRAM + 16 GB RAM class | supported ([official guide](https://github.com/ggml-org/llama.cpp/discussions/15396), [Unsloth](https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune)) |
| **Granite-4.0-H-Small** | 32B total / 9B active MoE; 4 attention + 36 Mamba2 layers, NoPE, 128k validated | KV from only 4 of 40 layers (roughly 10x+ smaller than dense-32B transformer) | Q4_K_M ~19-20 GB; A9B active means CPU-expert decode will be ~3x slower than A3B | supported since Oct 2025 ([IBM announcement](https://www.ibm.com/new/announcements/ibm-granite-4-0-hyper-efficient-high-performance-hybrid-models), [HF card](https://huggingface.co/ibm-granite/granite-4.0-h-small)) |
| Gemma 3 12B | 5:1 interleaved SWA-1024 | ~1/6 of dense equivalent | fits, dense so decode slower | SWA cache supported since [PR #13194](https://github.com/ggml-org/llama.cpp/pull/13194); `--swa-full` trades memory back for cache reuse |
| Gemma 4 | 5:1 SWA + shared KV between late layers, 10 of 60 global | small but reportedly 2-3x larger than same-gen peers | dense variants; sizes TBD for us | supported; no context shift ([HF blog](https://huggingface.co/blog/gemma4), [discussion #24944](https://github.com/ggml-org/llama.cpp/discussions/24944)) |
| Kimi-Linear-48B-A3B | 3:1 KDA linear + MLA | tiny | Q4 ~28 GB, would fit RAM | **PR still open as of early 2026** ([PR #17592](https://github.com/ggml-org/llama.cpp/pull/17592), [GGUF requires PR build](https://huggingface.co/ymcki/Kimi-Linear-48B-A3B-Instruct-GGUF)); re-check before investing |
| Qwen3-Next-80B-A3B | GDN hybrid | tiny | Q4 ~45 GB: does NOT fit 48 GB RAM with OS | skip |

Key intelligence on the Qwen3.5/3.6-35B-A3B for our rig:

- It is the direct successor of our flagship: same A3B active-parameter class (so the CPU-expert decode economics we know should transfer), one to two generations newer in quality, multimodal + hybrid thinking in 3.6 ([Qwen3.6 repo](https://github.com/QwenLM/Qwen3.6), [codersera guide](https://codersera.com/blog/how-to-run-qwen-3-6-locally-2026/)).
- Measured KV behavior confirms the arithmetic: going 4k to 262k adds only ~3 GB total memory on the 35B-A3B, vs ~16 GB on the 27B dense ([InsiderLLM guide](https://insiderllm.com/guides/qwen35-local-guide-which-model-fits-your-gpu/)).
- Risk 1, performance: early llama.cpp Gated DeltaNet kernels were immature; 3.5-35B initially ran ~35% slower than Qwen3-30B-A3B, and one report class puts 8 GB VRAM + expert-offload rigs at low-teens t/s ([InsiderLLM](https://insiderllm.com/guides/qwen35-local-guide-which-model-fits-your-gpu/)). Counter-evidence: a May 2026 recipe claims **Qwen3.6-35B-A3B at ~30 t/s on 6 GB VRAM + 32 GB RAM with llama.cpp** ([Medium recipe, paywalled](https://mychen76.medium.com/run-qwen3-6-35b-a3b-on-6gb-vram-using-llama-cpp-30-tps-a89032e5a60c)). The disagreement is exactly what our A/B/A protocol resolves.
- Risk 2, cache semantics: no context shift, no cache-reuse shifting (section 3); mitigate with `--ctx-checkpoints`.
- Risk 3, KV quant on hybrids: Unsloth advises bf16 cache if degradation appears; with only 10 attention layers the KV is already small, so we may not need KV quant at all (64k f16 = 1.25 GiB).

---

## Candidate experiments for our rig

All candidates follow protocol law: A/B/A flanking, warm cache (`copy /b model NUL`), `--mlock`, `-t 12` MoE / `-t 8` dense, verify feature engagement from server logs, check tensor dtypes before byte math, cross-session claims need thermal control (E032: +-10%).

### C1. 64k context on the current flagship (q8_0 KV, experts all on CPU)

- **What**: Qwen3-30B-A3B-2507, `--cpu-moe -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -c 65536 --mlock`. Measure: VRAM fit, decode t/s at depth 0/16k/32k/48k/60k (llama-batched-bench or -gp sweeps), PPL paired f16-vs-q8_0 KV on ppl-text.txt, and re-test the E014 `-fa auto` server slowdown.
- **Why**: zero download; native 262k model card; arithmetic says it fits (3.19 GiB KV in a ~5 GB budget). Prediction to falsify: decode degrades from ~30 t/s to ~15-22 t/s at full depth from GPU KV reads. 16x context over our current 4k default would be a capability unlock even at the degraded floor.
- **Needs**: nothing new. Hours.
- **Class**: capability-upgrade.

### C2. Qwen3.6-35B-A3B as flagship successor (hybrid GDN, 20 KiB/token KV)

- **What**: download Q4_K_M (~21.4 GB, ~2 h at 3 MB/s), port our -ncmoe recipe, A/B/A vs flagship: decode t/s (short and 32k/64k depth), quality battery, VRAM/RAM occupancy at 64k and 128k, GDN-layer CPU cost, `--ctx-checkpoints` behavior, MTP GGUF as stretch (ties to untested `--spec-type draft-mtp` from the speculative domain).
- **Why**: the only candidate that is potentially model-upgrade AND capability-upgrade at once: newer-generation quality at the same A3B decode economics, with 4.8x smaller KV making 128k+ genuinely reachable on 8 GB (128k f16 = 2.5 GiB). Community evidence is split (low-teens vs ~30 t/s on 6 GB VRAM), which makes it a real experiment, not a foregone conclusion.
- **Needs**: 21.4 GB download; b10064 already supports it. Days.
- **Class**: model-upgrade.

### C3. Cross-session KV persistence (slot save/restore + --cache-ram) on the flagship

- **What**: llama-server with `--slot-save-path` + `--cache-ram`; build a 16k and a 32k document context; measure (a) re-prefill time cold, (b) `--cache-ram` in-session restore, (c) disk restore after full server restart, (d) saved-file size vs KV dtype, (e) verify from logs the prompt cache actually engages on our MoE hybrid config. Wire the winner into Start-30B-AI.bat as a "resume session" path.
- **Why**: prefill is our real long-context pain (CPU-bound experts); NVMe at ~5 GB/s restores a 3 GiB slot in seconds vs minutes of reprocessing, and PR #16391 claims up to 93% TTFT cuts. Converts long-document work from unusable to instant-resume on this rig.
- **Needs**: nothing new. Hours.
- **Class**: capability-upgrade (TTFT/workflow, not decode t/s; by our strict decode definition it is incremental, but it makes 32k-doc sessions practically usable for the first time).

### C4. KV-quant quality ladder and the K/V asymmetry test

- **What**: paired-PPL ladder on ppl-text.txt for {f16/f16, q8/q8, q8/q4, q4/q8, q4/q4, q5_1 mids} on Qwen3-8B and the flagship, plus decode t/s at 16k/32k to test the "q4_0 KV up to 92% slower at long context" claim on Blackwell. Publishes a falsification-grade table nobody has for MoE-on-consumer.
- **Why**: sources conflict on whether K or V tolerates 4-bit worse (KIVI theory vs measured blog tables); our C1/C2 configs depend on the answer; cheap and entirely local.
- **Needs**: nothing new. Hours.
- **Class**: incremental (guards and tunes C1/C2).

### C5. gpt-oss-20b: the cheap 131k-context reasoner

- **What**: download gpt-oss-20b MXFP4 GGUF (~12 GB, ~1.1 h), run with attention+KV on GPU and experts on CPU (`-ncmoe` sweep), measure decode t/s (3.6B active should beat the flagship's 3.3B-active speeds if kernels cooperate), KV footprint at 64k/128k (dense half grows at ~24 KiB/token; SWA half fixed), and reasoning quality vs flagship at low/medium effort.
- **Why**: native learned attention sinks + SWA-128 halves KV growth by design; MXFP4 experts are small (12 GB total leaves huge RAM headroom); it is the only OpenAI-lineage reasoning model that fits this rig, and long-reasoning chains are exactly what eats context.
- **Needs**: 12 GB download. Days (includes quality battery).
- **Class**: capability-upgrade.

---

## Watch list (not testable now)

- **ElasticKV** llama.cpp eviction hook: revisit if/when the CUDA toolkit gets installed and we accept WSL2 ([repo](https://github.com/infolake/elastickv)).
- **KVzip**: if a llama.cpp port or GGUF-compatible implementation appears, it jumps straight to candidate ([repo](https://github.com/snu-mllab/kvzip)).
- **Kimi-Linear-48B-A3B**: re-check [PR #17592](https://github.com/ggml-org/llama.cpp/pull/17592) merge status next sweep; 48B at Q4 (~28 GB) fits our RAM and the KDA architecture is the strongest published long-context throughput story.
- **Per-head adaptive KV quant** ([issue #21385](https://github.com/ggml-org/llama.cpp/issues/21385)) and **TurboQuant** ([discussion #20969](https://github.com/ggml-org/llama.cpp/discussions/20969)): if merged, re-run C4 ladder.

## Dead ends confirmed this survey

- SnapKV/H2O-class eviction on our stack: no runnable Windows llama.cpp implementation exists (section 2).
- Qwen3-Next-80B-A3B: does not fit 48 GB RAM at usable quants.
- Unbounded "infinite context" via context shift on the newest hybrid models: architecturally disabled ([discussion #24944](https://github.com/ggml-org/llama.cpp/discussions/24944)).

---

## Feasibility verdicts

Adversarial review 2026-07-19. Verified against the LOCAL b10064 binary (`llama-server.exe --help`) and the local `llama.cpp-src` source tree, not just the survey's links. Disk free at review time: ~393 GB (survey context said ~416; both are plenty).

- **C1 (64k on flagship, q8_0 KV): GO.** Every flag verified in the local binary (`-ctk/-ctv` list all 9 types, FA required for quantized V per E014 note); zero download; KV math re-checked and correct (96 KiB/token f16, 3.19 GiB at 64k q8_0). One honest tightness note: 3.19 GiB KV + ~2.5 GB non-expert weights + CUDA compute buffer + Windows DWM VRAM lands ~6.5-7.3 GB on an 8 GB card, so keep `-ub` moderate and have a `-c 49152` fallback ready. Not a duplicate: E013/E023 never left short context.
- **C2 (Qwen3.6/3.5-35B-A3B successor): GO, with one mandatory pre-flight.** `qwen35`/`qwen35moe` arches and CUDA Gated DeltaNet kernels (`ggml-cuda/gated_delta_net.cu`) are confirmed present in our local source, so the hybrid runs GPU-side; `--spec-type draft-mtp` confirmed in the binary for the MTP stretch. But there is NO `qwen36` arch string in local `llama-arch.cpp`: before burning the 2 h / 21.4 GB download (math checks out at 3 MB/s), verify the 3.6 GGUF's declared arch metadata loads on b10064, and fall back to Qwen3.5-35B-A3B (definitively supported) if it does not. The low-teens-vs-30 t/s split is the experiment, not a feasibility blocker. Not a duplicate.
- **C3 (cross-session KV persistence): GO.** `--slot-save-path`, `--cache-ram` (default 8192 MiB, so it may ALREADY be silently active in our launchers, which makes the log-verification step mandatory, not optional), and `--cache-reuse` all confirmed in the local binary; zero download; flagship is full-attention so none of the hybrid slot-restore caveats apply; the vision-model save bug is irrelevant to us. Not a duplicate.
- **C4 (KV-quant ladder, K/V asymmetry): GO.** All cache types confirmed locally; paired-PPL harness and `datasets/ppl-text.txt` already exist from E023; zero download; genuinely unpublished territory (sources conflict on K-vs-V 4-bit sensitivity) and it directly de-risks C1/C2. Cheapest experiment in the sweep. Not a duplicate: E021-E028 was CPU matmul bandwidth, never KV dtype quality.
- **C5 (gpt-oss-20b): GO.** `gpt-oss` arch confirmed in local source; 12 GB / ~1.1 h download math is honest; MXFP4 experts-on-CPU leaves large RAM headroom and the dense-half KV math (24 KiB/token) re-checks correctly. Weakest of the five on upside (an Aug-2025 model, quality vs the 2507 flagship is genuinely uncertain), but feasibility is clean. Not a duplicate.

No KILLs: the survey already self-culled the infeasible items (SnapKV-class eviction, ElasticKV, Kimi-Linear, Qwen3-Next-80B) into the watch list and dead-ends sections before they reached candidate status, and the local-binary spot-checks all came back clean. Suggested order by risk-adjusted value per hour: C1 -> C4 -> C3 -> C2 -> C5.
