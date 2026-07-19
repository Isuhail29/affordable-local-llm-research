# Domain survey: new models in our class (late 2025 through mid 2026)

Sweep: 2026-07-sweep-01. Date: 2026-07-19.
Scope: MoE models under ~40 GB quantized with small active counts (A1B to A4B), plus exceptional dense models under 15 GB.
Baseline to beat: Qwen3-30B-A3B-Instruct-2507 Q4_K_M at 31-42 t/s with `--n-cpu-moe` (E013/E023).

Rig constraints applied throughout: 48 GB DDR5 (~60 GB/s ceiling, 37-42 GB/s extracted by CPU matmuls per E021-E028), RTX 5060 Laptop 8 GB, AVX2 only, Windows 11, llama.cpp b10064 official + instrumented build, ~3 MB/s internet (~10.8 GB/hour), ~416 GB free disk.

---

## Tier 1: direct hits on our class

### 1. Qwen3.6-35B-A3B (Alibaba, April-May 2026) - the new default flagship candidate

The direct successor to our baseline. 35B total, 3B active (256 experts, 8 routed + 1 shared), Apache 2.0, native 262K context extensible to 1M via YaRN. Architecture is hybrid: 40 layers arranged as 10 blocks of 3x (Gated DeltaNet -> MoE) then 1x (Gated Attention -> MoE), so most attention is linear-time GDN. Trained with a native multi-token-prediction (MTP) head for self-speculative decoding.

- Model card: https://huggingface.co/Qwen/Qwen3.6-35B-A3B and repo https://github.com/QwenLM/Qwen3.6
- Blog: https://qwen.ai/blog?id=qwen3.6-35b-a3b
- Benchmarks from the model card: MMLU-Pro 85.2, GPQA Diamond 86.0, SWE-bench Verified 73.4, SWE-bench Pro 49.5. Our baseline Qwen3-30B-A3B-Instruct-2507 sits around MMLU-Pro 78, GPQA ~70-73, SWE-bench Verified 22 (the 22.0 figure is from the GLM-4.7-Flash comparison table on Unsloth). This is a generational jump at the same active parameter count.
- GGUF: bartowski https://huggingface.co/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF and Unsloth. UD-Q4_K_XL is 21 GB (https://unsloth.ai/docs/models/qwen3.6). 3-bit ~17 GB, 6-bit ~30 GB.
- MTP variant: https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF (~1 GB extra footprint). Unsloth claims 1.4-2.2x faster generation at unchanged accuracy; JarvisLabs measured a more sober 1.17x on an RTX PRO 6000 (https://jarvislabs.ai/blog/qwen36-mtp-llamacpp-rtxpro6000). Recommended `--spec-draft-n-max 2`.
- llama.cpp status: qwen_next architecture support merged long ago (PR #16095, release b7186); spec decoding for Qwen3.5/3.6 MoE enabled by PR #19493, merged 2026-04-19 (https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090). Our b10064 postdates both.
- Critical third-party data point: the RTX 3090 study above tested 19 spec-decode configurations (ngram-cache, ngram-mod, classic draft with Qwen3.5-0.8B) and found NO net speedup on Ampere + A3B MoE, which independently replicates our E014 expert-union finding. MTP was not in that study; it is a different mechanism (native head, no expert-union across a separate draft model's tokens, draft cost is one extra tiny head pass).
- Known footgun: Unsloth warns CUDA 13.2 produced gibberish with this model. We run CUDA 13.3; verify output sanity on first load anyway.
- Someone ran the 35B-A3B at ~30 t/s on a 6 GB GPU with llama.cpp CPU-MoE offload (https://mychen76.medium.com/run-qwen3-6-35b-a3b-on-6gb-vram-using-llama-cpp-30-tps-a89032e5a60c), which is our exact deployment pattern with a weaker GPU. Our 31-42 t/s class should hold.
- Download: ~21 GB (Q4) = ~2 hours. MTP variant ~22 GB.

Fit verdict: excellent. Same active budget as baseline, +5 GB total weights (35B vs 30B at Q4: 21 vs 18.6 GB), well within 48 GB RAM with `--n-cpu-moe`. The GDN layers shrink KV-cache cost, so long-context should degrade far less than baseline. Open risk: GDN CPU/CUDA kernel maturity in our b10064 profiler build (custom patches may not touch the new ops, but our instrumentation hooks should still count them).

### 2. GLM-4.7-Flash (Z.ai, January 2026) - agentic/coding specialist in the same weight class

30B total, ~3.6B active MoE reasoning model, 202,752-token context, built explicitly for local deployment.

- Unsloth guide: https://unsloth.ai/docs/models/tutorials/glm-4.7-flash (also https://unsloth.ai/docs/models/glm-4.7-flash)
- GGUF: https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF, UD-Q4_K_XL ~17-18 GB.
- Benchmarks vs same-class models (Unsloth table): SWE-bench Verified 59.2 vs Qwen3-30B 22.0 and GPT-OSS-20B 34.0; GPQA 75.2 vs Qwen3-30B 73.4 and GPT-OSS-20B 71.5. Weak spot: AIME25.
- llama.cpp: supported; a January 21 fix corrected scoring_func softmax -> sigmoid, so quants and builds must postdate that (our b10064 does; re-download quants if grabbed early). Needs `--jinja`, tool-call parser `glm47`, repeat penalty disabled (https://www.datacamp.com/tutorial/run-glm-4-7-locally, https://community.frame.work/t/super-simple-llama-cpp-vulkan-glm-4-7-flash-setup/80057).
- Download: ~18 GB = ~1.7 hours.

Fit verdict: excellent mechanical fit (same pattern as baseline, A3.6B may cost a few t/s). It is a reasoning-first model, so tokens-to-answer matters: at 25-35 t/s, long thinking traces are usable but not free. Best positioned as the coding/agentic flagship next to Qwen3.6 as the generalist.

### 3. Qwen3.5 small series, 9B (Alibaba, March 2, 2026) - the full-GPU wildcard

Released as the local-first tail of the Qwen3.5 line (0.8B, 1.7B, 4B, 9B). Hybrid Gated DeltaNet + sparse MoE, early-fusion multimodal (vision+text), 262K native context, 1M via interpolation.

- Coverage: https://medium.com/data-science-in-your-pocket/qwen-3-5-small-model-series-released-7a5ed34fcbb3 and review with numbers: https://computertech.co/qwen-3-5-small-review-2026/
- Claimed benchmarks for the 9B: GPQA Diamond 81.7 (above GPT-OSS-120B's 80.1), MMLU-Pro 82.5, MMMLU 81.2. Weaker on competitive coding (LiveCodeBench 65.6 vs GPT-OSS-120B 82.7). Community signal agrees it is strong: a Granite 4.1 HN thread cites "qwen3.5 9b outperforms granite 4.1 30b by a huge amount (32 vs 15 on artificialanalysis)" (https://news.ycombinator.com/item?id=47960507).
- Q4 footprint: ~6 GB. That fits ENTIRELY in our 8 GB VRAM with room for context. No CPU offload, no RAM bandwidth wall, no thermal TDP sharing between CPU and GPU (the E032/E014 confound largely disappears).
- llama.cpp: supported since March 2026 (our b10064 qualifies; verify arch loads). Note: Ollama reportedly cannot load these GGUFs due to the multimodal projector, irrelevant for us.
- Download: ~6 GB = ~35 minutes. Cheapest experiment in this sweep.

Fit verdict: potentially the biggest practical win per GB. If the quality claims survive our own falsification (vendor benchmarks always need it), this is baseline-class quality at GPU-resident speeds, plausibly 60-100+ t/s on the 5060 given ~1-2B active parameters over 256 GB/s GDDR7. Also our first shot at local vision input.

### 4. NVIDIA Nemotron 3 Nano Omni 30B-A3B (April 28, 2026) - the capability play

Omni-modal MoE: text, image, video, and audio INPUT in one 30B-total, ~3B-active model. Hybrid Mamba2-Transformer (mostly Mamba-2 + MLP, only four attention layers), 256-300K context.

- Announcement: https://developer.nvidia.com/blog/nvidia-nemotron-3-nano-omni-powers-multimodal-agent-reasoning-in-a-single-efficient-open-model/
- GGUF: https://huggingface.co/unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF. Q4_K_M 23.9 GB, MXFP4_MOE 21.7 GB, Q8_0 33.6 GB. Text-only variant also exists: https://huggingface.co/unsloth/Nemotron-3-Nano-30B-A3B-GGUF
- Q4_K_M reportedly recovers 100% median accuracy vs BF16 on target benchmarks (https://www.buildfastwithai.com/blogs/nvidia-nemotron-3-nano-omni-2026). Multimodal scores: MathVista_MINI 82.8, VideoQA 72.2, OCR 67.04.
- llama.cpp: runs via llama-cli/llama-server; audio/image input depends on mtmd support in the build and the mmproj files. This needs verification on b10064 before committing to the 24 GB download.
- Download: ~24 GB Q4 = ~2.2 hours, plus mmproj.

Fit verdict: good mechanical fit (A3B, Mamba2 layers are CPU-cheap). This is not a smarter text model than Qwen3.6; its value is a NEW capability: local audio+vision understanding at usable speed. Falls squarely under breakthrough definition (c).

### 5. Gemma 4 26B-A4B (Google DeepMind, April 2, 2026) - the QAT-backed alternative

Google's first MoE Gemma. 26B total, 4B active, 256K context, 140+ languages, official quantization-aware-trained Q4_0 GGUFs from Google themselves.

- Docs: https://unsloth.ai/docs/models/gemma-4 and QAT: https://unsloth.ai/docs/models/gemma-4/qat
- Official QAT GGUF: https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-gguf; community: https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF
- Sizes: Q4_K_M 16.87 GB (https://knightli.com/en/2026/05/01/gemma-4-local-vram-quantization-table/). Roundups report ~85 t/s on consumer hardware and rate it the best all-round consumer MoE (https://benchlm.ai/best/local-llm).
- Family: E2B (3.11 GB), E4B (4.98 GB), 12B dense (June 3, 2026, ~7.5 GB Q4), 26B-A4B, 31B dense (18.32 GB).
- llama.cpp: supported (LM Studio and llama.cpp guides exist: https://avenchat.com/blog/run-gemma-4-with-llama-cpp); early GGUFs had a chat-template bug, use re-uploaded quants (https://www.openaitoolshub.org/en/blog/gemma-4-gguf-chat-template-fix).
- Download: ~17 GB = ~1.6 hours.

Fit verdict: good. A4B active means ~33% more bytes per token through the 37-42 GB/s CPU path than A3B, so expect ~25-32 t/s, still above our 25 t/s bar. QAT Q4_0 is scientifically interesting for us: a vendor-trained 4-bit model removes one layer of quantization guesswork. Benchmarks vs Qwen3.6-35B-A3B are not clearly in Gemma's favor, so this is the backup generalist, not the headline.

### 6. Qwen3-Coder-Next 80B-A3B (Alibaba, February 2026) - the 70B-class capability stretch

80B total, 3B active (512 experts, 10 routed), hybrid GDN + gated attention, 262K context, trained for long-horizon agentic coding.

- GGUF: https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF. Q2_K 29.2 GB, Q3_K_M 38.3 GB, Q4_K_M 48.5 GB (Q4 does NOT fit our 48 GB RAM).
- Base Qwen3-Next-80B-A3B GGUFs also exist (https://huggingface.co/bartowski/Qwen_Qwen3-Next-80B-A3B-Thinking-GGUF, https://huggingface.co/lefromage/Qwen3-Next-80B-A3B-Instruct-GGUF) but Qwen3.6-35B and Coder-Next have largely superseded that line.
- llama.cpp: qwen_next arch in since b7186; Feb 2026 tool-calling and compatibility fixes noted on the Unsloth page.
- Download: Q3_K_M 38.3 GB = ~3.5 hours. Close to our single-model disk comfort zone but fine with 416 GB free.

Fit verdict: marginal but tantalizing. Q3_K_M at 38.3 GB leaves ~4-6 GB Windows headroom after pushing ~3 GB of shared/attention tensors to VRAM; mlock policy would need relaxing or a pagefile safety net, and our cross-session +-10% error bar discipline matters here. A3B active at Q3 should still decode 20-30 t/s. If it works, that is 80B-class coding quality on a laptop, breakthrough definition (c).

---

## Tier 2: noted, not headline candidates

- **GPT-OSS-20B** (OpenAI, Aug 2025): 21B total, 3.6B active, native MXFP4 ~12.1 GB, 131K context. Strong reasoning/math at high effort, weaker world knowledge; comparisons: https://llm-stats.com/models/compare/gpt-oss-20b-vs-qwen3-30b-a3b, https://artificialanalysis.ai/models/comparisons/gpt-oss-20b-vs-qwen3-30b-a3b-instruct. Still a fine model, but Qwen3.6-35B-A3B and GLM-4.7-Flash have passed it in its own weight class. Not worth a dedicated experiment now.
- **LFM2.5-8B-A1B** (Liquid AI, May 28, 2026): 8B total, ~1B active, reasoning-only, 128K context, day-one llama.cpp GGUF. Remarkable agentic scores for its size (Tau2 Telecom 88.07 vs Qwen3-30B 21.93) per https://www.liquid.ai/blog/lfm2-5-8b-a1b. Predecessor LFM2-8B-A1B: https://www.liquid.ai/blog/lfm2-8b-a1b-an-efficient-on-device-mixture-of-experts. Quality ceiling is 3-4B-dense class, below our flagship needs, but it is a candidate draft-model donor and a sub-6 GB GPU-resident agent.
- **Ling-mini-2.0 / Ring-mini-2.0** (inclusionAI, late 2025): 16B total, ~1.4B active, 128K context, Q4_K_M 9.91 GB, llama.cpp b6709+ (lfm2moe... distinct arch note in card): https://huggingface.co/inclusionAI/Ring-mini-2.0-GGUF, https://huggingface.co/bartowski/inclusionAI_Ling-mini-2.0-GGUF. Sub-10B-dense-class quality; superseded in headline terms but could fit fully in VRAM at Q2/Q3. A known llama.cpp think-tag rendering bug exists (https://github.com/ggml-org/llama.cpp/issues/17832).
- **Granite 4.0 / 4.1** (IBM): 4.0-H-Tiny 7B-A1B (too small), 4.0-H-Small 32B-A9B (A9B active = ~10-13 t/s on our CPU path, below the bar). Granite 4.1 pivoted to dense 3B/8B/30B; community testing places even the 30B far below Qwen3.5-9B (https://news.ycombinator.com/item?id=47960507, https://www.ibm.com/new/announcements/ibm-granite-4-0-hyper-efficient-high-performance-hybrid-models). Pass.
- **Mistral Small 4** (March 16, 2026): 119B-A6B MoE, Apache 2.0, merges Magistral/Pixtral/Devstral (https://mistral.ai/news/mistral-small-4/). Q4 is ~65 GB: does not fit 48 GB RAM. **Ministral 3** (Dec 2, 2025) dense 3B/8B/14B: the 14B at ~8.5 GB Q4 is a fine dense option but not exceptional vs the MoE field.
- **Gemma 4 12B dense** (June 3, 2026): ~7.5 GB at Q4 sits right at our VRAM edge; Q3/Q4_0 with short context might squeeze into 8 GB fully resident. Worth a note as a GPU-resident fallback if Qwen3.5-9B disappoints.
- **Qwen3.6-27B dense**: SWE-bench Verified 77.2 and the roundups' coding pick for 24 GB cards (https://www.kdnuggets.com/top-7-coding-models-you-can-run-locally-in-2026), but a dense 27B on our rig is CPU-bandwidth-bound at ~2-3 t/s. Dead on arrival here; listed to show WHY our class filter is MoE-first.
- **Llama 4 Scout**: 109B total. Q4 ~55-60 GB. Does not fit. Pass.
- **ERNIE-4.5-21B-A3B** (Baidu, mid-2025): ~12 GB Q4 (https://huggingface.co/bartowski/baidu_ERNIE-4.5-21B-A3B-PT-GGUF), decent but now outclassed within its own size band. Pass.
- **Mellum2-12B-A2.5B-Thinking** (JetBrains): sparse MoE coder, scores within 6 points of 24 GB-card models per https://benchlm.ai/best/local-llm. Niche; watch, do not test yet.
- **GLM-4.6/4.7 full and GLM-4.5-Air**: 355B and 106B-A12B respectively; both far beyond 48 GB at usable quants (https://unsloth.ai/docs/models/tutorials/glm-4.6-how-to-run-locally). Flash is the only GLM that fits.

## Cross-cutting observations

- **The A3B class consolidated hard.** Qwen3.6-35B, GLM-4.7-Flash, Nemotron 3 Nano, Gemma 4 26B-A4B all landed within ~6 months on the exact config we optimized for in E013/E023. Our `--n-cpu-moe` + 12-thread protocol transfers directly to all of them.
- **Hybrid linear attention (GDN/Mamba2) is now standard** in this class. Good for us: KV cache shrinks, long context gets cheaper, and the CPU-resident expert path we understand stays the bottleneck. Risk: newer ops may have less-optimized AVX2/CUDA kernels; our instrumented profiler should measure, not assume.
- **Spec decode on A3B MoE keeps failing for others too.** The RTX 3090 Qwen3.6 study (19 configs, net-negative) independently confirms E014's mechanism. The untested exception is native MTP (draft-mtp in b10064), which avoids a separate draft model entirely. That is the one spec-decode door still open on this hardware.
- **Vendor benchmark inflation is the main epistemic risk** in this sweep (Qwen3.5-9B "beats GPT-OSS-120B" claims especially). Every candidate gets our own falsification battery before any leaderboard claim enters the repo.

---

## Candidate experiments for our rig

Ranked by expected value per download-hour.

### C1. Qwen3.5-9B fully GPU-resident (speed + quality falsification)
Download ~6 GB (35 min): Qwen3.5-9B instruct GGUF Q4 + mmproj. Verify arch loads on b10064. Run our A/B/A battery vs Qwen3-30B-A3B-2507: decode t/s at 0/8K/32K context, plus our quality probes. Hypothesis: >=60 t/s GPU-resident (2x baseline speed) at comparable quality. Also first vision-input smoke test. If quality holds, this redefines the daily-driver config. Effort: hours.

### C2. Qwen3.6-35B-A3B Q4 as the new flagship (model upgrade)
Download 21 GB (~2 h). UD-Q4_K_XL, `--n-cpu-moe`, -t 12, mlock, A/B/A flanked against baseline. Hypothesis: >=28 t/s (within 10% of baseline) with a generational quality jump (GPQA-D 86.0 vs ~70-73 class). Verify GDN layer placement and expert offload engagement from logs; watch for the CUDA gibberish issue reported on 13.2. Long-context bonus test: decode at 32K where baseline's KV cost bites but GDN should not. Effort: days.

### C3. Qwen3.6-35B-A3B-MTP + draft-mtp (the untested spec-decode door)
Depends on C2 rig setup; extra download ~22 GB for the MTP GGUF (or delta if Unsloth ships both in one repo). E014 killed draft-simple and ngram-simple; draft-mtp is explicitly untested in b10064 and MTP avoids the expert-union mechanism (native head, `--spec-draft-n-max 2`). Public data: 1.17x on RTX PRO 6000, unknown on CPU-expert hybrids. Hypothesis to falsify: MTP yields >=20% net decode gain on OUR hybrid config. A clean negative extends the E014 law to native-head spec decode; a positive is breakthrough (b). Effort: days.

### C4. GLM-4.7-Flash as coding/agentic flagship (model upgrade, cheap)
Download ~18 GB (~1.7 h). Needs `--jinja`, sigmoid scoring fix confirmed in b10064, repeat penalty off. A/B/A decode benchmark plus SWE-style task probes vs baseline and vs C2 winner. Hypothesis: >=25 t/s and clearly better agentic coding than Qwen3-30B-A3B (SWE-V 59.2 vs 22.0 says yes). Reasoning-token overhead measured as tokens-to-answer, not just t/s. Effort: days.

### C5. Nemotron 3 Nano Omni 30B-A3B (capability: local ears and eyes)
Precheck FIRST at zero download cost: confirm b10064 mtmd supports this arch's audio+image projectors (read llama.cpp docs/PRs, load a tiny mtmd model). Then download 23.9 GB Q4_K_M + mmproj (~2.3 h). Hypothesis: usable multimodal chat (image caption, audio transcription+reasoning) at >=20 t/s text decode with experts on CPU. This is breakthrough (c): a capability the rig has never had. Effort: days, gated on the precheck.

### Stretch (only if C2/C3 disappoint): Qwen3-Coder-Next 80B-A3B Q3_K_M
38.3 GB download (~3.5 h), 48 GB RAM is genuinely tight (relax mlock, measure paging). 80B-class agentic coder at 20-30 t/s would be breakthrough (c), but run the RAM budget math on paper first: weights minus GPU-resident tensors plus KV plus Windows baseline must stay under ~44 GB. Effort: week+ including stability work.

---

## Feasibility verdicts

Adversarial review 2026-07-19. Every candidate's existence, GGUF availability, and llama.cpp status re-verified against primary sources (HF repos fetched directly). None duplicates E013/E023, E014 (draft-simple/ngram-simple only), or E021-E028.

- **C1 Qwen3.5-9B GPU-resident: GO.** Official card confirms hybrid GDN + sparse MoE, 262K context, vision (https://huggingface.co/Qwen/Qwen3.5-9B); GGUFs from bartowski (built on b9222, well before our b10064) at Q4_K_M 6.17 GB. Caveats: 6.17 GB weights + mmproj + KV + CUDA context on an 8 GB card leaves little headroom, plan Q4_K_S or short context for the first pass; the 60-100 t/s hypothesis rests on an unpublished active-param count, treat it as the thing under test.
- **C2 Qwen3.6-35B-A3B flagship: GO.** Model card, Apache 2.0, April 2026 release, and Unsloth/bartowski GGUFs all confirmed; UD-Q4_K_XL is 22.9 GB not 21 GB (~10 min more download, still ~2.2 h); same --n-cpu-moe pattern as E013 with a new model is a model-upgrade test, not a duplicate. One correction: mainline MTP/spec support is PR #22673 (merged 2026-05-16, flag renamed to --spec-type draft-mtp on 2026-05-13), not PR #19493 as cited above.
- **C3 Qwen3.6-35B-A3B-MTP + draft-mtp: GO.** MTP GGUF repo exists (UD-Q4_K_XL 22.9 GB, a full separate download as stated); draft-mtp is in mainline since ~b9180/PR #22673 and our b10064 ships it; E014 tested only draft-simple and ngram-simple, so this is the one open spec-decode door. Honesty note: MTP avoids a separate draft model but batch VERIFICATION through CPU-resident experts still pays expert-union bandwidth, so the public 1.17x (full-GPU, RTX PRO 6000) is an upper bound here; a clean negative is still publishable.
- **C4 GLM-4.7-Flash: GO.** Repo confirmed (UD-Q4_K_XL 17.5 GB, Q4_K_M 18.3 GB), 30B-A3B, llama.cpp supported with llama-server one-liner; the Jan 21 GGUF re-upload requirement is real (Unsloth: looping/poor outputs fixed, re-download), so grab current quants. Download ~1.7 h honest.
- **C5 Nemotron 3 Nano Omni: MAYBE.** Repo, quant sizes (Q4_K_M 23.9 GB, MXFP4_MOE 21.7 GB), and mmproj files (mmproj-F16.gguf 1.59 GB) all confirmed in the GGUF repo, which is strong evidence for llama.cpp image input; AUDIO via mtmd on this arch is unverified anywhere and vendor docs steer multimodal to vLLM/SGLang/TRT-LLM. The zero-download precheck gate already in C5 is mandatory: if b10064 mtmd loads the audio projector, upgrade to GO; if image-only, the capability pitch shrinks and Qwen3.5-9B vision may cover it for 6 GB instead of 25.5 GB.
- **Stretch Qwen3-Coder-Next Q3_K_M: MAYBE.** Repo and sizes confirmed exactly (Q2_K 29.2 / Q3_K_M 38.3 / Q4_K_M 48.5 GB, so Q4 indeed does not fit); llama.cpp support real (Feb 4 fix noted, b10064 postdates it). Blockers are physics not existence: ~35 GB CPU-resident weights + ~5-6 GB Windows baseline + KV/buffers lands at ~43-45 GB of 48, mlock must be relaxed, and paging noise collides with our +-10% cross-session error bar. Only after C2/C3/C4, exactly as ranked, and only at Q3 with quality checked against C4 first.

Cross-cutting: all six run on stock llama.cpp b10064 on Windows/AVX2 (no AVX-512/AMX dependency anywhere); all download estimates check out within ~10% at 3 MB/s; total Tier-1 download budget if everything runs is ~95 GB against 416 GB free, fine.
