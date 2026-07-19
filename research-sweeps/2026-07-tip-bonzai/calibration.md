# Calibration sweep: fact-checking the video's supporting claims

Date: 2026-07-19
Scope: The Bonzai video cites three ecosystem claims as context. Each one verified or debunked shifts the prior on the headline "Prism ML Bonzai 27B" claim. This file records what each claim maps to in reality, with primary sources.

## Bottom line

All three supporting claims are REAL. The proper nouns are garbled in a pattern consistent with AI narration or auto-transcription, but the numbers are accurate to near-exact:

| Video said | Reality | Status |
|---|---|---|
| "Kolibri", 744B on 25 GB | Colibri (JustVugg/colibri), GLM-5.2 744B MoE on ~25 GB RAM | Real, numbers exact |
| "Hi3 in 1-bit", 295B to 83 GB | Tencent Hunyuan Hy3, 295B MoE, official 1-bit IQ1_M at 85.5 GiB (89.4 to 91.8 GB) | Real, size slightly off |
| "Angel Slim" toolkit | Tencent/AngelSlim, Hunyuan AI Infra team | Real, name exact minus spacing |

Bonus finding: the headline claim also resolves. "Prism ML Bonzai 27B" is **prism-ml/Ternary-Bonsai-27B-gguf** on Hugging Face (Bonsai, not Bonzai). Existence confirmed; quality claims not yet verified here.

## Claim 1: "Kolibri" streams experts from SSD, 744B on a 25 GB machine

**Verdict: REAL.** The project is **Colibri** (Italian spelling, with a K in the video's rendering).

- Repo: https://github.com/JustVugg/colibri (Apache 2.0, full source, no binary-only releases)
- What it is: a ~2,400-line pure C, zero-dependency inference engine that runs GLM-5.2 (744B MoE by Z.ai, ~40B active per token) on roughly 25 GB of RAM
- How: dense components (~17B params: attention, shared experts, embeddings) stay resident at int4 (~9.9 GB); the ~370 GB of routed experts stream from disk on demand with async readahead and an LRU cache that pins hot experts
- Traction: 16.2k stars, 1.5k forks as of this sweep; released around July 10, 2026; single-author project
- Honest about limits: 0.05 to 1.23 tok/s depending on hardware, quality benchmarking still incomplete per its own README
- Coverage: https://www.noze.it/en/insights/colibri-glm-5-2-locale/ and https://medium.com/data-science-in-your-pocket/run-glm-5-2-in-just-25-gb-ram-colibri-6e2d5b7bb51d

The video's numbers (744B, 25 GB) match exactly. Only the name is garbled (K for C).

## Claim 2: "Hi3 in 1-bit", 295B flagship down to 83 GB on a single GPU

**Verdict: REAL, size slightly garbled.** The model is Tencent Hunyuan **Hy3**, a 295B total / 21B active MoE flagship.

- Official announcement: Tencent Hunyuan on X, 1-bit and 4-bit Hy3 servable on a single GPU with llama.cpp and MTP: https://x.com/TencentHunyuan/status/2076953120765280284 (around July 14, 2026)
- Official quants live under the AngelSlim org on Hugging Face: https://huggingface.co/AngelSlim/Hy3-GGUF
- 1-bit (IQ1_M): 89.4 GB, or 91.8 GB with MTP heads (equals 85.5 GiB, a 6.7x cut from the 598 GB original). The video's "83 GB" is close but not exact
- 4-bit (Q4_K_M): 182 GB, 185 GB with MTP
- Coverage: https://www.remio.ai/post/tencent-hunyuan-hy3-quantized-release-1bit-single-card-deployment-4bit-near-full-performance

**Detail that matters for us:** Hy3 runs on MAINLINE llama.cpp. Support was merged as PR #25395 (any commit after 505b1ed). The HF page explicitly says earlier GGUFs built against a patched llama.cpp will not load on today's upstream. So the real ecosystem pattern is custom-fork-first, then upstreaming. The video's claim that Bonzai "requires Prism ML's fork" fits the first phase of that pattern but should be re-checked against mainline support status before we touch anything.

## Claim 3: "Angel Slim" open-source compression toolkit

**Verdict: REAL.** It is **AngelSlim** by Tencent's Hunyuan AI Infra team.

- Repo: https://github.com/Tencent/AngelSlim (1.5k stars, 167 forks, active with v0.5.0 on June 22, 2026 and FP8 for Hy3 in July 2026)
- Toolkit paper: https://arxiv.org/abs/2602.21233
- Supports Qwen3, Hunyuan, DeepSeek-R1/V3, GLM-4.6 and more; FP8/INT8/INT4 quant, Eagle3 speculative decoding, pruning

**Directly relevant to the Bonzai prior:** AngelSlim contains **Tequila**, a ternary quantization method (weights in -1/0/+1) accepted at ICLR 2026:

- Paper: https://arxiv.org/abs/2509.23809 (Tequila: Trapping-free Ternary Quantization for Large Language Models)
- Code: https://github.com/Tencent/AngelSlim/tree/tequila/TernaryQuant
- Claims: fixes deadzone trapping by repurposing trapped weights as dynamic biases; on ARC, over 4% above the SOTA ternary baseline and within 1% of full precision, with 3.0x speedup

So credible, peer-reviewed ternary work claiming near-lossless quality exists in this exact ecosystem right now. Note Tequila involves gradient updates (quantization-aware fine-tuning), not pure post-hoc conversion, so "post-hoc ternarization is impossible" is now a weaker prior than our pre-sweep assumption, but the mechanism still is not free lunch PTQ.

## Bonus: the headline claim has a real referent

Searching for the garbled name found the real one. "Prism ML Bonzai 27B" resolves to:

- https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf (ternary, 1.71 bits/weight claimed, Qwen3.6-27B base, claims 95% retention, GGUF Q2_0_g128 with custom llama.cpp kernels)
- https://huggingface.co/prism-ml/Bonsai-27B-gguf (the ~3.9 GB 1-bit phone-class build)
- https://huggingface.co/prism-ml/Ternary-Bonsai-27B-mlx-2bit (MLX variant, matching the video)
- Third-party coverage: https://www.marktechpost.com/2026/07/14/prismml-releases-bonsai-27b-1-bit-and-ternary-builds-of-qwen3-6-27b-that-run-on-laptops-and-phones/
- Hosted API by Together AI: https://www.together.ai/models/prism-ml-ternary-bonsai-27b
- Independent benchmark attempt on GitHub: https://github.com/ArmanJR/PrismML-Bonsai-vs-Qwen3.5-Benchmark

This is existence verification only. The 95% retention number, the 1M downloads figure, the account age of the prism-ml org, and the state of the custom llama.cpp fork are for the dedicated Bonzai sweep to verify.

## What this does to the prior

- **The video is a garbler, not a fabricator.** Every checkable claim mapped to a real event in the July 10 to 14, 2026 window. Substance accurate, proper nouns mangled (Kolibri/Colibri, Hi3/Hy3, Bonzai/Bonsai), one number slightly off (83 vs 85.5 GiB).
- **Prior on Bonzai existing: sharply up.** It exists on Hugging Face under prism-ml with the exact variant lineup the video described, plus third-party coverage and an API host.
- **Prior on the quality claim: modestly up, unproven.** Tequila (ICLR 2026, Tencent) shows near-lossless ternary is a live research direction, but it uses fine-tuning, and Bonsai's own 95% figure is still a vendor claim.
- **Security posture unchanged.** Hy3's history (patched fork GGUFs that later broke on upstream) confirms the fork-first pattern is normal but messy. We still build any fork from source after reading the diff. No prebuilt binaries.
