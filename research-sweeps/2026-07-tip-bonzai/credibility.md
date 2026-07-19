# Bonsai 27B technical credibility sweep

Sweep date: 2026-07-19. Scope: independent evaluations, how the "95% intelligence retained" number is produced, published literature on post-training ternarization at this scale, and whether PrismML's technical report holds up.

Bottom line: **the release is real and the technical story broadly holds up, with caveats.** The headline 95% figure is vendor-measured on a vendor-chosen suite. Independent testing lands a few points lower (roughly 88-92% relative retention) but nowhere near the collapse that naive post-hoc ternarization produces. The compression method itself is undisclosed ("proprietary Caltech intellectual property"), so the strongest claim in the tip, "post-hoc ternarization with no retraining", is not actually established anywhere, including by PrismML.

Naming corrections vs the YouTube video: the model is **Bonsai** (not "Bonzai"), the base is **Qwen3.6-27B** (a real July 2026 hybrid-attention release, ~75% linear attention), and the SSD-streaming engine is **Colibri** (not "Kolibri").

---

## 1. What actually exists (verified primary sources)

- **Hugging Face org `prism-ml`**: 28+ model repos. The 27B family (created 2026-07-04) includes GGUF, MLX 1-bit/2-bit, AWQ 4-bit, and unpacked variants. Download counts via the HF API on 2026-07-19: **Bonsai-27B-gguf 1,262,894** and **Ternary-Bonsai-27B-gguf 338,945** downloads last month; org-wide total ~1.79M. The org is not new: Bonsai 8B/4B/1.7B repos date to March-April 2026 and image models to May 2026. https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf
- **Whitepaper**: 24-page PDF, `bonsai-27b-whitepaper.pdf` in https://github.com/PrismML-Eng/Bonsai-demo (downloaded and read in full for this sweep; local copy in the session scratchpad). Two earlier whitepapers (1-bit Bonsai 8B, March 2026; Ternary Bonsai 8B/4B/1.7B, April 2026) also ship there.
- **llama.cpp fork is full source, not binary-only**: https://github.com/PrismML-Eng/llama.cpp branch `prism`, MIT, ~9,600 commits, 359 stars, 70 forks, 13 open issues. Implements Q2_0 (ternary g128) and Q1_0 (binary g128). README notes "Ternary (Q2_0) support is migrating into mainline llama.cpp backend-by-backend". The Bonsai-demo repo (1.8k stars) pushes prebuilt binaries from GitHub Releases for convenience, but our build-from-source path exists and is viable.
- **Third-party hosting**: Together AI serves a Ternary Bonsai 27B API endpoint. https://www.together.ai/models/prism-ml-ternary-bonsai-27b
- **Press coverage**: 9to5Mac (2026-07-14), Decrypt, Gigazine (2026-07-15), AlphaSignal, MarkTechPost. All rewrite the vendor announcement; none add independent measurements.
- **HN front page**: "Bonsai 27B: A 27B-Class model that runs on a phone", 700 points, 250 comments, 2026-07-14. https://news.ycombinator.com/item?id=48910545

## 2. How the "95%" is measured, and by whom

Per the whitepaper (Sections 7, B, C), the number is **PrismML's own evaluation**:

- 15 benchmarks, thinking mode: MMLU-Redux, MuSR, GSM8K, MATH-500, AIME25, AIME26, HumanEval+, MBPP+, LiveCodeBench, IFEval, IFBench, BFCL v3, tau2-Bench, MMMU-Pro, OCR Bench v2.
- Harness: EvalScope + vLLM on H100, same pipeline for all variants. Rule-based scoring first, Gemini 3 Flash judge as fallback. AIME averaged over 8 samples.
- Result: Qwen3.6-27B FP16 averages 85.07; Ternary Bonsai 80.49 (**94.6%, rounded up to "95%"**); 1-bit Bonsai 76.11 (89.5%).
- Baselines they beat: Qwen3.6-27B IQ2_XXS collapses to 72.73 (85.5%) at 9.4 GB; Gemma-4-31B Q2_K_XL collapses to 73.31 at 11.8 GB. Their strongest claim is that the collapse-prone benchmarks (AIME, LiveCodeBench, tau2-Bench) hold: ternary keeps AIME25/26 at 90.84/87.50 vs 93.29/93.33 FP16.

Methodological caveats in the vendor's own numbers:

- **Self-reported.** Nobody independent has rerun the 15-suite. Kubesimplify explicitly states: "These are PrismML's evaluations, not results I independently reproduced in full."
- **Sampling asymmetry**: Bonsai models run at temperature 0.7, the baselines at 1.0 ("each model's recommended setting"). Defensible, but it is a per-model tuned setting inside the comparison.
- **Baseline choice**: the conventional-quant comparators are GGUF IQ2_XXS / Q2_K_XL, which are known-weak at that bit range. No comparison to the actual academic sub-2-bit competition (PT2-LLM, TWLA, OneBit, TernaryLLM, BiLLM).
- The "size" story mixes ideal and deployed numbers: ternary is marketed at 5.9 GB (ideal 1.71 bpw) but the shipped GGUF is **7.17-7.2 GB** because current kernels store ternary values in 2-bit slots. The whitepaper does disclose this plainly (Table 3, Section 9).

## 3. Independent evaluations found

1. **ArmanJR / PrismML-Bonsai-vs-Qwen3.5-Benchmark** (102 stars, independent): 13 models, 98 questions x 7 categories x 3 runs, on a Jetson Orin 30 GB. Results: Qwen3.5-27B 95.7%, **Qwen3.6-27B 94.2%, Ternary-Bonsai-27B 86.2%, Bonsai-27B (1-bit) 82.9%**, Bonsai-8B 78.9%. That is ~91.5% relative retention for ternary and ~88% for 1-bit on this suite, a few points below the vendor's 94.6%/89.5%. Author's own framing: "rough directional signals, not definitive rankings". Notable finding: "The entire accuracy loss is concentrated in exactly two places: Math and Persian (multilingual)", which partially contradicts the vendor's "math is nearly untouched" story. https://github.com/ArmanJR/PrismML-Bonsai-vs-Qwen3.5-Benchmark
2. **Hacker News thread (48910545)**, mixed hands-on signals:
   - verdverm ran a small eval harness: baseline wikitext ppl 8.00 vs 16.75 on a downloaded "Bonsai 4-bit" build (gsm8k run flagged as an eval bug); concluded it "quant'd too hard" and warned that "bigger quant'd harder is not always better than a model of more modest size".
   - ch_sm: ~24 tok/s on M2 Max with ternary, quality "slightly underwhelmed" vs Qwen3.6-35B at 4-bit.
   - jedbrooke: "They aren't training at all. They are quantizing existing models" (commenter's inference, not a PrismML statement).
   - Multiple users hit loading failures in stock LM Studio / llama.cpp, consistent with the custom-fork requirement.
3. **Earlier HN thread on Ternary Bonsai 8B (47812749)**: Reubend "verified that if you naively quantize to 1 bit from the original Qwen model... it just spits out gibberish", i.e. whatever PrismML does is demonstrably more than naive rounding. londons_explore: "PrismML has not released any actual info on how they trained". armanj measured Ternary-Bonsai-1.7B beating Qwen3.5-0.8B by 12 points at ~5% smaller disk size.
4. **Kubesimplify (Saiyam Pathak)**: independent throughput benchmark, RTX PRO 6000 (ternary 120.7 tok/s tg, 1-bit 145.5) and DGX Spark (28.5 / 42.8). Quality checked only via smoke tests (passed, with one token-budget failure on 1-bit). Found speculative decoding hurt on DGX Spark (-37%). Did not reproduce the accuracy suite. https://blog.kubesimplify.com/bonsai-27b-rtx-pro-6000-dgx-spark
5. **wavect.io** ran a phone-focused review (not deeply mined in this sweep). https://wavect.io/blog/bonsai-27b-phone-local-ai-review/

Net: independent parties confirm the models load, run at the claimed speeds, and are far from gibberish. The only independent accuracy suite (ArmanJR) puts retention a few points below the vendor claim. Nobody independent has replicated the 15-benchmark thinking-mode suite.

## 4. Literature check: post-hoc ternarization of dense transformers

The tip's framing ("PTQ ternary of dense transformers is known to collapse") was correct through 2025 but is slightly stale for mid-2026:

- **Trained ternary works**: BitNet b1.58 (arXiv 2402.17764), but the largest natively trained 1-bit models remain ~2B params (BitNet b1.58 2B4T). PrismML's whitepaper itself makes this point and positions Bonsai as the opposite path.
- **Naive PTQ ternary still collapses**: confirmed anecdotally in the HN 8B thread (gibberish).
- **State of published PTQ art**: PT2-LLM (arXiv 2510.03267, ICLR-track) achieves sub-2-bit ternarization "competitive with SOTA 2-bit PTQ" on LLaMA-class models. **TWLA (arXiv 2606.13054, June 2026)** is the best published anchor: ternary weights + 4-bit activations retains "exceeding 92% of FP16" zero-shot on LLaMA2-70B and ~90% of FP16 on MMLU/HumanEval/GSM8K for Qwen3-32B-Instruct. CAT-Q (arXiv 2606.26650) is in the same space. Neither claims ~95% on a hard thinking-mode suite.
- **Where Bonsai sits**: claiming 94.6% retention at 1.71 bpw on a suite that includes AIME and LiveCodeBench in thinking mode is **ahead of anything published, but by a margin of a few points, not an order of magnitude**. Given TWLA's June 2026 numbers, the claim is aggressive-but-plausible rather than physically absurd. The plausibility gap narrows further if PrismML does recovery training.
- **Critical unknown**: the whitepaper never says the pipeline is training-free. It says Bonsai "starts from an off-the-shelf pretrained model and moves it into a binary or ternary representation" built on "proprietary Caltech intellectual property, a mathematically grounded framework rather than a collection of heuristics". One whitepaper line hints at optimization pressure: the models "have already been shaped to tolerate exactly this kind of noise" (re KV-cache tolerance). Distillation or QAT-style recovery on top of the pretrained checkpoint would be fully consistent with everything published and would make the numbers unremarkable rather than frontier-breaking. The video's "post-hoc, no retraining" framing is its own embellishment.

## 5. Does the technical report hold up?

Strengths: unusually candid for a vendor paper. It discloses deployed-vs-ideal sizes (7.2 vs 5.9 GB), admits the ternary build does not fit phones, reports the true average bit-widths of competing GGUF quants correctly (Q4_K_XL is really 5.2 bpw, IQ2_XXS 2.8 bpw), documents harness, budgets, judge model, temperatures and sampling repeats, reports energy per token with instrumentation details, and lists real references (BitNet, GIDD, BABILong, EvalScope, HQQ, Gemlite). The per-benchmark appendix is complete for all 8 evaluated models.

Weaknesses: **the core compression algorithm is a black box**. No math, no ablations, no calibration-data description, no compute budget, no comparison against academic sub-2-bit methods, and no statement on whether recovery training is involved. "Whitepaper" is accurate; this is a product technical report, not a peer-reviewed paper, and it is not on arXiv. The 95% headline rounds up 94.6%, measured by the vendor, on a suite the vendor chose, at a per-model-tuned temperature.

## 6. Scorecard vs the video's claims

| Video claim | Verdict |
|---|---|
| "Prism ML released Bonzai 27B" | Real (PrismML, Bonsai 27B) |
| Qwen 3.6 27B base | Real (Qwen3.6-27B, hybrid attention) |
| ~7 GB ternary | True (7.17 GB deployed GGUF; 5.9 GB is the ideal-packing figure) |
| 95% intelligence retention | Vendor-reported 94.6% on own suite; best independent suite shows ~91.5% relative; directionally right, modestly inflated |
| Post-hoc ternarization (no retraining) | Unestablished. Vendor discloses no method; naive PTQ demonstrably fails; recovery training cannot be ruled out and is likely |
| 1M downloads in 72 hours | Unverifiable as stated; HF shows ~1.6M last-month downloads across the two 27B GGUF repos (created 2026-07-04), so the magnitude is plausible |
| Requires custom llama.cpp fork | True, but the fork is full source (MIT) and Q2_0 is being upstreamed; prebuilt binaries are a convenience, not the only path |
| 3.9 GB 1-bit phone build, 11 tok/s iPhone 17 Pro Max | Matches whitepaper; Kubesimplify/HN confirm speed claims on other hardware |
| MLX / AWQ variants, vision mmproj | All exist on HF (mmproj is HQQ 4-bit, 0.63 GB) |
| "Kolibri" 744B on 25 GB | Real but misspelled: **Colibri** (JustVugg/colibri, 2026-07-10), streams GLM-5.2 744B MoE experts from SSD at 0.05-1.06 tok/s, i.e. a tech demo, not usable throughput |
| "Hi3 in 1-bit" 295B to 83 GB | Not found under that name in this sweep. Note 83 GB for 295B params is ~2.25 bpw, so the video's own "1-bit" arithmetic does not hold |
| "Angel Slim" toolkit | AngelSlim is Tencent's real, pre-existing compression toolkit, unrelated to PrismML |

## 7. Implications for our security posture

- We do not need their prebuilt binaries. **PrismML-Eng/llama.cpp branch `prism` is full source under MIT** and can be diffed against upstream ggml-org/llama.cpp before building. Q2_0 upstreaming into mainline is reportedly underway, which may soon remove the fork requirement entirely.
- The models themselves are GGUF/safetensors weight files (no code execution) from a 4-month-old HF org with 2k followers and third-party API hosting; standard weight-file hygiene applies.
- Before adopting: wait for or run an independent replication of at least a subset of the 15-suite (GSM8K + LiveCodeBench + IFEval on the ternary GGUF would be a cheap discriminator), and compare against Qwen3.6-27B IQ2_XXS and a 4-bit quant of a mid-size model on our own tasks. HN commenter ch_sm's observation (a 4-bit ~35B beat ternary Bonsai subjectively) is the practical alternative to beat.

## Sources

- https://prismml.com/news/bonsai-27b (vendor announcement)
- https://github.com/PrismML-Eng/Bonsai-demo (whitepapers, demo scripts; 1.8k stars)
- https://github.com/PrismML-Eng/llama.cpp (source fork, branch `prism`)
- https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf and https://huggingface.co/api/models?author=prism-ml (cards, download counts)
- https://news.ycombinator.com/item?id=48910545 (Bonsai 27B thread, 700 pts / 250 comments)
- https://news.ycombinator.com/item?id=47812749 (earlier Ternary Bonsai thread)
- https://github.com/ArmanJR/PrismML-Bonsai-vs-Qwen3.5-Benchmark (independent accuracy suite)
- https://blog.kubesimplify.com/bonsai-27b-rtx-pro-6000-dgx-spark (independent throughput)
- https://www.together.ai/models/prism-ml-ternary-bonsai-27b (third-party hosting)
- https://arxiv.org/abs/2510.03267 (PT2-LLM), https://arxiv.org/html/2606.13054 (TWLA), https://arxiv.org/html/2606.26650 (CAT-Q), https://arxiv.org/abs/2402.17764 (BitNet b1.58)
- https://github.com/JustVugg/colibri (Colibri, the "Kolibri" of the video)
- Press: https://9to5mac.com/2026/07/14/prismml-releases-bonsai-27b-claiming-first-major-ai-model-of-its-size-fit-for-iphone/ , https://gigazine.net/gsc_news/en/20260715-bonsai-27b/ , https://decrypt.co/373578/meet-bonsai-first-27b-ai-model-fits-phone , https://www.marktechpost.com/2026/07/14/prismml-releases-bonsai-27b-1-bit-and-ternary-builds-of-qwen3-6-27b-that-run-on-laptops-and-phones/
