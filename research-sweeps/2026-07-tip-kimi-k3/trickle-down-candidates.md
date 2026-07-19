# Kimi K3 trickle-down: smaller-model candidates for our rig

Sweep date: 2026-07-19. Rig: 48 GB RAM + 8 GB VRAM, llama.cpp b10064 CUDA.
Question: does the K3 announcement produce anything we can actually run, now or soon?

## Bottom line

No smaller K3 variants are announced. But the K3-shadow model already exists and has existed since late 2025: **Kimi-Linear-48B-A3B-Instruct**, Moonshot's own architecture preview of the exact hybrid linear attention (Kimi Delta Attention) that K3 uses. It is MIT licensed, merged into mainline llama.cpp in February 2026, has bartowski GGUFs that fit our RAM at Q4, and is the single best queue candidate from this whole tip.

## 1. K3 family: no small variants announced

- K3 launched July 16, 2026 with exactly two variants: **K3 Max** and **K3 Swarm Max**. Both are the full 2.8T-A50B model (Swarm is a parallel sub-agent serving mode, not a different size). Sources: [Tom's Hardware](https://www.tomshardware.com/tech-industry/artificial-intelligence/moonshot-releases-2-8-trillion-parameter-kimi-k3), [Latent Space AINews](https://www.latent.space/p/ainews-kimi-k3-28t-a50b-the-largest), [DataCamp](https://www.datacamp.com/blog/kimi-k3).
- Weights promised on Hugging Face by **July 27, 2026** under modified MIT. Vendor promise, not yet verifiable.
- No mini/air/lite/distill mentioned anywhere in launch coverage. Community requests for a sub-300B MoE exist in comment threads but there is zero signal from Moonshot.
- 2.8T total, 16 of 896 experts active (~1.8%). At any quant this is 350 GB+ minimum. Permanently out of scope for this rig.

## 2. The shadow model already exists: Kimi-Linear-48B-A3B

[moonshotai/Kimi-Linear-48B-A3B-Instruct](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct)

- **48B total, 3B active** MoE. **Kimi Delta Attention (KDA)**, a refined Gated DeltaNet, in a **3:1 KDA-to-full-MLA hybrid**. 1M token context. MIT license. 5.7T training tokens. Released as an open research model in late 2025, months before K3.
- The video's headline "6.3x faster decode deep in a 1M window" is a **vendor number from this model's card** ("6.3x faster TPOT compared to MLA"), which Moonshot carried into K3 marketing. So the flashiest K3 claim was actually measured on the 48B model in our size class.
- KV cache reduced up to 75% vs full attention. On an 8 GB VRAM card this matters more than raw speed: long contexts stop eating memory we do not have.
- **llama.cpp support is in mainline**: [PR #18755](https://github.com/ggml-org/llama.cpp/pull/18755) (backend-agnostic KDA + MLA KV cache) merged **Feb 6, 2026**, superseding the earlier [PR #17592](https://github.com/ggml-org/llama.cpp/pull/17592). Original feature request: [issue #16930](https://github.com/ggml-org/llama.cpp/issues/16930). Our b10064 build postdates the merge.
- **GGUFs**: [bartowski/moonshotai_Kimi-Linear-48B-A3B-Instruct-GGUF](https://huggingface.co/bartowski/moonshotai_Kimi-Linear-48B-A3B-Instruct-GGUF), imatrix quants made on build b7966. Sizes: IQ4_XS **26.5 GB**, Q4_K_S 29.0 GB, Q4_K_M 30.1 GB, Q3_K_XL 24.0 GB. IQ4_XS or Q4_K_S fits our 48 GB RAM with room for context; run with experts on CPU and attention/shared tensors on the 4060. Same A3B active budget as the Qwen3.6-35B-A3B hybrid-GDN we just measured at +32%, so throughput should land in the same band.
- Caveat: this is a 2025-vintage research model tuned to showcase architecture, not a frontier-distilled instruct model. Model card benchmarks (MMLU-Pro 51.0 at 4k, RULER 84.3 at 128k with 3.98x speedup) are vendor-reported. Independent community quality signal is thin; treat it as an architecture benchmark twin for our hybrid-linear testing first, daily driver second.

## 3. What happened with K2 distills (history check)

- **Moonshot has never shipped a small variant of any flagship.** The whole K2 line stayed at 1T-1.1T through every refresh: K2-Instruct (1T), [K2.5](https://huggingface.co/unsloth/Kimi-K2.5-GGUF) (1.1T, Apr 2026), [K2.6](https://huggingface.co/unsloth/Kimi-K2.6-GGUF) (1.1T, May 2026), Kimi-K2.7-Code (1.1T, June 2026). Their small models are separate research lines: Moonlight-16B-A3B (Feb 2025), Kimi-VL-A3B (16B vision), Kimi-Linear-48B-A3B, plus Kimi-Dev-72B (a Qwen2.5-72B coding fine-tune, not a distill).
- **Community distills of K2 were small and thin.** The visible ones are [TeichAI/Qwen3-8B-Kimi-K2-Thinking-Distill-GGUF](https://huggingface.co/TeichAI/Qwen3-8B-Kimi-K2-Thinking-Distill-GGUF) and [TeichAI/Qwen3-4B-Thinking-2507-Kimi-K2-Thinking-Distill-GGUF](https://huggingface.co/TeichAI/Qwen3-4B-Thinking-2507-Kimi-K2-Thinking-Distill-GGUF): SFT on roughly **1,000 K2-Thinking traces** onto Qwen3 bases. That is a style transfer, not a capability distill. GGUFs exist and run trivially here, but they never showed benchmark evidence of beating their Qwen3 base models. No DeepSeek-R1-style official distill wave ever happened for K2.
- Context worth remembering: in Feb 2026 Anthropic accused Moonshot (with DeepSeek and MiniMax) of industrial-scale distillation of its models ([TechCrunch](https://techcrunch.com/2026/07/18/kimi-threat-or-menace/)). Whatever the merits, it makes Moonshot officially blessing downstream distills of K3 less likely, not more.

## 4. What fits our rig today (Kimi family)

| Model | Size / active | Quant fit | llama.cpp | Worth queueing? |
|---|---|---|---|---|
| Kimi-Linear-48B-A3B-Instruct | 48B / 3B | IQ4_XS 26.5 GB in RAM | Mainline since Feb 2026 | **Yes, top pick** |
| Kimi-VL-A3B-Thinking-2506 | 16B / 2.8B | Q4 ~10.5 GB | Yes, [ggml-org GGUF](https://huggingface.co/ggml-org/Kimi-VL-A3B-Thinking-2506-GGUF) | Optional, only if we want a local vision model |
| Moonlight-16B-A3B-Instruct | 16B / 3B | Q4 ~9 GB | Yes | No, superseded by Qwen3 class |
| Kimi-Dev-72B | 72B dense | Q4 ~40 GB, barely | GGUFs exist ([unsloth](https://huggingface.co/unsloth/Kimi-Dev-72B-GGUF)) | No, dense 72B means ~1-2 t/s here |
| TeichAI K2-Thinking distills | 4B / 8B | Trivial | Yes | Low priority, thin distills |
| K2 / K2.5 / K2.6 / K2.7 / K3 | 1T-2.8T | Impossible | n/a | Never |

## 5. Realistic timeline for a 20-40B K3-shadow

- **The pattern is preview-before-flagship, not distill-after.** Kimi-Linear-48B (the KDA preview) shipped roughly 8 months before K3. If Moonshot repeats this, the next small model appears as a research preview of the K4-cycle architecture (AttnRes, SiTU activation, LatentMoE are the new K3 pieces the 48B preview lacks), not as a K3 shrink. Speculative window: late 2026 to mid 2027. No official signal of any kind; this is pattern extrapolation, clearly labeled as such.
- **Community K3 distills will appear within weeks of the July 27 weights drop** if the drop happens, because K2-Thinking got them. Expect Qwen-base SFT-on-traces in the 4B-30B range. History says they will be flavor transfers of limited value.
- **An official K3-mini is the least likely outcome.** Zero precedent across four flagship releases.

## 6. Verification notes

- Vendor claims (unverified independently): 6.3x decode, 75% KV reduction, AttnRes ~25% training efficiency, K3 benchmark suite, July 27 weights date.
- Independent or third-party: Artificial Analysis index 57, GDPval Elo 1668, Frontend Code Arena #1 at 1679 (as reported by [Latent Space](https://www.latent.space/p/ainews-kimi-k3-28t-a50b-the-largest) and launch press; AA page itself not re-fetched this sweep).
- Could not verify: any Moonshot statement about smaller K3 models (because none exists), independent quality evals of Kimi-Linear-48B beyond the model card.
- Channel calibration: the video's K3 specs cross-check almost perfectly against launch coverage (2.8T vs "~3T", 16/896 experts exact, $3/$15 exact, AA 57 exact, GDPval 1668 exact, July 27 exact, both variant names right). "Garbler not fabricator" holds; this one was barely even garbled.
