# Kimi K3: existence and claims verification

Sweep: 2026-07-tip-kimi-k3 | Checked: 2026-07-19 | Video claims vs primary and independent sources

## Verdict

**The video is substantially accurate.** Kimi K3 is real, announced by Moonshot AI on July 16, 2026. Every specific number in the video that we could check matched a primary or independent source: 16/896 experts, 6.3x decode speedup, +25% training efficiency, $3/$15 pricing, July 27 weights date, AA index 57 vs ~60 for top closed, GDPval ELO 1668, and the 2-of-6 coding benchmark record. The only garbles are the name ("Kimmy" for Kimi) and rounding 2.8T to "~3T", and Moonshot itself markets the model as "the world's first open 3T-class model", so even that matches vendor framing. Nothing was invented. The "modified MIT" license is reported and consistent with the K2 line precedent but cannot be fully verified until the weights ship.

## Claim-by-claim

| Video claim | Verdict | Reality and source type |
|---|---|---|
| Moonshot announced "Kimi K3" | Accurate | Launched July 16, 2026 (product live on kimi.com, Kimi Work, Kimi Code, API). AA evaluation published July 17. |
| ~3T total params MoE | Rounded, fair | 2.8T total. Moonshot's own blog calls it the "world's first open 3T-class model". Vendor claim. |
| 16 of 896 experts active | Accurate | Exact match to Moonshot blog ("effectively activating 16 out of 896 experts", Stable LatentMoE). Coverage pegs active params around 50B (AINews headline: "2.8T-A50B"). Vendor claim. |
| Kimi Delta Attention, hybrid linear attention | Accurate | KDA is the hybrid linear attention mechanism, a fine-grained-gated refinement of Gated DeltaNet (arXiv 2510.26692, Kimi Linear). Vendor architecture description. |
| 6.3x faster decode deep in a 1M window | Accurate | Blog: "up to 6.3x faster decoding in million-token contexts". Vendor claim, not independently measured. |
| Attention residuals, +25% training efficiency | Accurate | Blog: AttnRes delivers "~25% higher training efficiency at <2% additional cost". Overall K3 claims ~2.5x scaling efficiency vs K2. Vendor claim. |
| Native vision and video | Mostly accurate | Native vision confirmed everywhere. The Moonshot blog claims K3 "understands text, images, and video within the same model" (demos include video editing). But Artificial Analysis lists modality as text plus image input, text output. Video understanding is vendor-claimed, not independently benched. |
| Two variants: K3 Max, K3 Swarm Max | Accurate | Both shipped at launch as deployment variants of the same weights: K3 Max for chat and single-agent, K3 Swarm Max extends the Agent Swarm multi-agent framework to K3 scale. AA benched the "Kimi K3 Max" configuration (57.1). Thinking effort defaults to max; low and high effort modes referenced. |
| $3 / $15 per M tokens | Accurate | $3.00/MTok input (cache miss), $15.00/MTok output, plus $0.30/MTok cache-hit input the video omitted. Moonshot reports >90% cache hit rate in coding workloads. |
| Weights on Hugging Face July 27, 2026 | Accurate (pending) | Blog: "full model weights will be released by July 27, 2026", with a technical report. As of 2026-07-19 the weights are NOT up; Moonshot's HF org still tops out at Kimi-K2.7-Code. |
| Modified MIT license | Plausible, unverified | The blog does not name the license. Coverage expects Modified MIT following the K2.5/K2.6/K2.7 precedent; the final license ships with the files. Verify on July 27. |
| Artificial Analysis index 57, top closed ~60 | Accurate, independent | K3 scores 57 (K3 Max config 57.1), rank #4 of 187, effectively #3 counting best config per family. Claude Fable 5 is #1 at 60 under index v4.1 (the launch-day 64.9 figure was a retired methodology). Comparable to Opus 4.8 and GPT-5.5, behind Fable 5 and GPT-5.6 Sol. |
| GDPval ELO 1668 | Accurate, independent | GDPval-AA v2 (Artificial Analysis's run, not OpenAI's original leaderboard): K3 1668 vs Fable 5 1760, Opus 4.8 1600, GLM-5.2 1514, GPT-5.5 1494. Up from K2.6's 1190. Also AA-Briefcase ELO 1547 (+732 vs K2.6) and #1 on AutomationBench-AA at 53%. |
| Wins 2 of 6 coding benchmarks vs top closed | Accurate | Of the 6 coding benchmarks in Moonshot's table, K3 beats BOTH Fable 5 and GPT-5.6 Sol on exactly 2: Program Bench (77.8 vs 76.8 / 77.6) and SWE Marathon (42.0 vs 35.0 / 39.0). Splits the rest: Terminal-Bench 2.1 88.3 beats Fable 5 (84.6) but not Sol (88.8); KCB 2.0 and FrontierSWE each beat one of the two; loses DeepSWE to both (67.5 vs 70.0 / 73.0). Separately #1 on Frontend Code Arena (1,679, 76% pairwise wins), passing Fable 5. Vendor-reported table plus arena leaderboard. |

## What does not exist or cannot be verified yet

- **No weights on HF** as of 2026-07-19. Announced open-weight, currently hosted-only.
- **No license text** published. Modified MIT is expectation, not fact.
- **No technical report** yet (promised with the weights).
- **No smaller or distilled K3 family members** announced. Only the 2.8T model, in Max and Swarm Max deployment forms.
- **6.3x decode and +25% AttnRes numbers** are vendor-only; no independent replication.

## Vendor claims vs independent measurements

- Vendor (Moonshot blog): params, expert counts, KDA 6.3x, AttnRes 25%, video understanding, the 6-benchmark coding table, weights date.
- Independent (Artificial Analysis): index 57, GDPval-AA v2 ELO 1668, AA-Briefcase 1547, AutomationBench #1, output speed 62 tok/s, $0.94 cost per task.
- Third-party press (Bloomberg, CNBC, Tom's Hardware, VentureBeat) corroborates the announcement, positioning, and pricing.

## Lab relevance (48 GB RAM + 8 GB VRAM, llama.cpp)

1. **K3 itself is permanently out of reach.** 2.8T total, ~50B active. Even the native MXFP4 weights are on the order of 1.4+ TB. Not a target, ever.
2. **The architecture trend is the story.** KDA is a fine-grained-gated Gated DeltaNet variant, the same hybrid linear attention family as the Qwen3.6-35B-A3B hybrid-GDN we just measured at +32% on this rig. A frontier flagship built on it confirms hybrid linear attention is the direction, which means more runnable-class models with this architecture are coming.
3. **The runnable relative already exists: Kimi-Linear-48B-A3B-Instruct** (Moonshot's KDA testbed, arXiv 2510.26692). 48B-A3B is squarely our class. llama.cpp support was feature request #16930, blocked behind the Qwen3-Next GDN work (#16095); our b10064 build already runs hybrid-GDN, so check whether KDA landed upstream since.
4. **July 27 checklist:** confirm the license, read the technical report for KDA and AttnRes details, and watch for any smaller K3-family or distilled releases.
5. **Native MXFP4 weights with MXFP8 activations** continues the native low-precision release trend, which is good news for quantization quality at our sizes.
6. **Channel calibration:** upgrade this channel. Every checkable number in this video was correct; the only defects were name pronunciation and rounding. Consistent with, and stronger than, the prior "garbler not fabricator" rating from the bonzai tip.

## Sources

- Moonshot tech blog: https://www.kimi.com/blog/kimi-k3
- Moonshot announcement on X: https://x.com/Kimi_Moonshot/status/2077830229968683203
- Artificial Analysis K3 evaluation: https://artificialanalysis.ai/articles/kimi-k3-achieves-3-in-the-artificial-analysis-intelligence-index-comparable-to-opus-4-8-and-gpt-5-5
- Artificial Analysis K3 model page: https://artificialanalysis.ai/models/kimi-k3
- Artificial Analysis on X (57 score): https://x.com/ArtificialAnlys/status/2077832874183860404
- AA index v4.1 methodology (Fable 5 = 60): https://artificialanalysis.ai/articles/artificial-analysis-intelligence-index-v4-1
- AINews roundup (2.8T-A50B, benchmarks): https://www.latent.space/p/ainews-kimi-k3-28t-a50b-the-largest
- Benchmark breakdown vs Fable 5 / GPT-5.6 Sol: https://officechai.com/ai/kimi-k3-benchmarks/
- Variant explainer (Max, Swarm Max): https://lorphic.com/kimi-k3-benchmark-api-and-more/
- Bloomberg: https://www.bloomberg.com/news/articles/2026-07-17/china-s-powerful-new-moonshot-ai-model-closes-gap-with-us-rivals
- CNBC: https://www.cnbc.com/2026/07/17/moonshot-ai-kimi-k3-model-openai-anthropic-china.html
- Tom's Hardware: https://www.tomshardware.com/tech-industry/artificial-intelligence/moonshot-releases-2-8-trillion-parameter-kimi-k3
- VentureBeat: https://venturebeat.com/technology/chinas-moonshot-ai-releases-kimi-k3-the-largest-open-source-model-ever-rivaling-top-u-s-systems
- Moonshot HF org (K2.7 still latest as of check): https://huggingface.co/moonshotai/Kimi-K2.7-Code
- Kimi Linear paper (KDA): https://arxiv.org/abs/2510.26692
- Kimi Linear repo: https://github.com/MoonshotAI/Kimi-Linear
- llama.cpp KDA feature request: https://github.com/ggml-org/llama.cpp/issues/16930
