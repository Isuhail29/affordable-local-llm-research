# Kimi K3 architecture, through our lens

Date: 2026-07-19. Trigger: YouTube tip video claiming "Kimi K3" (channel garbles the name as "Kimmy"). Same channel previously calibrated as "garbler not fabricator" on the Bonzai tip. This file covers the architecture claims only; scope is what matters for a 48 GB RAM + 8 GB VRAM llama.cpp lab.

Bottom line: **K3 is real, announced July 16 2026, and its architecture claims check out against primary sources.** The 2.8T model itself is permanently out of our class, but it is the third major lab to ship the same hybrid-linear-attention recipe we just measured winning on our own rig. The direction is confirmed. The runnable family member for us is Kimi-Linear-48B-A3B, and llama.cpp already supports it.

## 1. What Kimi Delta Attention (KDA) actually is

Primary sources: [Kimi Linear paper, arXiv 2510.26692](https://arxiv.org/abs/2510.26692) (Oct 2025), [MoonshotAI/Kimi-Linear on GitHub](https://github.com/MoonshotAI/Kimi-Linear), [Kimi-Linear-48B-A3B-Instruct on HF](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct).

- KDA is a **linear attention module that extends Gated DeltaNet** with a finer-grained gating mechanism: per-channel decay instead of one scalar forget gate per head. The paper frames it as a hardware-efficient special case of DPLR (diagonal-plus-low-rank) transition matrices with a bespoke chunkwise kernel.
- **Kimi Linear = layerwise hybrid, 3 KDA layers : 1 full-attention layer.** The full-attention interleave is MLA (DeepSeek-style latent attention) with no position encoding; KDA carries position information.
- Proof-of-concept model: **48B total, 3B active MoE** (exactly our proven A3B class). Paper claims: beats full MLA under identical training recipes, **up to 75% KV cache reduction, up to 6x decode throughput at 1M context**. Vendor-measured, but the paper predates K3 by 9 months and the model + kernels (in the flash-linear-attention library) are public.
- **K3 scales this up:** 2.8T total, 16 of 896 experts active ("Stable LatentMoE"), KDA + "Gated MLA" hybrid, native text/image/video, 1M context, MXFP4 weights. Vendor claims 6.3x faster decode deep in the 1M window. Source: [Moonshot K3 blog](https://www.kimi.com/blog/kimi-k3), [Moonshot announcement thread on X](https://x.com/Kimi_Moonshot/status/2077830234955816983), [MarkTechPost writeup](https://www.marktechpost.com/2026/07/16/moonshot-ai-releases-kimi-k3-a-2-8-trillion-parameter-open-moe-model-with-kimi-delta-attention-and-1m-context/).

## 2. Relation to the Gated DeltaNet hybrid in Qwen3.5/3.6

Same family tree, one refinement apart:

- DeltaNet (delta rule) -> **Gated DeltaNet** (adds a scalar forget gate per head; the linear layer in Qwen3-Next, Qwen3.5, Qwen3.6) -> **KDA** (per-channel gate + faster chunkwise kernel).
- Both labs converged on the **same 3:1 hybrid ratio**. Qwen3.6-35B-A3B: 40 layers as 10 cycles of 3 Gated DeltaNet + 1 gated full-attention block, 8 of 256 experts + 1 shared ([Qwen3.6-35B-A3B on HF](https://huggingface.co/Qwen/Qwen3.6-35B-A3B), [architecture overview](https://huggingface.co/blog/EXDai/qwen36-35b-a3b-architecture-overview)). Kimi: 3 KDA + 1 MLA.
- The difference is the full-attention interleave (Qwen: gated attention with partial RoPE; Kimi: MLA with NoPE) and the gate granularity (scalar per head vs per channel).
- **Our lab already measured this class winning:** Qwen3.6-35B-A3B hybrid-GDN at +32% vs our previous flagship on this exact rig. KDA is a sibling of what we benchmarked, not a new bet.

## 3. "Attention residuals" (AttnRes): real paper, vendor numbers

- Real Moonshot paper: [Attention Residuals, arXiv 2603.15031](https://arxiv.org/abs/2603.15031) (Kimi Team, March 2026).
- Idea: replace the fixed identity residual stream with **learned softmax attention over depth**, so a layer selectively retrieves from earlier layers instead of accumulating everything uniformly. Block AttnRes is the scalable variant (memory/communication O(Nd) instead of O(Ld)).
- Claimed: **~25% training-compute advantage (1.25x) at under 2% extra cost**; validated on Kimi Linear 48B-A3B at 1.4T tokens (GPQA-Diamond +7.5, MATH +3.6, HumanEval +3.1). These are Moonshot's own numbers; no independent replication yet.
- For us this is a **training-efficiency story, not an inference story**: under 2% inference overhead, and it does not change the KV cache math. It matters mainly as evidence Moonshot ships its research papers into production models.

## 4. llama.cpp support status

Verified via GitHub API on 2026-07-19:

- **PR #17592** (the one from the tip stream, "Feature/kimi linear support" by cacaview): **closed, NOT merged** (Nov 2025 - Feb 2026). Community follow-up [PR #18381](https://github.com/ggml-org/llama.cpp/pull/18381) built on its branch, also closed unmerged.
- The support that actually landed: **[PR #18755](https://github.com/ggml-org/llama.cpp/pull/18755) "Kimi-Linear support (backend agnostic + MLA KV cache)" by ymcki, merged 2026-02-06.** Follow-up fixes all merged: #19531 (conv state fix, 02-13), #19668 (Kimi Linear folded into the unified delta-net path shared with Qwen3-Next, 02-19), #19827 (chunk size 16, 03-05). Original feature request [issue #16930](https://github.com/ggml-org/llama.cpp/issues/16930) closed.
- **The delta-net code path is shared between Qwen3-Next-lineage GDN and Kimi Linear KDA** (unified delta net). Our Qwen3.6 experience transfers directly.
- Backend note: Metal GDN kernel merged (#20361, March). [PR #18102](https://github.com/ggml-org/llama.cpp/pull/18102) (dedicated CUDA Delta-Net kernel for Qwen3-Next) was still **open** as of its last April update, and two CUDA GDN optimization PRs (#20448, #20449) closed unmerged. So CUDA-side delta-net speed still has headroom upstream. Our +32% Qwen3.6 result was measured under whatever the current state is, so this is upside, not risk.
- **Our b10064 build postdates all the merged support**, so Kimi-Linear should load today. GGUFs exist: [bartowski/moonshotai_Kimi-Linear-48B-A3B-Instruct-GGUF](https://huggingface.co/bartowski/moonshotai_Kimi-Linear-48B-A3B-Instruct-GGUF) (Q4_K_M roughly 29 GB, fits our 48 GB RAM with the --cpu-moe pattern).
- **K3 itself is a different story:** kimi_k3 as an arch does not exist in llama.cpp yet, and K3 adds AttnRes, Gated MLA, LatentMoE and MXFP4 on top of KDA. Moot for us at 2.8T regardless (MXFP4 weights alone are roughly 1.4 TB).

## 5. Video claim scorecard (channel calibration)

| Claim | Verdict |
|---|---|
| ~3T total params | Garble: actual 2.8T |
| 16 of 896 experts active | Correct |
| KDA hybrid linear attention | Correct |
| 6.3x faster decode deep in 1M window | Matches Moonshot's claim (vendor number, not independent) |
| Attention residuals, +25% training efficiency | Real paper (arXiv 2603.15031); +25% is Moonshot's own number |
| Native vision/video | Correct |
| Variants "K3 Max" and "K3 Swarm Max" | Half-garble: real names, but they are serving/agent modes of one model (single-agent vs multi-agent swarm), not model variants |
| $3 / $15 per M tokens | Correct (cache-miss input $3, output $15; omits the $0.30 cache-hit tier) |
| Weights on HF July 27 2026, modified MIT | Date confirmed ("by July 27" per Moonshot, plus technical report). Modified MIT is the expected K2-lineage license but NOT yet confirmed; weights not up as of today. Video stated it as fact, slight overreach |
| Artificial Analysis intelligence index 57 | Confirmed independent: [AA article](https://artificialanalysis.ai/articles/kimi-k3-achieves-3-in-the-artificial-analysis-intelligence-index-comparable-to-opus-4-8-and-gpt-5-5), [AA model page](https://artificialanalysis.ai/models/kimi-k3). Behind top closed (~59-60), ahead of Opus 4.8 (~56) |
| GDPval ELO 1668 | Confirmed: GDPval-AA v2 Elo 1668 per Artificial Analysis (vs K2.6's 1190) |
| Wins 2 of 6 coding benchmarks vs top closed | Directionally right (mixed coding results, beats Opus 4.8 and GLM-5.2 on most, trails Fable 5 / GPT 5.6 Sol); exact 2-of-6 count not verifiable in any source |

**Channel calibration: "garbler not fabricator" holds for a second video.** Every substantive claim traces to a real primary source; errors are rounding (3T vs 2.8T), naming ("Kimmy", variant framing), and stating one expected-but-unconfirmed detail (license) as fact.

## 6. Verdict for our lab

**Yes: K3 confirms hybrid linear attention as the direction consumer inference should bet on.** Three independent frontier labs (Alibaba Qwen3-Next/3.5/3.6, Moonshot Kimi Linear/K3, plus the MiniMax lightning-attention lineage before them) have now converged on delta-rule linear layers interleaved 3:1 with sparse full attention, and we have already measured the benefit class first-hand (+32% on Qwen3.6-35B-A3B, this rig). The 75% KV cache reduction attacks exactly our binding constraint (8 GB VRAM at long context; see E040's 64k work and the f16-KV governing law).

What K3 changes for us in practice: nothing at 2.8T, everything at the family level. Watch items below.

## Watch list

1. **July 27 2026:** K3 weights + technical report on HF. Check the actual license text, and specifically whether any smaller/distilled K3 family member appears. None is announced today; the video's hope for one is speculation.
2. **Candidate experiment:** Kimi-Linear-48B-A3B-Instruct Q4_K_M (~29 GB, bartowski GGUF) vs Qwen3.6-35B-A3B on this rig. Same A3B active class, same unified delta-net llama.cpp path, KDA vs GDN head-to-head, and its 75%-smaller KV cache is directly relevant to the long-context track. Queue as an R-number if we want it.
3. **Upstream:** merge status of CUDA delta-net kernels (#18102). Any merge is free speed for both Qwen3.6 and Kimi-Linear on our GPU.
