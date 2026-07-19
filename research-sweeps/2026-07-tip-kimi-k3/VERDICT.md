# VERDICT: Kimi K3 tip (2026-07 sweep)

Sweep: 2026-07-tip-kimi-k3 | Written: 2026-07-19 | Inputs: existence-and-claims.md, architecture-for-us.md, trickle-down-candidates.md

**One-line verdict: The video is real and almost entirely accurate. Kimi K3 (2.8T, announced July 16 2026) checks out against primary sources; K3 itself can never run here, but it confirms the hybrid linear attention bet we already measured winning, and its runnable sibling Kimi-Linear-48B-A3B is a top queue candidate today.**

## 1. Video reliability scorecard

Tally: **10 accurate, 3 garbled or overreached, 0 invented.** Every substantive claim traces to a real primary source.

| Video claim | Rating | Notes |
|---|---|---|
| Moonshot announced Kimi K3 | Accurate | Launched July 16 2026. Product live, AA eval published July 17. |
| ~3T total params MoE | Garbled (rounding) | Actual 2.8T. Moonshot itself markets "3T-class", so the garble matches vendor framing. |
| 16 of 896 experts active | Accurate | Exact match to Moonshot blog (Stable LatentMoE, ~50B active). |
| Kimi Delta Attention, hybrid linear attention | Accurate | KDA is a refined Gated DeltaNet (arXiv 2510.26692), 3:1 hybrid with full attention. |
| 6.3x faster decode deep in a 1M window | Accurate (vendor number) | Matches the blog. Notably first measured on the 48B Kimi-Linear model, our size class, then carried into K3 marketing. Not independently replicated. |
| Attention residuals, +25% training efficiency | Accurate (vendor number) | Real paper (arXiv 2603.15031). ~25% is Moonshot's own figure, no independent replication. |
| Native vision and video | Mostly accurate | Vision confirmed everywhere. Video understanding is vendor-claimed only; AA lists text+image in, text out. |
| Two variants: K3 Max, K3 Swarm Max | Half-garbled | Real names, but they are serving/agent modes of one 2.8T model, not different sizes. |
| $3 / $15 per M tokens | Accurate | Exact. Omits the $0.30/MTok cache-hit input tier. |
| Weights on HF July 27 2026 | Accurate (pending) | Blog says "by July 27" with a technical report. Not up as of 2026-07-19. |
| Modified MIT license | Overreach | Stated as fact; the blog names no license. Expected from K2-line precedent, verify with the files. |
| AA intelligence index 57 vs ~60 top closed | Accurate, independent | 57 (K3 Max 57.1), rank #4 of 187. Fable 5 leads at 60 under index v4.1. |
| GDPval ELO 1668 | Accurate, independent | GDPval-AA v2 (Artificial Analysis run): 1668 vs Fable 5's 1760, up from K2.6's 1190. |
| Wins 2 of 6 coding benchmarks vs top closed | Accurate (vendor table) | Verified against Moonshot's table: beats both Fable 5 and GPT-5.6 Sol on exactly Program Bench and SWE Marathon; splits or loses the rest. Vendor-reported numbers. One sweep doc could not reproduce the exact count from press coverage alone; the direct table check resolves it. |

**Channel calibration: upgrade.** Second video from this tip stream, second time everything substantive checks out. Defects are pronunciation ("Kimmy"), rounding (3T), variant framing, and stating one expected-but-unconfirmed detail (license) as fact. "Garbler not fabricator" holds and this one was barely garbled.

## 2. What K3 changes for our lab NOW

**K3 itself: nothing, ever.** 2.8T total, ~50B active, native MXFP4 weights around 1.4 TB, no llama.cpp arch, and no smaller K3 variant announced. Permanently out of reach on 48 GB RAM + 8 GB VRAM. Moonshot has never shipped a small version of any flagship, so do not wait for a K3-mini.

**Queue additions (concrete, runnable today):**

1. **Kimi-Linear-48B-A3B-Instruct head-to-head vs Qwen3.6-35B-A3B on this rig.** It is Moonshot's own KDA preview, MIT licensed, merged in mainline llama.cpp since Feb 2026 (PR #18755 plus follow-ups; our b10064 postdates all of it, same unified delta-net path as Qwen3.6). bartowski GGUFs fit: IQ4_XS 26.5 GB or Q4_K_S 29.0 GB with --cpu-moe. Same A3B active class as our +32% winner, so this is KDA vs GDN on identical hardware, plus its up-to-75% KV cache reduction is aimed straight at our 8 GB VRAM long-context ceiling. Treat it as an architecture benchmark first, daily driver second (2025-vintage research tune, thin independent quality data).
2. **July 27 check (dated task).** When K3 weights land on HF: read the actual license text (modified MIT is expectation, not fact), read the technical report for KDA/AttnRes details, and scan for any smaller or distilled family member. Community Qwen-base trace distills will likely follow within weeks; history says they are flavor transfers, low priority.

**Watch items (no action yet):**

- **llama.cpp CUDA delta-net kernels**: PR #18102 still open, two optimization PRs closed unmerged. Any merge is free speed for both Qwen3.6 and Kimi-Linear here. Upside, not risk; our +32% was measured under the current state.
- **AttnRes** is a training-efficiency story (under 2% inference overhead), not something that changes our inference math. File it as evidence Moonshot ships its papers into production.
- **Kimi-VL-A3B** (Q4 ~10.5 GB) only if we later want a local vision model. Everything else in the Kimi family is either superseded or impossible here.

## 3. The architecture trend, in plain terms

Every chatbot you have used is built on "attention", the mechanism that lets a model look back at everything you have said. The catch is that classic attention gets more expensive the longer the conversation gets, both in compute and in memory, because the model keeps a growing scratchpad (the KV cache) of everything it has seen. On a small machine like ours, that scratchpad is exactly what fills up first: it is why long documents are the thing an 8 GB graphics card struggles with, not short questions. A newer family of "linear attention" designs replaces most of that ever-growing scratchpad with a fixed-size running summary that gets updated as text streams past, which keeps memory flat and speed steady no matter how long the context gets. The trade-off used to be quality, so the winning recipe turned out to be a hybrid: mostly linear layers, with a full-attention layer kept every fourth layer as a safety net for the hard lookups.

What this sweep confirms is that the hybrid recipe is no longer an experiment, it is where the industry is converging. Three separate frontier labs (Alibaba with Qwen3-Next through 3.6, MiniMax before them, and now Moonshot with Kimi Linear and K3) independently landed on nearly the same design, right down to the same 3:1 ratio of linear to full attention. Moonshot just bet its 2.8-trillion-parameter flagship on it, and the flashiest number in the video (6.3x faster at million-token contexts) was actually measured on a 48B model in our size class before being scaled up. We are not guessing that this helps small machines: we already measured a +32% gain on this exact rig with Qwen3.6's version of the idea. The practical takeaway is that the models built for giant data centers and the models we can run at home are, for once, moving in the same direction, and each new flagship on this recipe makes more runnable-class siblings likely. Our job is simply to keep benchmarking each one as it lands, starting with Kimi-Linear-48B.
