# Prior art: MoE expert memory layout and access optimization

Scope: work beyond what we already summarized (LLM-in-a-flash, PowerInfer, Fiddler, FlexGen). Focus: MoE-Lightning, 2024-2026 MoE offloading systems, expert popularity/skew evidence for Qwen-family MoE, routing locality measurements, and layout/pinning/prefetch techniques. Ends with what we can test on our box (llama.cpp b10064, Qwen3-30B-A3B Q4_K_M, `-ngl 99 --n-cpu-moe 40`) without touching llama.cpp source.

## 1. MoE-Lightning (ASPLOS 2025)

[MoE-Lightning, arXiv:2411.11217](https://arxiv.org/abs/2411.11217) targets throughput-oriented batch inference on memory-constrained GPUs. Two ideas matter for us:

- **CGOPipe**: a CPU-GPU-I/O pipeline with paged expert weights, so weight transfer, CPU-side work, and GPU compute overlap instead of serializing. Up to 10.3x over prior offloading systems for Mixtral 8x7B on a T4.
- **HRM (Hierarchical Roofline Model)**: they explicitly model CPU memory bandwidth, PCIe, and GPU as separate roofs and search for the policy that saturates the binding one. Their core diagnosis matches ours: in CPU-offloaded MoE decode the binding resource is host DRAM bandwidth, and most systems leave it underutilized because access is demand-driven and unoverlapped.

Caveat: it is a batch-throughput system (many concurrent requests raise arithmetic intensity per weight byte). For single-stream decode its main transferable lesson is pipelining plus "treat host bandwidth as the roof you must saturate."

## 2. Offloading, caching, and prefetch systems 2024-2026

- [MoE-Infinity](https://arxiv.org/abs/2401.14361): traces sequence-level Expert Activation Matrices; finds expert activation is sparse and has strong per-sequence temporal locality, and uses that for activation-aware prefetch and cache retention.
- [ProMoE, arXiv:2410.22134](https://arxiv.org/pdf/2410.22134): learned predictor for upcoming experts, proactive prefetch coordinated with inference. Notes that naive reactive caching drops below 50 percent hit rate under tight VRAM.
- [HOBBIT](https://arxiv.org/abs/2411.01433): mixed-precision expert caching (low-bit copy of cold experts, high precision for hot ones) and multi-layer-ahead prefetch driven by gate-input similarity across adjacent layers.
- [AdapMoE](https://arxiv.org/abs/2408.10284): adaptive per-layer cache budget allocation plus sensitivity-aware expert skipping.
- [ExpertFlow, arXiv:2410.17954](https://arxiv.org/html/2410.17954v1): transformer-based routing-path predictor plus predictive expert caching and token scheduling.
- [MoE-Gen, arXiv:2503.09716](https://arxiv.org/html/2503.09716v1): module-based batching so expert weight reads are amortized over larger token batches.
- [DALI, arXiv:2602.03495](https://arxiv.org/html/2602.03495v1): workload-aware offloading specifically for local PCs, closest deployment profile to ours.
- [ST-MoE-style spatio-temporal prefetch, arXiv:2606.15453](https://arxiv.org/abs/2606.15453): measures that expert requests "exhibit strong correlation across both adjacent MoE layers and consecutive decoding tokens" and prefetches on both axes.
- [SpecMD, arXiv:2602.03921](https://arxiv.org/pdf/2602.03921) and [MoE-SpeQ](https://www.researchgate.net/publication/397739048_MoE-SpeQ_Speculative_Quantized_Decoding_with_Proactive_Expert_Prefetching_and_Offloading_for_Mixture-of-Experts): use speculative decoding drafts as an oracle for which experts to prefetch.
- Router-side fixes: [ReMoE, arXiv:2605.27081](https://arxiv.org/html/2605.27081) fine-tunes the router with a temporal-locality loss; [Oracle-MoE](https://proceedings.mlr.press/v267/zhou25b.html) (ICML 2025) restructures routing to preserve locality across consecutive tokens; [Sticky Routing, arXiv:2607.08780](https://arxiv.org/abs/2607.08780) trains with a routing consistency loss. These need training, so they are out of scope for us, but their measurement sections are the best published routing-locality data.

## 3. Do Qwen3 MoE models have hot experts?

Best available answer: **near-uniform in aggregate, skewed per domain and per sequence.** Qwen3 MoE was trained with the global-batch load-balancing loss from Qwen's own paper [Demons in the Detail, arXiv:2501.11873](https://arxiv.org/abs/2501.11873) (see also the [Qwen blog post](https://qwenlm.github.io/blog/global-load-balance/) and the [Qwen3 Technical Report, arXiv:2505.09388](https://arxiv.org/pdf/2505.09388)). Global-batch LBL balances expert load at the corpus level, deliberately allowing per-domain specialization. The paper shows clearly different expert selection frequencies on different domain data. Consequence: over one coding session or one conversation, a subset of experts runs hot; averaged over everything, the histogram flattens.

Supporting observations:

- [Alloc-MoE, arXiv:2604.08133](https://arxiv.org/pdf/2604.08133) profiled Qwen3-30B-A3B over 200 inference requests and reports several layers with noticeable expert imbalance (a small subset activated disproportionately often). Layer-dependent, not global.
- The [LMSYS AMD MI300X blog on Qwen3](https://www.lmsys.org/blog/2026-02-11-Qwen-latency/) reports expert hotspots on certain datasets for Qwen3-235B (specific layers showing hot experts), motivating EPLB-style expert replication in production serving.
- [REAP (Cerebras), arXiv:2510.13999](https://arxiv.org/pdf/2510.13999) prunes 25 percent of Qwen3-30B-A3B experts by router-weighted saliency with under 1 point quality loss, direct evidence that expert importance is far from uniform even if activation counts look balanced.
- Anecdotal but on-topic: [llama.cpp issue #20757](https://github.com/ggml-org/llama.cpp/issues/20757) (two-tier GPU+RAM expert cache RFC) claims roughly 15-20 percent of experts handle about 80 percent of tokens in practice (GPT-OSS-120B PoC, 98-100 percent steady-state hit rate). Unverified for Qwen3, but consistent with the per-domain skew picture.

## 4. Routing locality across consecutive tokens: published numbers

- ReMoE measures a baseline Expert Overlap Ratio (step-to-step expert overlap) of **27.3 percent** on DeepSeek-V2-Lite, and an LRU unique-hit-rate of about **32 percent** with cache size equal to top-k. Their fine-tuning lifts EOR to 34.5 percent. They also validate on Qwen1.5-MoE-A2.7B.
- Surveys of consecutive-token patterns report that at least one expert repeats from the previous token with **40-60 percent probability per layer**, and of those, roughly 23 percent share an expert with the previous two tokens (see the measurement sections of [arXiv:2512.16473](https://arxiv.org/abs/2512.16473) and related caching papers).
- HOBBIT and ST-MoE both find adjacent-layer gate similarity strong enough to prefetch one or more layers ahead.

Takeaway: locality exists but is moderate. A pure LRU expert cache sized near top-k gets roughly a third of reads for free; the rest needs prediction or skew exploitation. No published measurement of consecutive-token overlap specifically for Qwen3-30B-A3B was found; this is a gap we could fill ourselves.

## 5. Relevance to our measured gap, and what we can test without source changes

Important framing: most of this literature optimizes CPU-to-GPU expert transfer. Our config computes experts **on the CPU in RAM** (`--n-cpu-moe`), so our 34 vs 55.6 GB/s gap is about CPU read efficiency of `ggml_mul_mat_id` gathering 8 expert rows per layer from the fused `blk.N.ffn_{up,gate,down}_exps.weight` 3D tensors, not PCIe. Skew and locality data still matter: they bound what any caching or pinning scheme could recover.

**Testable now, no source modification:**

1. **Reduce active experts via KV override.** `--override-kv qwen3moe.expert_used_count=int:6` (or 4). Expert bytes per token scale linearly with top-k, so this is a direct bandwidth cut. AdapMoE and REAP both suggest quality degrades slowly. Measure perplexity or task quality vs t/s. Community documentation: [DavidAU's MoE guide](https://huggingface.co/DavidAU/How-To-Set-and-Manage-MOE-Mix-of-Experts-Model-Activation-of-Experts); open feature request for a first-class flag: [issue #19528](https://github.com/ggml-org/llama.cpp/issues/19528).
2. **Speculative decoding as batch amortization.** `llama-server -md Qwen3-0.6B` style drafting verifies several tokens per forward pass, so each expert row read serves multiple tokens (the MoE-Gen and MoE-Lightning insight, plus the SpecMD prefetch-oracle angle). With 27-40 percent consecutive-token expert overlap, the union of experts for K drafted tokens is well under 8K, so effective GB/s per token drops. Win depends on acceptance rate; test on our workload.
3. **Thread and affinity sweep.** Achievable DRAM bandwidth on Raptor Lake depends heavily on which cores issue the reads. Sweep `--threads` 6 to 16 and use `--cpu-mask`/`--cpu-strict` to compare P-core-only vs mixed; E-core participation can reduce effective bandwidth for latency-bound gathers.
4. **Rule out paging overhead.** Compare mmap default vs `--mlock` vs `--no-mmap`. Soft page faults and TLB pressure on first-touch of 18.6 GB of experts can masquerade as low bandwidth.
5. **Quant layout comparison.** Requantize to Q4_0 and IQ4_XS and compare effective expert-read bandwidth vs Q4_K_M. K-quant super-block scale layout produces less sequential access per row than Q4_0; measuring this isolates how much of the gap is format, not scheduling.
6. **Layer placement experiments with `-ot`.** `--override-tensor` works at whole-tensor granularity, so we can move specific layers' expert tensors between CPU and GPU and find the true optimum split vs the blunt `--n-cpu-moe 40`.
7. **ik_llama.cpp fork (zero code written by us, but a different binary):** `-fmoe` fused MoE ops, `-rtr` runtime repacking to interleaved layouts, `-ser` smart expert reduction (drop low-probability experts dynamically). See the [DocShotgun CPU+GPU MoE optimization guide](https://gist.github.com/DocShotgun/a02a4c0c0a57e43ff4f038b46ca66ae0).

**Not testable without source changes:** per-expert hot pinning or expert reordering. All 128 experts of a layer live in one fused GGUF tensor, and `-ot` cannot split it, so PowerInfer/HOBBIT-style hot-expert placement needs code (this is exactly what [issue #20757](https://github.com/ggml-org/llama.cpp/issues/20757) requests and what [discussion #12071](https://github.com/ggml-org/llama.cpp/discussions/12071) and [issue #11532](https://github.com/ggml-org/llama.cpp/issues/11532) circle around). Similarly, an expert-granularity LRU cache and routing-aware prefetch inside `ggml_mul_mat_id` are source-level projects.

Uncertainty notes: the 15-20/80 skew figure is a community PoC on GPT-OSS, not Qwen3; no published consecutive-token overlap number exists for Qwen3-30B-A3B specifically; ik_llama.cpp flag behavior changes frequently between releases.
