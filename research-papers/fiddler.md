# Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models

## Citation

Keisuke Kamahori, Tian Tang, Yile Gu, Kan Zhu, Baris Kasikci (University of Washington, SyFI Lab). "Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models." arXiv:2402.07033. First posted February 2024, revised versions through May 2025. Published at ICLR 2025. Code: Apache-2.0, github.com/efeslab/fiddler.

- arXiv: https://arxiv.org/abs/2402.07033

## Problem

MoE models have huge total parameter counts but small per-token active counts. Mixtral-8x7B is over 90 GB in fp16, yet each token only activates 2 of 8 experts per layer. On a single consumer or workstation GPU (the paper targets 24 GB cards), the weights simply do not fit, so you must offload.

The standard offloading approaches at the time were bad at latency-oriented, single-user inference:

- **Weight-streaming offloading** (DeepSpeed-MII ZeRO-Infinity style, Mixtral-Offloading): when a needed expert is not in GPU memory, copy its weights over PCIe, then compute on GPU. One Mixtral expert is more than 300 MB in fp16, and the paper measures around 50 ms to copy one expert to a Quadro RTX 6000 or L4. For decoding a single token, that transfer dwarfs the actual math.
- **Naive CPU fallback** ignores that CPUs and GPUs have very different latency profiles, so it leaves performance on the table for prefill and batched cases.

The core question: for each expert invocation, should you move weights to the compute, or move the compute to the weights?

## Key Idea

Keep expert weights resident in CPU memory and execute those experts **on the CPU**, shipping only the activations across PCIe. For single-token decode, an activation is a few KB while an expert is 300+ MB, so the data movement problem nearly vanishes; the paper reports activation copies cost less than 1 percent of CPU execution latency. The GPU holds all non-expert weights (attention, embeddings, router) plus as many of the most popular experts as fit. A simple latency model then decides, per expert and per batch, whether CPU execution or weight transfer plus GPU execution is faster.

This is exactly the "experts on CPU" pattern that llama.cpp later exposed via tensor override flags, which is why this paper is the direct blueprint for our E013 experiment.

## Mechanism

**Placement (initialization).** Fiddler profiles expert popularity offline on calibration data, then greedily fills spare GPU memory with the most frequently routed experts to maximize hit rate. In the 24 GB environments it fits roughly 52 to 56 of Mixtral's 256 experts (32 layers x 8) on the GPU; on a 48 GB card it fits about 125. Everything else stays in CPU RAM.

**Per-expert execution decision.** For an expert that is missing from GPU memory, there are two options:

1. Copy weights CPU to GPU, compute on GPU. Cost is dominated by the transfer: 300+ MB per expert, about 50 ms on their PCIe Gen3 x16 setups, which they note is 2 to 5 times the actual computation time.
2. Copy activations GPU to CPU, compute on CPU, copy the result back. Transfer cost is negligible; compute cost scales roughly linearly with the number of tokens routed to that expert, because CPU GEMM on a handful of rows is essentially memory-bandwidth-bound streaming of the expert weights.

The decision rule (Algorithm 1 in the paper) compares `cpu_lat(n_tokens)` against `gpu_lat + weight_transfer_lat`. GPU latency is nearly flat in input size thanks to parallelism, while CPU latency grows with tokens per expert. So decode (1 token per expert) goes to the CPU, and long prefill (hundreds of tokens hitting the same expert) flips to weight transfer plus GPU compute. Fiddler makes this choice dynamically per layer based on the router output.

**Prefill.** For 512 to 4096 token prompts, many tokens land on every expert, so CPU compute becomes prohibitive and Fiddler streams expert weights to the GPU instead. This is the same physics behind our measured 18x prefill speedup from llama.cpp CUDA builds streaming weights over PCIe even at -ngl 0.

**Parallelism.** CPU expert execution for one layer overlaps with GPU work where possible, and beam search batches all beams that hit the same expert into one CPU GEMM, which is where its largest wins come from.

## Reported Results

Numbers below are from the paper and repo; the baselines changed between versions, so I list both eras.

**v1 (Feb 2024), Mixtral-8x7B fp16 (90+ GB) on single 24 GB GPUs:**

- Over **3 tokens/s** end-to-end decode on a single 24 GB GPU, an order of magnitude over the then-standard offloaders.
- **8.2x** (Quadro RTX 6000 env) and **10.1x** (L4 env) faster than Mixtral-Offloading.
- **19.4x** and **22.5x** faster than DeepSpeed-MII.

**ICLR 2025 version, with stronger baselines including llama.cpp b2956:**

- Single-batch decode: **1.26x** over the best baseline.
- Long prefill (512 to 4096 tokens): **1.30x** over the best baseline, 1.65x over Mixtral-Offloading.
- Beam search (width 4 to 16): **11.57x** over llama.cpp.

The shrink from 8-19x to 1.26x is informative: llama.cpp's partial-offload approach (CPU-resident layers computed on CPU) already captures most of the benefit of not streaming weights during decode. Fiddler's remaining edge comes from expert-granularity placement by popularity and the dynamic CPU-vs-GPU decision, and it stays large only where baselines handle batching badly (beam search).

## Limitations and Caveats

- **fp16 only in the evaluation.** Quantized variants were listed as future work in the repo. Quantization changes the arithmetic: 4-bit experts are 4x cheaper to copy and to stream from RAM, compressing Fiddler's advantage.
- **Server-class CPUs.** Evaluation used 32 to 112 core Xeons with AVX512; the repo warns performance degrades without AVX512. A consumer 16-core laptop CPU has far less compute and memory bandwidth.
- **Assumes non-expert weights fit on GPU** (under 2B parameters for Mixtral). Fine for us at Q4 on 8 GB, but tight for bigger MoEs with large attention stacks.
- **Expert popularity profiling needs calibration data** and assumes routing skew is stable across workloads. Mixtral's routing is known to be fairly balanced, which caps the value of popularity-based placement; the paper does not report strong skew statistics.
- **All experts must fit in CPU RAM.** Fiddler has no SSD tier; Mixtral fp16 needs 90+ GB of system memory.
- The headline "3 tokens/s" is for uncompressed fp16; it is not directly comparable to quantized llama.cpp numbers.

## What This Means for Our Project

**E013 (Qwen3-30B-A3B, experts on CPU): this paper is the blueprint.** llama.cpp's `--n-cpu-moe` / `-ot "exps=CPU"` flags implement Fiddler's placement: attention, embeddings, router, and KV cache on the RTX 5060, expert FFNs resident in our 48 GB DDR5 and computed by the i7-14650HX. Fiddler validates the core bet: for decode, moving KB-scale activations beats moving expert weights.

**Our decode ceiling is set by the 50.8 GB/s RAM figure.** CPU expert execution is a memory-bandwidth-bound stream of the active expert weights. Back-of-envelope (ours, not the paper's): Qwen3-30B-A3B activates about 3.3B params per token, of which roughly 2.4B are expert weights; at Q4_K_M that is about 1.3 to 1.5 GB touched per token, so 50.8 GB/s gives a hard ceiling around 30 to 38 t/s for the CPU portion, with real-world results likely well under half of that. Either way it should beat our 10.1 t/s all-CPU dense 8B baseline per active parameter, and the GPU handles attention.

**The copy-vs-compute crossover looks different on our hardware.** A Mixtral expert is 300+ MB, so copying was hopeless. A Qwen3-30B-A3B expert at Q4 is only a few MB, and our PCIe Gen5 x8 (about 32 GB/s, coincidentally matching the paper's Gen3 x16 environment) is the same order as our 50.8 GB/s RAM bandwidth. So on our machine, streaming expert weights to GPU during decode is no longer absurd, just slightly worse than CPU compute. This is worth measuring directly in E013.

**Prefill: route to GPU.** Fiddler's finding that long prefill should flip to weight streaming plus GPU compute matches our measured 18x prefill speedup from CUDA weight streaming at -ngl 0. E013 configs should keep CUDA prefill streaming enabled rather than forcing experts to compute on CPU during prompt processing.

**E010 (offload placement):** Fiddler's cost model (CPU latency linear in tokens per expert, GPU latency flat plus transfer) is the exact calculus for deciding what to pin where. Its popularity-based expert placement suggests measuring routing skew on our workloads before assuming uniform placement is optimal.

**E020/E021/E022 (mmap cliff, SSD prefetch, expert caching):** Fiddler stops at the RAM boundary. For MoEs bigger than 48 GB of RAM, we need the SSD tier it lacks; its popularity profiling is the natural admission policy for an SSD-backed expert cache (E022), with our 5 GB/s SN5000S as the third rung under 50.8 GB/s RAM and 312.7 GB/s VRAM.

## Links

- Paper: https://arxiv.org/abs/2402.07033
- HTML (v2): https://arxiv.org/html/2402.07033v2
- OpenReview (ICLR 2025): https://openreview.net/forum?id=WX7lxohjFe
- Code: https://github.com/efeslab/fiddler
- Lab page: https://syfi.cs.washington.edu/publications/fiddler/
