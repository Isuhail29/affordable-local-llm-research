# FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU

## Citation

Ying Sheng, Lianmin Zheng, Binhang Yuan, Zhuohan Li, Max Ryabinin, Daniel Y. Fu, Zhiqiang Xie, Beidi Chen, Clark Barrett, Joseph E. Gonzalez, Percy Liang, Christopher Re, Ion Stoica, Ce Zhang. "FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU." ICML 2023 (oral). arXiv:2303.06865, submitted March 2023, revised June 2023. [https://arxiv.org/abs/2303.06865](https://arxiv.org/abs/2303.06865)

## Problem

Running a model far larger than GPU VRAM requires offloading weights, activations, and KV cache to CPU DRAM and disk. Existing offloading systems in 2023 (DeepSpeed ZeRO-Inference, Hugging Face Accelerate) inherited schedules designed for training or for low-latency single-request inference. They traverse the computation "row by row": finish one batch end to end, then start the next. That order reloads every layer's weights from slow storage once per batch, so on a 16 GB GPU they managed only about 0.01 token/s on OPT-175B. The paper asks: if you give up on latency entirely and only care about total tokens per second across a big batch (benchmarking, data cleaning, extraction, synthetic data generation), how fast can one commodity GPU go?

## Key Idea

Treat offloading as a formal search problem over a three-tier memory hierarchy (GPU, CPU, disk). Define a policy space covering both the computation schedule (what order to walk the batch x layer grid) and the placement of every tensor type (what fraction of weights, activations, and KV cache lives on each tier), then use a cost model plus a linear programming solver to pick the policy that maximizes throughput under the hardware's memory constraints. The core trick is amortization: load each layer's weights once, run them against a huge effective batch, and only then move on, so weight I/O per generated token shrinks by the batch size.

## Mechanism

**Schedule: zig-zag block computation.** Picture the work as a grid with layers on one axis and batches on the other. Row-by-row (all layers for batch 1, then batch 2) is latency-friendly but reloads all weights for every batch. Column-by-column (layer 1 for all batches, then layer 2) reuses weights perfectly but activations and KV cache for every in-flight batch pile up until memory overflows. FlexGen's compromise walks a column downward for a block of batches until memory fills, then zig-zags to the next block. The authors prove this block schedule is within 2x of the I/O-optimal schedule. A theoretically better diagonal schedule is described but not implemented due to complexity.

**Policy: 11 variables.** Block size (number of batches sharing one weight load, typically under 20), GPU batch size (multiples of 4), and nine placement percentages: weights split across GPU/CPU/disk (wg, wc, wd), KV cache split (cg, cc, cd), activations split (hg, hc, hd), with each group summing to 100 percent.

**Cost model and LP search.** FlexGen profiles the machine's bandwidths, then estimates per-block latency assuming perfect overlap of the pipeline: T = max(disk-to-CPU, CPU-to-GPU, GPU-to-CPU, CPU-to-disk, compute). It minimizes T per generated token subject to peak memory limits on all three tiers. The outer loop enumerates (block size, GPU batch size) pairs; the inner LP solves the placement percentages. This makes the system portable: change the hardware profile and the solver re-derives the policy.

**CPU delegation of decode attention.** When the KV cache lives in DRAM, moving the cache to the GPU for attention costs b x s x h1 x 4 bytes per layer, while moving the activations to the CPU costs only b x h1 x 4 bytes, a factor of s (sequence length) less traffic. Decode attention is memory-bandwidth-bound with almost no FLOPs, so the slow CPU is fast enough, and I/O drops massively. Prefill still runs on the GPU because it is compute-bound.

**4-bit compression.** Weights and KV cache are quantized with fine-grained group-wise asymmetric quantization, group size 64 (weights grouped along the output channel, cache along the hidden dimension), no calibration needed. Accuracy loss is negligible: OPT-175B Lambada accuracy 0.758 to 0.756, WikiText perplexity 10.82 to 10.94. Compression shrinks every tier's footprint and every transfer.

**Hardware.** Experiments ran on a Google Cloud instance with one NVIDIA T4 (16 GB), an Intel Xeon at 2.0 GHz with 208 GB DRAM, and a 1.5 TB NVMe SSD (roughly 2 GB/s read, 1 GB/s write).

## Reported Results

- **OPT-175B on one 16 GB T4:** 0.69 token/s generation throughput without compression, 1.12 token/s with 4-bit compression at an effective batch size of 144. Baselines (DeepSpeed ZeRO-Inference, HF Accelerate) manage about 0.01 token/s, so FlexGen is up to 100x faster.
- **OPT-30B, prompt 512, generate 32:** 7.32 token/s vs about 0.60 to 0.62 token/s for the baselines, roughly 12x.
- **Latency-throughput frontier:** at an equal latency budget of 5,000 seconds for 32 output tokens, FlexGen runs an effective batch of 64 and beats DeepSpeed by more than 40x. Relaxing latency to 12,000 seconds allows batch 256 and 69x; compression pushes it to 100x. Note what "throughput" means here: 1.12 token/s is the aggregate over 144 concurrent sequences, so each individual request waits over an hour for 32 tokens.
- **Practical demo:** benchmarking a 30B model on 7 HELM scenarios completed in 21 hours on the single T4.
- Reported GPU utilization: about 82 percent during prefill, about 13 percent during decoding, confirming decode is I/O-bound even with an optimal policy.

## Limitations and Caveats

- **Throughput-only by design.** Every gain comes from amortizing weight I/O over a giant effective batch. At batch size 1 the block schedule degenerates to row-by-row and the advantage disappears entirely.
- **Cost model idealism.** The LP assumes perfect compute/transfer overlap and linear costs; the paper notes the solver sometimes produces policies that run out of memory and need manual adjustment, and hand-tuned policies occasionally beat the search.
- **Dense decoder-only models, padded batches.** Everything is OPT with uniform sequence lengths; MoE models, variable-length production traffic, and modern attention variants are out of scope.
- **CPU attention and quantization interact.** CPU delegation works on the uncompressed cache; combining it with 4-bit cache compression adds conversion overhead.
- **2023 baselines.** The 100x headline is against offloading systems of early 2023. Modern llama.cpp with a quantized model and partial GPU offload is a much stronger baseline for the local use case.

## What This Means for Our Project

**FlexGen formalizes what E010 does by hand.** Our offload placement experiment is a manual search over the same policy space: which tensors sit in 8 GB of VRAM (312.7 GB/s), which in 48 GB DDR5 (50.8 GB/s), which page from the SN5000S (about 5 GB/s). FlexGen says: measure those three bandwidths, write the cost model, let a solver pick placements instead of grid-searching llama.cpp flags. Even a spreadsheet version of their LP could predict our E010 results before running them.

**Our own numbers confirm the bandwidth-bound decode model.** Our GPU/CPU decode ratio for Qwen3-8B Q4_K_M is 62.2 / 10.1 = 6.2x, almost exactly our bandwidth ratio 312.7 / 50.8 = 6.2x. That is FlexGen's cost model working in miniature: batch-1 decode speed is bytes-touched-per-token divided by the bandwidth of wherever those bytes live. It also sets a hard ceiling for any SSD tier: weights streamed from the SN5000S at 5 GB/s cap a 5 GB dense model near 1 token/s, no scheduling cleverness can fix that at batch 1.

**The batch tricks do not transfer, but three components do.** The zig-zag schedule and effective batch 144 are useless for interactive chat, where per-request latency of an hour is absurd. What transfers: (1) the cost-model-driven placement search (E010); (2) overlapped next-layer weight prefetch from Algorithm 1, which is exactly the layer-ahead SSD prefetching hypothesis of E021 and the managed alternative to the mmap paging cliff we are probing in E020, since scheduled sequential reads beat random page faults; (3) 4-bit compression of weights and KV cache, which is why quantization multiplies the value of every tier.

**CPU delegation is the ancestor of our MoE experiments.** "Move the small tensor to the big tensor" is FlexGen's justification for CPU attention (activations are s times smaller than the KV cache) and it is precisely why E013 experts-on-CPU works for Qwen3-30B-A3B: shipping activations to DRAM-resident experts costs far less than shipping expert weights over PCIe. E022's expert caching on SSD is FlexGen's three-tier placement applied per-expert, and its cost model approach suggests weighting placement by measured expert reuse frequency rather than treating all experts equally.

**One caution for our prefill findings.** FlexGen keeps prefill on the GPU because it is compute-bound; llama.cpp's CUDA build already streams weights over PCIe for prefill even at -ngl 0 (our measured 18x prefill speedup). So the prefill side of FlexGen's insight is already built into our stack; the open territory for us is decode-side placement and prefetch.

## Links

- arXiv abstract: [https://arxiv.org/abs/2303.06865](https://arxiv.org/abs/2303.06865)
- HTML full text: [https://arxiv.org/html/2303.06865](https://arxiv.org/html/2303.06865)
- ICML 2023 proceedings page: [https://proceedings.mlr.press/v202/sheng23a.html](https://proceedings.mlr.press/v202/sheng23a.html)
- Code (repo renamed from FlexGen): [https://github.com/FMInference/FlexLLMGen](https://github.com/FMInference/FlexLLMGen)
