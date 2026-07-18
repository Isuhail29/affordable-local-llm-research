# PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU

## Citation

Yixin Song, Zeyu Mi, Haotong Xie, Haibo Chen (IPADS, Shanghai Jiao Tong University). "PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU." arXiv:2312.12456, December 2023. Later published at SOSP 2024. Code: SJTU-IPADS/PowerInfer on GitHub, built directly on the llama.cpp/ggml codebase.

## Problem

Serving a large LLM on a single consumer GPU forces most weights out of VRAM. The two standard answers both hit the same wall. Layer-level offloading (llama.cpp style) runs the CPU-resident layers at host memory bandwidth, which is an order of magnitude slower than GDDR. Weight-streaming (FlexGen style) pulls weights over PCIe every token, and at batch size 1 the paper measures that over 99.5% of FlexGen's processing time is spent transferring weights. DejaVu exploits activation sparsity but assumes the whole model fits in GPU memory, so it does not help the offload case. The question PowerInfer asks: which bytes actually deserve the fast memory?

## Key Idea

Neuron activations in ReLU-family LLMs follow a power law. A small set of "hot" neurons fires on almost every input, while the majority of "cold" neurons fire rarely and input-dependently. PowerInfer profiles this offline, pins the hot neurons in GPU memory permanently, leaves cold neurons in host memory for the CPU to compute, and uses small online predictors to skip neurons that will not activate at all. GPU and CPU each compute their own resident neurons and merge partial results, so weights almost never cross PCIe during decode. This is offloading at neuron granularity, placed by activation frequency instead of by layer position.

## Mechanism

**Measured sparsity.** In ReLU models roughly 80% of neurons stay inactive for any given token, and fewer than 10% of activation map elements are non-zero. The skew is strong: in the MLP blocks, 26% of neurons account for 80% of all activations in OPT-30B and 43% in LLaMA2-70B with ReGLU, but 69% in LLaMA2-70B with SwiGLU, which is why PowerInfer only supports ReLU/ReGLU models. Model-wide, about 17% of OPT-30B neurons and 26% of ReGLU LLaMA-70B neurons cover 80% of activations.

**Offline placement as an ILP.** PowerInfer runs the model on generic corpora (C4, Wikipedia) and counts per-neuron activation frequency. Placement is then an integer linear program: maximize total activation frequency of GPU-resident neurons, subject to GPU memory capacity, one placement per neuron, and a communication constraint that a layer only gets GPU neurons if GPU compute plus synchronization beats pure CPU compute for that layer. Neurons are grouped into batches of 64 with similar impact to keep the NP-complete ILP tractable; solving takes about 10 seconds.

**Online predictors.** Per Transformer layer, two small MLP predictors (one for attention, one for FFN) forecast which neurons will activate for the current token. Naive fixed-size predictors for OPT-175B would need about 27 GB, so PowerInfer sizes each predictor adaptively to the layer's sparsity and skew, landing at roughly 10% of total model parameters, with above 95% activation prediction accuracy and under 10% of total inference time.

**Neuron-aware sparse operators.** Instead of cuSPARSE-style formats that need runtime conversion, PowerInfer's operators index directly into rows/columns for predicted-active neurons. On CPU (AVX2), this beats dense matmul even below 10% sparsity, where conventional sparse libraries need roughly 87% sparsity to break even. On GPU it matches PIT while also covering the CPU side.

**Hybrid execution.** The model is compiled to an operator DAG with a CPU executor and a GPU executor working concurrently. A key measurement (their Insight 2) motivates the whole design: at batch sizes under 32 on an RTX 4090, transferring a cold neuron's weights over PCIe and computing on GPU is slower than just computing it on the CPU. Merging of partial results happens on the GPU, and synchronization is skipped when the CPU had no activated neurons.

## Reported Results

Hardware: PC-High (i9-13900K, 192 GB DDR at 67.2 GB/s, RTX 4090 24 GB, PCIe 4.0) and PC-Low (i7-12700K, 64 GB at 38.4 GB/s, RTX 2080Ti 11 GB, PCIe 3.0). Models: OPT 6.7B to 175B, LLaMA (ReGLU) 70B, Falcon (ReLU) 40B, in FP16 and INT4.

- **FP16, PC-High:** average 8.32 t/s, peak 16.06 t/s; average 7.23x over llama.cpp, up to 11.69x (Falcon-40B).
- **INT4, PC-High:** average 13.20 t/s, peak 29.08 t/s; average 2.89x over llama.cpp, up to 4.28x.
- **PC-Low:** average 5.01x, peak 7.06x. Gains shrink because 11 GB of VRAM holds fewer hot neurons, pushing more work to the CPU.
- **vs. a server A100:** on OPT-30B generation, the 4090 running PowerInfer lands within 18% of an A100 running vLLM-class serving, versus llama.cpp being 93% slower.
- **Batching:** advantage holds below batch 32 (average 6.08x), decaying to 4.38x at batch 32.
- **Accuracy:** negligible task deltas on PIQA, Winogrande, RTE, COPA (e.g. OPT-175B PIQA 79.65% to 79.26%), because predictor misses only skip marginal neurons.

## Limitations and Caveats

- **ReLU/ReGLU only.** The power law is far weaker under SwiGLU (69% of neurons for 80% of activations), and modern models (Llama 3, Qwen3, Mistral) use SwiGLU/SiLU. Using PowerInfer today means using specially retrained models (ReluLLaMA, ProSparse, Bamboo, TurboSparse), some of which show quality regressions from insufficient retraining.
- **Predictor tax.** Roughly 10% extra parameters and a per-model training step that takes hours.
- **Decode-phase tool.** Long prompts collapse effective sparsity (many tokens union their active sets), so prefill gains little and the CPU becomes the bottleneck.
- **Custom format.** Requires PowerInfer's own GGUF variant plus predictor and activation-statistics files; it is a fork, not a llama.cpp flag.
- **Small-VRAM scaling is the open question for us.** Their own PC-Low result shows the technique degrades as VRAM shrinks; 8 GB is below anything they evaluated.

## What This Means for Our Project

**E010 (offload placement): placement should follow expected reads per byte, not layer order.** PowerInfer is essentially a proof that frequency-weighted placement dominates contiguous layer splits. Our measured hierarchy is GPU 312.7 GB/s vs CPU 50.8 GB/s, a 6.2x ratio, and our Qwen3-8B decode numbers (62.2 t/s GPU vs 10.1 t/s CPU) track that ratio almost exactly, confirming the bandwidth-centric cost model PowerInfer's ILP assumes. For E010 with llama.cpp's `--override-tensor`, the transferable move is to rank tensors by how often their bytes are actually read per token and pack VRAM greedily by that metric. Their Insight 2 also matches our PCIe finding from the opposite direction: streaming weights over PCIe wins for prefill (our measured 18x prefill speedup at -ngl 0) but loses to local CPU compute for batch-1 decode, which is exactly why llama.cpp only streams during prefill.

**E013 (Qwen3-30B-A3B experts on CPU): MoE is PowerInfer with free predictors.** Qwen3's SwiGLU FFNs lack ReLU sparsity, but the MoE router already gives us structured, exact activation sparsity at expert granularity, with no predictor overhead and no accuracy risk. Experts-on-CPU is the PowerInfer hot/cold split where "hot" is the dense attention plus shared tensors (pinned in 8 GB VRAM) and "cold" is the expert pool in DDR5. PowerInfer's concurrent CPU+GPU execution with GPU-side merging is the execution pattern to compare against llama.cpp's current sequential handoff.

**E022 (expert caching on SSD): profile expert frequency before assuming a power law.** PowerInfer's offline profiling recipe (run generic corpora, count activations, solve placement under a capacity constraint) ports directly to experts: count per-expert routing frequency, pin hot experts in VRAM, keep warm ones in the 48 GB of DDR5, and leave cold ones on the SN5000S at ~5 GB/s. One honest caveat: MoE training uses load-balancing losses that deliberately flatten expert usage, so the skew may be much milder than PowerInfer's 26%-covers-80% neuron curve. Measuring that histogram for Qwen3-30B-A3B should be step one of E022, because if usage is near-uniform, caching wins come from temporal locality across consecutive tokens rather than from a static hot set. The follow-up paper PowerInfer-2 (arXiv 2406.06282) applies exactly this hot/cold plus storage-tier design on smartphones and reports 11.68 t/s on TurboSparse-Mixtral-47B, evidence the idea survives a flash tier.

## Links

- arXiv abstract: https://arxiv.org/abs/2312.12456
- arXiv HTML full text: https://arxiv.org/html/2312.12456v1
- SOSP 2024 paper PDF: https://ipads.se.sjtu.edu.cn/_media/publications/song-sosp24.pdf
- Code: https://github.com/SJTU-IPADS/PowerInfer
- Follow-up (PowerInfer-2, smartphone + flash tier): https://arxiv.org/abs/2406.06282
