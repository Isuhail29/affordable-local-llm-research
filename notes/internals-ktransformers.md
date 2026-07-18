# KTransformers vs llama.cpp for MoE on consumer hardware

Research note for the Qwen3-30B-A3B project. Question: what does KTransformers (kvcache-ai) actually do differently, what needs Intel AMX, and what transfers to our AVX2-class Raptor Lake i7-14650HX running llama.cpp on Windows.

Primary sources: the [KTransformers repo](https://github.com/kvcache-ai/ktransformers), their [AMX kernel doc](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/AMX.md), the [kt-kernel README](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/README.md), the [DeepSeek R1/V3 tutorial](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/DeepseekR1_V3_tutorial.md), the [SOSP '25 paper](https://dl.acm.org/doi/10.1145/3731569.3764843) ([PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)), and the [LMSYS/SGLang integration blog](https://www.lmsys.org/blog/2025-10-22-KTransformers/).

## 1. Expert placement strategy

KTransformers pioneered the split we already copied via `--n-cpu-moe`: attention, KV cache, dense layers, and shared experts on GPU; routed expert FFNs in CPU RAM. Two refinements go beyond llama.cpp:

- **Expert-granular, not layer-granular, GPU placement.** In the SGLang integration, `--kt-num-gpu-experts N` pins the N statistically hottest routed experts (from offline activation profiling) on the GPU across all layers, with the cold tail on CPU. llama.cpp can only move whole per-layer expert tensors (`--n-cpu-moe`, `-ot` regex), because all 128 experts of a layer live in one fused 3D GGUF tensor. KT's placement cuts CPU DRAM traffic proportionally to hot-expert hit rate.
- **Reduced expert count.** Their V0.3 DeepSeek numbers use top-6 instead of top-8 experts "based on offline profile results" with claimed negligible quality loss. That is a straight 25 percent cut in expert bytes read per token.

## 2. CPU kernels: what is AMX and what is not

- **v0.2 and the LLAMAFILE backend.** KT's original CPU expert kernels are literally llamafile (tinyBLAS/iqk lineage) GEMMs operating on GGUF quants, the same kernel class llama.cpp uses. kt-kernel still ships this as the "AVX2/AVX512-based MoE backend built on Llamafile for universal CPU deployment" consuming Q4_K/Q5_K GGUF directly. So at the per-core kernel level, KT on AVX2 hardware is not fundamentally faster than llama.cpp; the v0.2 wins came from scheduling and layout (below).
- **v0.3 AMX kernel (AMX-only).** Weights are preprocessed at load into tile-shaped sub-matrices matching AMX tile registers, 64-byte aligned, with symmetric group-wise int8 quantization. The cache hierarchy is exploited so that "every data element of expert weights and output activations accesses DRAM only once" ([AMX.md](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/AMX.md)). Claimed 21 TFLOPS BF16 / 35 TOPS int8 on Xeon 4th gen, about 4x PyTorch's AMX path. This is the engine behind their headline prefill numbers and it does not exist on Raptor Lake.
- **Arithmetic-intensity-aware switching.** The SOSP paper's key kernel insight: AMX only wins when arithmetic intensity is high (prefill, batched decode). Single-token decode is memory-bound, so they switch to a lightweight AVX-512 vector-matrix kernel. In other words, even KT concedes that decode speed is set by DRAM bandwidth, not matrix throughput. Their decode-side gains come from layout, threading, and overlap, not AMX.
- **AVX2-only backend (v0.5.3, 2025).** [PR #1892](https://www.phoronix.com/news/KTransformers-0.5.3) added AVX2 MoE kernels for BF16, FP8 (256-entry LUT dequant via `_mm256_i32gather_ps`), GPTQ-INT4, and RAWINT4, including an AVX-VNNI-256 variant that quantizes activations to int8 on the fly and uses `dpbusd` ([AVX2 tutorial](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/AVX2-Tutorial.md), [DeepWiki source walkthrough](https://deepwiki.com/kvcache-ai/ktransformers/7.3-avxavx512-support)). Notable for us: Raptor Lake has AVX-VNNI (256-bit), so the `dpbusd` int8 trick is available on our chip, and llama.cpp's Q8 dot products already use it when built with `-march=native` or `GGML_AVX_VNNI`. The tutorial publishes no tokens/s numbers for AVX2-only hardware and says plainly "memory bandwidth is often the bottleneck."

## 3. Scheduling and expert parallelism

These are the parts most relevant to our 34 vs 55.6 GB/s gap, because they attack overheads rather than FLOPS:

- **Fused per-layer expert tasks with dynamic task stealing.** All activated experts' matmuls in a layer are fused into one large task set, column-partitioned across threads, with work stealing to absorb skewed expert activation. Contrast ggml's `mul_mat_id`, which historically used static row partitioning plus a full thread barrier per op (3 ops x 48 layers per token), where one slow thread (an E-core on our hybrid CPU) stalls everyone.
- **Dedicated CPUInfer thread pool** with busy polling, decoupled from the GPU graph, instead of ggml's fork-join per graph node.
- **CUDA Graph capture spanning CPU ops.** GPU kernel launch overhead in hybrid decode dropped "from over 20 percent to nearly zero" ([LMSYS blog](https://www.lmsys.org/blog/2025-10-22-KTransformers/)).
- **Expert Deferral.** A few experts per layer are deferred into the next layer's window so CPU expert GEMV overlaps GPU attention, raising CPU utilization from under 75 percent to near 100 percent, for "up to 1.45x" decode throughput at under 0.5 percent accuracy delta (SOSP paper). llama.cpp strictly serializes GPU attention then CPU experts each layer, so its effective bandwidth number is diluted by idle gaps. Part of our measured 34 GB/s "extraction" gap is likely this serialization, not the kernel's streaming rate.
- **NUMA-aware placement** (weight duplication in v0.2, slice placement now, `--kt-threadpool-count` per node), worth "up to 63 percent" on dual-socket. Irrelevant on our single-socket laptop.

## 4. Claimed numbers vs llama.cpp, with caveats

From the [DeepSeek tutorial](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/DeepseekR1_V3_tutorial.md), DeepSeek-V3/R1 Q4_K_M on 2x Xeon Gold 6454S (64 cores, 2 NUMA) + RTX 4090D:

- llama.cpp (all 8 experts): prefill 10.31 t/s, decode 4.51 t/s
- KT v0.2 (6 experts, dual NUMA): prefill 97.32 t/s, decode 13.69 t/s; single socket 65.14 / 10.31
- KT v0.3-preview (AMX, 6 experts): prefill 286.55 t/s, "up to 27.79x" llama.cpp; decode unchanged vs v0.2

Qwen3-30B-A3B: "up to 347 tokens/s prefill" on a Xeon4 workstation via AMX int8 ([AMX.md](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/AMX.md)). DeepSeek-V3 end-to-end 418 t/s on Xeon4 + 4090 (SGLang, multi-concurrency).

Caveats: the llama.cpp baseline is early 2025, before `--override-tensor`/`--n-cpu-moe` and subsequent MoE improvements; KT used 6 experts vs llama.cpp's 8; decode gains lean heavily on dual-socket NUMA. The honest read: KT's decode advantage on comparable single-socket hardware is roughly 1.5 to 2x from scheduling and overlap, while the 10 to 28x headlines are AMX prefill.

## 5. Windows

Native Windows support was announced Aug 9, 2024 for the old injection framework ([issue #4](https://github.com/kvcache-ai/ktransformers/issues/4)), with community wheels and a [win-builder project](https://github.com/knilink/ktransformers-win-builder), but it was always painful. The current kt-kernel + SGLang stack is explicitly "Linux x86-64 only; Windows is not supported" ([kt-kernel README](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/README.md)). Practical route on our machine would be WSL2, with reported instability on recent versions ([issue #2016](https://github.com/kvcache-ai/ktransformers/issues/2016)). Qwen3-30B-A3B is a documented kt-kernel model, but the AVX2 BF16 path wants 64 GB RAM (we have 48) and the smaller GPTQ-INT4 path targets 24 GB VRAM GPUs in their examples.

## 6. What transfers to our setup

Uncertain items flagged: I could not extract the SOSP paper's exact DRAM bandwidth utilization figures (PDF not parseable here), and llama.cpp `mul_mat_id` threading details are version-dependent; verify against b10064 source before acting.

- AMX tile kernels, int8 35 TOPS, tile layouts: do not transfer. No AMX, no AVX-512 on 14650HX.
- Fewer-experts trick: transfers today via `--override-kv qwen3moe.expert_used_count=int:6`.
- Task stealing vs static partition + barriers: transfers as a config experiment (P-core-only threads) or an upstream patch; likely a real cause of our bandwidth gap on hybrid cores.
- CPU/GPU overlap (Expert Deferral): transfers conceptually, but needs llama.cpp implementation work; nothing user-facing exists.
- Contiguous streaming layout with one-DRAM-touch guarantee: partially present in GGUF already (per-expert slabs are contiguous); the delta is scheduling, prefetch, and avoiding Q4_K superblock scale-decode overhead. ik_llama.cpp's fused MoE and runtime repack are the nearest Windows-friendly implementation of KT-style ideas ([discussion #242](https://github.com/ikawrakow/ik_llama.cpp/discussions/242)).
