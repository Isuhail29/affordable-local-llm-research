# Phase 1 Docs: Understanding the Inference Pipeline

Read in order. Each doc teaches one layer of the stack, computes real numbers for our machine, and ends with why it matters for the research goal: running models far larger than 8 GB VRAM on consumer hardware.

| # | Doc | Topic |
|---|---|---|
| 00 | [00-hardware-baseline.md](00-hardware-baseline.md) | The research machine, bandwidth hierarchy, napkin-math predictions |
| 01 | [01-transformer-architecture.md](01-transformer-architecture.md) | What a decoder layer computes, where 8.2B parameters live |
| 02 | [02-inference-pipeline.md](02-inference-pipeline.md) | Tokenization, prefill vs decode, sampling |
| 03 | [03-kv-cache.md](03-kv-cache.md) | The KV cache: memory math, GQA, cache quantization |
| 04 | [04-quantization.md](04-quantization.md) | Why 4-bit works, GGUF quant families, quality measurement |
| 05 | [05-gguf-format.md](05-gguf-format.md) | GGUF layout, memory mapping, the OS page cache |
| 06 | [06-llamacpp-architecture.md](06-llamacpp-architecture.md) | ggml, backends, layer offloading, thread scheduling on hybrid CPUs |
| 07 | [07-memory-hierarchy-and-bandwidth.md](07-memory-hierarchy-and-bandwidth.md) | Why bandwidth rules inference: the core systems doc |
| 08 | [08-rope-and-context.md](08-rope-and-context.md) | RoPE, positional encoding, context extension |

The mental model in one paragraph: a transformer generates one token at a time; producing each token requires streaming every active weight through the compute unit once; therefore tokens/sec is capped by (memory bandwidth) / (model bytes); everything else in this project, offloading, quantization, caching, prefetching, MoE routing, speculative decoding, is an attempt to either shrink the bytes, raise the effective bandwidth, or dodge the read entirely.
