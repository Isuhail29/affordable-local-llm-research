# Baseline Model Selection

Decision date: 2026-07-18

## Decision: Qwen3-8B, Q4_K_M GGUF (official Qwen release)

Source: https://huggingface.co/Qwen/Qwen3-8B-GGUF (Qwen3-8B-Q4_K_M.gguf, 5.03 GB)

## Candidates considered

| Model | License | Age | llama.cpp support | Notes |
|---|---|---|---|---|
| **Qwen3-8B** | Apache 2.0 | Apr 2025 | First-class, official GGUFs | Best-in-class quality at 8B, GQA, 32k native context |
| Llama 3.1 8B | Llama license (restricted) | Jul 2024 | Excellent | Solid but a generation behind on benchmarks |
| Gemma 3 | Gemma license (restricted) | Mar 2025 | Good | No 8B variant exists (1B/4B/12B/27B); 12B is too big for a clean full-GPU baseline on 8 GB |
| Mistral 7B | Apache 2.0 | Sep 2023 | Excellent | Historically important, no longer competitive |

## Why Qwen3-8B wins for this project

- **Open weights, permissive license.** Apache 2.0 means no usage restrictions on research, redistribution of results, or derived tooling. Llama and Gemma both attach conditions.
- **Right size for a controlled baseline.** Q4_K_M is 5.03 GB, so it fits entirely in our 8 GB VRAM with room for KV cache. That gives us a clean best-case reference before we start starving it of VRAM deliberately.
- **Architecture is representative of the current generation.** GQA attention (big KV cache savings), SwiGLU FFN, RMSNorm, RoPE with extension support. Findings transfer to its bigger siblings.
- **A full family ladder for scaling experiments.** Qwen3 ships 0.6B/1.7B/4B/8B/14B/32B dense plus 30B-A3B and 235B-A22B MoE. Same architecture family from "fits in VRAM" to "impossible", which is exactly the gradient this research climbs. The small models also serve as speculative-decoding drafts.
- **First-class llama.cpp support and official GGUFs.** No conversion step, no community-quant uncertainty for the baseline. (Community quants like unsloth/bartowski become interesting later for imatrix comparisons.)
- **Active community.** Most-quantized, most-benchmarked 8B family right now, so we can sanity-check our numbers against public results.

## Checked against the mid-2026 landscape

A search of current comparisons (July 2026) shows nothing in the ~8B class that clearly supersedes Qwen3-8B for our purposes. DeepSeek R1 distills are interesting for reasoning but are Qwen/Llama fine-tunes anyway, and reasoning models burn tokens on thinking, which muddies throughput benchmarking. Qwen3's ability to toggle thinking mode off (`/no_think`) is a benchmarking advantage in itself.

## Key architectural facts (for benchmark math)

- 36 transformer layers, hidden size 4096
- GQA: 32 query heads, 8 KV heads, head_dim 128
- Native context 32,768 (extendable to 128k with YaRN)
- Vocab ~151k (BPE, tiktoken-style)
- ~8.2B total parameters

## Revisit triggers

Re-evaluate this choice if: a new Apache-licensed ~8B model clearly beats Qwen3-8B on quality, a new architecture (hybrid attention, SSM mix) becomes the local-inference standard, or our experiments shift to MoE-first questions (then Qwen3-30B-A3B becomes the primary subject).
