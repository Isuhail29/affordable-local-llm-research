# Affordable High-Performance Local LLM Research

A long-term research and engineering project to discover, prototype, benchmark, and document techniques for running large open-source LLMs on inexpensive consumer hardware.

We do not assume we already know the answers. Every optimization gets measured. Every experiment produces data.

## The core problem

Modern open-weight models are bigger than the memory of affordable machines. The interesting question is not "which GPU should I buy" but "how far can software push the hardware I already have". Token generation is dominated by memory bandwidth, so this project is really a study of the memory hierarchy: VRAM, RAM, PCIe, and SSD, and how intelligently we can move model data between them.

## Research machine

| Component | Spec | Why it matters |
|---|---|---|
| CPU | Intel i7-14650HX, 16C/24T (8P+8E) | CPU inference, threading experiments |
| RAM | 48 GB DDR5-5600 dual channel | ~89.6 GB/s theoretical bandwidth ceiling |
| GPU | RTX 5060 Laptop, 8 GB GDDR7 (Blackwell) | ~384 GB/s VRAM bandwidth, the "too small VRAM" scenario |
| SSD | WD SN5000S 1 TB PCIe 4.0 NVMe | ~5 GB/s reads, the layer-streaming tier |
| OS | Windows 11 | llama.cpp prebuilt CUDA 13.3 binaries |

Full measured baseline: [docs/00-hardware-baseline.md](docs/00-hardware-baseline.md)

## Methodology

Observe → Measure → Understand → Hypothesize → Prototype → Benchmark → Compare → Document → Repeat.

Rules:

- Every optimization must be benchmarked. Never trust assumptions.
- Measure tokens/sec, first token latency, RAM/VRAM usage, SSD throughput, CPU/GPU utilization, and accuracy impact.
- Every experiment is documented in `experiments/` with Goal, Background, Hypothesis, Implementation, Expected Result, Actual Result, Benchmark, Lessons Learned, Possible Improvements, Next Steps.

## Directory map

| Directory | Purpose |
|---|---|
| `docs/` | Educational deep dives on the inference pipeline (Phase 1 output) |
| `notes/` | Decisions, ideas, running observations |
| `experiments/` | One folder per experiment, numbered, full template each |
| `benchmarks/` | Raw benchmark outputs (llama-bench JSON, logs) |
| `scripts/` | Benchmark and automation scripts |
| `models/` | GGUF model files (gitignored territory, large) |
| `llama.cpp/` | llama.cpp prebuilt binaries |
| `research-papers/` | Papers and summaries |
| `results/` | Processed results, tables, charts |
| `diagrams/` | Architecture and pipeline diagrams |
| `datasets/`, `logs/`, `tools/`, `archive/` | Support material |

## Current status (2026-07-18)

- Phase 1 (Understanding) complete: educational docs 00-08 in `docs/`, technically reviewed
- Baseline model: **Qwen3-8B Q4_K_M** (selection rationale in [notes/model-selection.md](notes/model-selection.md))
- Runtime: **llama.cpp b10064**, CUDA 13.3 Windows build
- **Experiment 001 complete:** 62.2 t/s decode full GPU, 10.1 t/s CPU (8 threads optimal), linear hybrid scaling confirmed, and a surprise: CUDA builds GPU-accelerate prefill 18x even at -ngl 0 via PCIe weight streaming. Details in [experiments/001-baseline-benchmark](experiments/001-baseline-benchmark/README.md)
- Experiment backlog: [notes/research-questions.md](notes/research-questions.md)
- **Experiment 002 complete:** the 89.6 GB/s RAM spec is really a 55.6 GB/s practical ceiling, inference extracts 91% of it, and physical page placement (Flex Mode) is a ±21% lottery. Protocol: mmap always, duplicate runs. [experiments/002-ram-bandwidth](experiments/002-ram-bandwidth/README.md)
- **Experiment 013 complete, thesis validated:** Qwen3-30B-A3B (MoE) runs at **38.5 t/s decode on the 8 GB GPU** with experts in RAM (-ncmoe 36), 27% faster than naive offload, and even pure CPU hits 20 t/s. Sparse models are the budget-hardware architecture. [experiments/013-moe-experts-on-cpu](experiments/013-moe-experts-on-cpu/README.md)
- Paper library: LLM-in-a-flash, PowerInfer, Fiddler, FlexGen summarized in [research-papers/](research-papers/README.md)
- **Experiment 021 complete, core discovery:** the MoE expert-read gap is 100% llama.cpp implementation, not memory physics (random 0.25 MB reads hit the same ~60 GB/s as sequential). No config knob reaches it; fixing it needs source work. Bonus: the soft-fault cliff (15-17 GB/s) explains the RAM-pressure slowdowns. [experiments/021-expert-scatter](experiments/021-expert-scatter/README.md)
- **Experiment 023 complete, shipped as turbo mode:** running 6 of 8 experts per token = **+21% speed for +2.4% perplexity** (top-4 rejected at +17.8%). Start-30B-AI-Turbo.bat, ~40 t/s. [experiments/023-expert-count](experiments/023-expert-count/README.md)
- **Experiment 014 complete, hypothesis refuted (a good day for science):** speculative decoding loses on this hardware, both for the MoE (expert-union cost, down to 0.36x) and the dense 8B (TDP-shared laptop GPU). The chase still paid: MoE thread optimum is 12, cache pre-warming is worth 1.5-3x, and a KV-quant config trap that halved server speed is documented. [experiments/014-speculative-decoding](experiments/014-speculative-decoding/README.md)

## Reproducing this setup

Large binaries are not in this repo. To recreate the environment:

1. **llama.cpp b10064** prebuilt Windows CUDA binaries: download `llama-b10064-bin-win-cuda-13.3-x64.zip` and `cudart-llama-bin-win-cuda-13.3-x64.zip` from [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases/tag/b10064), extract both into `llama.cpp/bin/`.
2. **Models** into `models/`:
   - [Qwen/Qwen3-8B-GGUF](https://huggingface.co/Qwen/Qwen3-8B-GGUF): `Qwen3-8B-Q4_K_M.gguf` (5.03 GB)
   - [unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF](https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF): `Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf` (18.6 GB)
   - [Qwen/Qwen3-0.6B-GGUF](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF): `Qwen3-0.6B-Q8_0.gguf` (E014 draft model)
3. Double-click `Start-30B-AI.bat` (full quality) or `Start-30B-AI-Turbo.bat` (+21% speed, -2.4% perplexity). Numbers in the experiment reports assume the hardware in `docs/00-hardware-baseline.md`; your ceilings will differ, but the methods transfer.

## Phase roadmap

1. **Understand** the full inference pipeline (docs, diagrams, baseline measurements)
2. **Characterize** the machine: bandwidth ceilings, offload behavior, SSD streaming reality
3. **Optimize**: hypothesis-driven experiments on caching, prefetching, KV compression, MoE offloading, speculative decoding, hybrid execution
4. **Scale up**: apply what works to models far larger than 8 GB VRAM (14B, 32B dense, 30B+ MoE)
