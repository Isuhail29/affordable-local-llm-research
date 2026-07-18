# Research Questions and Experiment Backlog

The charter's questions, expanded and turned into testable experiments. Ordered roughly by (knowledge gained) / (effort), and by dependency. Numbers are tentative experiment IDs; they get real IDs when an experiment folder is created.

## Tier 1: Characterize the machine (prerequisites for everything)

- **E001 Baseline offload + thread sweep** (running). Ground truth for all comparisons.
- **E002 Memory bandwidth reality check.** STREAM-style benchmark of the mixed 16+32 GB Flex Mode RAM: does the top 16 GB region really run single channel? Does Windows place the model there under memory pressure? Directly affects every CPU inference number.
- **E003 Cold start vs warm start.** mmap from SSD, first run vs page-cache-warm run: first token latency, load time, SSD read pattern (sequential or random). Tests whether GGUF tensor layout matches access order.
- **E004 PCIe link characterization.** Negotiated gen/width (nvidia-smi -q), measured host-to-device bandwidth, and what that means for any weight-streaming scheme.
- **E005 Prefill batch scaling.** pp at ubatch 32/64/.../2048: where does each processor hit its compute roof? Determines the crossover where GPU prefill pays for its PCIe transfer cost in hybrid setups.

## Tier 2: The 8 GB VRAM wall (core project territory)

- **E010 Offload placement quality.** llama.cpp puts the *last* N layers on GPU by default. Does it matter which layers? (-ot tensor overrides let us test first-N vs last-N vs attention-only-on-GPU vs FFN-only.) Hypothesis: attention on GPU + FFN on CPU beats layer-granularity splits at equal VRAM, because KV cache stays on GPU.
- **E011 KV cache placement and quantization.** --cache-type-k/v q8_0/q4_0 at long context: quality (perplexity) vs VRAM saved vs speed. When does spilling KV to CPU RAM beat quantizing it?
- **E012 Bigger models through the wall.** Qwen3-14B and Qwen3-32B at various quants and offload levels: chart tokens/sec vs (model bytes on CPU side). Validates or breaks the linear hybrid-decode model from doc 07.
- **E013 MoE as the budget cheat code.** Qwen3-30B-A3B: only ~3B params active per token. With --n-cpu-moe (expert tensors on CPU, everything else on GPU), can a 30B-class model hit 20+ t/s on this machine? This is potentially the single highest-value result for the project's thesis.
- **E014 Speculative decoding.** Qwen3-0.6B/1.7B draft + Qwen3-8B verify: acceptance rates and net speedup at various draft lengths. Then the budget-relevant version: draft on GPU, verify model partially on CPU.

## Tier 3: SSD as a memory tier (the hard research)

- **E020 The mmap cliff, measured.** Force a model bigger than RAM (e.g. Qwen3-32B Q8_0, ~35 GB, or artificially cap RAM) and measure exactly how bad paging gets, and whether the access pattern is prefetch-friendly (sequential per token) or hostile.
- **E021 Can SSD latency be hidden?** Layers are used in a fixed order every token. Prefetching layer N+k from SSD while computing layer N is the obvious idea; the question is whether Windows' prefetcher already does it, whether PrefetchVirtualMemory calls help, and whether NVMe queue depth can be kept full. Target: beat naive mmap paging by 2x+.
- **E022 MoE expert caching on SSD.** Experts have skewed usage frequency. LRU-cache hot experts in RAM, stream cold ones from SSD. Related work: Mixtral offloading papers, PowerInfer, LLM-in-a-flash (Apple). Reproduce their core ideas on consumer Windows hardware.

## Tier 4: Quality and efficiency levers

- **E030 Quant ladder for fixed hardware.** For THIS machine: 8B-Q8 vs 14B-Q4 vs 32B-IQ2 at equal memory budget. Which wins on actual task quality? (Perplexity + a small eval set.)
- **E031 Thread/core scheduling.** P-core only vs E-core only vs mixed, thread affinity, and whether Windows scheduler hurts us. Follows from E001's thread sweep.
- **E032 Power and thermals.** Sustained t/s over 10 minutes vs first 30 seconds on a laptop chassis; per-token energy CPU vs GPU if measurable.

## Standing questions without an experiment yet

- Can two cheap machines cooperate over LAN (llama.cpp RPC) without the network becoming the new SSD?
- Is context-reuse (prompt caching to disk) a bigger practical win for local use than raw t/s?
- Do importance-matrix quants shift the quality/size frontier enough to change E030's answer?
