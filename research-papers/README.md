# Research Paper Summaries

Literature review for the central question of this project: can consumer SSD, RAM, and GPU tiers cooperate to run models larger than VRAM at usable speed on a machine like ours (RTX 5060 Laptop 8 GB at 312.7 GB/s, 48 GB DDR5 at 50.8 GB/s, WD SN5000S NVMe at ~5 GB/s, PCIe Gen5 x8)?

## Papers

| Paper | One-line idea | Informs |
|---|---|---|
| [LLM in a Flash](llm-in-a-flash.md) (Apple, arXiv:2312.11514) | Treat flash as a real memory tier: predict which FFN neurons fire, cache the recent working set in DRAM, and lay weights out so every read is large and contiguous | E020, E021, E022 |
| [PowerInfer](powerinfer.md) (SJTU, arXiv:2312.12456) | Neuron activations follow a power law, so pin the hot neurons in VRAM, compute cold ones on CPU, and let small predictors skip inactive ones entirely | E010, E013, E022 |
| [Fiddler](fiddler.md) (UW, arXiv:2402.07033) | For MoE offloading, move KB-scale activations to CPU-resident experts instead of moving 300+ MB experts over PCIe, and flip to GPU streaming only when prefill batches make it worthwhile | E010, E013, E022 |
| [FlexGen](flexgen.md) (Stanford et al., arXiv:2303.06865) | Formalize GPU/CPU/disk placement and scheduling as a cost-model-driven search, then amortize weight I/O over huge batches for throughput-oriented workloads | E010, E020, E021 |

## Synthesis

The literature converges on a clear yes, with conditions. Batch-1 decode is memory-bandwidth-bound: tokens per second is bytes touched per token divided by the bandwidth of the tier those bytes live in. Our own baseline confirms this exactly (62.2 / 10.1 = 6.2x GPU/CPU decode ratio versus a 312.7 / 50.8 = 6.2x bandwidth ratio). So running past VRAM at usable speed reduces to one question: how few bytes can you touch per token, and from which tier?

The papers give three answers that stack. FlexGen shows placement across tiers should be a cost model, not folklore, though its batch-amortization trick is useless for interactive use. PowerInfer and LLM in a Flash show sparsity is the real lever: if you can predict which weights a token needs, the slow tiers only serve a small, mostly-reused working set, and both DRAM-over-VRAM and flash-over-DRAM become viable. Fiddler shows the data-movement rule that makes hybrid execution work: ship activations to resident weights for decode, ship weights to the GPU for prefill. Our measured 18x prefill speedup from llama.cpp PCIe streaming at -ngl 0 is the same physics.

The catch is that the strongest results all lean on ReLU-style activation sparsity, which modern SwiGLU models (Qwen3, Llama 3) do not have. That is precisely why MoE matters for us: the router hands us exact, structured sparsity for free, no predictor training, no accuracy risk. E013 is Fiddler's design implemented through llama.cpp flags, and E022 is LLM in a Flash's flash tier with experts as natural multi-MB contiguous bundles that sit in the efficient region of the SSD bandwidth curve.

The open territory is real. First, nobody in this set characterizes the SSD tier on a consumer Windows laptop: the mmap paging cliff (E020) and managed large-read prefetching versus 4 KB page faults (E021) are measured on Apple silicon or server boxes, not our stack. Second, all four papers assume either abundant RAM or ReLU sparsity; an SSD-backed expert cache for a balanced-routing MoE (E022) sits in neither regime, and the first thing to measure is whether Qwen3-30B-A3B expert usage shows any exploitable skew or only temporal locality. Third, the copy-versus-compute crossover shifts on our hardware: quantized experts are only a few MB and PCIe Gen5 x8 is close to our RAM bandwidth, so Fiddler's "never stream during decode" rule deserves re-testing (E013, E010). Reproducing the placement cost model with our measured constants and extending it below the RAM floor is a contribution none of these papers makes.
