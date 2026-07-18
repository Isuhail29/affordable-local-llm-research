# LLM in a Flash: Efficient Large Language Model Inference with Limited Memory

## Citation

Keivan Alizadeh, Iman Mirzadeh, Dmitry Belenko, Karen Khatamifard, Minsik Cho, Carlo C Del Mundo, Mohammad Rastegari, Mehrdad Farajtabar (Apple). "LLM in a flash: Efficient Large Language Model Inference with Limited Memory." arXiv:2312.11514. v1 December 2023, v3 July 2024. Published at ACL 2024.

- Abstract: https://arxiv.org/abs/2312.11514
- Full text (HTML): https://arxiv.org/html/2312.11514v3

## Problem

The model does not fit in DRAM, so weights must live on flash storage and be pulled in on demand. The naive approach, loading the full set of needed weights from flash for every forward pass, is catastrophically slow because flash is both lower bandwidth than DRAM and terrible at small random reads. The paper asks: how far can you get if you treat flash as a first-class tier in the inference memory hierarchy and design the loading policy around flash's actual performance characteristics, instead of letting the OS pager or a naive loader thrash?

This is exactly the situation our project hits with models above roughly 40 GB on a 48 GB DDR5 machine, and it is the closest prior art to our E020 (mmap paging cliff) and E021 (SSD prefetching) experiments.

## Key Idea

Build an explicit inference cost model with three latency terms: flash I/O, memory management (inserting and evicting weights in DRAM), and compute. Then optimize the two levers flash cares about:

1. **Transfer less.** Exploit FFN activation sparsity so you only load the neurons that will actually fire, and cache recently used neurons so consecutive tokens reuse most of what is already resident (windowing).
2. **Transfer in bigger chunks.** Restructure the weight layout on flash (row-column bundling) so each neuron's data is one contiguous read of twice the size, because flash random-read throughput scales strongly with chunk size.

With about half the model held in DRAM, this runs models up to 2x DRAM capacity at 4-5x the naive speed on CPU and 20-25x on GPU.

## Mechanism

**The flash bandwidth curve is the whole game.** On the M1 Max SSD the paper measures sequential reads at 6+ GB/s for 1 GB transfers, but random reads at only about 1.25 GB/s with 4 KB chunks, improving to about 2.25 GB/s with 32 KB chunks (their Figure 2b). Their stated rule: it is often worth "reading more than needed in larger chunks and then discarding" because latency-to-first-byte dominates small reads. They treat 32 KB as roughly the minimum sensible chunk. Throughput also scales with the number of parallel outstanding reads (multiple threads issuing I/O concurrently).

**Sparsity prediction.** ReLU-family FFNs are extremely sparse per token: 97% of FFN neurons inactive for OPT 6.7B, 95% for a "relufied" Falcon 7B, and about 90% for Llama 2 with FATReLU fine-tuning. A small low-rank predictor per layer looks at the attention output and predicts which FFN neurons will be nonzero, before their weights are loaded. Training cost about 4 hours per layer on an A100 using 10k C4 samples; the predictor adds only 1.25% to model size. False negatives are around 5% for OPT 6.7B with minimal zero-shot accuracy impact.

**Windowing.** Keep a sliding-window cache in DRAM of the neurons activated by the last k tokens (they use k = 5 in the main method, k = 4 in the DRAM budget appendix). Because active-neuron sets overlap heavily between nearby tokens, each new token only needs to load the incremental difference. For OPT 6.7B with a k = 4 window, the per-token FFN load drops to about 2.4% of FFN weights instead of the ~10% you would load with sparsity alone and no cache.

**Row-column bundling.** The i-th column of the FFN up-projection and the i-th row of the down-projection both correspond to intermediate neuron i, so the paper stores them concatenated on flash. One read of size 2 x d_model fetches everything neuron i needs, doubling the chunk size and moving throughput from the 1.25 GB/s regime toward 2.25 GB/s. A follow-up idea, bundling each neuron with its most frequently co-activated "closest friend," failed: the most active neurons are everyone's closest friend, so they got loaded repeatedly. The negative result is in Appendix D.

**Memory management.** The DRAM neuron cache is a preallocated fixed-size matrix. Eviction swaps the deleted row with the last valid row and decrements a counter, so deletes are O(c) and there are no reallocations or large copies. This matters: on M1 Max for OPT 6.7B the per-token latency breakdown is roughly 105 ms I/O plus 57 ms memory management, so naive cache bookkeeping would eat much of the I/O win.

**DRAM budget.** For OPT 6.7B the resident set totals 52.1% of model size: embeddings 3%, attention weights 32.3% (attention is always resident, only FFN weights stream), predictors 1.25%, and the windowed FFN cache about 15.5%.

## Reported Results

- Models: OPT 6.7B, relufied Falcon 7B, Persimmon 8B, plus Phi-2 and Llama 2 experiments in v3. Decode only: 128-token prompt, 256 generated tokens, batch size 1.
- CPU (M1 Max): OPT 6.7B per-token latency drops from 3182 ms naive to 669 ms, about 4.8x.
- Metal GPU (M1 Max): 2389 ms to 565 ms, about 4.2x.
- RTX 4090 (bfloat16): 2218 ms to 84 ms, about 26.4x versus naive flash loading; adding speculative decoding with 4 draft tokens gives a further 1.4x.
- Flash I/O latency per token stays flat out to 1000 generated tokens, so the windowing cache does not degrade over long generations.
- All runs assume only about half the model fits in DRAM.

Note the baselines: "naive" reloads all needed weights from flash every forward pass, and the "hybrid" baseline keeps half the model resident and streams the other half without sparsity. The 20-25x GPU number is against the naive flash baseline, not against a fully-in-memory model. A fully resident model is still faster; this is about making the impossible-to-fit case usable.

## Limitations and Caveats

- **The sparsity lever needs ReLU-style activations.** OPT's 97% sparsity does not exist in modern SiLU/SwiGLU models like Qwen3 or Llama 3 without relufication or FATReLU fine-tuning, which requires training we cannot do. This is the biggest gap between the paper and our stack.
- **Predictors require per-model training** (hours per layer on an A100) and add a small accuracy risk from false negatives.
- **Decode only.** Prefill, batching, and long-context KV cache pressure are out of scope.
- **Apple silicon numbers dominate**; the unified-memory M1/M2 machines and their fast SSDs differ from a Windows laptop with discrete GPU, though the RTX 4090 result shows the approach transfers.
- Attention weights always stay resident, so the "2x DRAM" claim depends on FFN dominating the parameter count (about two thirds in these models).

## What This Means for Our Project

**E020 (mmap paging cliff): this paper predicts the cliff and explains it.** Windows demand paging of an mmapped GGUF faults in 4 KB pages, which is exactly the 1.25 GB/s worst-case regime on their SSD curve; our WD SN5000S peaks near 5 GB/s sequential but will show the same order-of-magnitude collapse at 4 KB random. E020 should measure our own throughput-vs-chunk-size curve (4 KB, 32 KB, 128 KB, 1 MB, queue depth 1 vs high QD) with fio before interpreting llama.cpp paging behavior. The paper's "read big and discard" rule says large readahead should beat precise faulting.

**E021 (SSD prefetching): windowing is the design template.** Their result that consecutive tokens reuse most of the working set justifies prefetching the stable hot set once and only streaming the cold tail. A practical port: pin attention plus hot layers in RAM, and issue large asynchronous sequential reads for streamed tensors rather than relying on page faults. Their 57 ms/token memory-management cost is a warning that the bookkeeping layer must be cheap or it erases the I/O win.

**E022 (MoE expert caching on SSD): the strongest mapping.** We cannot get ReLU sparsity in dense Qwen3 FFNs, but MoE routing gives us the same thing for free: the router is a perfect "sparsity predictor" (it tells us exactly which experts fire), and experts are natural row-column bundles, contiguous multi-MB blobs that sit in the efficient region of the SSD curve. An LRU expert cache in RAM is the direct analog of their neuron window; the co-activation caveat maps to hot shared experts, which should simply be pinned rather than cached.

**E013 and E010 (experts on CPU, offload placement): the cost model transfers.** Their three-term latency model (I/O, management, compute) is the right frame for our placement decisions across the 312.7 GB/s VRAM, 50.8 GB/s DRAM, roughly 8 GB/s effective PCIe Gen5 x8 in practice, and ~5 GB/s NVMe tiers. Their finding that attention should always stay in the fastest tier while conditional FFN/expert weights ride the slower tier matches the E010/E013 layout we are already testing, and gives us paper-backed numbers for how much reload traffic a k-token reuse window saves.

**Bottom line:** with a 48 GB machine, this paper says an SSD tier is viable only if reads are large, contiguous, and predicted ahead of compute. For dense SwiGLU models the sparsity trick is unavailable, so our SSD experiments should focus on MoE experts (E022), where routing hands us the predictor and the bundling for free.

## Links

- Paper abstract: https://arxiv.org/abs/2312.11514
- Full text HTML (v3): https://arxiv.org/html/2312.11514v3
- PDF: https://arxiv.org/pdf/2312.11514
- ACL 2024 version: https://aclanthology.org/2024.acl-long.678/
