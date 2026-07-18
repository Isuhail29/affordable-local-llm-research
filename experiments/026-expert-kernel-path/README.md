# Experiment 026: The Expert Kernel Deficit

Date: 2026-07-19
Status: In progress (design phase)

## Goal

Recover the CPU expert-path kernel deficit identified by E021+E025: dense decode extracts ~47.5 GB/s equivalent from kernels that are compute-bound at matvec granularity, the expert path only ~30. Closing even half the gap is worth roughly +15-25% end-to-end decode on the flagship config, stacking with turbo mode.

## Background (established facts)

- E021: memory physics is innocent; steady-state RAM delivers ~60 GB/s at any chunk size. The gap is in code.
- E025: barrier stragglers falsified (work-stealing patch worth <= 5%). Same-toolchain control revealed 10% compiler sensitivity on the expert path: it is ALU/kernel-bound, not bandwidth-bound.
- Suspects, to be confirmed against source: (a) the optimized llamafile_sgemm engine may be unavailable to mul_mat_id and possibly to Q4_K entirely; (b) the down expert matrices have 768-column rows = only 3 Q4_K superblocks, amplifying per-row kernel setup costs; (c) per-node fixed overhead (activation re-quantization, single-threaded id-grouping, barriers) across ~144+ expert nodes per token.

## Method discipline

Three source-study agents are reading the actual b10064 tree (sgemm support matrix and guards, Q4_K vec_dot anatomy with a short-row cost model, mul_mat_id overhead accounting). Their outputs (notes/e026-*.md) feed a design ladder ordered by information-per-effort. Hypotheses for each rung get pre-registered HERE before any implementation is benchmarked. No rung is implemented before its falsification threshold is written down.

## Candidate ladder (pending source confirmation)

- A: requantize to an sgemm-supported type, A/B with zero code changes
- B: per-expert llamafile_sgemm invocation inside mul_mat_id with fallback
- C: multi-row / short-row-specialized vec_dot for the 768-col down matrices
- D: fused gate+up expert pass
- E: node-overhead reduction (batching the three expert matmuls, fewer barriers)

## Source-study verdicts (notes/e026-*.md)

- **sgemm is a dead end** (notes/e026-sgemm-support.md): no Q4_K/Q8_K support, hard n<2 guard (prefill-only by design), no call site in mul_mat_id, and dense decode does not use it either. Ladder rungs A and B are eliminated before implementation.
- **The deficit mechanism** (notes/e026-vecdot-anatomy.md): ~38% of the Q4_K vec_dot kernel is per-superblock bookkeeping (hits dense and experts equally; explains 47.5 vs 60). The expert-specific loss is serial per-row boundary latency (indirect call + scalar scale unpack + horizontal reduction, a ~40-50 cycle poorly-overlapped chain) amplified on 3-superblock down rows: ~1.38M vec_dot calls per token.
- **The cure already exists in-tree**: the repack path (block_q4_Kx8 + ggml_gemv_q4_K_8x8_q8_K) processes 8 interleaved rows per call, shares activation loads and scale unpacks, has no per-row reduction, and supports MUL_MAT_ID. Both expert shapes qualify (768 and 2048 rows, divisible by 8). It is unreachable because --n-cpu-moe pins experts to the plain CPU buffer and the -ot parser only enumerates default buffer types.
- **The unlock**: ~12 lines in common/arg.cpp exposing device extra buffer types (mirroring llama-model.cpp:923-930), making `-ot "<regex>=CPU_REPACK"` work from the CLI. Patch: patches/e026-ot-extra-bufts.patch.

## Hypothesis (pre-registered before any benchmark)

1. Repacked experts recover a large share of the boundary losses: pure-CPU MoE decode (our-build stock baseline 17-19 t/s at t12/16) gains +15-35%; the hybrid chat split (plain ncmoe-40 class ~33 t/s) reaches 38-44. Falsification: under +8% kills the boundary-latency model and shifts suspicion to per-node overheads (notes/e026-mmid-overhead.md).
2. Costs: load time rises materially (repack at load, no mmap for repacked tensors; cold start +30-120 s) and expert memory becomes private allocations (~17 GB committed; page-cache eviction immunity like mlock, but E002 placement lottery applies).
3. Quality unchanged: repack is an exact re-layout, not requantization; a story-prompt output must be coherent and byte-similar in meaning to plain runs.
4. Engagement is verified, not assumed: the load log must show expert tensors in a CPU_REPACK buffer; absent that line, results are void (E014 lesson).

## Actual Result

(pending)
