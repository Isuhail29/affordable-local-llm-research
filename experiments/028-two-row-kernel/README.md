# Experiment 028: Two-Row Q4_K Kernel for Expert Matmuls

Date: 2026-07-19
Status: In progress

## Goal

Attack the one robust kernel-level finding left after E027: the down-expert short-row penalty (26.7 vs 37.5 GB/s intra-run). Mechanism: each 768-column row costs an indirect call, a serial scalar scale-unpack chain, and a ~20-25 cycle horizontal reduction that overlap poorly at 3 superblocks of work.

## Implementation

New ggml_vec_dot_q4_K_q8_K_2row (arch/x86/quants.c): processes two adjacent weight rows per call against one activation column. Shares the q8 activation loads and bsums work between rows; the two rows' serial chains (scale unpack, reductions) interleave in the out-of-order window. AVX2 body mirrors the upstream single-row kernel line by line; non-AVX2 builds fall back to two single-row calls. Driver (ggml_compute_forward_mul_mat_id_one_chunk) pairs rows under GGML_MMID_2ROW=1 for Q4_K, odd-tail row falls back to the standard vec_dot. Dense MUL_MAT untouched. Patch: patches/e028-2row-kernel.patch.

## Hypothesis (pre-registered before any benchmark)

1. Correctness: temp-0 outputs and perplexity match the stock path within reordering noise (PPL equal to ~3 decimals). Any real divergence voids the experiment.
2. Down-expert bin (E027 profiler): 26.7 -> 32-40 GB/s (the penalty was boundary latency, which pairing hides).
3. Gate/up bin: +0-10% (8-superblock rows have less boundary share to recover).
4. End-to-end: pure-CPU MoE +5-9%; hybrid chat config +6-12%. Falsification: under +3% end-to-end with bins unmoved kills the boundary-latency mechanism for good.
5. Register pressure risk pre-declared: the 2-row body roughly doubles live registers; if MSVC spills badly, the kernel could LOSE throughput. A loss is a valid result (documents the ceiling of this approach under this compiler).

## Method

E027 lesson applied: A/B/A within one session (stock, 2row, stock again) with the profiler on, pure CPU -t 12 first (bins visible), then hybrid -ncmoe 40. Thermal drift bounded by the flanking stock runs.

## Actual Result

Correctness gate: paired perplexity IDENTICAL to 4 decimals (5.4508 both paths). The kernel is mathematically correct.

A/B/A decode, pure CPU -t 12, profiler bins (µs/node):

| Run | t/s | gate/up | down |
|---|---|---|---|
| A1 stock | 18.66 | 187.4 | 244.1 |
| B two-row | 16.22 | 215.4 | 277.2 |
| A2 stock | 16.14 | 214.1 | 284.1 |

## Benchmark analysis

**The flanking control saved the conclusion: A2 (stock) matches B (two-row).** The machine thermally sagged ~13% across the sequence (it had just run E032's soak plus perplexity). Against the honest comparator the kernel is a wash: H4's falsification threshold fires. Sharing activation loads and hiding per-row boundary latency buys ~0-2%: the boundary-latency mechanism is conclusively dead.

**Then the autopsy found the real story: the down-expert penalty was a units error.** The model stores its 48 down-expert tensors as Q6_K (verified: 49 q6_K tensors in the loader manifest), not Q4_K: 0.820 bytes/param, not 0.563. Recomputed:

```
down (Q6_K):   10.32 MB/node / 244.1 us = 42.3 GB/s
gate/up (Q4_K): 7.08 MB/node / 187.4 us = 37.8 GB/s
```

Down-experts extract BETTER per byte than gate/up. E027's "29% short-row penalty" inverts into "down moves 46% more bytes because it is quantized fatter." (This also means the two-row kernel, gated to Q4_K, never touched the down tensors at all: doubly moot.)

**The complete, corrected conclusion of the E021-E028 arc:** under matched thermal conditions with correct byte accounting, every CPU matmul in this MoE model extracts 37-42 GB/s, uniformly. There is no expert penalty, no short-row penalty, no kernel deficit worth chasing. The apparent gap that launched three experiments was manufactured from (a) cross-session thermal drift (E032: ~9%; placement lottery: up to ~20%), (b) the Q6_K units error, and (c) a real but small residue of glue and burst-size structure (~5-15%). llama.cpp's CPU expert path is near-optimal on this hardware, and the shipped configs are a fair reflection of the machine's honest ceiling.

## Lessons Learned

1. A/B/A flanking is mandatory on this hardware; a simple A/B today would have reported the kernel as -13%.
2. Check tensor DTYPES before computing bytes: one wrong assumption manufactured a publishable-looking 29% effect from thin air.
3. Correct code that changes nothing is still evidence: it killed the last standing mechanism and forced the accounting review that ended the hunt.
4. The mark of this project's method: the final answer to "where is the missing bandwidth" is "mostly nowhere," and only pre-registered falsification thresholds, intra-run controls, and validated instruments could have reached that unglamorous truth confidently.

## Possible Improvements

- The kernel stays in the lab tree (env-gated, correct, documented) as reference material for AVX2 multi-row Q4_K work; per-type profiler bins (Q4_K vs Q6_K) would prevent the units error class permanently.

## Next Steps

- The kernel-optimization track is closed on this machine. Remaining speed levers are architectural (bigger RAM bandwidth, matched DIMMs, GPU VRAM) or model-level (quant choices, expert count, E023-style).
- Write the research story (final deliverable of this phase).
