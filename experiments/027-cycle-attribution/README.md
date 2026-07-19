# Experiment 027: Cycle Attribution (where do 50 ms/token go?)

Date: 2026-07-19
Status: In progress

## Goal

Directly attribute the pure-CPU MoE token budget (~50 ms at ~20 t/s) across operation types, ending the hypothesis phase of the expert-deficit hunt. E026 eliminated kernels, granularity, stragglers, and (bounded) barriers; ~13 ms/token vs dense-class extraction is unattributed.

## Method

Instrument the CPU backend's node-execution loop (ggml-cpu.c): thread 0 accumulates wall time per op type across the run (ggml_time_us around each node incl. its closing barrier), env-gated by GGML_OP_PROFILE, table printed at process exit. MUL_MAT_ID split into two shape bins (gate/up: ne01=768; down: ne01=2048). Profile two runs, same protocol (warm, -t 12, GPU hidden, tg128 r=3):

- P1: 30B MoE pure CPU (the mystery)
- P2: dense 8B pure CPU (the control that extracts 47.5 GB/s)

## Hypothesis (pre-registered)

1. MUL_MAT_ID holds >75% of MoE token time. If glue ops (norms, rope, attention, router softmax/topk, quantize) hold a large share instead, the deficit was never in expert matmuls and every kernel theory was chasing the wrong op.
2. The dense control shows MUL_MAT at its known extraction rate, validating the instrument against an established number.
3. Bins: gate/up (2 nodes/layer, longer rows) vs down (1 node/layer, short rows) per-byte times differ by <25% (the granularity curve says short rows are mostly fine); a larger skew reopens the short-row case with direct evidence.

## Falsification stance

Whatever the table says, wins: every outcome retires at least one live theory. The instrument is validated or invalidated by the dense control against 47.5 GB/s.

## Actual Result

385 tokens profiled per run; instrument validated: profile totals match measured wall time to within rounding on BOTH runs (MoE 53.7 ms/token profiled vs 54.0 measured at 18.52 t/s; dense 126.2 vs 125.9 at 7.94 t/s).

### MoE 30B pure CPU, per token

| Op | ms/token | Share | Effective GB/s |
|---|---|---|---|
| MUL_MAT_ID gate/up experts | 17.7 | 33% | 37.5 |
| MUL_MAT_ID down experts | 12.4 | 23% | **26.7** |
| MUL_MAT (attention + router + head) | 19.5 | 36% | 36.9 |
| Glue (ADD, norms, rope, attention-ext, softmax, argsort...) | 4.1 | 8% | n/a |

Dense 8B control: MUL_MAT 98.0% of time, extraction 39.9 GB/s this session.

## Benchmark analysis

**H1 falsified, and that is the discovery: the deficit was never expert-specific.** Expert matmuls hold only 56% of the MoE token budget; the model's ordinary dense attention matmuls run at the same depressed rate (36.9 GB/s) as the gate/up experts (37.5). Under matched conditions, everything in the MoE except the down-experts extracts within ~10% of the dense-8B control.

**H2 confirmed**: the control behaved exactly as required, validating the instrument.

**H3 falsified in the informative direction**: down-experts (768-column, 3-superblock rows) extract 26.7 GB/s vs 37.5 for their gate/up siblings in the same run: a 29% intra-run penalty, immune to thermal confounds. Worth ~3.9 ms/token; fixing it entirely would buy ~7% pure-CPU and an estimated 9-12% on the hybrid config where experts dominate CPU time.

**The self-correction this experiment forced: our historical cross-session comparison (47.5 vs 30 GB/s) was partially a thermal artifact.** The dense control ran at 39.9 GB/s today vs 47.5 in an earlier, cooler session; hours of continuous benchmarking shift the machine's whole performance level. Intra-run contrasts (down vs gate/up) are trustworthy; cross-session absolute comparisons need thermal control. This retroactively shrinks the "13 ms mystery" that motivated E026/E027: it decomposes into ~4 ms of genuine short-row penalty, ~4 ms of glue and attention structure, and session-level thermal variance that inflated the rest.

## Lessons Learned

1. Attribution before optimization: three experiments of kernel theories (E025, E026) targeted ops holding 56% of the budget; 20 minutes of instrumentation would have scoped them correctly from the start.
2. A validated instrument beats any inference chain: profile totals matching wall time end-to-end is what makes every row of the table trustworthy.
3. Intra-run contrasts are robust to environmental drift; cross-session comparisons on a laptop are not. All prior effective-GB/s comparisons across sessions carry an unquantified thermal error bar until E032 runs.
4. The instrument (GGML_OP_PROFILE, patches/e027-op-profile.patch) is permanent lab equipment now: any future change gets judged by its per-op table, not aggregate t/s alone.

## Possible Improvements

- Re-run the E026 repack A/B WITH the profiler: does repack help the down bin specifically while losing elsewhere (netting the observed tie)?
- Extend bins: per-op effective-GB/s computed inside the profiler instead of by hand.

## Next Steps

- **E032 (now scientifically required): sustained thermal characterization**: 10-minute continuous decode logging t/s + clocks + temperature per minute, quantifying the level-shift that contaminates cross-session numbers.
- **E028 candidate: the down-expert short-row fix**: the one legitimate kernel target left (batch multiple short rows per kernel invocation in the MUL_MAT_ID driver, or a fused two-row Q4_K vec_dot). Ceiling: ~9-12% on the shipped hybrid config.
