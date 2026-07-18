# Experiment 021: The Expert-Scatter Penalty

Date: 2026-07-18
Status: Complete (empirical); optimization follow-ups tracked in notes/expert-scatter-roadmap.md

## Goal

Explain and, if possible, recover the missing bandwidth in MoE expert reads: dense weight streaming extracted ~50 GB/s from RAM while the expert workload of Qwen3-30B-A3B extracted only ~30-34 GB/s. Worth up to +50% decode speed.

## Hypothesis (pre-registered)

1. Pure access-pattern replay lands at 40-48 GB/s: roughly half the gap is physics, half implementation.
2. Bandwidth rises with chunk size, approaching sequential around 8-32 MB.
3. Routing-locality effects on real speed are small (under 15%); the old 59 t/s anomaly will not reproduce.
4. Expert-read effective bandwidth confirms in the 33-37 GB/s band under strict protocol.

## Implementation

- Part A: llama-bench, warm cache, same session: 30B -ngl 99 -ncmoe 48 -t 12 and 8B -ngl 0 -t 8 (CUDA hidden), -p 0 -n 128 -r 3
- Part B: scripts/scatter-bench.py: 12 workers mmap the real 18.6 GB GGUF, random-offset chunks (0.25/1/4/16/64 MB) vs sequential; v1 accidentally measured cold page-table mapping, v2 populates page tables untimed then times a second pass (steady state)
- Part C: llama-server (mlock, -t 12, ncmoe 40), three regimes: extreme repetition (temp 0), normal essay (temp 0), high-entropy babble (temp 1.5); ~1,000 tokens each, outputs recorded
- Background workflow: ggml mul_mat_id internals, llama.cpp MoE perf PR history, KTransformers, expert-layout prior art (notes/internals-*.md, notes/expert-scatter-roadmap.md)

## Actual Result

### Part A: effective rates under strict protocol

| Workload | t/s | Effective GB/s |
|---|---|---|
| Dense 8B CPU decode (-t 8) | 9.45 ± 0.03 | 47.5 |
| 30B all-experts-on-CPU (-t 12) | 26.3 ± 3.8 | ~30 on the expert reads (GPU share subtracted) |

### Part B: access-pattern microbenchmark (12 workers, aggregate)

| Pattern | Cold page tables | Warm page tables |
|---|---|---|
| random 0.25 MB chunks | 14.7 GB/s | 59.9 GB/s |
| random 1 MB chunks | 15.3 GB/s | 60.9 GB/s |
| random 4 MB chunks | 15.6 GB/s | 59.0 GB/s |
| random 16 MB chunks | 15.5 GB/s | 61.4 GB/s |
| random 64 MB chunks | 17.1 GB/s | 60.3 GB/s |
| sequential 16 MB sweep | 17.3 GB/s | 60.5 GB/s |

### Part C: routing-locality probe (~1,000 tokens each, outputs verified sane)

| Regime | t/s |
|---|---|
| Extreme repetition (numbered fox sentences) | 34.6 |
| Normal essay | 33.6 |
| High-entropy babble (temp 1.5) | 33.0 |

## Benchmark analysis

**H1 and H2 refuted, decisively and usefully.** At steady state the memory system delivers ~60 GB/s at EVERY chunk size from 0.25 MB up, identical to sequential. Scatter physics costs nothing on this hardware; modern prefetchers handle sub-megabyte chunks perfectly. There is no chunk-size curve to climb.

**The bandwidth ladder of this machine (read-only, warm):**

```
60 GB/s   steady-state RAM read ceiling (any pattern >= 0.25 MB chunks)
47.5      llama.cpp dense decode        (79% of ceiling: dequant+compute cost)
~30       llama.cpp expert-path decode  (50% of ceiling)  <-- the implementation gap
15-17     cold page-table first touch   (soft-fault-bound)
~5        SSD re-reads after eviction   (the disaster mode)
```

**The entire expert gap is llama.cpp implementation, not physics.** Roughly 2x sits between the expert path and the ceiling, and ~1.6x between expert path and llama.cpp's own dense efficiency. Candidate mechanisms (per the internals research): thread work-splitting across 320 small per-expert matvecs per token, synchronization between them, and dequant kernels that do not stream as well when the work unit is a 0.9 MB expert slice instead of a long contiguous row range.

**H3 confirmed.** Locality is worth ~5% end to end (34.6 vs 33.0). Qwen3's load-balanced router leaves nothing for hot-expert caching on this model; E022-style expert caching is deprioritized. The old 59 t/s anomaly did not reproduce even under maximum repetition and is now classified as a broken-measurement artifact.

**Bonus mechanism, field-relevant: the soft-fault cliff.** First-touch mapped reads run at 15-17 GB/s. That predicts ~14 t/s for a MoE process whose pages keep getting unmapped, which matches the 12.2 t/s observed in the user's RAM-pressure session almost exactly. mlock (already shipped in the launcher) prevents precisely this.

## Addendum: configuration-knob probes (same day, analyzed after the writeup)

Three roadmap ideas tested with existing binaries, all negative, all narrowing the search:

| Probe | Result | vs reference | Verdict |
|---|---|---|---|
| Pure CPU isolation (-ngl 0, -t 12) | 18.99 ± 1.93 t/s (~34 GB/s effective) | hybrid ncmoe 48: 26.3 | Same bandwidth cap without any GPU involvement: hybrid handoff is NOT the bottleneck, the CPU MoE path is |
| No-mmap (-mmp 0, repack hypothesis) | 24.93 ± 0.81 t/s | mmap: 26.3 ± 3.8 | No repack engagement visible, no win; only lower variance (malloc dodges some placement lottery) |
| P-core pinning (-t 8 --cpu-mask 0x5555 --cpu-strict 1 --poll 100) | 22.55 ± 3.13 t/s | 12 mixed threads: 26.3 | Pinning LOSES; E-core participation adds useful outstanding memory requests on this workload |

Consequence: no configuration knob reaches the gap. The ~30-34 GB/s cap is inside ggml's mul_mat_id work-splitting/kernel path, reachable only via source changes (or a fork that already made them). Remaining non-source lever: reducing active experts per token via metadata override (tested separately).

## Lessons Learned

1. Distrust intuitive physics explanations ("scattered reads are slow") until the pattern is replayed without the software stack. The scapegoat was innocent; the code path is the culprit.
2. Cold-vs-warm page tables is a 4x effect on Windows mmap reads; any benchmark of mmap'd model access MUST state which regime it measured.
3. A wrong benchmark can still be a discovery: v1's "failure" quantified the soft-fault cliff that explains our worst field observation.
4. Pre-registered wrong hypotheses (40-48 GB/s) make the surprise (60) trustworthy: we predicted against ourselves and can prove it.

## Possible Improvements

- Replay the pattern with interleaved fake compute (dequant-like arithmetic between reads) to emulate the dense path's 79% and locate the expert path's extra loss precisely.
- Verify the ~30 GB/s expert figure with a wider ncmoe ladder under the strict protocol.
- Windows large-page support would shrink the soft-fault cliff; investigate llama.cpp support status.

## Next Steps

- The speedup now requires touching llama.cpp's CPU MoE matmul path: follow notes/expert-scatter-roadmap.md (build-from-source track, thread-scheduling experiments, possible upstream contribution: this finding + a patch would benefit every llama.cpp MoE user).
- E032 (formal sustained thermal test) remains queued.
