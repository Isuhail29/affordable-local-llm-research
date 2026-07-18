# Experiment 002: RAM Bandwidth Decomposition

Date: 2026-07-18
Status: Complete

## Goal

Explain the gap between theoretical RAM bandwidth (89.6 GB/s) and the 50.8 GB/s effective bandwidth inferred from CPU decode in E001. How much is generic DDR5 efficiency loss, and how much is the mixed-module Flex Mode configuration (16 GB + 32 GB, where the top 16 GB physical region runs single channel at 44.8 GB/s theoretical)?

## Hypothesis

1. Multiprocess STREAM copy peaks at 55-70 GB/s aggregate.
2. A single worker cannot saturate the controller: 15-30 GB/s.
3. Saturation by 4-8 workers.
4. With 28 GB ballast pushing the model toward the single-channel region, CPU decode drops toward 6-8 t/s.

## Implementation

- scripts/ram-bandwidth.py: numpy multiprocess STREAM copy (read+write counted), 2 GiB per worker
- scripts/ballast.py: allocate and hold N GiB
- llama-bench CPU decode probes (CUDA_VISIBLE_DEVICES=-1, -t 8, -p 0 -n 128 -r 3), varying weight backing (mmap vs malloc) and pre-existing memory state
- winsat mem produced no output without elevation; skipped

## Actual Result

### STREAM sweep

| Workers | Aggregate GB/s | Per-worker |
|---|---|---|
| 1 | 28.6 | 28.6 |
| 2 | 42.3 | ~21 |
| 4 | 47.9 | ~12 |
| 8 | 52.5 | ~6.6 |
| 12 | **55.6** | ~4.6 |

### CPU decode across memory states (all -t 8, tg128)

| Weight backing | Memory state | t/s | Implied GB/s |
|---|---|---|---|
| mmap (E001) | warm page cache | 10.07-10.10 | 50.8 |
| malloc + 28 GB ballast first | ballast in upper region | 9.84 | 49.5 |
| mmap, cache re-faulted after purge | mixed | 9.37 | 47.1 |
| malloc, no ballast (x2, reproducible) | mostly free RAM | 8.33, 8.34 | 41.9 |

## Benchmark analysis

**Hypotheses 1-3 confirmed** (55.6 GB/s ceiling, 28.6 single worker, gradual saturation). **Hypothesis 4 refuted in direction but vindicated in mechanism.**

The decisive observation is the internal consistency of the two extremes:

```
malloc no ballast:  41.9 GB/s = 93.5% of the single-channel region's 44.8 GB/s
best placement:     50.8 GB/s = 91.4% of the 55.6 GB/s practical machine ceiling
```

Interpretation: Windows hands out fresh anonymous pages preferentially from the upper (single-channel) physical region when memory is free, so a plain malloc'd model lands in the slow region. A large ballast allocated first absorbs those pages and forces the model into the dual-channel region. Both endpoints run at ~90+% of their region's practical ceiling, so llama.cpp itself is nearly optimal; physical placement decides the rest.

**E001's mystery is resolved.** The 56.7%-of-theoretical efficiency decomposes into: theoretical 89.6 → practical STREAM ceiling 55.6 (client controller + Flex Mode averaging, a platform property) → 50.8 during inference (91% extraction, excellent). Nothing was wrong with the inference stack.

**The placement lottery is worth 21%.** Identical benchmark, identical flags, minutes apart: 8.33 to 10.10 t/s depending only on invisible physical page placement. User space cannot pin placement on Windows; we can only bias it.

## Lessons Learned

1. **Use mmap (the default).** It was never the slowest state and malloc cost up to 17% in the common case. Never benchmark with -mmp 0 unless malloc behavior is the subject.
2. **Benchmark protocol must control memory state.** From now on: default mmap, run twice, report both, and record free RAM + rough page cache state before each CPU run. A 20% "regression" can be pure placement noise.
3. **Mixed-capacity RAM kits have a hidden cost on this workload.** A matched 2x32 GB kit would make all physical memory dual channel (~55 GB/s ceiling everywhere) and remove the lottery: cheap, meaningful upgrade advice for budget builds, worth stating in the final writeups.
4. **Verify what a "failed" control actually did.** The first ballast attempt died silently from PowerShell argument-splitting and the wait loop polled a dead process; the second failed because Windows trimmed the working set below a too-strict threshold. Both were harness bugs, not experiment results.

## Possible Improvements

- Sysinternals RAMMap could directly show which physical ranges back the model pages and confirm the placement story instead of inferring it.
- A read-only STREAM kernel (sum instead of copy) would match decode's traffic pattern better; copy understates read-only bandwidth.
- Repeat the matrix a few more times for distributions rather than point pairs.

## Next Steps

- Adopt the benchmark protocol above for all CPU-side experiments (E013 onward).
- Optional follow-up: test whether a small "placement ballast" at model-load time is a practical trick to reliably win the lottery (allocate, load model, free ballast).
