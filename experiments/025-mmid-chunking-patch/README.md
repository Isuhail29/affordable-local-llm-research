# Experiment 025: mul_mat_id Chunking Patch (first source-level experiment)

Date: 2026-07-19
Status: In progress (awaiting toolchain)

## Goal

Test whether restoring work-stealing in llama.cpp's CPU expert-matmul path recovers part of the expert-read bandwidth gap proven in E021 (~30 GB/s extracted vs ~60 GB/s machine ceiling).

## Background

Reading ggml/src/ggml-cpu/ggml-cpu.c (b10064), ggml_compute_forward_mul_mat_id splits each active expert's matvec into chunks. At single-token decode (nr1 = 1) the chunk size is bumped to 64 rows, which for our 768-row gate/up expert matrices yields only 12 chunks. The guard `nchunk0 * nchunk1 < nth * 4` then rejects chunking and falls back to one static slice per thread, and the stealing loop exits immediately (`nth >= nchunk0 * nchunk1`). Consequence: every one of the ~144 expert-matmul graph nodes per token (3 matmuls x 48 layers) runs at the pace of the SLOWEST of 12 threads. On a hybrid 8P+8E CPU with OS-scheduled (unpinned) threads, E-core stragglers stall each node barrier.

This explains two E021 observations: why 12 mixed threads beat 8 pinned P-cores (more workers shrink each static slice, softening the straggler), and why pinning could not fix it (pinning reduced worker count instead of fixing the split).

The patch (patches/e025-mmid-chunk16.patch, 5 lines): remove the single-token chunk-size bump so expert matvecs always chunk at 16 rows. Gate/up: 48 chunks (passes the nth*4=48 guard exactly at 12 threads); down: 128 chunks. Work-stealing survives; fast threads absorb straggler slack. Prefill (nr1 large) and the dense mul_mat path are untouched; small-expert models (nr0 under ~64x threads/4) fall back to old behavior.

## Hypothesis

1. Patched decode at -ncmoe 48 -t 12 gains 10-30% (26.3 to 29-34 t/s). Below 5% falsifies the straggler theory and points the residual gap at dequant kernel efficiency on small work units.
2. The patched build's thread curve shifts: 16 threads should now HELP (stealing turns stragglers into contributors), possibly beating 12.
3. Control: dense 8B CPU decode unchanged within noise (its code path is untouched).
4. Added atomics cost is negligible (~55k relaxed fetch_adds/token, well under 1% of token budget).

## Implementation

- Source: llama.cpp b10064 clone (llama.cpp-src/), patch applied, otherwise pristine
- Build: cmake + VS Build Tools 2022 + CUDA toolkit, -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120, Release, targets llama-bench/llama-server/llama-cli (scripts/build-patched.ps1)
- Benchmark: E021 strict protocol (warm cache, -p 0 -n 128 -r 3): stock vs patched at ncmoe 48 x threads {8, 12, 16, 20}, plus dense 8B control, plus ncmoe 40 chat config
- Same binaries directory layout so only the build differs

## Expected Result

If the straggler theory is right: patched ncmoe 48 around 30-34 t/s at t12-16, and the chat config (ncmoe 40 turbo) proportionally faster. Honest uncertainty: the E021 isolation test capped pure-CPU at ~34 GB/s too, which a straggler fix alone may not fully explain.

## Actual Result

Same-toolchain A/B (MSVC 19.44, Ninja, AVX2 native build; 30B pure CPU -ngl 0, tg128, warm cache, r=3):

| Threads | Stock | Patched |
|---|---|---|
| 12 | 17.10 ± 1.29 | 16.32 ± 3.50 |
| 16 | 17.84 ± 0.63 | 18.74 ± 0.34 |
| 20 | 15.32 ± 0.51 | 15.00 ± 0.70 |

## Benchmark analysis

**Hypothesis 1 falsified by its own pre-registered threshold.** The patch gains at most +5% (t16) and nothing at other thread counts. Restoring work-stealing does not recover the expert-path bandwidth gap; node-barrier stragglers were not the bottleneck. H2 weakly supported (patched optimum moved to 16 threads, tighter variance) but the effect is marginal. H3/H4 moot.

**The control arm produced the real finding.** Our stock rebuild (17.1 t/s at t12) is ~10% slower than the official b10064 binary on the identical code path (19.0), differing only in compiler (MSVC vs Clang). A purely DRAM-bandwidth-bound workload is compiler-insensitive; a 10% compiler effect means the expert path is COMPUTE-bound in its kernels, not memory-bound. The "34 GB/s extraction" is therefore not a memory-system limit at all: it is dequant/vec_dot ALU throughput at matvec granularity.

**Source re-inspection under that lens found the likely mechanism:** the dense mul_mat path can route through llamafile_sgemm (optimized tiled kernels); ggml_compute_forward_mul_mat_id has no sgemm path and always walks the plain vec_dot loop. Dense weights get the fast engine, experts get the naive loop. This coherently explains E021's dense-vs-expert efficiency split (47.5 vs ~30 GB/s equivalent) without invoking memory behavior at all.

## Lessons Learned

1. Same-toolchain A/B was essential: comparing our patched MSVC build against the official Clang binary would have misread the patch as -14%.
2. A falsified pre-registered hypothesis with a clean control is real progress: two candidate mechanisms (scatter physics in E021, barrier stragglers here) are now eliminated, and the compiler-sensitivity observation localizes the cost to kernel compute.
3. Keep negative-result patches in the repo (patches/e025-mmid-chunk16.patch): they document the eliminated branch and the +5%/t16 residual is harmless to keep in our private build.

## Possible Improvements

- Confirm compute-boundedness directly: perf-counter or IPC sampling during expert decode, or a vec_dot microbenchmark at 768x2048 matvec shape vs long-row dense shape.
- Build with clang-cl to erase the compiler deficit and re-baseline.

## Next Steps

- E026 candidate (the promising one): give mul_mat_id an sgemm/tiled path per active expert, or batch the 8 active experts' matvecs to reach sgemm-friendly shapes. Larger change; requires careful design against ggml conventions.
- Complete the CUDA build of the current tree once the CUDA toolkit is installed, so server-config (ncmoe) A/Bs become possible.
