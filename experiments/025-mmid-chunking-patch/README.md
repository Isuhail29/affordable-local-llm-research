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

(pending build)

## Benchmark

(pending)

## Lessons Learned

(pending)

## Possible Improvements

(pending)

## Next Steps

(pending)
