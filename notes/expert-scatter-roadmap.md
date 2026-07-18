# Expert-read bandwidth roadmap: closing the 34 to 51 GB/s gap

Target: Qwen3-30B-A3B Q4_K_M on llama.cpp b10064, `-ngl 99 --n-cpu-moe 40`, i7-14650HX (8P+8E, AVX2 + AVX-VNNI, no AVX-512, no AMX), 48 GB DDR5-5600, RTX 5060 Laptop 8 GB, Windows 11. Measured: ~34 GB/s effective expert reads vs 50.8 GB/s dense streaming, 55.6 GB/s practical ceiling. Sources: our four internals notes ([ggml MoE CPU path](internals-ggml-moe-cpu.md), [upstream perf survey](internals-moe-perf-issues.md), [KTransformers](internals-ktransformers.md), [layout prior art](internals-expert-layout-art.md)).

## 1. Most likely causes, ranked by evidence

1. **CPU/GPU serialization dilutes the average.** Decode is layer-serial: GPU attention, copy to CPU, CPU expert FFN, copy back, roughly 40 round trips per token via `ggml_backend_sched` splits. RAM is idle during all GPU time and transfer latency, so a kernel streaming at 50 GB/s in its window averages ~35 GB/s over the token. Evidence: structural (by design, no PR fixes it), KTransformers' Expert Deferral gets "up to 1.45x" decode purely from overlapping CPU experts with GPU attention ([SOSP '25 paper](https://dl.acm.org/doi/10.1145/3731569.3764843)), and our 850 MB/token of expert reads at 33 t/s already implies ~28 GB/s from weights alone. Strongest candidate, and the cheapest to confirm (Section 2, experiment 1).
2. **Static thread split on hybrid cores.** At batch size 1 both CPU `mul_mat_id` implementations degenerate to a static equal row split: the generic path in [ggml-cpu.c](https://github.com/ggml-org/llama.cpp/blob/b10064/ggml/src/ggml-cpu/ggml-cpu.c) re-chunks to `nchunk0 = nth` (killing the [#11666](https://github.com/ggml-org/llama.cpp/pull/11666) work-stealing loop), and the repack path in [repack.cpp](https://github.com/ggml-org/llama.cpp/blob/b10064/ggml/src/ggml-cpu/repack.cpp) uses a fixed `(ith*ne01)/nth` band. Every op then finishes at the slowest E-core's pace while P-cores spin. slaren himself noted heterogeneous-core systems benefit most from dynamic scheduling in #11666.
3. **Barrier and serial-section tax.** ~240 full-threadpool spin barriers per token (2 per `mul_mat_id` x 3 ops x 40 layers) plus thread-0-only row grouping, and in the repack path thread-0-only Q8_K activation quantization at bs=1. [PR #20596](https://github.com/ggml-org/llama.cpp/pull/20596) attacks exactly this and reports 4-8% TG on EPYC; the maintainer-adjacent confirmation that this tax is real on `--n-cpu-moe` configs.
4. **Q4_K dequant ALU cost, especially on E-cores.** The Q4_K x Q8_K vec_dot does real compute per byte; an E-core band can turn compute-bound and extract less than its assigned DRAM share. ik_llama.cpp's ~1.9x MoE CPU TG over mainline ([issue #19480](https://github.com/ggml-org/llama.cpp/issues/19480)) via interleaved repack plus fused up+gate kernels shows the kernel headroom is real.
5. **Secondary: paging and repack quirks.** 4 KB file-backed mmap pages (no large pages on Windows), and repack's GEMV was tuned for GEMM with reported TG regressions on some systems ([#12759](https://github.com/ggml-org/llama.cpp/issues/12759)). Worth an A/B, unlikely to be the main gap.

Ruled out for us: NUMA (single socket), PCIe expert transfer (experts compute in RAM), AMX-style tile kernels (hardware absent).

## 2. Experiments runnable today, no compiling

Run each with `llama-bench -n 64` or a fixed prompt in `llama-cli`, and recompute effective expert GB/s (850 MB/token x t/s).

1. **CPU-only isolation (the decision fork).** Same GGUF, `-ngl 0`, best CPU thread config. If effective bandwidth jumps to 45+ GB/s, the gap is mostly cause 1 (serialization) and the lever is overlap and split placement, not kernels. If it stays ~35, causes 2-4 dominate and kernel/threading work pays. Do this first; everything else branches on it.
2. **Thread topology sweep.** Compare `-t 16` (default) vs `-t 8 --cpu-mask 0x5555 --cpu-strict 1` (one thread per physical P-core; verify logical CPU numbering with Task Manager first, HT pairs are usually even/odd on 0-15 with E-cores at 16-23) vs `-t 12`. Add `--poll 100` so threads spin through GPU splits instead of sleeping 40 times per token ([threadpool PR #8672](https://github.com/ggml-org/llama.cpp/pull/8672)). Because of the static split, 8 uniform fast threads can beat 16 mixed ones.
3. **mmap and repack A/B.** Four runs: default, `--no-mmap`, `--no-repack`, both. Check the load log for `CPU_REPACK` buffers to confirm which path you were actually measuring. `--mlock` optional (Windows VirtualLock needs working-set headroom).
4. **Cut expert traffic directly.** `--override-kv qwen3moe.expert_used_count=int:6` is a straight 25% cut in expert bytes; KTransformers ships 6-expert configs and [REAP](https://arxiv.org/pdf/2510.13999) shows Qwen3-30B-A3B tolerates expert reduction. Measure perplexity before adopting.
5. **Rebalance the split.** With flash attention on and KV cache quantized (`-fa -ctk q8_0 -ctv q8_0`), try `--n-cpu-moe 38` or 36; each layer moved to GPU removes ~340 MB/token of RAM reads. Also try `-ot` regexes to pick which layers stay on CPU.
6. **Speculative decoding.** Draft with Qwen3-0.6B; accepted tokens share expert reads (27-40% consecutive-token expert overlap in the literature, see [layout prior art note](internals-expert-layout-art.md)). Win depends on acceptance rate on your workload.
7. **ik_llama.cpp A/B.** The fork ships Windows release binaries; run the same GGUF with `-fmoe -rtr` ([repo](https://github.com/ikawrakow/ik_llama.cpp), [DocShotgun guide](https://gist.github.com/DocShotgun/a02a4c0c0a57e43ff4f038b46ca66ae0)). If it recovers most of the gap, that quantifies the fused-kernel share. Flag: flag behavior changes between releases.
8. **Watch actual DRAM counters.** HWiNFO64 (memory read bandwidth sensor) or Intel VTune's memory-access analysis during decode shows instantaneous vs average bandwidth, directly separating "kernel is slow" from "kernel is idle".

## 3. Experiments needing a source build

**Windows build (one-time setup).** Install Visual Studio 2022 Build Tools (C++ workload), CMake, Ninja, and CUDA Toolkit 12.8 or newer (RTX 5060 is Blackwell, compute 12.0; older toolkits will not target it). Then:

```
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout b10064
cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j 16
```

Use the "x64 Native Tools" prompt so MSVC is found. `GGML_NATIVE=ON` should enable AVX2 + AVX-VNNI; confirm with the flags line at startup. If CUDA setup fights you, a CPU-only build is enough for experiments 1-3 below.

1. **Cherry-pick [PR #20596](https://github.com/ggml-org/llama.cpp/pull/20596).** `git fetch origin pull/20596/head && git cherry-pick <commits>` onto b10064 (may need conflict fixes; the PR may have merged since, in which case just build a newer tag). Expected: single-digit percent from removed barriers and skipped row mapping. Our hybrid-core box is exactly where reviewers lacked data.
2. **Fix the decode-time chunking degeneration.** In `ggml_compute_forward_mul_mat_id` (ggml-cpu.c, ~line 1534), the `chunk_size = 64` re-chunk collapses to a static split whenever `nchunk0 < 4*nth` (768 rows / 64 = 12 chunks < 64). Drop chunk_size to 8 or 16 for the `nr1 == 1` case so the atomic work-stealing loop survives and P-cores absorb E-core stragglers. This is a 5-line experiment and the single most direct test of cause 2. Also try forcing the generic path over repack per-build to compare kernels cleanly.
3. **Parallelize the serial sections.** In repack.cpp `forward_mul_mat_id` (~line 4386), spread the bs=1 Q8_K activation quantization across threads (the generic path already splits by 256-block; repack splits by row and a single row lands on thread 0). Low ceiling (~2048 floats) but it sits inside 120 barriers per token.
4. **Instrument before optimizing further.** Wrap `ggml_barrier` and the mul_mat_id chunk loop with rdtsc timestamps, or set `GGML_SCHED_DEBUG=2` to dump splits; produce a per-token time budget (GPU, copies, CPU kernel, barriers). This turns the cause ranking from inference into measurement.
5. **Prefetch on ids arrival (stretch).** Once the routed ids reach the CPU, issue `_mm_prefetch` sweeps over the 8 expert slabs before the gate matmul begins. Cheap to hack into the row-grouping step; benefit uncertain since hardware prefetchers already handle the sequential inner streams.

Not feasible on this box: AMX tile kernels, AVX-512 paths, NUMA mirroring. KTransformers itself is Linux-only now (kt-kernel README); WSL2 is possible but its AVX2 backend wants 64 GB RAM for the BF16 path, so skip it.

## 4. Upstream contribution opportunities

1. **Hybrid-core benchmark data for #20596.** Reviewers questioned how much barrier cost matters; a Raptor Lake 8P+8E data point on the most popular consumer MoE model is exactly the missing evidence. Lowest-effort real contribution.
2. **Decode-time dynamic chunking PR.** If experiment 3.2 shows gains, a small PR making `mul_mat_id` chunking effective at `nr1 == 1` (smaller chunk size or per-arch tuning, as slaren suggested in [#11666](https://github.com/ggml-org/llama.cpp/pull/11666)) is well-scoped and maintainer-aligned.
3. **Measured numbers for [#19480](https://github.com/ggml-org/llama.cpp/issues/19480).** That issue documents the same 40-60% extraction efficiency with no root-cause statement; our CPU-only vs hybrid isolation plus VTune counters would be the first decomposed measurement in the thread.
4. **Qwen3-30B-A3B routing locality dataset for [#20757](https://github.com/ggml-org/llama.cpp/issues/20757).** No published consecutive-token expert-overlap numbers exist for this model; logging ids per token over real workloads and posting hit-rate curves would ground the two-tier expert cache RFC (and tell us whether hot-expert pinning is worth anything here).
5. **CPU gated-FFN fusion track ([discussion #22315](https://github.com/ggml-org/llama.cpp/discussions/22315), [#22423](https://github.com/ggml-org/llama.cpp/pull/22423)).** Fused MoE FFN is the stated next target and the ik_llama.cpp results bound the prize at up to ~1.9x; testing and benchmarks on AVX2 consumer hardware are welcome even without writing kernels.

Uncertainty notes: line numbers are b10064-specific; #20596 and #20757 status should be rechecked before acting; the 30% GPU-time share behind cause 1 is an estimate that experiment 2.1 exists to replace with a measurement.
