# ggml CPU backend internals: how MoE expert matmuls (mul_mat_id) execute

Scope: llama.cpp tag `b10064`, CPU backend, decode batch size 1, Qwen3-30B-A3B geometry (n_embd 2048, n_ff_exp 768, 128 experts, 8 active, Q4_K). All line references checked against the b10064 tag directly.

## TL;DR

- There are two CPU implementations of `GGML_OP_MUL_MAT_ID`: the generic one in `ggml-cpu.c` and a repacked one in `repack.cpp`. With `--n-cpu-moe`, our Q4_K expert tensors are eligible for the repack path (Q4_K 8x8 interleaved, AVX2), because CPU overrides deliberately consider extra buffer types.
- Reads of an active expert are contiguous. The gather is indirect only at the level of "which 884 KB expert slab", not per row.
- At batch size 1, both paths degenerate to a static equal split of expert rows across all threads with no work stealing. On a hybrid 8P+8E CPU that means every op waits for the slowest E-core.
- Each `mul_mat_id` pays 2 full-threadpool spin barriers (one inside the op, one per graph node), and the row grouping plus (in the repack path) the whole activation quantization runs on thread 0 while the other threads spin. That is roughly 240+ barriers per token for our 40 CPU layers.
- A large share of the 34 vs 55.6 GB/s gap is not inside `mul_mat_id` at all: it is RAM sitting idle while the GPU runs attention and while activations hop GPU->CPU->GPU every layer. The op itself streams well.

## Where the code lives

Generic CPU path, in [ggml/src/ggml-cpu/ggml-cpu.c](https://github.com/ggml-org/llama.cpp/blob/b10064/ggml/src/ggml-cpu/ggml-cpu.c):

- `ggml_compute_forward_mul_mat_id` (lines ~1534-1707): setup, activation quantization, row grouping, per-expert chunk loop.
- `ggml_compute_forward_mul_mat_id_one_chunk` (lines ~1463-1524): inner tiled vec_dot loop.
- `ggml_barrier` (line ~575): spin barrier with seq_cst fences, used after every graph node in `ggml_graph_compute_thread` (barrier at line ~3116).

Repack path, in [ggml/src/ggml-cpu/repack.cpp](https://github.com/ggml-org/llama.cpp/blob/b10064/ggml/src/ggml-cpu/repack.cpp):

- `tensor_traits<...>::forward_mul_mat_id` (line ~4386): its own mul_mat_id with a static row split.
- `repack_q4_K_to_q4_K_8_bl` (line ~3231): converts standard Q4_K rows to `block_q4_Kx8` (8 rows interleaved), from [PR #12332](https://github.com/ggml-org/llama.cpp/pull/12332).
- AVX2 kernels `ggml_gemv_q4_K_8x8_q8_K` / `ggml_gemm_q4_K_8x8_q8_K` in [arch/x86/repack.cpp](https://github.com/ggml-org/llama.cpp/blob/b10064/ggml/src/ggml-cpu/arch/x86/repack.cpp) (gemv at line ~1464).

Which path runs: `llama_model_loader::create_tensor` in [src/llama-model-loader.cpp](https://github.com/ggml-org/llama.cpp/blob/b10064/src/llama-model-loader.cpp) (lines ~1165-1197). When a tensor override targets the CPU buffer (which is exactly what `--n-cpu-moe` does), it calls `select_weight_buft` with the CPU buft list, which is ordered ACCEL -> GPU host -> **CPU extra (repack)** -> plain CPU (`make_cpu_buft_list` in [src/llama-model.cpp](https://github.com/ggml-org/llama.cpp/blob/b10064/src/llama-model.cpp), line ~884). So on an AVX2 machine with `ne01 % 8 == 0` (768 and 2048 both qualify), the expert tensors land in the `CPU_REPACK` buffer unless repacking is disabled. The loader even logs a warning that CPU overrides plus mmap should consider `--no-mmap`, because repacked tensors cannot be mmapped and get copied at load. Check the load log for which buffer the exps tensors went to; everything below covers both paths.

The graph side (`llm_graph_context::build_moe_ffn` in [src/llama-graph.cpp](https://github.com/ggml-org/llama.cpp/blob/b10064/src/llama-graph.cpp), line ~1799) emits per layer: gating matmul, top-k select (`ggml_argsort_top_k`), `ggml_get_rows` for weights, then **three `mul_mat_id` calls** (up, gate, down) for Qwen3 (a single merged gate_up `mul_mat_id` exists only for models that ship a fused tensor).

## Storage of fused expert tensors

Since [PR #6505](https://github.com/ggml-org/llama.cpp/pull/6505) all experts of a layer live in one 3D tensor, e.g. `blk.N.ffn_gate_exps.weight` with shape [2048, 768, 128]. In memory that is 128 back-to-back 2D expert matrices; expert `e` starts at `src0->data + e*nb02`. For Q4_K, one gate/up row is 2048/256 super-blocks x 144 bytes = 1152 bytes, so one expert slab is 768 x 1152 = 884 KB (down: 2048 x 432 = same 884 KB). Per token, 8 experts x 3 projections x 40 CPU layers = about 850 MB of weight reads, which at your 33 t/s is about 28 GB/s of expert traffic alone. Consistent with your 34 GB/s measurement.

## Execution at batch size 1

1. **Activation quantization.** src1 (F32) is quantized to the vec_dot type of Q4_K, which is Q8_K. Generic path: all threads split the single row by 256-wide blocks (2048 elements = 8 blocks, so with 16 threads half of them do nothing). Repack path: rows are distributed by `i11 % nth`, and at bs=1 the gate/up input has ne11=1, so **thread 0 quantizes alone while 15 threads head to the barrier**.
2. **Row grouping (the gather).** Thread 0 alone reads the `ids` tensor (top-k output) and builds `matrix_row_counts[n_as]` plus `matrix_rows` (`mmid_row_mapping {i1, i2}`), bucketing every (expert_slot, token) pair by expert id. At bs=1 that is 8 entries. Then one `ggml_barrier`.
3. **Per-expert loop.** All threads sweep `cur_a = 0..127`, skipping the 120 experts with zero rows. There is no barrier between experts, so threads drift through the list independently.
4. **Thread split.** Generic path: nr0 = 768 rows, nr1 = 1, chunk_size = 64, so nchunk0 = 12. Since 12 < 4*nth, the chunking added in [PR #11666](https://github.com/ggml-org/llama.cpp/commit/a394039db004c6ee00098250d160b5aa018c2314) is abandoned and it re-chunks to nchunk0 = nth: a **static split of 48 rows per thread (16 threads), and the work-stealing `atomic_fetch_add` loop breaks immediately**. Same for down (2048 rows -> nchunk0 = 32 < 64). Repack path: always a static `(ith*ne01)/nth` band, aligned to 8 rows, with zero stealing by construction. So at decode time neither path load-balances across P and E cores.
5. **Inner kernel.** Generic: `vec_dot` (`ggml_vec_dot_q4_K_q8_K`) one row at a time in 16-row tiles (the 2-row `num_rows_per_vec_dot` trick of `mul_mat` is absent here, though on x86 Q4_K nrows is 1 anyway). Repack: `ggml_gemv_q4_K_8x8_q8_K` over the thread's band; nibble unpack + `maddubs` integer dot, never materializing floats.

## Are expert reads contiguous?

Yes. Each thread reads one contiguous band (about 55 KB with 16 threads) inside the contiguous 884 KB expert slab, and the Q4_Kx8 interleaved format is also strictly sequential. Sixteen sequential streams into one slab is a friendly DRAM pattern; hardware prefetchers handle it. Cache reuse is zero by nature (each byte used once per token), so this op is pure bandwidth plus dequant ALU. TLB: with mmap on Windows these are 4 KB file-backed pages and no large pages, so there is a dTLB miss every page, but sequential access keeps page walks cheap; this is a secondary effect, not the main gap.

## Known inefficiencies (ranked for our setup)

1. **CPU/GPU serialization, not the kernel.** With `-ngl 99 --n-cpu-moe 40` ([PR #15077](https://github.com/ggml-org/llama.cpp/issues/15263)), every layer does GPU attention -> copy hidden state + ids to CPU -> CPU expert FFN -> copy back. Decode is algorithmically serial (layer n attention needs layer n-1 FFN), so RAM is idle during all GPU time and all transfer latency. If GPU plus sync is about 30 percent of token time, a kernel streaming at 50 GB/s shows up as about 35 GB/s average. This likely explains most of the measured gap.
2. **Static split plus hybrid cores.** Every op finishes at the pace of the slowest E-core band; P-cores then spin in `ggml_barrier`. No stealing at bs=1 (see above).
3. **Barrier count.** 2 spin barriers per mul_mat_id x 3 ops x 40 layers = 240 full 16-thread rendezvous per token, plus barriers for the CPU-resident gating/glu ops. Each is a seq_cst fence plus spin; on hybrid cores wakeup jitter adds up (order of microseconds each).
4. **Serial sections.** Row grouping always on thread 0; in the repack path at bs=1 the gate/up Q8_K quantization too.
5. **Q4_K dequant ALU cost.** The in-kernel nibble+scale decode is heavier than dense streaming; on E-cores (narrower AVX2 throughput) a band can turn compute-bound, lowering per-core extraction below the DRAM share it was assigned. This is why a pure read benchmark hits 50.8 GB/s but the GEMV does not.
6. **Repack quirks.** Repack was tuned for GEMM (prompt processing); [issue #12759](https://github.com/ggml-org/llama.cpp/issues/12759) reports slow load-time repacking and token-generation regressions on some systems, and repack forfeits mmap for those tensors. Whether 8x8 GEMV beats the generic vec_dot on Raptor Lake is empirical, so test both. Minor extra: in `forward_mul_mat_id` a thread whose aligned band is empty does `return` and skips all remaining experts (harmless at our dims, 768/16 = 48 rows).

## What to try

- **Verify which path runs**: look for `CPU_REPACK` buffers and the "tensor overrides to CPU are used with mmap enabled" warning in the load log. A/B: default vs `--no-repack`, and mmap vs `--no-mmap`.
- **Thread topology sweep**: `-t 8` (P-cores only) vs 12 vs 16, with `--cpu-mask`/`--cpu-strict 1` to pin one thread per physical P-core. Because of the static split, 8 fast uniform threads can beat 16 mixed ones; also try `--poll 100` to cut barrier latency.
- **Isolate the serialization share**: run the same GGUF CPU-only (`-ngl 0`) with `llama-bench -n 64`; compute effective expert GB/s. If CPU-only extraction jumps to 45+ GB/s, the hybrid schedule, not the kernel, is the bottleneck, and the lever is shrinking GPU/sync time (fewer CPU MoE layers, smaller KV, flash attention on).
- **Rebalance --n-cpu-moe**: each layer moved fully to GPU removes about 340 MB of expert reads from RAM and about 21 ms/s of CPU work; with 8 GB VRAM even 2-3 more layers is a few percent.
- **Track upstream / forks**: [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) fuses the MoE up+gate ops and has faster K-quant CPU GEMV; community offload guides ([Doctor-Shotgun's guide](https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide)) match this analysis.

Uncertainty notes: line numbers and behavior are from tag b10064 and move fast in this codebase; whether repack engages on your exact binary depends on build flags (`GGML_CPU_REPACK`) and the `--no-repack` default of your release; the 30 percent GPU-time estimate is an assumption to be measured, not a measurement.
