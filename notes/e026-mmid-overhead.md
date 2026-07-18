# E026: Per-token overhead accounting in the CPU expert path (mul_mat_id)

Question: beyond the Q4_K dot products themselves, how much fixed per-token cost does the CPU expert path carry, and can it explain a meaningful part of the 47.5-vs-30 GB/s-equivalent gap? All references are to the local tree (llama.cpp b10064 plus the chunk_size 16 patch).

## 1. What happens inside one mul_mat_id node at batch 1

File: `ggml/src/ggml-cpu/ggml-cpu.c`, `ggml_compute_forward_mul_mat_id` (lines 1534-1707).

**src1 to Q8_K quantization (lines 1585-1620, active branch 1606-1619).** All threads run the loop; each row of src1 is split along its Q8_K superblocks (`bs = 256`, lines 1610-1612) across `nth = 16` threads.
- gate/up node: src1 is the hidden state reshaped to [2048, 1, 1]. 2048 floats = 8 superblocks, so exactly 8 of 16 threads quantize one 256-float block each; the other 8 do nothing. Output is 8 x 292 B = 2336 B of wdata. Note the same 2048-float vector is re-quantized twice per layer (once in the up node, once in the gate node).
- down node: src1 is the swiglu output [768, 8, 1]. 768 cols = only 3 superblocks per row, so per row just 3 of 16 threads do work (with the `(ith*3)/16` split, threads 5, 10, 15). 24 blocks total, 7008 B of wdata.
Total conversion per token: (2048 + 2048 + 6144) x 48 = 491,520 floats. Even poorly parallelized this is well under 0.2 ms.

**wdata bookkeeping (lines 1568-1583, plan at 2856-2874).** Per node: `matrix_row_counts` 128 x 8 B, `matrix_rows` 128 x 8 x 8 B = 8 KB, and 128 cache-line chunk counters = 8 KB. About 19-24 KB touched per node, 144 nodes per token = ~3 MB of bookkeeping writes. Negligible bandwidth.

**ids grouping pass, single-threaded (lines 1622-1637).** `ith == 0` memsets 1 KB of counts and walks `ids` (8 entries at batch 1) writing `mmid_row_mapping`. Sub-microsecond. It overlaps with the other threads' quantization slices, and everyone then meets at the internal `ggml_barrier` (line 1645). Cost is bounded by the barrier itself, not the serial work.

**Chunk counter reset (lines 1640-1643).** Each thread resets 8 of the 128 per-expert atomic counters. Trivial.

**Expert loop and gather (lines 1647-1706).** The loop visits all 128 experts, skipping the 120 with `cne1 == 0` (line 1650). For each active expert, `one_chunk` (lines 1463-1524) looks up the row mapping via `MMID_MATRIX_ROW` (line 1496), computes offsets (1509-1514), calls `vec_dot` per output row into `tmp[16]` (1516-1518), then memcpys to dst (1520). The gather itself is a few instructions per 16 rows.

**Chunking surprise.** With chunk_size 16 (line 1664): gate/up have nr0 = 768 rows, so nchunk0 = 48, and 48 < nth*4 = 64, so the fallback at lines 1672-1675 fires: 16 static chunks of 48 rows, one per thread, and the `nth >= nchunk0*nchunk1` break (line 1700) disables stealing entirely. Only the down node (nr0 = 2048, 128 chunks) gets live work stealing. The patch comment (lines 1661-1663) claims nchunk0 >= 4*nth at single-token matvecs; that holds for down only. Consistent with E025: stealing barely mattered.

## 2. Graph nodes per MoE layer at decode

Qwen3MoE FFN is built in `src/models/qwen3moe.cpp` (build_moe_ffn call, lines 136-151: SILU, norm_w = true, softmax gating) with the body in `src/llama-graph.cpp::build_moe_ffn` (lines 1799-2148). Executed compute nodes at batch 1 (reshape/view nodes are skipped without a barrier, `ggml-cpu.c:3091-3094`, `ggml-impl.h:90-101`):

1. router mul_mat (llama-graph.cpp:1831), 2048x128
2. soft_max (1849)
3. argsort for top-k (1915; `ggml.c:5346-5360`, full 128-way std::sort of one row, single-threaded, `ops.cpp:8360-8381`)
4. get_rows for weights (1929), forced n_tasks = 1 (`ggml-cpu.c:2336-2343`)
5. sum_rows (1943), 6. clamp (1947), 7. div (1950)
8. mul_mat_id up (1996), 9. mul_mat_id gate (2009)
10. swiglu_split (2055)
11. mul_mat_id down (2098)
12. mul by weights (2111)
13-19. seven adds aggregating the 8 expert views (2123-2138)

That is 19 compute nodes per layer, of which only 3 carry real work. CPU op fusion covers only RMS_NORM+MUL (`ggml-cpu.c:3026-3058`), so nothing here fuses.

## 3. Barriers

The graph loop places a full 16-thread barrier after every compute node (`ggml-cpu.c:3115-3117`). mul_mat_id has one internal barrier (1645) and plain mul_mat one (1364, unconditional). `ggml_barrier` (575-611) is a pure spin: seq_cst fetch_add on entry (587), seq_cst increment by the last arrival (594), and everyone else spins on `_mm_pause` (599-601, 524-526). No yield, no futex. On Raptor Lake with 16 threads across P and E clusters a reasonable per-barrier cost is 1-3 us (one hot cache line bouncing through two coherency domains).

Per MoE layer: 19 end-of-node + 4 internal (3 mm_id + router) = 23 barriers. Per token: 23 x 48 = 1104 barriers in the FFN path alone. A dense-FFN-equivalent graph would need about 7 (3 matmuls + swiglu + internals), so the MoE structure adds roughly 16 x 48 = 768 extra barriers per token.

## 4. The arithmetic vs the 30 ms budget

Fixed costs per token, central estimates (pessimistic bound in parens):

- 1104 MoE-FFN barriers x 1.5 us = 1.7 ms (3.3 ms at 3 us)
- serial small-node work (softmax, argsort ~1-2 us, get_rows, adds): ~0.2 ms
- Q8_K quantization: ~0.1-0.2 ms
- ids grouping, counter resets, expert-loop skips, gather: < 0.1 ms

Total ~2 ms, upper bound ~4 ms, i.e. 7-13 percent of the 30 ms token.

Gap framing: active expert weights are ~21.2 MB x 48 = ~1.02 GB per token. At the dense path's 47.5 GB/s equivalent that is 21.5 ms; at the observed 30 GB/s it is ~34 ms. The gap is ~12.5 ms. Fixed overhead of ~2 ms (max ~4) explains roughly 15-20 percent of it, up to a third under pessimistic barrier costs. It cannot explain the majority.

Where the rest plausibly lives: inside `vec_dot` itself. Per token the expert path makes 28,672 vec_dot calls per layer (768 x 8 x 2 gate/up + 2048 x 8 down) = 1.38 M calls. Down rows are only 3 Q4_K superblocks (768 cols), so the kernel prologue/epilogue (scale unpack, accumulator setup, horizontal sum, function-pointer call, tmp store) is amortized over 3 iterations instead of the 8+ a dense 2048-col row gets. At 10-20 ns per call that is another 0.9-1.7 ms wall, and it scales with work, which is exactly the kernel-compute signature the MSVC-vs-Clang 10 percent delta points at.

## 5. Verdicts

- Fixed per-node overhead plus the ith==0 serial sections are real but secondary: ~2 ms of a ~12.5 ms gap.
- The dominant per-node fixed cost is the barrier count (23 per MoE layer), not the serial compute inside nodes.
- The ids grouping pass is sub-microsecond and hides behind the internal barrier; not a target.
- Actionable oddities for E026+: gate/up mul_mat_id silently loses work stealing at batch 1 (fallback at ggml-cpu.c:1672-1675); the activation vector is quantized to Q8_K twice (up and gate nodes); the 7-add aggregation plus router/softmax/topk chain costs ~16 barriers a layer that a fused MoE-FFN op would eliminate.
- Testable: dropping to 8 threads (P-cores only) should shrink barrier cost superlinearly; if tokens/s barely moves at equal kernel throughput, the barrier bill is confirmed small, reinforcing the kernel-compute conclusion.
