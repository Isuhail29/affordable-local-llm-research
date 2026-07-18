# E026: Anatomy of ggml_vec_dot_q4_K_q8_K on AVX2 and the short-row cost model

All references are to the local tree `llama.cpp-src` (b10064 plus our chunk_size patch).

## The kernel and how it is driven

The AVX2 Q4_K dot kernel is `ggml_vec_dot_q4_K_q8_K` at `ggml/src/ggml-cpu/arch/x86/quants.c:2038-2214` (AVX2 branch 2057-2120). It asserts `nrc == 1` at line 2040: one weight row, one activation column per call, no multi-row variant on x86.

The MoE driver `ggml_compute_forward_mul_mat_id_one_chunk` at `ggml/src/ggml-cpu/ggml-cpu.c:1463-1524` tiles work as `blck_0 = blck_1 = 16` (lines 1486-1487), accumulates 16 outputs in `float tmp[16]` (line 1489), calls the kernel once per row through the `type_traits_cpu[type].vec_dot` function pointer (lines 1483, 1517) with the last argument hardcoded to literal `1`, then copies the tile out with one memcpy (line 1520). Our patched chunking (chunk_size 16, `ggml-cpu.c:1664`) feeds it 16-row chunks per work-steal grab.

## Per-superblock cost (identical for dense and expert paths)

One Q4_K superblock is 256 weights in 144 bytes (`ggml/src/ggml-common.h:327-338`); the Q8_K activation block is 292 bytes (`ggml-common.h:371-375`), reread from L1 for every row.

Inner loop, 4 iterations of `j` per superblock (`arch/x86/quants.c:2091-2110`), per iteration:

- 5 loads: 2 shuffle-constant loads via `get_scale_shuffle_k4` (32 B table loads, `quants.c:527-539`), 1 q4 load, 2 q8 loads
- 11 vector ops: 2 vpshufb, 2 vpand + 1 vpsrlw (nibble unpack), 2 vpmaddubsw, 2 vpmaddwd, 2 vpaddd

That is roughly 16 uops per 64 weights, ~64 uops per superblock, of which only the 8 madd instructions are the actual dot product. Note the kernel loads 256 B of shuffle constants per superblock against 128 B of weight qs: it moves twice as many constant bytes as weight bytes through the load ports.

Per-superblock header (`quants.c:2066-2087` plus tail 2112-2113), roughly 37-40 uops:

- 2 fp16 conversions and scale mults (2066-2067), via F16C (`simd-mappings.h:58`)
- the 6-bit scale unpack: 12-byte memcpy plus ~10 serial scalar shift/mask ops (2069-2074)
- GPR-to-XMM transfer `_mm_set_epi32` + `cvtepu8` (2079), a serial 3-cycle-latency chain
- bsums/mins correction: load, 2 extracts, vphaddw, vpmaddwd, cvt, broadcast, fmadd (2081-2084)
- scales broadcast (2086-2087) and the superblock fmadd tail (2112-2113)

Total ~100-104 uops per superblock. About 38 percent is per-superblock bookkeeping that does no multiply-accumulate work. This fraction amortizes per superblock, not per row, so it hurts dense and experts equally; it explains why Q4_K extraction in general sits well below the 60 GB/s RAM ceiling, not why experts are worse than dense.

## Per-row cost and the 8-vs-3 superblock comparison

Per call overhead: indirect call and prologue, accumulator init (2061-2062), the final horizontal reduction (2117-2120, ~10 ops forming a ~20-25 cycle serial dependency chain), the tmp store, and driver pointer math (`ggml-cpu.c:1493-1517`). Call it ~30 uops of instructions plus a 20-25 cycle latency tail that overlaps poorly across the call boundary.

- gate/up row, 2048 cols = 8 superblocks: 8x104 + 30 = ~862 uops per 1152 B, overhead ~3.5 percent
- down row, 768 cols = 3 superblocks: 3x104 + 30 = ~342 uops per 432 B, overhead ~8.8 percent
- dense 4096-col row, 16 superblocks: ~1694 uops per 2304 B, overhead ~1.8 percent

Instruction counts alone predict the expert byte mix (gate/up/down carry equal bytes; down is 2048 of 3584 calls per expert-layer) at only ~4 percent worse than dense. The observed gap is ~37 percent (30 vs 47.5). So the deficit is dominated by what instruction counting cannot see: every row starts with the serial scalar scale-unpack chain of its first superblock and ends with the serial horizontal reduction, bracketed by an indirect call. A 3-superblock row body is only ~80-110 cycles at IPC 3-4; if even half of the ~40-50 cycles of boundary latency fails to overlap out-of-order across calls, that is a 20-30 percent loss on down rows versus under 5 percent on dense rows, compounded across 1.38M vec_dot calls per token (48 layers x 8 experts x 3584 rows). This is consistent with E025 (stragglers were not it) and with the MSVC result (codegen quality of exactly these serial chains moves the number).

## Concrete short-row inefficiencies

1. Per-row horizontal reduction with no cross-row accumulation (`quants.c:2117-2120`): pure serial latency paid 3584 times per expert-layer.
2. Indirect call per row (`ggml-cpu.c:1517`) prevents inlining and cross-row software pipelining.
3. Scalar scale unpack on the critical path at every superblock (`quants.c:2069-2079`), unamortized GPR-to-vector transfers.
4. 256 B of shuffle-table loads per superblock (`quants.c:2093-2094` hitting the table at 527-539).
5. tmp buffer plus memcpy per 16-row tile (`ggml-cpu.c:1489, 1520`), minor.

## Multi-row verdict (vec_dot_num_rows)

There is no nrc=2 Q4_K kernel on x86: `arch/x86/quants.c:2040` asserts nrc==1, and `type_traits_cpu[GGML_TYPE_Q4_K].nrows` is 2 only under `__ARM_FEATURE_MATMUL_INT8` (`ggml-cpu.c:310-318`). The only real nrc==2 implementation needs SVE plus i8mm (`arch/arm/quants.c:2334-2374`). Worse, the MoE path could not use one anyway: `mul_mat_id_one_chunk` passes literal 1 (`ggml-cpu.c:1517`), unlike dense which plumbs `num_rows_per_vec_dot` (`ggml-cpu.c:1437-1443, 1242-1243`), and even dense forces it back to 1 at decode because ne11=1 fails the parity check at line 1441.

The real multi-row machinery on x86 is the repack path: Q4_K weights interleaved 8 rows into `block_q4_Kx8` (`ggml/src/ggml-cpu/repack.cpp:3231`, traits at 4536, selected on AVX2 when ne01 % 8 == 0 at 4600-4605; 768 and 2048 both qualify). It supports MUL_MAT_ID (`repack.cpp:4168-4172, 4194-4196`, `forward_mul_mat_id` at 4386-4520) and drives `ggml_gemv_q4_K_8x8_q8_K` (`arch/x86/repack.cpp:1464-1677`): one call covers the thread's whole row slice, 8 rows per iteration, activation loads and bsums shared across 8 rows (1604-1612, 1533-1535), scale unpack amortized 4x (1575-1588), and zero per-row reduction: 8 outputs stay in one register, stored with a single permute (1674-1675).

Why we are not running it: `--n-cpu-moe` pins `ffn_*_exps` to the plain CPU buffer type via `llm_ffn_exps_cpu_override` (`common/common.h:1079-1081`), and repack's `supports_op` requires src0 to live in the repack buft (`repack.cpp:4791-4795`). `-ot` cannot name it either: `parse_tensor_buffer_overrides` (`common/arg.cpp:249-259`) enumerates only device default bufts, so `CPU_REPACK` (`repack.cpp:4746`) is unreachable from the CLI in b10064. A one-line local patch pointing the override at `ggml_backend_cpu_repack_buffer_type()` (declared in `ggml/src/ggml-cpu/repack.h:11`) unlocks it. Costs: repack happens at load via set_tensor (`repack.cpp:4760`), so slower startup and no mmap for those tensors, and the repack mul_mat_id splits rows statically with no work stealing (`repack.cpp:4484-4489`), which E025 suggests is an acceptable risk.

E026 proposal: patch the override, verify the load log shows a CPU_REPACK buffer, and measure decode. This is the highest-leverage test of the kernel-compute-bound theory, far cheaper than hand-writing an nrc=2 AVX2 kernel.
