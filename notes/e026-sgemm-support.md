# E026: llamafile sgemm (tinyBLAS) support audit

Question: is llamafile_sgemm a usable lever for the Q4_K expert path on our AVX2 machine?
Answer: no. It does not support Q4_K, it refuses n=1 by design, and mul_mat_id never calls
it at all. All line numbers below are from the local tree at llama.cpp-src (b10064 plus our
chunk_size patch), read directly, not from upstream memory.

## 1. Type support on x86 AVX2

The dispatch is one big switch on Atype in `llamafile_sgemm`,
ggml/src/ggml-cpu/llamafile/sgemm.cpp:3699-4058. Cases that exist:

- `GGML_TYPE_F32` (line 3723): requires Btype F32. AVX2 kernel at 3732-3737.
- `GGML_TYPE_BF16` (line 3787): on plain AVX2 requires Btype BF16 (3804-3811).
- `GGML_TYPE_F16` (line 3851): on AVX2+F16C requires Btype F16 (3860-3867).
- `GGML_TYPE_Q8_0` (line 3935): requires Btype Q8_0 (3936-3937). AVX kernel
  `tinyBLAS_Q0_AVX<block_q8_0, block_q8_0, float>` at 3938-3945.
- `GGML_TYPE_Q4_0` (line 3972), `GGML_TYPE_Q5_0` (line 4009), `GGML_TYPE_IQ4_NL`
  (line 4025): all require Btype Q8_0, all served by `tinyBLAS_Q0_AVX`.
- `default:` return false (lines 4041-4042).

So on AVX2: F32, F16, BF16, Q8_0, Q4_0, Q5_0, IQ4_NL. That is the complete list.

**Q4_K is not supported.** There is no `GGML_TYPE_Q4_K` case anywhere in the switch, and no
K-quant kernel exists in the file (the quantized class `tinyBLAS_Q0_AVX`, sgemm.cpp:1351-1353,
is templated only over Q0-style blocks with a `d` scale and 32-value `qs`). Q8_K, the
vec_dot_type of Q4_K (ggml-cpu.c:310-318), is likewise not accepted as a Btype anywhere.
Q8_0 IS supported, but only when both A and B are Q8_0.

Unsupported types are handled by returning false. There is no fallback inside sgemm; the
comment at sgemm.cpp:3673-3675 says it plainly: work is only performed when a handwritten
kernel exists, otherwise the caller must fall back. Two additional universal guards: Ctype
must be F32 (3718-3719), and asserts require lda/ldb >= k (3706-3707).

## 2. When the dense path invokes sgemm

Both call sites live in `ggml_compute_forward_mul_mat`, ggml/src/ggml-cpu/ggml-cpu.c, inside
`#if GGML_USE_LLAMAFILE`:

- **Site 1** (lines 1295-1320): fires only when src1 is contiguous (`src1_cont`, lines
  1300-1302), before any activation quantization. Passes `src0->type, src1->type` (src1 is
  F32 here). If any 2D slice returns false it jumps to `UseGgmlGemm1` and the whole op goes
  down the generic path. For our Q4_K weights, Atype=Q4_K hits `default` immediately, so
  this site never services K-quants.
- **Site 2** (lines 1366-1388): fires only when `src1->type != vec_dot_type`, after src1 has
  been quantized into wdata (lines 1322-1357) and after the barrier at 1364. Passes
  `src0->type, vec_dot_type`. This is the site that serves Q4_0/Q5_0/Q8_0/IQ4_NL weights
  (vec_dot_type Q8_0). For Q4_K, vec_dot_type is Q8_K and Atype is Q4_K, so it returns false
  and control drops into the vec_dot chunk loop at 1390-1451.

**Batch-size guard: sgemm never fires at n=1 on x86.** sgemm.cpp:3712-3716:

    // only enable sgemm for prompt processing
    #if !defined(__MMA__)
        if (n < 2)
            return false;
    #endif

`__MMA__` is POWER10 only. On our machine every single-token decode matvec (n = ne11 = 1)
is rejected before the type switch is even reached. sgemm is prefill-only by explicit
design. So during token generation the dense path always uses the vec_dot chunk loop, and
during prefill it uses sgemm only for the type combinations listed above. For our Q4_K
model, sgemm is never used at any stage on the FFN/attention weight matmuls; the only
candidates are pure F32xF32 ops, which in practice are not the hot matmuls.

## 3. Could mul_mat_id use sgemm per expert at decode?

**It currently cannot: there is no call.** `ggml_compute_forward_mul_mat_id`
(ggml-cpu.c:1534-1707) and its worker `ggml_compute_forward_mul_mat_id_one_chunk`
(1463-1524) contain no `GGML_USE_LLAMAFILE` block. Grep confirms `llamafile_sgemm` appears
only at lines 1305 and 1373, both in the dense function. Expert matmuls always take the
vec_dot loop at 1516-1518, at decode and at prefill alike.

**Would sgemm accept a per-expert (m=768 or 2048, k, n=1) call if we added one?** No, twice
over: the n<2 guard at 3714-3715 rejects n=1, and Q4_K/Q8_K hit the default case. To get
any kernel to run you would have to delete the guard AND repack experts to Q4_0/Q8_0 with
Q8_0 activations.

**And if we did all that, would it be faster than the vec_dot chunk loop?** Expect equal at
best, likely slower:

- At n=1 tinyBLAS has no advantage to offer. Its whole win is register-tiled outer products
  that amortize each A load across several B columns (the 4xN kernels at sgemm.cpp:1524-1574).
  With one B column there is nothing to amortize; the n=1 tile shapes (cases 0x41/0x31/0x21/
  0x11, lines 1464-1511) degenerate into 4-rows-at-a-time dot products using the same AVX2
  instruction mix (`_mm256_sign_epi8` + `updot` + `hsum`) as `ggml_vec_dot_q4_0_q8_0`. The
  guard comment "only enable sgemm for prompt processing" is the authors saying exactly this.
- Work division is worse for us. tinyBLAS splits tiles statically per thread
  (`duty = (tiles + nth - 1) / nth`, sgemm.cpp:1527-1532) with no atomic work-stealing.
  On the 8P+8E hybrid that reintroduces the straggler exposure our chunked mul_mat_id loop
  (ggml-cpu.c:1664-1705) exists to bound. E025 showed stragglers are not the current
  bottleneck, but a static split can only be worse, never better.
- Repacking to Q4_0 forfeits Q4_K quality for zero expected speed, and our decisive E-series
  clue says the expert path is kernel-compute-bound; the lever is per-superblock vec_dot
  efficiency on 3-superblock rows (768 cols = 3 Q4_K superblocks), which sgemm does not touch.

The one real gap sgemm-style tiling could address is expert PREFILL (many tokens per expert,
n >> 1, still going through vec_dot), but that would require writing a Q4_K/Q8_K tinyBLAS
kernel that does not exist today.

## 4. Is GGML_USE_LLAMAFILE on in this build?

Yes, on by default. Top-level CMakeLists.txt:140-142 sets `GGML_LLAMAFILE_DEFAULT ON`
whenever the user did not define `GGML_LLAMAFILE`; ggml/CMakeLists.txt:197 makes the option
default to that; ggml/src/ggml-cpu/CMakeLists.txt:80-86 then defines `GGML_USE_LLAMAFILE`
and compiles llamafile/sgemm.cpp into the CPU backend. The only place it is force-disabled
is ggml-cpu.c:45-47 for ARM SVE / MATMUL_INT8, irrelevant on x86. Runtime confirmation:
`ggml_cpu_has_llamafile` (ggml-cpu.c:3722-3728) drives the `LLAMAFILE = 1` field in the
system_info line both our MSVC rebuild and the official binary print.

## Verdict

sgemm is a dead end for E026's decode question: wrong types (no Q4_K/Q8_K), hard n<2 gate,
and no call site in mul_mat_id. It neither explains nor can it fix the 30 vs 47.5 GB/s
expert-path gap. The compute-bound clue keeps pointing at the Q4_K vec_dot kernel itself on
short 3-superblock rows, not at missing GEMM tiling.
