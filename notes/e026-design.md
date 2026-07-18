# E026 design: experiment ladder for the CPU expert-path kernel deficit

Goal: recover the ~37 percent gap (30 vs 47.5 GB/s equivalent) in the CPU expert path, given that E025 killed the straggler theory and the MSVC-vs-Clang delta says the path is kernel-compute-bound. Ladder is ordered by information gained per unit effort. All references are to the local tree llama.cpp-src (b10064 plus the chunk_size patch), read directly.

## A correction that reorders everything

The vecdot-anatomy doc concluded that repack (`block_q4_Kx8` plus `ggml_gemv_q4_K_8x8_q8_K`) is unreachable under `--n-cpu-moe` without a local patch to `llm_ffn_exps_cpu_override` (common/common.h:1079-1081). That is wrong for this tree. `llama_model_loader::create_tensor` special-cases CPU overrides: when `overrides->buft == ggml_backend_cpu_buffer_type()` it re-selects via `select_weight_buft` over `buft_list_cpu` (src/llama-model-loader.cpp:1164-1181, remap at 1170-1172), and that list puts extra bufts ahead of plain CPU (`make_cpu_buft_list`, src/llama-model.cpp:884-932, extra bufts added at 915-932). Extra bufts are on by default (`use_extra_bufts` default true, src/llama-model.cpp:2329; `no_extra_bufts` default false, common/common.h:584; flag `-nr/--no-repack`, common/arg.cpp:2220-2227). Repack accepts our exact case: MUL_MAT_ID, 3D src0, F32 src1 (repack.cpp:4791-4806), Q4_K on AVX2 with ne[1] % 8 == 0 (repack.cpp:4600-4605; 768 and 2048 both qualify), verified through the dummy-buffer trick in `weight_buft_supported` (llama-model-loader.cpp:1030-1035).

So it is possible the E-series numbers were measured WITH repack already live, in which case the vec_dot short-row anatomy describes a kernel that is not even running. Or something blocks selection and the anatomy stands. This is a factual question that costs minutes to answer, so it is rung 1.

## The ladder

### Rung 1: ground-truth the live kernel, then A/B repack (do this first)

Change or measure: zero code. Run the standard decode benchmark twice, `--repack` vs `--no-repack` (or env `LLAMA_ARG_REPACK`, common/arg.cpp:2227), and read the load log for a `CPU_REPACK` buffer line (name from repack.cpp:4745-4749). Add `--no-mmap` on the repack run since repack loads via set_tensor anyway (repack.cpp:4751-4765; warning at llama-model-loader.cpp:1173-1177).

Predicted outcome: if prior runs had repack off, turning it on lifts expert-path throughput from ~30 toward 40+ GB/s equivalent (gemv amortizes every boundary cost the anatomy doc lists: 8 rows per pass, shared activation loads and bsums, 4x scale-unpack amortization, no per-row horizontal reduction, arch/x86/repack.cpp:1464-1677). Falsification threshold: with CPU_REPACK confirmed in the log, a decode gain under 8 percent kills the short-row boundary-cost theory as the dominant deficit driver. Opposite branch: if the log shows repack was already live, then `-nr` should cost at least 10 percent; if it changes nothing, the traits dispatch (ggml-cpu.cpp:435-436) is not firing and that bug becomes the experiment.

Risk: low. Slower load (no mmap for ~16 GB of exps), ~16 GB resident instead of mapped, and the repack mul_mat_id uses a static row split with no work stealing (repack.cpp:4484-4489); E025 says stragglers cost at most 5 percent, acceptable.

### Rung 2: thread-scaling probe to bound the barrier bill

Change or measure: zero code. Same benchmark at 16 threads vs 8 P-core threads (and one 12-thread point). From the mmid-overhead doc: 1104 FFN barriers per token, estimated 1.7-3.3 ms of a 30 ms token.

Predicted outcome: 8 P-threads reach 90-97 percent of 16-thread tokens/s. Falsification thresholds: if 8 threads matches or beats 16, cross-cluster barrier cost is at the pessimistic end and rung E-impl gets revived; if 8 threads loses more than 15 percent, compute still scales with cores and the barrier bill is confirmed small.

Risk: none.

### Rung 3: sampling profile of steady-state decode

Change or measure: 30 s VTune or WPA sample during decode, whichever binary rung 1 says is faster. Attribute cycles to `ggml_vec_dot_q4_K_q8_K` (arch/x86/quants.c:2038-2214) or `ggml_gemv_q4_K_8x8_q8_K` (arch/x86/repack.cpp:1464-1677), `ggml_barrier` (ggml-cpu.c:575-611), and quantize_row_q8_K.

Predicted outcome: one dot/gemv symbol holds 70+ percent of cycles, confirming kernel-compute-bound and pinpointing intra-kernel hotspots (scale unpack vs madds vs reduction) before any kernel is hand-written. Falsification: if spin time in `ggml_barrier` exceeds ~15 percent, the overhead ledger in the mmid doc is wrong and E-impl moves up.

Risk: none beyond tool setup.

### Rung 4 (conditional): kernel work, shaped by rungs 1-3

Only if repack is on and still ~30 GB/s: optimize `ggml_gemv_q4_K_8x8_q8_K` guided by the rung 3 profile. Only if repack proves unusable: revive rung C as a hand-written multi-row AVX2 Q4_K vec_dot plus plumbing (the MoE driver hardcodes nrc 1 at ggml-cpu.c:1517; x86 kernel asserts nrc == 1 at quants.c:2040). Effort is days either way; do not start it before rungs 1-3 say where the cycles are.

Micro-rung, optional: gate/up mul_mat_id silently loses work stealing at batch 1 (48 chunks < nth*4 = 64 trips the fallback at ggml-cpu.c:1672-1675; the break at 1700 then disables stealing). Setting chunk_size 8 restores it. Predict under 2 percent from E025; only worth folding into some other rebuild.

## Killed rungs

- (A) Requantize for sgemm: doomed as framed. sgemm rejects n < 2 on x86 by design (sgemm.cpp:3712-3716), so it never runs at decode for any type, and mul_mat_id has no sgemm call site at all (only ggml-cpu.c:1305 and 1373, both dense). A requantize A/B would compare vec_dot kernels, not sgemm, confounded by type change. Salvageable variant if ever needed: a Q8_0 requant as a pure bandwidth probe (1.89x bytes per weight; if decode does not slow ~1.9x, compute-bound is reconfirmed), but rung 1 answers the same question cheaper.
- (B) Per-expert llamafile_sgemm at decode: doomed three times over. The n < 2 guard, no Q4_K or Q8_K case in the type switch (sgemm.cpp:3699-4058, default false at 4041-4042), and at n = 1 tinyBLAS degenerates to the same AVX2 dot-product mix with a worse static tile split (sgemm.cpp:1527-1532).
- (D) Fuse gate+up: ceiling is one duplicate Q8_K quantization of a 2048-float vector plus about 2 barriers per layer, roughly 0.2-0.3 ms of a 30 ms token, ~1 percent, below noise. Requires graph surgery in build_moe_ffn (src/llama-graph.cpp:1799-2148) or model reconversion to `ffn_gate_up_exps`. Kill.
- (E) Overhead reduction as implementation: measured ceiling ~2 ms central, ~4 ms pessimistic, of a ~12.5 ms gap (mmid-overhead doc, section 4). Keep the measurement (rung 2), kill the implementation unless rung 2 or 3 revives it.

## Recommendation

Implement rung 1 first. It is a config-only A/B that either delivers most of the recoverable deficit outright (repack was off) or invalidates the current kernel anatomy and retargets all further work (repack was on). Every other rung's interpretation depends on its answer.
