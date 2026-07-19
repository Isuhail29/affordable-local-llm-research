# Quantization Frontier 2025-2026: Sub-4-Bit Formats and the 48 GB Question

Sweep: 2026-07-sweep-01 | Written: 2026-07-19 | Status: survey complete, candidates proposed

## TL;DR

- The trellis (QTIP-class) revolution is real and runnable today, but NOT in mainline llama.cpp. It lives in two places: EXL3 (GPU-only, exllamav3) and ik_llama.cpp's KT quants (CPU+CUDA, GGUF-shaped but fork-only). Mainline's door was explicitly closed in Feb 2026 (PR #19726 rejected).
- The key question has a yes answer on paper: Qwen3.5-122B-A10B (released spring 2026) has published mainline-compatible 2.4-3 bpw GGUFs at 36.6-41.8 GB that fit our 48 GB RAM with the usual experts-on-CPU split, and its 10B-active MoE plus linear-attention hybrid should decode at an estimated 13-19 t/s on our measured 37-42 GB/s CPU bandwidth. The blocker risk is a documented llama.cpp CPU inefficiency for this architecture family (issue #19480).
- Separately, a near-free model upgrade appeared: Qwen3.6-35B-A3B (same A3B active class as our daily driver, one generation newer, 22.3 GB at Q4_K_M) plus merged MTP self-speculative decoding (PR #22673, May 2026) that directly fills the biggest untested gap from our E014.
- GLM-4.5-Air is now a marginal fit (42.7 GB at UD-IQ2_XXS for a 110B) and is superseded on every axis by Qwen3.5-122B-A10B. gpt-oss-120B (~61-63 GB) still does not fit 48+8 GB.

---

## 1. The format landscape, mid-2026

### 1.1 Mainline llama.cpp GGUF: K-quants + IQ + imatrix, no trellis

Mainline's official low-bit menu is unchanged in kind: K-quants (Q2_K at ~3.16 bpw effective through Q6_K) and the importance-based IQ series (IQ1_S 2.0 bpw up to IQ4_NL 4.68 bpw), calibrated with an importance matrix. See the [quantize README](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md). MXFP4 entered the tree with gpt-oss (Aug 2025) and is now used as an `MXFP4_MOE` recipe by quant publishers for other MoEs; ternary TQ1_0/TQ2_0 remain niche (BitNet-style models only).

A useful January 2026 reference: [Which Quantization Should I Use? A Unified Evaluation of llama.cpp Quantization on Llama-3.1-8B-Instruct](https://arxiv.org/abs/2601.14277) benchmarks 3-8 bit K-quant and legacy formats across downstream tasks, PPL, and CPU throughput. Practical guide for picking per-tensor recipes; it confirms the broad rule that Q4_K_M-class is the knee of the curve for dense models.

Crucially, QTIP-style trellis quants are NOT in mainline and are not coming soon:

- [Discussion #10125](https://github.com/ggml-org/llama.cpp/discussions/10125) (QTIP integration) never produced a merged implementation.
- [PR #19726](https://github.com/ggml-org/llama.cpp/pull/19726), which ported ik_llama.cpp's IQ2_K...IQ6_K quants to mainline (CPU backend, strong KLD results: mean KLD ~0.005-0.012, same-top-p 93-95%), was closed unmerged on 2026-02-23 by ggerganov after licensing/attribution friction between the two projects. The trellis KT quants were never even in scope.

Consequence for us: on mainline, the best sub-3-bit we can get is the IQ2/IQ1 series plus dynamic per-tensor recipes (below). Anything QTIP-class requires a fork or a GPU-only stack.

### 1.2 Dynamic / importance-driven recipes: Unsloth UD 2.0, bartowski, ubergarm

- [Unsloth Dynamic 2.0](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs) ([blog](https://unsloth.ai/blog/dynamic-v2)) is now the de facto standard for low-bit MoE GGUFs: per-layer bit allocation, real-use calibration sets, important layers upcast. Unsloth claims UD-Q4_K_XL sits on the KLD Pareto frontier (99.9% figure cited in their [Qwen3.5 GGUF benchmarks](https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks)). Their UD-IQ1/IQ2 MoE quants are what makes 100B-class models land under 48 GB at all.
- [bartowski](https://huggingface.co/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF) continues imatrix-calibrated standard recipes with online repacking for AVX2 (relevant to us; our E021-E028 found repack a tie on this rig, so no free lunch there).
- [ubergarm](https://huggingface.co/ubergarm/Qwen3.5-122B-A10B-GGUF) publishes ik_llama.cpp-only recipes with measured PPL per size, the best public quality-per-GB data at 2-3 bpw (numbers in section 2).

### 1.3 Trellis / QTIP class: the real sub-3-bit frontier

[QTIP (arXiv 2406.11235)](https://arxiv.org/abs/2406.11235) replaced vector-codebook quantization (QuIP#/AQLM lineage) with trellis-coded quantization plus incoherence processing. Two production descendants:

- EXL3 in [exllamav3](https://github.com/turboderp-org/exllamav3): a streamlined QTIP variant ([format doc](https://github.com/turboderp-org/exllamav3/blob/master/doc/exl3.md)), 1-8 bpw, Hadamard transforms + LDL + trellis. Headline result: Llama-3.1-70B coherent at 1.6 bpw; 70B at ~3 bpw runs in under 16 GB VRAM. GPU-only (NVIDIA). As of v1.0.0 (2026-07-14) there are official Windows wheels ([releases](https://github.com/turboderp-org/exllamav3/releases), [PyPI](https://pypi.org/project/exllamav3/)); Blackwell works via an xformers monkey-patch (v0.0.34 notes, 2026-05-09). On our 8 GB card this caps out around a 27B dense at ~2 bpw or a 14B at ~3.5 bpw, so it cannot host a 70B-class model here, but it is the strongest per-bit format that runs on this machine at all.
- KT quants in [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp): IQ1_KT/IQ2_KT/IQ3_KT/IQ4_KT (1.75-4.0 bpw), an integer-based trellis designed for CPU efficiency (claimed 3-4x faster on CPU than a faithful QTIP float trellis; good CUDA performance too). Quality: IQ2_KT beats IQ2_KS on PPL at slightly lower bpw (2.125 vs 2.1875). The fork also carries the IQ*_K/KS series, fused MoE ops, and MoE-focused runtime flags. No official Windows binaries; [docs/build.md](https://github.com/ikawrakow/ik_llama.cpp/blob/main/docs/build.md) covers Windows+CUDA source builds, which matches our existing from-source workflow. Caveat for us: some of the advertised MoE speedups lean on AVX-512, which our 14650HX lacks.

### 1.4 Vector-quantization lineage: effectively dormant for local use

- [AQLM (arXiv 2401.06118)](https://arxiv.org/abs/2401.06118) and [QuIP# (arXiv 2402.04396)](https://arxiv.org/abs/2402.04396): superseded by QTIP-class methods; no GGUF path; GPU inference via research kernels only.
- [VPTQ (arXiv 2409.17066)](https://arxiv.org/abs/2409.17066), [repo](https://github.com/microsoft/VPTQ): Microsoft's 2-bit VQ with good 70B demos, but inference is a torch/CUDA path, no GGUF integration (only exploratory fork discussions), and its 70B 2-bit checkpoints (~20+ GB) exceed our 8 GB VRAM by 2.5x. Paper-and-demo tier for our purposes.
- [HIGGS (arXiv 2411.17525)](https://arxiv.org/abs/2411.17525): Hadamard + grid, integrated in HF transformers, GPU-only, no GGUF.

### 1.5 Calibration-free and QAT tracks

- [SINQ (arXiv 2509.22944)](https://arxiv.org/abs/2509.22944) (Huawei, Sinkhorn-normalized, calibration-free): competitive with HQQ at 4 bit, transformers integration, no llama.cpp/GGUF path. Not runnable in our stack.
- [ParetoQ (arXiv 2502.02631)](https://arxiv.org/abs/2502.02631) (Meta): the scaling-law argument that sub-4-bit wants QAT, not PTQ. Released artifacts are small (MobileLLM class). Its practical descendant is vendor QAT: gpt-oss shipping MXFP4-native, and quant-aware releases like Qwen's official GGUFs. Watch for QAT checkpoints of models we care about; nothing actionable today beyond what publishers already bake into UD 2.0.

---

## 2. The key question: a 2.5-3 bit ~100B-class MoE in 48 GB RAM

Our budget: 48 GB DDR5-5600 (37-42 GB/s measured extractable by CPU matmuls, per E021-E028), 8 GB VRAM (~6.5-7 GB usable after display/driver), Windows 11 idle ~5-6 GB RAM. Realistic weight budget: ~38-40 GB RAM (mlocked) + ~5-6 GB VRAM.

### 2.1 Qwen3.5-122B-A10B: the first real fit (primary target)

Released spring 2026 as part of the [Qwen3.5 family](https://unsloth.ai/docs/models/qwen3.5) (0.8B/2B/4B/9B dense, 27B dense, 35B-A3B, 122B-A10B, 397B-A17B). Architecture per the [unsloth GGUF card](https://huggingface.co/unsloth/Qwen3.5-122B-A10B-GGUF): 122B total, 10B active; hybrid 16 x (3 x (Gated DeltaNet -> MoE) + 1 x (Gated Attention -> MoE)); 256 experts, 8 routed + 1 shared; 262K native context. The linear-attention hybrid means tiny KV cache (only 16 of 64 blocks carry full attention with 2 KV heads), which is exactly what our 8 GB card wants.

Mainline-compatible sizes ([unsloth/Qwen3.5-122B-A10B-GGUF](https://huggingface.co/unsloth/Qwen3.5-122B-A10B-GGUF)):

| Quant | Size | Fits 38-40 GB RAM + 5-6 GB VRAM? |
|---|---|---|
| UD-IQ1_M | 34.2 GB | Yes, comfortably |
| UD-IQ2_XXS | 36.6 GB | Yes |
| UD-IQ2_M | 39.1 GB | Yes, tight |
| UD-Q2_K_XL | 41.8 GB | Borderline |
| UD-IQ3_XXS | 44.7 GB | No (would need mmap paging) |

Fork-only quality anchors ([ubergarm/Qwen3.5-122B-A10B-GGUF](https://huggingface.co/ubergarm/Qwen3.5-122B-A10B-GGUF), requires ik_llama.cpp): IQ5_KS 77.3 GB = PPL 4.83 (quality ceiling); IQ2_KL 43.3 GB (3.047 bpw) = PPL 5.10 (+5.7% over ceiling); smol-IQ2_KS 35.3 GB (2.485 bpw) = PPL 5.46 (+13%); IQ1_KT 30.2 GB (2.126 bpw, trellis) = PPL 5.78 (+20%). A 3.0 bpw recipe of a 122B holding within ~6% PPL of its own 5.4 bpw is precisely the "sub-4-bit formats that hold quality" frontier arriving for a model class we can physically fit.

Speed estimate for our rig (routed experts on CPU, everything else + KV on GPU): ~7B routed-active per token at 2.4-3.0 bpw is ~2.1-2.6 GB/token from RAM; at our 37-42 GB/s that is ~14-19 t/s, with GPU work overlapping. A hybrid 64 GB + 12 GB rig reports 17.5-19 t/s on this size class ([HF discussion](https://huggingface.co/Qwen/Qwen3.5-122B-A10B/discussions/3)).

The known risk: llama.cpp's CPU path for this architecture family has a documented inefficiency. [Issue #19480](https://github.com/ggml-org/llama.cpp/issues/19480) (Feb 2026) measured Qwen3-Coder-Next (80B-A3B, same Gated DeltaNet family) at 7.74 t/s where bandwidth math predicts 20-30+, i.e. the CPU path read far more than the active params require; cross-referenced EPYC data (issue #17936) showed a 5.4x gap vs an equal-active-params standard MoE. [PR #17996](https://github.com/ggml-org/llama.cpp/pull/17996) (Qwen3-Next autoregressive pass optimization) landed since and the issue is closed, but nobody has published post-fix consumer-CPU numbers for the 122B. This is the single most important thing for us to measure, and our A/B/A + feature-engagement-verification protocol is built for exactly this.

### 2.2 The alternatives, and why they lose to the 122B

- GLM-4.5-Air 110B-A12B ([unsloth GGUF](https://huggingface.co/unsloth/GLM-4.5-Air-GGUF)): UD-IQ1_M 40.1 GB, UD-IQ2_XXS 42.7 GB, UD-Q2_K_XL 47.4 GB. Even 1-bit barely fits our budget, A12B active means ~25% more bytes per token than the Qwen (slower), it is a July 2025 model, and no Air-class successor exists (GLM-4.6/4.7/5.x shipped only at 355B+, per [unsloth's GLM guides](https://unsloth.ai/docs/models/tutorials/glm-4.6-how-to-run-locally)). Ruled marginal: only worth testing if the Qwen 122B falls to the #19480 problem, since GLM4MOE uses standard attention with a mature llama.cpp path.
- Qwen3-Next-80B-A3B / Qwen3-Coder-Next ([official GGUF](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Thinking-GGUF)): fits easily at Q3 (~34-36 GB) and A3B would be fast, but it is the very model with the measured 5x CPU shortfall above, and Qwen3.5-122B dominates it on quality per GB in our budget. Useful only as a cheaper probe of the same architecture risk.
- gpt-oss-120B: ~61-63 GB MXFP4 against our 56 GB combined ceiling. Still ruled out; mmap paging over a 5 GB/s NVMe with random expert access would collapse decode.
- Kimi-Linear-48B-A3B: GGUF exists only via an experimental custom llama.cpp branch ([ymcki HF repo](https://huggingface.co/ymcki/Kimi-Linear-48B-A3B-Instruct-GGUF)); not mainline-merged. Skip until upstreamed.

### 2.3 The adjacent free win: Qwen3.6-35B-A3B and merged MTP

Not a 70B-class answer, but it changes our flagship baseline:

- [Qwen3.6-35B-A3B](https://huggingface.co/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF) (spring 2026, arch `qwen35moe`, supported by current mainline including our b10064 vintage): Q4_K_M 22.3 GB, IQ4_XS 19.7 GB. Same A3B active class as our Qwen3-30B-A3B daily driver, one generation newer, 256K context. A low-end 6 GB VRAM + 32 GB RAM machine reports ~30 t/s ([writeup](https://mychen76.medium.com/run-qwen3-6-35b-a3b-on-6gb-vram-using-llama-cpp-30-tps-a89032e5a60c)); a 243-run GB10 study found Q4 and MXFP4_MOE equal-best ([Subterra](https://www.subterratechnologies.com/blog/qwen3-6-35b-on-nvidia-gb10-243-llama-cpp-runs-to-find-the-best-local-quant)).
- MTP (multi-token prediction) self-speculative decoding merged into llama.cpp via [PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673) (2026-05-16, with [cleanup #23269](https://github.com/ggml-org/llama.cpp/pull/23269)): `--spec-type draft-mtp --spec-draft-n-max 2-3` on MTP-bearing GGUFs ([unsloth/Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF)). Reported ~75% acceptance and up to 2x on GPU dense ([Qwen3.6 guide](https://unsloth.ai/docs/models/qwen3.6) claims 1.4-2.2x dense, 1.15-1.2x MoE). Directly relevant to our E014 gap: draft-mtp was on our UNTESTED list. Counter-evidence to keep us honest: a 19-configuration RTX 3090 study of the non-MTP spec types on Qwen3.6-35B-A3B found no net speedup on that hardware class ([thc1006 benchmark](https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090)), which matches our E014 mechanism (expert-union verification cost on MoE). MTP differs in the two ways that matter: no second model competing for TDP, and much higher acceptance than ngram. Genuinely falsifiable on our rig.

---

## 3. Candidate experiments for our rig

### C1. Qwen3.6-35B-A3B as new flagship (model upgrade, lowest risk)

Download bartowski or unsloth UD Q4 (~20-23 GB, ~2 h at our 3 MB/s). Run our full harness: A/B/A vs Qwen3-30B-A3B Q4_K_M at -t 12, --n-cpu-moe, mlock, sustained-thermal protocol. Hypothesis: >=25 t/s (evidence: 30 t/s on a weaker 6 GB/32 GB machine) at materially better quality (one generation newer). Falsifier: decode below 25 t/s or no measurable quality gain on our eval set. Effort: hours once downloaded. Needs: nothing new in the stack.

### C2. MTP self-speculative decoding on Qwen3.6-35B-A3B (speed upgrade, fills the E014 gap)

Requires the MTP-bearing GGUF (separate ~22-23 GB download) and our b10064 (`--spec-type draft-mtp`, verify engagement in logs per protocol law). Sweep --spec-draft-n-max 1-3. Hypothesis: >=20% net sustained decode gain; mechanism risk is E014's expert-union cost during batch verification, now offset by ~75% acceptance and zero draft-model contention. Either outcome is publishable: first consumer-CPU-offload MTP numbers we know of. Effort: hours (after C1). Pairs with C1 into one download session.

### C3. Qwen3.5-122B-A10B UD-IQ2_XXS on mainline (capability upgrade: 100B-class in 48 GB)

Download 36.6 GB (~3.5 h). Split: routed experts CPU (mlock, -t 12), attention + DeltaNet + shared expert + KV on GPU. First measure the #19480 question on OUR rig: bytes-per-token via our instrumented build vs the ~2.2 GB/token theoretical, then sustained decode. Success: >=10 t/s with quality clearly above the 30B/35B daily driver (it should be; +13% PPL over its own ceiling still leaves a 122B far above a 35B). Falsifiers: architecture CPU path still reads 3-5x excess (we quantify and publish the deficit), or RAM budget forces paging. Fallback probe if it fails: GLM-4.5-Air UD-IQ1_M (standard attention, mature path) as the control for "is it the arch or the size". Effort: days. Needs: 36.6 GB download, careful RAM budgeting.

### C4. ik_llama.cpp Windows source build: trellis KT quants bake-off (format frontier + enabler)

Build ik_llama.cpp with CUDA on Windows (we already maintain an instrumented llama.cpp build; [build docs](https://github.com/ikawrakow/ik_llama.cpp/blob/main/docs/build.md)). Two tests: (a) our current flagship Qwen3-30B-A3B requantized to IQ*_K/KT recipes vs mainline Q4_K_M, quality per GB and t/s; (b) if C3 succeeded, ubergarm smol-IQ2_KS 35.3 GB or IQ2_KL 43.3 GB of the 122B vs unsloth UD-IQ2_XXS, testing whether fork quants + fused MoE beat mainline at equal footprint. Risks: no AVX-512 on our CPU (some fork gains assume it), Windows build friction, our profiler does not port. This is the only route to QTIP-class quality in our RAM-bound regime since mainline rejected the port (PR #19726). Effort: days. Needs: source build, possibly 35-43 GB download.

### C5 (stretch). EXL3 2 bpw dense on the 5060: GPU-only trellis probe

exllamav3 v1.0.0 Windows wheels + Blackwell monkey-patch. Fit test: Qwen3.5-27B at ~2.0 bpw (~7 GB) or a 14B at ~3.5 bpw fully in 8 GB VRAM. Question: can trellis 2 bpw dense-on-GPU match our MoE-on-CPU flagship quality while freeing ALL system RAM (agent/multitasking value) and decoding GPU-fast? Prior: probably loses on quality; but it is the only fully-GPU configuration this laptop can run above the 9B class, and nobody has published EXL3 numbers on an 8 GB Blackwell laptop. Effort: days. Needs: pip stack alongside llama.cpp, ~7 GB download, quant possibly self-made (EXL3 quantization of a 27B needs more VRAM than we have; must find pre-made 2 bpw uploads first).

---

## Priority call

C1+C2 first (one ~45 GB download session, near-certain model upgrade plus a real shot at the speed upgrade). C3 is the breakthrough swing and answers this survey's key question empirically. C4 follows C3. C5 only if idle time.

---

## Feasibility verdicts

Adversarial review 2026-07-19. Every load-bearing claim re-verified against primary sources.

- **C1: GO.** Verified: [bartowski repo](https://huggingface.co/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF) exists, Q4_K_M 22.29 GB / IQ4_XS 19.70 GB, arch `qwen35moe`, quantized with llama.cpp b9222 (our b10064 postdates it). Fits trivially, not a duplicate of any E0xx, download honest (~2 h). Lowest-risk item in the sweep.
- **C2: GO.** Verified: [PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673) merged 2026-05-16 with `--spec-type draft-mtp`; [unsloth MTP GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF) exists (UD-Q4_K_M 22.7 GB). draft-mtp is explicitly on our E014 UNTESTED list, so no duplication. Note the documented constraints: `-fa on`, `-np 1`, no `--mmproj`; MTP head adds <10% memory. Either outcome publishable.
- **C3: MAYBE.** Model and sizes verified ([unsloth repo](https://huggingface.co/unsloth/Qwen3.5-122B-A10B-GGUF), UD-IQ2_XXS 36.6 GB, fits the 38-40 GB RAM + 5-6 GB VRAM budget). But the survey's risk framing is wrong in our favor's opposite direction: [issue #19480](https://github.com/ggml-org/llama.cpp/issues/19480) is still OPEN, not closed, and [PR #17996](https://github.com/ggml-org/llama.cpp/pull/17996) merged 2025-12-16, BEFORE the issue was filed (Feb 2026). All #17996 benchmark gains reported are GPU token-gen (Vulkan/CUDA/ROCm); the 7.74 t/s consumer-CPU measurement is post-fix. Expect the deficit to still be there; plan for the "quantify and publish the deficit" outcome as the likely one, with >=10 t/s as upside, not baseline. Still worth the swing because the negative result is publishable on our instrumented build.
- **C4: MAYBE.** [ik_llama.cpp Windows+CUDA build docs](https://github.com/ikawrakow/ik_llama.cpp/blob/main/docs/build.md) verified real, but the toolchain is clang via VS Build Tools + CUDA 12.6, not our MSVC flow, and the documented flag set assumes AVX-512 (must be stripped for our 14650HX). Units bug in section 2.1: [ubergarm](https://huggingface.co/ubergarm/Qwen3.5-122B-A10B-GGUF) publishes GiB, not GB. IQ2_KL is 43.3 GiB = 46.5 GB and does NOT fit our budget; smol-IQ2_KS 35.3 GiB = 37.9 GB is the realistic 122B option (same units error class E021-E028 already burned us on). PPL numbers otherwise verified exact. Contingent on C3; test (a) on the 30B flagship stands alone.
- **C5: MAYBE (weak).** [exllamav3 v1.0.0](https://github.com/turboderp-org/exllamav3/releases) Windows wheels verified (cu128, torch 2.10, py3.10-3.14), but the survey's Blackwell claim is stale: v1.0.0 REMOVED the xformers dependency, so the v0.0.34 monkey-patch path no longer applies and release notes do not mention sm_120 explicitly. Fit math is the bigger problem: 27B at 2.0 bpw is 6.75 GB of weights alone before torch CUDA context (~0.6-1 GB), KV, and head tensors; on ~6.5-7 GB usable VRAM this likely OOMs. The 14B at ~3.5 bpw (~6.1 GB) is the only realistic config, and no pre-made EXL3 2 bpw Qwen3.5-27B upload was verified to exist. Keep only as the idle-time probe it claims to be, scoped to 14B-class.
