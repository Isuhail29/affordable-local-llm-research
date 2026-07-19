# MoE and Sparse-Inference Advances, 2025-2026

Domain survey for sweep 2026-07-sweep-01. Scope: expert offloading, expert caching/prediction, sparsity exploitation, and inference-time routing schemes, in both academic papers and runnable systems, assessed against OUR rig (i7-14650HX AVX2-only, 48 GB DDR5-5600 at ~55-60 GB/s practical, RTX 5060 Laptop 8 GB, llama.cpp b10064, Windows 11, ~3 MB/s internet, ~416 GB free disk).

Survey date: 2026-07-19. Every claim linked. "Runnable" means a Windows-feasible implementation exists today, not just a paper.

---

## 1. The headline shift since our last look: MTP self-speculation for MoE

The single biggest ecosystem change relevant to us is that **multi-token prediction (MTP) speculative decoding now works in llama.cpp for MoE models**, with the draft head bundled inside the GGUF (no separate draft model, no vocab matching, no second expert pool).

- Qwen's 3.6 generation (released spring 2026) ships models trained with MTP heads. llama.cpp gained MTP support around PR #19493 (merged 2026-04-19 per the [thc1006 benchmark repo](https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090)); our b10064 build already ships `--spec-type draft-mtp`, which we have never tested (E014 covered only draft-simple and ngram-simple).
- Usage is one flag pair: `--spec-type draft-mtp --spec-draft-n-max 2` against an MTP GGUF ([Unsloth Qwen3.6 guide](https://unsloth.ai/docs/models/qwen3.6), [mer.vin walkthrough](https://mer.vin/2026/05/run-qwen-3-6-mtp-in-llama-cpp-faster-local-inference-with-built-in-speculative-decoding/)).

### Reported results (all primary sources)

| Setup | Model | Result | Link |
|---|---|---|---|
| RTX 5060 Ti 16 GB, all-GPU | Qwen3.6-35B-A3B | 98 to 144 t/s, **1.47x** | [njannasch.dev](https://njannasch.dev/blog/mtp-speculative-decoding-qwen-3-6-5060ti/) |
| RTX PRO 6000, all-GPU | Qwen3.6-35B-A3B | **1.17x** | [jarvislabs.ai](https://jarvislabs.ai/blog/qwen36-mtp-llamacpp-rtxpro6000) |
| DGX Spark | Qwen3.6-27B dense | positive, workload-dependent | [NVIDIA forum](https://forums.developer.nvidia.com/t/mtp-llama-cpp-a-look-at-qwen3-6-27b/370298) |
| RTX 3090, vLLM MTP k=1 | Qwen3.6-35B-A3B | **+27.5%** decode | [thc1006 HackMD, April 2026 revision](https://hackmd.io/@thc1006/SJly6IE6Wx) |

### The negative result that maps directly onto our E014

[thc1006's RTX 3090 benchmark](https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090) tested 19 llama.cpp spec-decode configs on Qwen3.6-35B-A3B (ngram-cache, ngram-mod at six N values, classic draft with a vocab-matched Qwen3.5-0.8B): **all 19 net-negative or neutral, even at 100% draft acceptance**. His explanation is exactly our E014 expert-union finding: the model routes 8-of-256 experts per token, and verifying K drafted tokens in one batch loads the union of their expert sets, which eats the savings. Critically, **draft-mtp was NOT among the 19 configs**, and his April 2026 revision concedes that MTP with small K on identical hardware is net-positive (+27.5% under vLLM). The union penalty scales with K; classic llama.cpp drafting uses K=5-64, MTP uses K=1-3.

**Why this matters for us:** E014's expert-union cost is worst-case at large K with a separate draft model. MTP changes both terms: the draft head is a single bundled layer (no second model competing for TDP, the E014 dense-8B killer), and `--spec-draft-n-max 2` keeps the verify union at most ~24 of 256 routed experts versus 8 for one token, with overlap in practice. Whether the surviving union cost on CPU-resident experts beats the amortized attention savings on OUR asymmetric rig is a genuinely open, falsifiable question. Nobody has published MTP numbers for an experts-on-CPU hybrid config.

There is also a 2025 paper directly on this problem: [Accelerating Speculative Decoding with Sparse Computation in Verification](https://arxiv.org/pdf/2512.21911) (sparsifying the verification pass to cut the expert-union cost), no consumer runtime yet.

---

## 2. New models: what actually fits 48 GB RAM + 8 GB VRAM

### Qwen3.6-35B-A3B (the model-upgrade candidate)

Architecture, from the [Unsloth MTP GGUF card](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF): 35B total / 3B active, 40 layers, hybrid attention (3:1 Gated DeltaNet linear layers to gated full attention, same family as Qwen3-Next), **256 experts, top-8 routed + 1 shared**, 262,144-token native context, multimodal, hybrid thinking, MTP head bundled. Two generations newer than our Qwen3-30B-A3B-2507 flagship at the same A3B active size.

Quant sizes that fit us (from the same card): UD-Q4_K_XL **22.9 GB**, UD-IQ4_XS **18.2 GB**, UD-Q3_K_XL **17.2 GB**. At our 3 MB/s that is roughly 2.2 h / 1.75 h / 1.6 h downloads. A [byteshape MTP GGUF](https://huggingface.co/byteshape/Qwen3.6-35B-A3B-MTP-GGUF) also exists.

Evidence it runs well on rigs like ours: [~30 t/s reported on a 6 GB VRAM + 32 GB RAM machine with llama.cpp expert offload](https://mychen76.medium.com/run-qwen3-6-35b-a3b-on-6gb-vram-using-llama-cpp-30-tps-a89032e5a60c), and the [RTX 3060 12 GB `--n-cpu-moe` guide](https://knightli.com/en/2026/05/26/rtx-3060-llama-cpp-n-cpu-moe-local-35b/) treats 35B-A3B-class MoE as the standard budget target now.

Known risks, both llama.cpp-implementation, not physics:
- [Issue #19894](https://github.com/ggml-org/llama.cpp/issues/19894): Qwen3.5-35B-A3B decode 35% slower than Qwen3-30B-A3B on CUDA (38.4 vs 59.0 t/s), with CPU cores loading during GPU decode, suggesting DeltaNet ops falling back to CPU. Open, unconfirmed, April-era build.
- [Issue #19480](https://github.com/ggml-org/llama.cpp/issues/19480): Qwen3-Next-80B CPU-only decode ~7.7 t/s on a 96 GB DDR5-5600 Ryzen AI 370 (~5x below bandwidth expectation), same DeltaNet family. Pure-CPU only; hybrid keeps DeltaNet on GPU, which is our configuration.

The same hybrid-linear-attention design is also a capability lever: KV cache exists only for the 1-in-4 full-attention layers, so ~4x less KV memory per token of context than our current flagship. 100K+ context inside 8 GB VRAM becomes plausible for the first time on this rig.

### Others assessed

- **GPT-OSS-20B** (21B total, 3.6B active, native MXFP4, ~12.1 GB): well-trodden on 8 GB VRAM rigs at ~30 t/s via CPU offload, CPU-bound ([official llama.cpp guide](https://github.com/ggml-org/llama.cpp/discussions/15396), [2026 hardware guide](https://runaihome.com/blog/gpt-oss-20b-local-ai-hardware-guide-2026/)). Same speed class as our current 31-42 t/s flagship, not clearly smarter than Qwen3-30B-A3B-2507, so an alternative, not an upgrade.
- **Qwen3-Next-80B-A3B** ([official GGUF](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct-GGUF)): Q4 ~45+ GB does not fit 48 GB RAM with OS overhead; Q3-class ~35 GB fits but issue #19480's CPU-path numbers (7.7 t/s on comparable bandwidth) kill the >=25 t/s bar. Dead for now; superseded by 3.6-35B anyway.
- **REAP-pruned checkpoints** (see section 4): Qwen3-Coder-REAP-25B-A3B Q4_K_M at **15.1 GB**, and a community REAP-0.30 of Qwen3.6-35B-A3B already exists.

---

## 3. Engines and forks

### ik_llama.cpp (the serious one for our CPU-bound expert path)

[ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) is the fork with the exact optimizations mainline lacks on our bottleneck. Our E025/E026 arc concluded the expert path is kernel-compute-bound (mul_mat_id has no llamafile_sgemm fast path) and mainline's CPU path extracts 37-42 GB/s of our ~60 GB/s ceiling. ik targets precisely that gap:

- **Fused MoE** (`-fmoe`): fused gate+activation+up kernel for expert FFNs, cutting per-expert kernel overhead and memory traffic ([DeepWiki feature summary](https://deepwiki.com/ikawrakow/ik_llama.cpp/1.1-key-features-and-performance-improvements)).
- **Runtime repack** (`-rtr`) into row-interleaved `_R4/_R8` layouts; documented caveat: avoid in hybrid CUDA mode (missing CUDA kernels for repacked K-quants), which matches our own E026 finding that mainline repack is OFF in ncmoe hybrid.
- **Smart Expert Reduction** (`-ser`, added March 2025 per the [project wiki news](https://github.com/ikawrakow/ik_llama.cpp/wiki/Previous-Latest-News)): inference-time adaptive expert skipping, the shipped version of what our E023 static top-6 override approximates.
- **IQK quants and iqk_mul_mat**: claimed 1.8-5.2x prompt processing and 1.06-2.1x token generation over mainline on AVX2, per the author's [CPU comparison discussion #164](https://github.com/ikawrakow/ik_llama.cpp/discussions/164); community reports of ~1.9x MoE inference are echoed in [llama.cpp issue #19480](https://github.com/ggml-org/llama.cpp/issues/19480). AVX2 is fully supported (AVX-512 helps but is not required).
- Practitioner consensus (e.g. [DocShotgun's MoE offload guide](https://gist.github.com/DocShotgun/a02a4c0c0a57e43ff4f038b46ca66ae0), [Level1Techs comparison thread](https://forum.level1techs.com/t/llama-cpp-v-ik-llama-cpp-sanity-check-step-3-5-flash/246110)) is that ik wins mainly on CPU-heavy and hybrid MoE setups, exactly our regime.

Caveats for us: no official Windows binaries (source build; we have the working cmake+ninja+cl environment from E026, but recall our measured 10% MSVC-vs-Clang penalty, so a clang-cl build is preferred), and a full hybrid test needs the CUDA toolkit installed (still pending Sohail's admin install). A CPU-only A/B against mainline's pure-CPU 20.14 t/s baseline needs neither download nor CUDA.

### Mainline llama.cpp

b10064 is current-generation and already contains the year's MoE-relevant merges: `--n-cpu-moe`, the full spec-decode family including draft-mtp/draft-eagle3/ngram-* (mostly untested by us), and Qwen3-Next/3.5/3.6 architecture support (PR #16095 lineage for qwen_next, MTP via PR #19493). The DeltaNet CPU/CUDA kernels are the immature corner (issues [#19480](https://github.com/ggml-org/llama.cpp/issues/19480), [#19894](https://github.com/ggml-org/llama.cpp/issues/19894)).

### KTransformers-family research systems (not for us, but the frontier)

- [DALI (Feb 2026)](https://arxiv.org/html/2602.03495v1): workload-aware CPU/GPU expert assignment solved as runtime 0-1 optimization, residual-corrected expert prefetch, workload-scored cache eviction. 7.6x over llama.cpp decode on an EPYC + RTX 3090 box. Built on KTransformers; code release unclear; assumes 24 GB GPU and PCIe expert streaming.
- [CoX-MoE (May 2026)](https://arxiv.org/pdf/2605.17889): coalesced expert execution with CPU-GPU co-execution, requires Intel AMX. We have no AMX. Dead on arrival for this rig.
- [Prima.cpp](https://arxiv.org/pdf/2504.08791): 30-70B on home clusters of multiple devices. Wrong shape for a single laptop.
- [PowerInfer](https://github.com/Tiiny-AI/PowerInfer) (now maintained under Tiiny-AI): the ReLU-activation-sparsity engine line continues (their Pocket Lab device demoed GPT-OSS-120B at CES 2026), but the desktop engine still needs sparsity-finetuned model variants (TurboSparse etc.) and its Windows/consumer-Blackwell path is unmaintained relative to llama.cpp.

**Why the offloading-paper class mostly does not transfer to us:** HOBBIT ([arXiv 2411.01433](https://arxiv.org/pdf/2411.01433), mixed-precision expert fetching), ProMoE ([arXiv 2410.22134](https://arxiv.org/pdf/2410.22134), learned expert prediction + prefetch), ExpertFlow ([arXiv 2410.17954](https://arxiv.org/html/2410.17954v2), predictive caching + token scheduling), MoE-Infinity, Fiddler, DALI: all of them attack the PCIe-fetch bottleneck of keeping experts in GPU memory on demand. Our E013 architecture already sidesteps that: all experts live permanently in system RAM and the CPU computes them at RAM bandwidth. E021 additionally measured routing locality worth only ~5% on our rig, so prediction/caching has no headroom here. These papers are the right literature for 24 GB rigs, not 8 GB + 48 GB ones.

---

## 4. Expert reduction, pruning, and adaptive routing at inference time

This is the research lane our E023 (static top-6, +21% speed / +2.4% PPL) already lives in, and it moved fast in 2025-2026:

- **REAP** (Cerebras, [blog](https://www.cerebras.ai/blog/reap), [code](https://github.com/CerebrasResearch/reap)): one-shot router-weighted expert activation pruning. 25% expert removal stays within ~1 point of baseline on Qwen3-30B-A3B; 50% retains ~96% on code tasks. Released checkpoints include [Qwen3-Coder-REAP-25B-A3B](https://huggingface.co/cerebras/Qwen3-Coder-REAP-25B-A3B) (community [Q4_K_M GGUF, 15.1 GB](https://huggingface.co/danielus/Qwen3-Coder-REAP-25B-A3B-Q4_K_M-GGUF)) and community REAP runs of the newest models, e.g. [Qwen3.6-35B-A3B REAP-0.30](https://huggingface.co/groxaxo/Qwen3.6-35B-A3B-Heretic-REAP-0.30). Important honesty note for our rig: pruning shrinks the expert POOL (RAM footprint), not the per-token active set, so it buys memory headroom and page-cache slack rather than decode speed. Its value to us is fitting bigger-class models and de-risking RAM pressure (the 12 t/s field incident class), not t/s.
- **Ada-K routing** ([arXiv 2410.10456](https://arxiv.org/pdf/2410.10456), ICLR-track): learned per-token expert-count allocator, >20% inference speedup at preserved quality. Needs training the allocator; no GGUF-world implementation.
- **Alloc-MoE (2026)** ([arXiv 2604.08133](https://arxiv.org/pdf/2604.08133)): budget-aware expert activation allocation at inference without retraining. Paper-only.
- **Dynamic top-p routing** ([HF blog writeup](https://huggingface.co/blog/Spico/dynamic-routing)): replace fixed top-k with a router-probability-mass threshold at inference. Trivially small patch in llama.cpp terms; the shipped equivalent is ik_llama's `-ser`.
- **Matryoshka MoE** ([arXiv 2509.26520](https://arxiv.org/pdf/2509.26520)): training for elastic expert counts so k can be dialed at inference. Training-side, watch for models trained this way.
- **SMoE expert substitution** ([arXiv 2508.18983](https://arxiv.org/pdf/2508.18983)): substitute cold experts with cheaper stand-ins at the edge. Paper-only.
- **DynaExq** ([arXiv 2511.15015](https://arxiv.org/html/2511.15015)): hotness-driven dynamic expert precision (hot experts high-bit, cold low-bit). Conceptually applicable to our RAM-resident experts (down-experts are already Q6_K vs Q4_K, per our E028 dtype discovery), but no runnable consumer implementation.

The actionable synthesis for us: E023 proved static k-reduction pays on this rig; the 2025-2026 literature says ADAPTIVE k (threshold on router weights) dominates static k on the speed-quality frontier; ik_llama ships it (`-ser`) and mainline does not; and our instrumented build plus PPL harness makes a threshold-routing patch a small, well-scoped source change (llm_build router softmax, where our expert_used_count override already operates).

---

## 5. Activation sparsity for dense models (watched, still not our lane)

- **TEAL** ([OpenReview](https://openreview.net/forum?id=dGVZwyq5tV)): training-free magnitude sparsity, 40-50% model-wide, up to 1.53-1.8x decode on GPU kernels.
- **CHESS / CATS-line thresholding** ([arXiv 2409.01366](https://arxiv.org/pdf/2409.01366)), **Polar Sparsity** ([arXiv 2505.14884](https://arxiv.org/pdf/2505.14884)), **CoreInfer** ([arXiv 2410.18311](https://arxiv.org/pdf/2410.18311)).
- Most relevant to us: [Enabling Dynamic Sparsity in Quantized LLM Inference](https://arxiv.org/pdf/2511.04477) implements TEAL-style thresholded sparsity **inside llama.cpp (~8K lines C/C++) for quantized models**, the first time this tech met the GGUF world.

Why it stays low priority here: the wins accrue to dense-model CPU/low-bandwidth decode. Our dense baseline (Qwen3-8B) runs fully in VRAM at 62 t/s where sparsity buys little, and our flagship is already an A3B MoE whose FFN sparsity is architectural. The crossover to watch: if the 2511.04477 code lands publicly and supports MoE expert FFNs, thresholded sparsity INSIDE each active expert would stack on top of everything else on our CPU path.

---

## Candidate experiments for our rig

Ranked. Effort assumes our existing protocol stack (A/B/A flanking, warm cache, mlock, -t 12 MoE, PPL harness on datasets/ppl-text.txt, sustained-thermal context from E032).

### C1. Qwen3.6-35B-A3B as the new flagship (model upgrade, possibly capability too)

**What:** Download [unsloth/Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF) UD-Q4_K_XL (22.9 GB, ~2.2 h at our 3 MB/s; fallback UD-IQ4_XS 18.2 GB). Reproduce the E013 methodology: `-ngl 99 --n-cpu-moe` sweep, `-t 12`, mlock, llama-bench + real-chat soak, PPL versus the 30B flagship, and a long-context probe (the hybrid DeltaNet design keeps KV only on 1-in-4 layers, so 64K-128K context inside 8 GB VRAM is plausible).
**Why:** Two generations newer at the same A3B active size, 256K native context, multimodal, hybrid thinking. A ~30 t/s result on a 6 GB VRAM + 32 GB RAM machine is already published ([Medium](https://mychen76.medium.com/run-qwen3-6-35b-a3b-on-6gb-vram-using-llama-cpp-30-tps-a89032e5a60c)), so >=25 t/s on our stronger rig is likely.
**Kill criteria / risks:** [#19894](https://github.com/ggml-org/llama.cpp/issues/19894)-style DeltaNet CPU fallback stealing our expert threads (verify with the E027 profiler: attribute DeltaNet op time); decode <25 t/s sustained = fail as flagship, keep as long-context specialist.
**Needs:** 22.9 GB download, no build, no new hardware. **Effort:** days.

### C2. draft-mtp self-speculation on Qwen3.6-35B-A3B (the E014 rematch)

**What:** On the C1 model (MTP head is in the same GGUF, zero extra download): llama-server A/B/A with `--spec-type draft-mtp`, sweeping `--spec-draft-n-max 1..3`, on three workload classes (code, structured, chat), measuring net t/s at identical output plus acceptance rates from logs. Instrument the expert-union size per verify batch with our E027 profiler to publish the first union-cost measurement on an experts-on-CPU config.
**Why:** draft-mtp is explicitly untested in E014, and the math that killed E014 (large-K union, second model) inverts at K=1-2 with a bundled head: 1.47x on a 5060 Ti ([njannasch.dev](https://njannasch.dev/blog/mtp-speculative-decoding-qwen-3-6-5060ti/)), 1.17x on RTX PRO 6000 ([jarvislabs](https://jarvislabs.ai/blog/qwen36-mtp-llamacpp-rtxpro6000)), +27.5% vLLM MTP k=1 on the same 3090 where all 19 classic llama.cpp configs lost ([thc1006](https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090)). Nobody has published the experts-on-CPU case; either outcome is a publishable result for our repo.
**Kill criteria:** net t/s below the C1 baseline on all three workloads at every n-max = refutation, and the union-size measurement explains why.
**Needs:** nothing beyond C1. **Effort:** hours.

### C3. ik_llama.cpp on the expert path (the fork mainline refuses to be)

**What:** Stage 1 (no download, no CUDA): build ik_llama.cpp CPU-only with our E026 cmake+ninja environment (prefer clang-cl given our measured 10% MSVC penalty) and A/B/A it against mainline's pure-CPU 20.14 t/s on the existing Qwen3-30B-A3B Q4_K_M, testing `-fmoe`, `-rtr`, and `-ser` individually. Stage 2 (only if stage 1 clears +15%): install CUDA toolkit (Sohail admin task, already pending) and rebuild for the hybrid ncmoe config against the 31-42 t/s flagship number.
**Why:** ik's fused-MoE and iqk kernels attack exactly the deficit our E025/E026 arc proved (mul_mat_id kernel-compute-bound, no sgemm fast path); author-published AVX2 gains of 1.06-2.1x TG and much larger PP ([discussion #164](https://github.com/ikawrakow/ik_llama.cpp/discussions/164)); `-ser` is shipped adaptive expert reduction to benchmark against our E023 top-6.
**Kill criteria:** stage 1 <= +10% TG over mainline pure-CPU = fork overhead not worth carrying; document and close.
**Needs:** source build (no official Windows binaries); CUDA toolkit install for stage 2 only. **Effort:** days (stage 1), week+ (through stage 2).

### C4. Adaptive expert reduction: threshold routing versus static top-6

**What:** Patch our instrumented mainline build to replace the fixed top-k expert selection with a router-weight threshold (skip experts below a fraction of the top-1 weight, floor of 4, cap of 8), sweep the threshold, and plot the speed-PPL frontier against E023's static top-6 (+21% / +2.4% PPL) and top-4 (+17.8% PPL, rejected). Also run the E023-style static override on Qwen3.6 (256 experts, top-8) once C1 lands.
**Why:** The 2025-2026 literature (Ada-K [2410.10456](https://arxiv.org/pdf/2410.10456), Alloc-MoE [2604.08133](https://arxiv.org/pdf/2604.08133), [dynamic top-p routing](https://huggingface.co/blog/Spico/dynamic-routing)) consistently finds adaptive k dominates static k at equal compute; easy tokens spend 4 experts, hard tokens keep 8. Small, well-scoped source change at the exact spot our expert_used_count override already touches; extends our strongest shipped result.
**Kill criteria:** no threshold point beats the static top-6 frontier (PPL at equal speed or speed at equal PPL) = static is good enough, close the lane.
**Needs:** no downloads, our own build. **Effort:** days.

### C5. REAP-pruned checkpoints as headroom and quality-per-GB plays

**What:** Download [Qwen3-Coder-REAP-25B-A3B Q4_K_M (15.1 GB)](https://huggingface.co/danielus/Qwen3-Coder-REAP-25B-A3B-Q4_K_M-GGUF) (~1.4 h), benchmark decode and PPL against the 30B flagship, and measure the RAM-pressure margin (the 12 t/s page-eviction incident class) with realistic desktop load. If C1 succeeds, evaluate the community [Qwen3.6-35B REAP-0.30](https://huggingface.co/groxaxo/Qwen3.6-35B-A3B-Heretic-REAP-0.30) the same way.
**Why:** REAP is the one pruning line with released, GGUF-converted checkpoints of current models ([Cerebras blog](https://www.cerebras.ai/blog/reap)); 25% pruning stays within ~1 point on Qwen3-30B-A3B. Honest expectation set in section 4: this buys RAM headroom and eviction resistance, not decode t/s (active set unchanged), so it is an enabler, not a breakthrough.
**Kill criteria:** PPL/quality regression beyond ~3% versus the unpruned flagship, or no measurable eviction-resistance benefit under load.
**Needs:** 15.1 GB download. **Effort:** hours.

---

## Sequencing note

C1 and C2 share one download and one protocol setup; run them as a single arc (C2 is hours once C1 exists). C3 stage 1 and C4 need no downloads and can run while the C1 download trickles in at 3 MB/s. C5 is opportunistic filler. The combined bull case: Qwen3.6-35B-A3B at ~28-30 t/s baseline with a 1.2-1.4x MTP multiplier lands a two-generations-smarter, 256K-context, multimodal flagship at or above our current 31-42 t/s speed band, which would satisfy the model-upgrade AND speed-upgrade definitions simultaneously.

---

## Feasibility verdicts

Adversarial review, 2026-07-19. Each candidate checked for: 8 GB VRAM + 48 GB RAM + AVX2-only + Windows fit; implementation runnable on our b10064 stack (not paper-only); non-duplication of E013-E032; honest download/effort math.

Corrections found during verification: mainline MTP landed as [PR #22673, merged 2026-05-16](https://github.com/ggml-org/llama.cpp/pull/22673), not "PR #19493 merged 2026-04-19" as cited above; [issue #19894](https://github.com/ggml-org/llama.cpp/issues/19894) is now CLOSED (not open); the njannasch.dev 1.47x post returns HTTP 403 and could not be verified; the mychen76 "6 GB VRAM at ~30 t/s" article is member-gated, its quant and flags unverifiable. None of these kill a candidate, but the survey's MTP provenance was wrong.

- **C1 Qwen3.6-35B-A3B flagship: GO.** Verified real: [unsloth MTP GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF) exists with UD-Q4_K_XL at exactly 22.9 GB (fits 48 GB RAM with mlock, experts-on-CPU keeps the hybrid-attention non-expert weights inside 8 GB VRAM), b10064 postdates the 2026-05-16 MTP merge so the arch loads, download math honest, no duplicate; residual DeltaNet-CPU-fallback risk (#19894 class, now closed upstream) is covered by the stated profiler check and <25 t/s kill criterion.
- **C2 draft-mtp self-speculation: GO** (contingent on C1). Confirmed merged in mainline with the exact flags cited and tested on Qwen3.6-35B-A3B in the PR itself; not a duplicate (E014 tested only draft-simple/ngram-simple); zero extra download. Honest new caveats from the PR and field reports: prompt processing slows from device-to-host embedding transfers, n_parallel=1 only, MoE speedups smaller than dense, and long-running llama-server MTP crashes (~20 min under load) reported in early builds, so add a soak-stability check to the protocol.
- **C3 ik_llama.cpp: MAYBE.** Fork is real, active, AVX2-supported, and `-fmoe`/`-rtr`/`-ser` all confirmed in its docs, so stage 1 (CPU-only, no download, no CUDA) is genuinely cheap; downgraded from GO because the author's headline TG gains concentrate in IQK-format quants and prompt processing, putting the expected Q4_K_M/AVX2 token-gen gain near the +10-15% kill threshold, Windows is a documented-but-fiddly source build (no official binaries), and the stage 2 hybrid endpoint is both blocked on the pending CUDA toolkit install and undermined by ik's own repacked-quants-in-CUDA caveat. Run stage 1 as background filler with the kill gate enforced.
- **C4 adaptive threshold routing: GO.** No downloads, no new tooling, patches the exact code site the E023 override already touches in our instrumented build; extends (does not duplicate) the static top-6 result; literature direction (Ada-K, top-p routing, ik's shipped `-ser`) is consistent; clean falsifiable frontier comparison against E023. Cheapest real candidate in the sweep.
- **C5 REAP checkpoints: MAYBE.** [danielus GGUF](https://huggingface.co/danielus/Qwen3-Coder-REAP-25B-A3B-Q4_K_M-GGUF) verified at 15.1 GB, but its base is Qwen3-Coder-30B-A3B-Instruct, a coder model, so the proposed PPL comparison against our general-instruct flagship is cross-family and weak; by the survey's own admission REAP cannot touch decode t/s (fails all three breakthrough definitions); the "eviction resistance" measurement is soft against our +-10% cross-session error bar; and the Qwen3.6 follow-on ([groxaxo REAP-0.30](https://huggingface.co/groxaxo/Qwen3.6-35B-A3B-Heretic-REAP-0.30)) is a Heretic (abliterated) variant shipping BF16 safetensors at ~50 GB with thin GGUF coverage, adding a quality confound. Only worth running if C1 surfaces real RAM-pressure incidents.
