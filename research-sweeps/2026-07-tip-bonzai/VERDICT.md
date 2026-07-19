# VERDICT: "Bonzai 27B" (real name: PrismML Bonsai 27B)

Date: 2026-07-19. Consolidates provenance.md, credibility.md, fork-inspection.md, calibration.md.

## Verdict: REAL BUT OVERSOLD

The release exists exactly as the video described in substance, with every proper noun garbled and the two headline claims inflated. PrismML is a real vendor with a 5-month public track record, the models are on Hugging Face with 1.2M+ monthly downloads, Together AI hosts a paid API endpoint, and the llama.cpp fork is full MIT source that is actively being merged into mainline. What is oversold: the "95% intelligence retention" number (vendor-measured, rounded up from 94.6%, on a vendor-chosen suite; the only independent suite lands around 91.5%) and the framing that this is post-hoc ternarization with no retraining (nobody, including PrismML, actually claims that; the method is undisclosed and the evidence points to distillation or recovery training).

## The single strongest piece of evidence

**PrismML's Q2_0 ternary format has been merged into mainline llama.cpp by upstream maintainers.** PR #24448 (CPU, merged 2026-06-11) and PR #25419 (Metal, merged 2026-07-07) were authored by PrismML's own engineer (khosravipasha) and reviewed by ggerganov, who engaged constructively in discussion #22019 and shaped the format (group size 64). CUDA PR #25707 is open. Fake projects do not survive code review by the most scrutinized inference codebase in local AI. This single fact settles "real" beyond reasonable doubt.

The strongest evidence on the "oversold" side: the only independent accuracy suite (ArmanJR, 98 questions x 7 categories x 3 runs, 13 models) scored Ternary-Bonsai-27B at 86.2% vs Qwen3.6-27B at 94.2%, about 91.5% relative retention rather than 95%, with the entire loss concentrated in math and multilingual. That partially contradicts the vendor's "math is nearly untouched" story.

## What the honest numbers look like

| Claim in the video | Honest version |
|---|---|
| 95% intelligence retained | Vendor: 94.6% on its own 15-benchmark suite, own harness, per-model tuned temperatures. Best independent suite: ~91.5% relative. 1-bit build: vendor 89.5%, independent ~88%. Losses concentrate in math and multilingual. |
| Post-hoc ternary, no retraining | Never claimed by PrismML. Method is a black box ("proprietary Caltech IP"). Repo ships a trained DSpark drafter and the README references distillation. Naive PTQ ternary of Qwen demonstrably produces gibberish (verified by an HN commenter). So this is BitNet-adjacent trained/recovered low-bit work, not a physics-breaking PTQ discovery. Aggressive vs published art (TWLA, June 2026, ~90-92%), but by a few points, not an order of magnitude. |
| ~7 GB | True: 7.17 GB deployed GGUF. The marketed 5.9 GB is ideal 1.71 bpw packing that current kernels cannot store. |
| 1M downloads in 72 hours | Magnitude real (1.2M+ on the main repo, ~1.6-1.8M org-wide, 30-day HF counters, repos created 2026-07-04). The 72-hour framing is unverifiable and probably compressed for drama. GitHub fork downloads are in the hundreds. |
| Requires their custom fork | Outdated. Mainline llama.cpp already loads the g64 ternary GGUFs on CPU and Metal. The fork is only required for CUDA offload and for the original g128 files, and it is full source, not binary-only. |
| Kolibri / Hi3 / Angel Slim ecosystem | All real: Colibri (744B GLM-5.2 on 25 GB RAM, but 0.05-1.2 tok/s, a tech demo), Tencent Hy3 1-bit (89.4 GB, not 83), Tencent AngelSlim (which includes Tequila, a peer-reviewed ICLR 2026 ternary method). Names garbled, numbers near-exact. |

Community temperature check: HN hands-on reports are mixed ("slightly underwhelmed" vs a 4-bit 35B; "benchmaxxed" complaints on r/LocalLLaMA). Nobody independent has replicated the 15-benchmark thinking-mode suite.

## Is testing worth our time?

**Yes, one bounded afternoon, because the cheap path costs us zero new trust.** Mainline llama.cpp already supports the g64 files on CPU, so Phase 0 requires no fork, no new binaries, and no build. The model file itself is inert data (GGUF weights, no code execution).

The honest expectation for our rig tempers the excitement: Bonsai is a dense 27B, so every token reads all 7.17 GB. On our 8 GB RTX 5060 Laptop that means CPU-heavy inference at roughly 5-7 t/s CPU-only (50.8 GB/s / 7.17 GB ceiling), maybe 12-20 t/s with fork CUDA partial offload. Our E013/E023 champion (Qwen3-30B-A3B MoE Turbo) does ~40 t/s. So Bonsai only earns a slot if its quality clearly beats the MoE, which is exactly what a small eval settles.

## Safe test plan (source-build only, no vendor binaries ever)

Hard rules: never run PrismML release binaries; never run Bonsai-demo (it auto-downloads prebuilt binaries); weights-only downloads from HF are fine.

**Phase 0: zero-new-trust smoke test (existing official b10064 binary)**
1. Download `Ternary-Bonsai-27B-Q2_g64.gguf` (7.59 GB) from `prism-ml/Ternary-Bonsai-27B-gguf` into `models/`.
2. Run CPU-only on our existing mainline b10064 (true CPU via `CUDA_VISIBLE_DEVICES=-1`, per E001 gotcha). Pre-warm into page cache, `--mlock`, `-t 8` dense-thread optimum.
3. Smoke prompts plus paired perplexity on `datasets/ppl-text.txt` against Qwen3-8B Q4_K_M and Qwen3-30B-A3B Q4_K_M. A/B/A thermal flanking per protocol law.

**Phase 1: quality discriminator slice (same harness for all models)**
4. 100-200 GSM8K items + an IFEval slice, run identically on: Ternary-Bonsai-27B, Qwen3.6-27B IQ2_XXS (9.4 GB, the vendor's own comparison baseline), Qwen3-30B-A3B Q4_K_M, Qwen3-8B Q4_K_M. Record decode t/s and RAM/VRAM alongside scores.

**Phase 2: only if quality survives, fork source build for CUDA speed**
5. Clone `PrismML-Eng/llama.cpp` branch `prism` at a tagged release commit. Diff vs the upstream fork point (44 commits ahead; touches ggml CUDA/Metal kernels, `arch/x86/quants.c`, tests, CI). Read the whole diff; flag anything touching networking, process spawning, or file paths outside model I/O (none expected per fork-inspection.md).
6. Build with our known-good env: `cmake -G Ninja` inside vcvars64 (`tools/ninja/`). Blocker to clear first: CUDA toolkit is still not installed (needs Sohail's admin install). Cleaner alternative that fully preserves our posture: wait for mainline CUDA PR #25707 to merge and use official ggml-org binaries.

**Pre-registered hypotheses (write down before running)**
- H1: Bonsai ternary retains 88-92% of Qwen3.6-27B reference scores on our slice, below the vendor's 94.6%.
- H2: it clearly beats Qwen3.6-27B IQ2_XXS at comparable size (the vendor's favorite comparison is probably legitimate).
- H3: the largest deficit shows up in GSM8K/math, per ArmanJR.
- H4: CPU-only decode lands 5-7 t/s; fork-CUDA hybrid under 20 t/s.
- H5 decision rule: adopt only if Bonsai beats Qwen3-30B-A3B on the quality slice by a margin that justifies giving up more than half our tokens per second. Otherwise document and file as a reference point.

## What to tell the user about the video

The video is a garbler, not a fabricator. Every checkable claim traced back to a real event in the July 10-14 launch window; the proper nouns are mangled in a pattern typical of AI narration reading a feed (Bonzai/Bonsai, Kolibri/Colibri, Hi3/Hy3). But it repeated vendor marketing as fact, invented the "post-hoc, no retraining" mechanism the vendor never claimed, rounded the retention number up, and pushed prebuilt binaries as the required path when full source exists and mainline support is landing. Treat this channel as a lead generator, never as verification, and never run binaries it points to.
