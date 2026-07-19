# Fork inspection: PrismML llama.cpp fork (Bonsai 27B tip)

Date: 2026-07-19. Method: GitHub REST API + repo pages only. No release assets downloaded, no binaries executed.

## Bottom line

The fork is real, full source is public, the diff contains genuine quantization kernel work, and the author is actively upstreaming the format into mainline llama.cpp with constructive engagement from ggerganov. This lane supports "real project." Several details in the YouTube video are garbled or inflated (name is Bonsai not Bonzai, "requires their fork" is now only partially true, GitHub download counts are in the hundreds, not millions).

## 1. Does the repo exist? Who is behind it?

- Fork: https://github.com/PrismML-Eng/llama.cpp
  - True GitHub fork of ggml-org/llama.cpp (parent link intact). Created 2026-02-25, default branch `prism`, last push 2026-07-18 (daily activity).
  - 359 stars, 70 forks, 20 open issues. MIT license (inherited from upstream).
- Org: https://github.com/PrismML-Eng
  - Created 2026-02-03, 9 public repos, 238 followers, no website/email/location listed. Young org (about 5.5 months old at inspection).
- Other org repos: forks of mlx, mlx-swift, mlx-c, sglang (all with a `prism` branch), plus Bonsai-demo (1,823 stars, launcher that auto-downloads prebuilt binaries), Bonsai-Image-Demo (507 stars), image-studio.
- Principal engineer: khosravipasha (Pasha Khosravi), plus bri-prism. khosravipasha is also the author of the merged mainline PRs below, i.e. the same person contributes to ggml-org/llama.cpp under review by upstream maintainers.

## 2. Is full source public? What does the diff actually contain?

Compared via https://github.com/ggml-org/llama.cpp/compare/master...PrismML-Eng:llama.cpp:prism

- Status: diverged, 44 commits ahead, 516 behind upstream master. All 44 ahead commits are visible source commits, nothing hidden.
- The ahead commits are real kernel engineering, not a wrapper:
  - Q2_0 (2-bit ternary, group-size-128) type definition and kernels for CPU (AVX-512-VNNI, AVX-VNNI fast paths), Metal, CUDA, Vulkan.
  - Q1_0 (1-bit) ARM NEON DOTPROD/I8MM dot products and repack kernels; CUDA byte-permute extraction speedups.
  - Opt-in Hopper wgmma (sm_90a) tensor-core mul_mat paths for Q1_0/Q2_0 prefill (claimed 7 to 8 percent gains) with multi-GPU hardening.
  - KV-cache mean-centering for Q4_0 K-cache with a calibration tool, GGUF bias file format, docs, and invariance tests.
  - Release CI: `.github/workflows/release-prism.yml` builds the published binaries from this public source.
- Files touched include ggml CUDA/Metal sources, `arch/x86/quants.c`, ARM NEON quant sources, common/ and tools/ for the KV calibration utility, tests, and docs. This is consistent with a legitimate quantization-format fork.

## 3. Releases: binary-only or source?

- Releases page ships prebuilt binaries (example: `prism-b9596-9fcaed7`, published 2026-07-18): Windows CPU x64/arm64, CUDA 12.4, Vulkan, HIP/Radeon; Linux CPU/CUDA 12.4 and 12.8/Vulkan/ROCm; macOS arm64/x64; iOS xcframework. GitHub also auto-attaches source archives for every tag, and the full source branch is public, so this is binaries-plus-source, not binary-only.
- Download counts are modest: the largest asset (win CUDA) shows a few hundred downloads. The video's "1M downloads in 72 hours" is not supported by GitHub release stats (Hugging Face counts are a separate question for the model-page lane).
- Security signals: found no malware reports, antivirus flags, or community warnings about the binaries in issues or web search. Issue #71 complains some Windows release zips were missing executables (packaging sloppiness, fixed by PR #78). Our posture stands regardless: build from source, skip the binaries and skip Bonsai-demo's binary auto-download.

## 4. Issue and community history (legitimacy signals)

- Real outside users filing real bugs and PRs: merged outside PR #72 (AVX-VNNI Q2_0 fast path, 3.2x), #76, #79 (Blackwell int8 MMA proposal), #83 (fork runs non-ternary quants 4.5 percent slower than mainline, an honest regression report left open), #77 (infinite prompt-processing loop), #89 (Intel Arc Vulkan device resets).
- Third-party ecosystem on GitHub: ArmanJR/PrismML-Bonsai-vs-Qwen3.5-Benchmark (102 stars, independent benchmark repo), MiaAI-Lab/Ternary-Bonsai-27B-tool-eval-bench-results (tool-calling eval results), nisten/prism-ml-biturbo, community Windows-build repos (fabiomatricardi, froodx), a third-party CLI (nareshnavinash/bonsai), and Mintplex-Labs/prism-ml-llama.cpp. This is organic-looking, multi-actor activity, not a single sock-puppet cluster.

## 5. Does mainline llama.cpp already handle this? (key nuance)

- Mainline has had TQ1_0/TQ2_0 ternary types since Aug 2024 (compilade's PR #8151, for TriLM/BitNet trained-ternary models, group/row scaling different from Prism's format). Those types cannot load Prism's Q2_0 files.
- Prism opened mainline discussion #22019 (2026-04-16): https://github.com/ggml-org/llama.cpp/discussions/22019. ggerganov engaged constructively and recommended group size 64 instead of 128; the upstreamed format standardized on g64.
- Upstreaming status in ggml-org/llama.cpp:
  - PR #24448 "ggml: add Q2_0 quantization support (CPU)" by khosravipasha: merged 2026-06-11.
  - PR #25419 "metal: Q2_0 backend" by khosravipasha: merged 2026-07-07.
  - PR #25707 "CUDA: add Q2_0 support" by khosravipasha: open as of 2026-07-15 (a competing draft #25603 also open).
- Practical consequence for us: recent mainline builds can already load `*-Q2_0_g64.gguf` Bonsai files on CPU (and Metal). CUDA is not merged yet, so on our RTX machine mainline b10064 cannot GPU-offload the ternary layers; that is exactly what fork releases exist for. The fork's original `*-Q2_0.gguf` g128 files load only on the fork. The video's "mainline lacks the format, you must use their fork" is therefore outdated/oversimplified but was true before June 2026 and is still true for CUDA offload and for g128 files.

## 6. Buildable from source on Windows?

- Yes, by all appearances. The fork keeps the stock llama.cpp CMake build system and stock README/docs (only a Prism preamble added), and its own public CI workflow builds Windows CPU/CUDA/Vulkan/HIP binaries from the same source we would build.
- Independent parties have built Windows binaries from this source (fabiomatricardi/llamacpp-PrismML-Bonsai-1bit, froodx/prism-ml-llama.cpp with Windows build scripts), so a local MSVC or CUDA toolkit build following standard llama.cpp docs/build.md should work.
- Recommended path given our posture: clone PrismML-Eng/llama.cpp at a tagged release commit, read the 44-commit diff vs upstream (it is small and reviewable), then CMake-build with CUDA locally. Alternative that avoids the fork entirely: use g64 GGUFs on mainline CPU today, or wait for mainline CUDA PR #25707 to merge.

## 7. Caveats and what this lane does not establish

- This lane verifies the code and repo hygiene, not the quality claim. "94.6 percent of FP16 retained" (vendor number; the video rounded to 95) is Prism's own benchmark; whether the process is truly post-hoc PTQ or involves quantization-aware finetuning is not established by the repo alone. Independent benchmark repos exist (ArmanJR, MiaAI-Lab) but were not audited here.
- Naming in the video is wrong: model is "Bonsai" / "Ternary-Bonsai-27B", vendor is PrismML (org PrismML-Eng), base is Qwen3.6-27B per press coverage (MarkTechPost, 2026-07-14).
- Org is young (Feb 2026) with no listed website or identity details; reputation rests on the public code and the upstream maintainers' engagement, which is substantial.

## Links

- Fork: https://github.com/PrismML-Eng/llama.cpp
- Org: https://github.com/PrismML-Eng
- Diff vs upstream: https://github.com/ggml-org/llama.cpp/compare/master...PrismML-Eng:llama.cpp:prism
- Releases: https://github.com/PrismML-Eng/llama.cpp/releases
- Mainline format discussion: https://github.com/ggml-org/llama.cpp/discussions/22019
- Mainline CPU Q2_0 (merged): https://github.com/ggml-org/llama.cpp/pull/24448
- Mainline Metal Q2_0 (merged): https://github.com/ggml-org/llama.cpp/pull/25419
- Mainline CUDA Q2_0 (open): https://github.com/ggml-org/llama.cpp/pull/25707
- Historical mainline ternary types TQ1_0/TQ2_0: https://github.com/ggerganov/llama.cpp/pull/8151
- HF model (other lane): https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf
- Press: https://www.marktechpost.com/2026/07/14/prismml-releases-bonsai-27b-1-bit-and-ternary-builds-of-qwen3-6-27b-that-run-on-laptops-and-phones/
- Independent benchmark repo (unaudited): https://github.com/ArmanJR/PrismML-Bonsai-vs-Qwen3.5-Benchmark
