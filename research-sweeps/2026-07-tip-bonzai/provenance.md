# Provenance: Prism ML / "Bonzai 27B" (actual name: Bonsai 27B)

Sweep date: 2026-07-19. All data below pulled from primary sources (Hugging Face API, GitHub API, vendor site, Together AI) on this date.

## Bottom line

The release is real. The video got the name wrong ("Bonzai" is actually **Bonsai**) and mischaracterized the method (it is not naive post-hoc PTQ; Prism ML's own README references distillation and a trained drafter layer). Company, HF org, base model, file sizes, and the source-available llama.cpp fork all verify. Quality-retention numbers remain vendor-claimed with only thin independent verification so far.

## 1. Prism ML the company

- Site: https://prismml.com (live, product site for the Bonsai family), docs at https://docs.prismml.com, Discord discord.gg/prismml, contact@prismml.com, LinkedIn `prismml`, X `@PrismML`.
- Self-description: "building ultra dense intelligence" for edge devices. Announcement page (https://prismml.com/news/bonsai-27b) states: "PrismML emerged from a team of Caltech researchers and was founded with support from Khosla Ventures, Cerberus, and Google, with continuing support from Samsung." This is vendor-claimed; I found no independent funding coverage in this sweep. Site displays Caltech, Google, and Cerberus logos.
- GitHub org `PrismML-Eng`: created **2026-02-03** (GitHub API), 9 public repos, 238 followers.
- Hugging Face org `prism-ml`: 33 public models, 11 team members (names private), 2,032 followers, links back to prismml.com and PrismML-Eng.

## 2. Hugging Face repos (exact names, HF API data)

Org: https://huggingface.co/prism-ml

Key 27B repos (createdAt / downloads (30-day counter) / likes):

| Repo | Created | Downloads | Likes |
|---|---|---|---|
| prism-ml/Bonsai-27B-gguf (1-bit) | 2026-07-04 | 1,218,815 | 456 |
| prism-ml/Ternary-Bonsai-27B-gguf | 2026-07-04 | 301,893 | 753 |
| prism-ml/Bonsai-27B-mlx-1bit | 2026-07-04 | 20,639 | 128 |
| prism-ml/Ternary-Bonsai-27B-mlx-2bit | 2026-07-04 | 17,063 | 115 |
| prism-ml/Ternary-Bonsai-27B-AWQ-4bit | 2026-07-11 | 2,959 | 14 |

The org has a track record predating the 27B launch: Bonsai-8B-gguf created **2026-03-18**, then 1.7B/4B (2026-03-29), Ternary 1.7B/4B/8B (April), bonsai-image 4B models (May). 34 model repos total in the listing I pulled. This is a progressive release history, not a fresh account.

Download-claim check: the video said "1M downloads in 72 hours." The HF 30-day counter shows 1.22M on Bonsai-27B-gguf as of 2026-07-19, with repo creation 2026-07-04 and public announcement 2026-07-14 (MarkTechPost, Together AI both date it July 14). 1M+ since launch is confirmed; the exact 72-hour framing cannot be verified from HF's 30-day counter and may be compressed for drama.

## 3. File listing, Ternary-Bonsai-27B-gguf (HF tree API, exact bytes)

- Ternary-Bonsai-27B-Q2_0.gguf: **7,165,121,600 bytes = 7.17 GB. This matches the video's "~7 GB / 7.17 GB" file almost exactly** (video's filename "ternary_bonzai_27b_q2.gguf" is garbled; real name above).
- Ternary-Bonsai-27B-PQ2_0.gguf: 7,165,121,600 bytes
- Ternary-Bonsai-27B-Q2_g64.gguf: 7,585,330,240 bytes
- Ternary-Bonsai-27B-F16.gguf: 53,808,280,640 bytes (53.8 GB FP16 reference)
- Ternary-Bonsai-27B-mmproj-BF16.gguf: 931,145,760 bytes and mmproj-Q8_0: 629,246,880 bytes (**vision mmproj claim: confirmed**)
- Ternary-Bonsai-27B-dspark-bf16.gguf: 7,291,885,792 bytes and dspark-Q4_1: 1,946,393,568 bytes (trained "DSpark" drafter, see method note below)
- Plus .eval_results yamls (aime_2026, gsm8k, mmmu_pro), LICENSE (apache-2.0 tagged), README

MLX and AWQ variants exist as separate repos (confirmed above). The 3.9 GB 1-bit phone build is the Bonsai-27B (non-ternary) line; site and third-party pages consistently state ~3.9 GB (I did not pull that repo's tree in this sweep).

## 4. Base model: Qwen/Qwen3.6-27B exists

- https://huggingface.co/Qwen/Qwen3.6-27B, created **2026-04-21**, lastModified 2026-04-24, **5,395,520 downloads**, 1,994 likes, apache-2.0.
- Architecture `Qwen3_5ForConditionalGeneration` (model_type qwen3_5), multimodal image-text-to-text, no MoE markers in tags: consistent with a dense 27B. Bonsai repos carry the tag `base_model:quantized:Qwen/Qwen3.6-27B`, so the lineage claim is explicit on HF.

## 5. The llama.cpp fork (our build-from-source path exists)

- https://github.com/PrismML-Eng/llama.cpp: genuine fork of ggml-org/llama.cpp (parent: 120,889 stars), created **2026-02-25**, default branch **`prism`**, 359 stars, 70 forks, MIT, 20 open issues, last push 2026-07-18. Full source is available: the diff vs mainline is readable and buildable, so our no-prebuilt-binaries posture is workable.
- Releases DO ship prebuilt binaries (video claim confirmed): e.g. `prism-b9596-9fcaed7` published 2026-07-18 with ~20 assets across win/linux/macos/iOS (CUDA 12.4/12.8, ROCm, Vulkan, arm64, xcframework). Earlier releases 2026-07-17 and 2026-07-14. We ignore the binaries and build the `prism` branch from source after reading the diff.
- README instructs cloning the fork because mainline lacks the "Q2_0_g128 hybrid-attention kernels", consistent with the video's "custom fork required" claim.

## 6. Method: the video's "post-hoc" framing is wrong

Neither the model README, docs.prismml.com, nor the announcement describes naive post-hoc ternarization of a dense FP16 checkpoint. The README references **distillation** and a "**DSpark drafter layer trained against the low-bit target**", and points to a whitepaper for full methodology. Vendor states the low-bit representation runs end to end (embeddings, attention, MLPs, LM head) at a true 1.71 bpw ternary. So this is a training/distillation-assisted low-bit release (BitNet-adjacent territory), not the frontier-breaking PTQ the video implied. The extraordinary-claim alarm was aimed at the video's characterization, not at what Prism ML actually shipped.

Vendor-claimed retention: 94.6% for ternary (80.49 vs 85.07 FP16 across 15 thinking benchmarks), 89.5% for 1-bit. These are vendor numbers with eval yamls in-repo.

## 7. Independent signals (thin but nonzero)

- **Together AI hosts it**: https://www.together.ai/models/prism-ml-ternary-bonsai-27b, serverless endpoint `Prism-ML/Ternary-Bonsai-27B`, release dated July 14, 2026, 262K context. A major commercial host onboarding the model is a meaningful third-party signal.
- **Independent benchmark repo**: https://github.com/ArmanJR/PrismML-Bonsai-vs-Qwen3.5-Benchmark (community eval claiming the ternary build is comparable to Qwen3.6-27B NVFP4 for agentic workflows in the 12 to 16 GB VRAM class).
- **WebGPU community demo**: https://huggingface.co/spaces/webml-community/bonsai-webgpu-kernels (webml-community, not Prism ML).
- **Press**: MarkTechPost 2026-07-14 (https://www.marktechpost.com/2026/07/14/prismml-releases-bonsai-27b-1-bit-and-ternary-builds-of-qwen3-6-27b-that-run-on-laptops-and-phones/), technosports.co.in. Churn-tier outlets, but they corroborate the release date and claims.
- **Community reception is mixed**: r/LocalLLaMA testers reportedly called it "benchmaxxed" (strong on the published suite, weaker off-suite). Treat the 95% number as unproven until we run our own evals.

## 8. Not covered in this sweep

Kolibri (SSD expert streaming), "Hi3 in 1-bit", and Angel Slim were not mentioned anywhere in Prism ML's materials and were not investigated here; they are separate ecosystem claims from the video and need their own verification. Note Tencent's AngelSlim toolkit is a real, known project independent of Prism ML.

## Suggested next steps for our project

1. Clone PrismML-Eng/llama.cpp branch `prism`, diff against upstream ggml-org/llama.cpp at the fork point, review the Q2_0_g128 kernel code, then build from source (never the release binaries).
2. Pull Ternary-Bonsai-27B-Q2_0.gguf (7.17 GB) and run our own eval slice against our Qwen3-8B baseline before believing any retention number.
3. Watch r/LocalLLaMA for off-suite eval results; the "benchmaxxed" complaints are the main open risk.
