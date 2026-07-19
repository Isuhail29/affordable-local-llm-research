# Domain survey: uncensored image-generation models that run in 8 GB VRAM

Sweep: 2026-07-sweep-02-quality. Date: 2026-07-20.
Scope: open-weight text-to-image models that fit our RTX 5060 Laptop 8 GB (Blackwell sm_120), ranked by best quality achievable inside 8 GB, with censorship status stated plainly. Families covered: SD 1.5 community checkpoints, the SDXL family (Juggernaut, Pony V6, Illustrious, NoobAI), Flux.1 (dev/schnell/Krea and GGUF/NF4 quantization), Chroma, SD 3.5, and the 2025-2026 efficient releases (Z-Image, Qwen-Image, HunyuanImage).

Rig constraints applied throughout: RTX 5060 Laptop 8 GB GDDR7 (~256 GB/s, Blackwell sm_120, CUDA 13.3), 48 GB DDR5-5600, i7-14650HX (AVX2, no AVX-512/AMX), Windows 11, ~3 MB/s internet (~10.8 GB/hour), ~416 GB free disk.

---

## Framing: our LLM throughput law does NOT transfer to diffusion

Our whole LLM program is governed by one law: autoregressive token generation is memory-bandwidth-bound, capped at ~30-42 t/s by the ~60 GB/s DDR5 ceiling, and no orchestration trick raises it.

**Image diffusion is a different physics problem.** Denoising is iterative dense matmul/attention over a fixed latent, run for N steps. It is GPU-compute-bound, not host-bandwidth-bound. So two things follow that are the opposite of the LLM situation:

1. **If the model's transformer/UNet fits in the 8 GB of VRAM, it runs at full Blackwell GPU speed.** The 5060's FLOPS are the resource that matters, and they are decent. The 48 GB of system RAM does NOT gate throughput the way it gates our LLMs.
2. **The 8 GB wall is a capacity wall, not a speed wall.** The question for every model below is "does the denoiser (plus the bits that must be resident) fit in 8 GB", and "what do we offload to the 48 GB host to make it fit". Text encoders (T5-XXL is the repeat offender, ~9 GB in fp16) and the VAE can live on the CPU or be swapped in and out with a small latency cost. The denoiser wants to stay resident.

Practical consequence: our 48 GB RAM is a genuine asset here. It lets us CPU-offload the T5/CLIP encoders and VAE for Flux/Chroma/SD3.5 the same way `--n-cpu-moe` offloads experts, keeping the DiT resident in the 8 GB. "Quality per 8 GB" is the real axis, and wall-clock per image (seconds, not t/s) is the cost axis.

## Tooling reality on sm_120 / Windows

Two runnable-today stacks, both confirmed to support Blackwell-era cards in 2026:

- **ComfyUI** is the mainstream. It needs a PyTorch build on CUDA 12.8+ for sm_120 (RTX 50-series), which is standard now. GGUF loading for Flux/Chroma/Qwen-Image is via the **city96 ComfyUI-GGUF** custom node. `--lowvram` streams the denoiser layers from system RAM when a model is slightly too big; this is the 8 GB survival flag. Sources: [Next Diffusion low-VRAM GGUF guide](https://www.nextdiffusion.ai/tutorials/how-to-run-flux-dev-gguf-in-comfyui-low-vram-guide), [Apatero 8 GB GGUF guide](https://apatero.com/blog/flux-gguf-quantization-8gb-vram-guide-2026).
- **stable-diffusion.cpp (leejet)** is the ggml-native option and the natural extension of our existing llama.cpp expertise: pure C/C++, GGUF weights, CUDA/Vulkan backends, same quantization vocabulary we already speak. It explicitly supports SD, SDXL, Flux (1 and 2), Chroma, Qwen-Image, and Z-Image with preconverted GGUFs. Repo: [github.com/leejet/stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp). sm_120 needs a CUDA-13-arch build flag; unverified on our exact box, so this is a build-and-smoke-test item, not a given.

A note on the word "uncensored". Open diffusion weights are not classifier-locked the way a hosted API is. The base SD/SDXL pipelines ship an optional CLIP safety-checker that is trivially disabled and is not part of the weights. The real censorship axis is **what the training data contained**: a model trained on SFW-only data simply cannot render what it never saw, no flag fixes that. So "uncensored" below means the weights (or a community finetune of them) actually produce unrestricted output, not merely "the safety filter is off".

---

## Tier 1: best uncensored quality that genuinely fits 8 GB

### 1. Chroma1-HD (lodestones) - the best uncensored high-fidelity model that fits

The standout. A community Flux derivative purpose-built to be uncensored, at Flux-class quality, small enough to fit 8 GB after quantization.

- **What it is:** 8.9B-parameter text-to-image model, architecturally reduced from Flux.1-schnell's 12B (redundant modulation layers pruned), then **retrained from scratch on a 5M curated set (20M pool)** covering artistic, photographic, and niche styles. Not a LoRA on Flux, an actual retrain. Card: [huggingface.co/lodestones/Chroma1-HD](https://huggingface.co/lodestones/Chroma1-HD).
- **Censorship:** explicitly uncensored. The card states it "is released in a state as is and has not been aligned with a specific safety filter." This is the design intent, not a side effect.
- **License:** **Apache 2.0.** Fully open, commercial use fine. This is rare at this quality tier and a big deal versus Flux.1-dev's non-commercial license.
- **8 GB fit (GGUF, [silveroxides](https://huggingface.co/silveroxides/Chroma1-HD-GGUF) / [QuantStack](https://huggingface.co/QuantStack/Chroma1-HD-GGUF)):** Q4_K_S **5.43 GB**, Q4_K_M 5.57 GB, Q5_K_S **6.51 GB**, Q6_K 7.65 GB, Q8_0 9.74 GB, BF16 17.8 GB. Q5_K_S at 6.51 GB is the sweet spot: leaves ~1.5 GB for the working set at 1024x1024 with the **T5 text encoder offloaded to our 48 GB RAM**. Q4_K_S if headroom is tight.
- **Resolution:** 1024x1024 native (HD variant); the 512-res base is a separate repo ([Chroma1-Base](https://huggingface.co/lodestones/Chroma1-Base)).
- **Family:** Chroma1-Base (512), Chroma1-Flash (CFG "baked" for fewer steps), **Chroma1-Radiance** (pixel-space variant that skips the VAE to avoid compression artifacts, still WIP as of mid-2026: [sd.cpp discussion #971](https://github.com/leejet/stable-diffusion.cpp/discussions/971)). Civitai hub: [civitai.com/models/1330309](https://civitai.com/models/1330309/chroma).
- **Runnable today:** yes, both ComfyUI-GGUF and stable-diffusion.cpp list Chroma support with preconverted GGUFs.
- **Download:** ~6.5 GB (~1 hour) plus a T5 encoder (~5 GB fp8 or GGUF, one-time, shared with Flux).

**Fit verdict: the headline candidate.** Flux-class fidelity + genuinely uncensored + Apache 2.0 + fits 8 GB at Q5. Nothing else in the survey matches all four.

### 2. Flux.1 [dev] via GGUF - the prompt-adherence and in-image-text king

The reference for coherence, complex prompts, and legible text-in-image. Fits 8 GB only after quantization, and the license is the catch.

- **What it is:** 12B guidance-distilled rectified-flow transformer from Black Forest Labs. Best-in-class prompt following and text rendering among open models. Base GGUF repo: [city96/FLUX.1-dev-gguf](https://huggingface.co/city96/FLUX.1-dev-gguf).
- **Censorship:** the base is aesthetically/safety-aligned (not a hard classifier in the weights, but the training leans SFW). It is "lightly censored" in practice; the enormous community LoRA ecosystem plus the fact the weights aren't classifier-locked means NSFW is achievable, but it takes LoRAs/effort rather than working out of the box. This is exactly the gap Chroma was built to close.
- **License:** **flux-1-dev-non-commercial-license.** Fine for our research, NOT for commercial output. Chroma (Apache) is the commercial-safe alternative at similar quality.
- **8 GB fit (city96 GGUF):** Q2_K 4.03, Q3_K_S 5.23, Q4_0 **6.79**, Q4_K_S **6.81**, Q5_K_S 8.29, Q6_K 9.86, Q8_0 12.7, F16 23.8 GB. **Q4_K_S (6.81 GB) is the documented 8 GB sweet spot**; Q4_0 is equivalent size with wider node compatibility. NF4 (bitsandbytes 4-bit) is ~6-7 GB and can edge out GGUF-Q4 on quality at 4-bit, but is less flexible across nodes.
- **The T5 gotcha:** Flux needs the **T5-XXL text encoder (~9 GB fp16)** on top of the denoiser. On 8 GB you MUST use fp8/GGUF T5 and/or CPU-offload it. This is the single biggest 8 GB planning item for the whole Flux/Chroma/SD3.5 family, and where our 48 GB RAM earns its keep. Sources: [Local AI Master low-VRAM Flux](https://localaimaster.com/blog/run-flux-on-low-vram-gpu), [Apatero](https://apatero.com/blog/flux-gguf-quantization-8gb-vram-guide-2026).
- **Variants worth knowing:** **Flux.1 [schnell]** (Apache 2.0, 4-step distilled, faster but lower fidelity, same size class) and **Flux.1 Krea [dev]** (photorealism-tuned dev, same license/size). Schnell's Apache license is why Chroma chose it as a base.
- **Caution on Flux.2:** Black Forest Labs' **Flux.2 [dev]** (Nov 2025) introduced **mandatory safety filtering in both the license and the pipeline**, a real regression for uncensored use, and it is larger/harder to fit 8 GB. For our purpose Flux.1 (and Chroma) is the target, not Flux.2. Source: [Local AI Master uncensored guide](https://localaimaster.com/blog/uncensored-local-image-generation).
- **Download:** ~6.8 GB denoiser + ~5 GB T5 (~1.1 hours combined).

**Fit verdict: excellent for quality, license-limited for output.** Best coherence/text of anything that fits, but for uncensored-by-default and commercial use, Chroma beats it. Keep Flux.1-dev as the coherence/benchmark reference.

### 3. Z-Image Turbo (Alibaba Tongyi) - the 2025-2026 efficiency champion

The newest and most interesting entry: near-Flux quality from a 6B model, 8 steps, Apache 2.0, and it fits 8 GB with room to spare.

- **What it is:** 6B "Scalable Single-Stream DiT" (S3-DiT) from Alibaba's Tongyi Lab, released **Nov 27, 2025** (arXiv 2511.22699). The **Turbo** variant is distilled to **8 NFEs** (vs 50 for base Z-Image), so it is fast. Bilingual (EN/CN) text rendering, strong instruction adherence, photorealistic. Card: [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo).
- **Censorship:** effectively uncensored. Community testing notes "many concepts censored by other models are doable out of the box" ([Local AI Master](https://localaimaster.com/blog/z-image-turbo-comfyui)). No documented content filter on the card.
- **License:** **Apache 2.0**, commercial use fine.
- **8 GB fit:** ~12 GB bf16, **~8 GB FP8, ~6 GB GGUF** ([Z-Image low-VRAM quant on Civitai](https://civitai.com/models/2169712/z-image-turbo-quantized-for-low-vram)). Fits 8 GB in FP8 or GGUF. The 6B size plus 8-step schedule means it is both small and fast, the rare combination.
- **Standing:** reported top-ranked open-source image model on the Artificial Analysis leaderboard as of its release window ([RunDiffusion](https://www.rundiffusion.com/z-image)). Treat leaderboard claims as vendor-adjacent until we falsify, per our usual rule.
- **Runnable today:** yes, ComfyUI native + GGUF, and stable-diffusion.cpp lists Z-Image support.
- **Download:** ~6-8 GB (~1 hour).

**Fit verdict: the sleeper pick.** If the quality claim survives our own falsification, this is the best quality-per-second AND quality-per-GB on the list: Apache, uncensored, 8 steps, fits comfortably. Directly analogous to how Qwen3.5-9B was our LLM sleeper.

### 4. SDXL uncensored finetunes - the mature, fully-resident mainstream

The workhorse. SDXL is the only high-quality base that fits 8 GB with the denoiser **fully resident and no encoder-offload gymnastics**, and its community finetunes are the uncensored mainstream with the deepest LoRA/ControlNet ecosystem in existence.

- **What it is:** SDXL 1.0 base is 3.5B (2.6B UNet), 1024x1024 native, ~6.9 GB fp16 full checkpoint (UNet + VAE + dual CLIP, all self-contained, no T5). License CreativeML OpenRAIL++-M (permits NSFW; has behavioral-use clauses, not a true open license but universally used).
- **8 GB fit:** the entire checkpoint (~6.9 GB) loads resident; 1024x1024 generation is the standard target. 8 GB is the documented minimum; hires-fix, ControlNet, and heavy LoRA stacking want 12 GB, so on our card do those sparingly or at lower res. Our 48 GB RAM covers any spillover comfortably. Source: [offlinecreator VRAM-ranked list](https://offlinecreator.com/blog/best-uncensored-ai-models-civitai-2026), [TechTactician Illustrious comparison](https://techtactician.com/best-illustrious-xl-sdxl-anime-model-fine-tunes-comparison/).
- **Photorealistic, uncensored finetunes:** **Juggernaut XL** (v9/v10/X), **RealVisXL**, **AlbedoBase XL**, **DreamShaper XL**. All trained without content filters, all photoreal-capable.
- **Anime / illustration, the uncensored leaders:**
  - **NoobAI-XL (V-Pred 1.0)** is currently the strongest uncensored anime base by community ELO: best tag comprehension and anatomy. Cost: it is a **v-prediction** model, so it needs v-pred settings + Euler + CFG 4-5 + 28+ steps, not the SDXL defaults. Source: [offlinecreator Illustrious NSFW list](https://offlinecreator.com/civitai/illustrious-xl-nsfw-models-2026).
  - **Illustrious XL** (v1.1 is the practical daily driver with full LoRA back-compat; v2.0 is an intentionally-untuned merge/training base).
  - **Pony Diffusion V6 XL** remains the most thoroughly uncensored, tag-driven (score_9 etc.) anime/stylized base and the ecosystem anchor.
- **Pony V7 status:** still an extended development/early-access arc as of mid-2026, moving OFF SDXL to a larger base (AuraFlow was the leading candidate; the creator also weighed SD3). It has NOT displaced the SDXL Pony/Illustrious/NoobAI trio as the practical uncensored mainstream. Treat V7 as "watch", not "deploy". Sources: [Towards Pony V7](https://civitai.com/articles/5069/towards-pony-diffusion-v7), [Civitai newsletter reveal](https://newsletter.civitai.com/p/new-lora-training-options-sd3-5-generation-training-update-and-a-special-guest-interview-you-won-t-w).
- **Download:** ~6.9 GB per checkpoint (~40 min).

**Fit verdict: the safe, fast, unrestricted default.** Lower ceiling on prompt coherence and in-image text than Flux/Chroma/Z-Image, but fully resident, fastest to iterate, and the ecosystem (LoRAs, ControlNet, IP-Adapter, inpainting) has no rival. For pure uncensored output with control, this is the pragmatic pick.

---

## Tier 2: fits, but with a real quality or capability compromise

### 5. Qwen-Image (QwenLM/Alibaba) - text-rendering king, but too big for 8 GB except at low quant

- **What it is:** 20B MMDiT, **Apache 2.0**, best-in-class complex text rendering and precise editing (separate Qwen-Image-Edit variant). Qwen-Image 2.0 landed 2026-02-10. Repo: [github.com/QwenLM/Qwen-Image](https://github.com/QwenLM/Qwen-Image), GGUF: [QuantStack/Qwen-Image-GGUF](https://huggingface.co/QuantStack/Qwen-Image-GGUF), guide: [ComfyUI Wiki](https://comfyui-wiki.com/en/tutorial/advanced/image/qwen/qwen-image).
- **8 GB reality:** at 20B, 8 GB forces **Q2_K or Q3_K_S plus the Lightning LoRA**; output is "usable but noticeably softer" than higher precision. Q4 needs 12-13 GB and does not fit. So on our card it is a compromised experience, not the way to see the model at its best.
- **Censorship:** base is moderately filtered; Apache license and community finetunes exist. Not the uncensored standout, and the low-quant penalty on 8 GB makes it hard to recommend over Chroma/Z-Image.

**Fit verdict: marginal on 8 GB.** Its superpower (text/editing) is real, but you only get a degraded version of it at 8 GB. Revisit if we ever move to a 12-16 GB card.

### 6. SD 3.5 Medium (Stability AI) - fits, but underwhelming for our purpose

- **What it is:** 2.5B MMDiT, ~5.1 GB fp16 denoiser. Fits 8 GB with FP8/offload (needs ~9.9 GB excluding encoders at full precision, so FP8 or GGUF is mandatory on our card). Card family: [Introducing SD 3.5](https://stability.ai/news-updates/introducing-stable-diffusion-3-5), [SD 3.5 Medium VRAM](https://willitrunai.com/image-models/sd-3-5-medium).
- **License:** Stability AI Community License (free under $1M annual revenue).
- **Censorship:** base leans filtered; the SD3 line inherited the infamous anatomy weakness of the original SD3 Medium, and the community uncensored-finetune scene never rallied around it the way it did around SDXL and Flux/Chroma.
- **SD 3.5 Large (8.1B):** ~16 GB fp16, FP8 ~11 GB, does not fit 8 GB comfortably; GGUF-Q4 could squeeze in but the value proposition is weak versus Chroma at the same footprint.

**Fit verdict: fits but out-classed.** For uncensored quality per 8 GB, Chroma, Z-Image, and even good SDXL finetunes beat SD 3.5. Documented here for completeness and licensing contrast.

### 7. SD 1.5 community checkpoints - the 4 GB legacy speed floor

- **What it is:** the original 860M-UNet architecture, 512x512 native (upscale via hires-fix to 1024+), ~2-4 GB fp16 full checkpoints. Runs on 4 GB, so on our 8 GB it is trivially resident and the fastest iteration loop available. License CreativeML OpenRAIL-M.
- **Uncensored finetunes:** **Realistic Vision** (v5.1/v6), **CyberRealistic**, **DreamShaper**, **epiCRealism**, **ChilloutMix**. Thoroughly uncensored, enormous legacy LoRA library. Source: [offlinecreator VRAM list](https://offlinecreator.com/blog/best-uncensored-ai-models-civitai-2026).
- **Quality:** dated by 2026 standards (weaker coherence, hands, text) but still excellent for photoreal portraits/close-ups and unbeatable for speed and low footprint.

**Fit verdict: keep as the fast lane.** Not the quality pick, but the right tool for rapid iteration, batch generation, and anything where SDXL's extra seconds/image aren't worth it.

### Noted, does not fit cleanly

- **HunyuanImage (Tencent):** genuinely uncensored out of the box, no LoRA needed, bilingual, photoreal. But it is a **12 GB+ tier** model; only marginal at 8 GB with aggressive quant. Best-in-class "zero-config uncensored" if we had 12 GB. Source: [offlinecreator Flux vs SDXL vs Hunyuan](https://offlinecreator.com/blog/flux-vs-sdxl-vs-hunyuan-uncensored).
- **Flux.2 [dev]:** larger, and shipped mandatory safety filtering in license + pipeline. Regression for our purpose; skip in favor of Flux.1/Chroma.

---

## Ranking: best uncensored quality achievable in 8 GB

1. **Chroma1-HD (Q5_K_S, 6.5 GB)** - Flux-class fidelity, uncensored by design, Apache 2.0. Best overall.
2. **Flux.1 [dev] (Q4_K_S, 6.8 GB + T5 offload)** - top coherence/text; lightly aligned, non-commercial. Quality reference.
3. **Z-Image Turbo (FP8 ~8 GB / GGUF ~6 GB)** - near-Flux quality, 8 steps, Apache, uncensored. Best quality-per-second; falsify the leaderboard claim.
4. **SDXL uncensored finetunes (~6.9 GB, fully resident)** - NoobAI/Illustrious/Pony for anime, Juggernaut/RealVisXL for photoreal. Deepest control ecosystem, fully unrestricted, fastest to iterate.
5. **Qwen-Image (Q2_K/Q3 + Lightning)** - text/editing king but only a softened version fits 8 GB.
6. **SD 3.5 Medium (FP8)** - fits, out-classed for uncensored use.
7. **SD 1.5 finetunes (~2-4 GB)** - dated quality, unbeatable speed/footprint, fully uncensored.

Tooling note underpinning all of the above: everything in Tiers 1-2 is runnable today via ComfyUI (PyTorch cu128 for sm_120) and most via stable-diffusion.cpp (ggml/GGUF, our home turf). The only genuine 8 GB engineering task recurring across Flux/Chroma/SD3.5 is CPU-offloading the T5-XXL encoder into our 48 GB RAM.

---

## Candidate experiments for our rig

Ranked by expected value per download-hour. Cost axis is seconds-per-1024x1024-image and peak VRAM, NOT t/s (see framing section).

### E1. SDXL uncensored baseline (cheapest, establishes the floor)
Download one photoreal + one anime checkpoint (~6.9 GB each, ~40 min each): e.g. **Juggernaut XL** and **NoobAI-XL V-Pred 1.0**. Run in ComfyUI on the 5060. Measure: does it load fully resident in 8 GB, peak VRAM at 1024x1024, seconds/image at 28-30 steps, and whether hires-fix/ControlNet fit or OOM. Confirm NoobAI's v-pred + Euler + CFG 4-5 recipe. This is the reference every later model is judged against, and it validates the ComfyUI-on-sm_120 stack itself. Effort: hours.

### E2. Chroma1-HD Q5_K_S as the flagship uncensored model (the headline test)
Download Chroma1-HD Q5_K_S (~6.5 GB) + a quantized T5 encoder (~5 GB, reused later), ~1.1 h. Run via **ComfyUI-GGUF**, then repeat via **stable-diffusion.cpp** to compare our ggml-native path. Measure: peak VRAM with T5 CPU-offloaded, seconds/image, and a same-prompt quality A/B against the E1 SDXL baseline and against Flux.1-dev (E4). Hypothesis: Flux-class quality, genuinely uncensored output, fits 8 GB at Q5 with T5 on the host. This is the model most likely to become our default. Effort: days.

### E3. Z-Image Turbo speed/quality falsification (the sleeper)
Download FP8 or GGUF (~6-8 GB), ~1 h. 8-NFE schedule. Measure seconds/image (expect it to be the fastest quality model here by a wide margin) and blind-quality A/B versus Chroma (E2) and SDXL (E1) on identical prompts. Falsify the "top open model on Artificial Analysis" claim with our own eyes, exactly as we falsify LLM leaderboard claims. Hypothesis to test: best quality-per-second AND per-GB on the rig. Effort: hours-to-days.

### E4. Flux.1 [dev] Q4_K_S with T5 CPU-offload (the coherence reference)
Download Q4_K_S (~6.8 GB), reuse the E2 T5, ~1 h. Purpose is twofold: (a) establish the coherence/in-image-text ceiling that Chroma and Z-Image are measured against, and (b) quantify the `--lowvram` / encoder-offload penalty in seconds/image so we know the true cost of the T5 gotcha on 8 GB. Note the non-commercial license: this is an internal reference, not for shipped output. Effort: days.

### E5. stable-diffusion.cpp sm_120 build + ggml smoke test (infrastructure, high strategic value)
Near-zero extra download (reuse an E1/E2 checkpoint). Build **leejet/stable-diffusion.cpp** with the CUDA-13 / sm_120 arch flag on our box and generate one image per supported family (SDXL, Flux, Chroma, Z-Image). This extends our existing llama.cpp/ggml competence to image gen, gives us a scriptable CLI we can wire into the same orchestration layer as our LLM hub, and de-risks the whole survey against ComfyUI/PyTorch dependency churn. Confirms the one open tooling unknown (sm_120 build) at trivial cost. Effort: hours, gated on a clean CUDA build.

---

## Feasibility notes

- **All Tier-1 models are runnable today** on Windows/sm_120: existence, GGUF availability, and tooling support verified against primary sources (HF cards + repos fetched directly; sd.cpp and ComfyUI-GGUF support confirmed).
- **AVX2 is not a constraint here** the way it is for our LLMs: image inference is GPU-side, so the CPU's lack of AVX-512/AMX barely matters. Host CPU only carries text-encoder/VAE offload, which is light.
- **The recurring 8 GB engineering item is T5-XXL offload** (Flux, Chroma, SD3.5). Our 48 GB RAM makes this a non-issue in capacity terms; the only cost is a small per-generation latency when the encoder runs on CPU. SDXL and SD 1.5 sidestep it entirely (CLIP-only, self-contained checkpoints).
- **Vendor/leaderboard claims get our standard falsification** before entering the repo as fact, especially Z-Image's "top open model" and any Civitai ELO ordering.
- **Total Tier-1 download budget** if we run E1-E4 is ~35 GB (checkpoints + one shared T5) against 416 GB free, ~3.5 hours at 3 MB/s. Trivial.

---

## Feasibility verdicts

Adversarial review of 2026-07-20. Every fit/runnability/license claim below was re-checked against primary sources (HF cards, city96/silveroxides/QuantStack GGUF repos, leejet/stable-diffusion.cpp README + docs, BFL Flux.2 license, low-VRAM community guides). Two framing corrections precede the per-candidate calls:

- **"Zero per-token speed cost" is the wrong axis and the doc says so correctly.** Diffusion is GPU-compute-bound, not host-bandwidth-bound, so the LLM throughput law does not transfer. Confirmed. There is no per-token cost to flag; the honest cost axis is seconds/1024px-image and peak VRAM. No candidate secretly needs retraining on our part (Chroma's retrain was done by the community; we download finished weights) and none needs a second model resident concurrently (T5/CLIP encoders run in a separate phase and CPU-offload to the 48 GB host, they do not co-occupy the 8 GB with the denoiser).
- **Two honest caveats the framing understates.** (1) Tier-1 bullet "runs at full Blackwell GPU speed if it fits" is optimistic for Flux/Chroma at Q4/Q5: those need `--lowvram` layer-streaming, which community sources measure at ~20-30% slower and 60+ s/image on 8 GB. Fits and runs, but not at "full" speed. (2) Image gen shares the same 8 GB with our LLM hub, so image and LLM inference are mutually exclusive on this one GPU (they swap, exactly like the hub already swaps LLMs). This does not reduce LLM t/s when the LLM is loaded; it just means "generate image" is another swap target, not a concurrent service.

### Models

- **Chroma1-HD (Q5_K_S ~6.5 GB) - GO.** Verified: 8.9B, Apache 2.0, uncensored by design, GGUF live (silveroxides/QuantStack), sd.cpp has a `chroma.md`. Fits 8 GB only with T5 CPU-offloaded and a tight working set (expect occasional `--lowvram`); the headline pick stands.
- **Flux.1 [dev] (Q4_K_S ~6.8 GB + T5 offload) - GO, research-only.** Verified: city96 GGUF, ~6.8 GB Q4_K_S, fp8/GGUF T5 mandatory on 8 GB, `--lowvram` gives ~60+ s/image. Non-commercial license is real, so internal reference not shipped output. Effort honest (days).
- **Z-Image Turbo (FP8 ~6 GB / GGUF 5-6 GB) - GO.** Verified: 6B, arXiv 2511.22699 (Nov 2025), Apache 2.0, 8 NFEs, uses a small Qwen-3B-class text encoder (no T5 gotcha), FP8 runs in 8 GB and GGUF in 5-6 GB, sd.cpp added Z-Image Dec 2025, ~13-20 s/image on peer 8 GB cards. Best quality-per-second/GB; only open item is falsifying the leaderboard claim, which the doc already flags.
- **SDXL uncensored finetunes (~6.9 GB, fully resident) - GO.** Verified: 3.5B, CLIP-only self-contained checkpoint, no T5 offload, fully resident in 8 GB at 1024px. Juggernaut/RealVisXL/NoobAI-XL/Illustrious/Pony V6 all real and uncensored; NoobAI's v-pred + Euler + CFG 4-5 recipe is a genuine config nuance, not a blocker. Safest, fastest-to-iterate, deepest ecosystem.
- **Qwen-Image (Q2_K/Q3_K_S + Lightning LoRA) - MAYBE.** Verified: 20B, Apache 2.0, Q2_K ~7 GB is the only thing that fits 8 GB and only with Lightning LoRA; Q4 needs 12-13 GB. It runs today but only in a softened state, so its text/edit superpower is real but degraded on our card. Honest "marginal" call; revisit at 12-16 GB.
- **SD 3.5 Medium (FP8) - MAYBE.** Verified: 2.5B, fits with FP8/GGUF, runs today. Out-classed for uncensored use (SD3 anatomy weakness, no strong uncensored-finetune scene). Fits and runs, weak value; keep for licensing/completeness only.
- **SD 1.5 finetunes (~2-4 GB) - GO, fast lane only.** Verified: 860M UNet, trivially resident, fastest loop, thoroughly uncensored (Realistic Vision / CyberRealistic / epiCRealism). Dated coherence/hands/text by 2026, correctly positioned as iteration/batch tool, not the quality pick.
- **HunyuanImage - KILL (for 8 GB).** 12 GB+ tier; only marginal at 8 GB with aggressive quant, which throws away its zero-config-uncensored advantage. Revisit only on a 12 GB+ card. Correctly parked.
- **Flux.2 [dev] - KILL.** Verified: 32B, non-commercial license, and mandatory safety filtering in both license and pipeline (BFL did final safety fine-tuning). Does not fit 8 GB and is a censorship regression. Correctly excluded.

### Experiments

- **E1 SDXL baseline - GO.** Cheapest, fully-resident, validates the ComfyUI-on-sm_120 stack itself. Effort (hours) honest. Do first.
- **E2 Chroma1-HD flagship - GO.** Headline test; both the ComfyUI-GGUF and sd.cpp paths exist today. Effort (days) honest; note the tight-VRAM/`--lowvram` reality when measuring seconds/image.
- **E3 Z-Image Turbo falsification - GO.** Highest expected value per download-hour: small, fast, Apache, no T5. Effort (hours-to-days) honest.
- **E4 Flux.1 [dev] reference - GO, internal-only.** Establishes the coherence ceiling and quantifies the real `--lowvram`/T5 penalty. Keep output internal (non-commercial license). Effort (days) honest.
- **E5 stable-diffusion.cpp sm_120 build - MAYBE.** High strategic value (ggml-native, scriptable into our hub) and sd.cpp genuinely lists SDXL/Flux/Chroma/Z-Image support. But the sm_120 CUDA-13 build is the one unverified tooling unknown on our exact box; the "hours" effort is optimistic since Blackwell arch-flag/CMake debugging can eat a day. Worth doing, gated on a clean build, with ComfyUI (PyTorch cu128, sm_120 supported today) as the fallback so the survey does not depend on E5 landing.
