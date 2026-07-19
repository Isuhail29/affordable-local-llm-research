# Experiment 043: First Local Image Generation (Queue Track B)

Date: 2026-07-20
Status: First image succeeded
Source: research-sweeps/2026-07-sweep-02-quality/QUEUE.md Track B

## Goal

Add uncensored image generation to the rig, using stable-diffusion.cpp (the ggml sibling of llama.cpp) so it fits our existing no-Python ecosystem and can sit behind the same hub.

## Setup

- Built stable-diffusion.cpp from source with CUDA sm_120 (Blackwell), Ninja + MSVC, `-DSD_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120`. Build trap: `stable-diffusion.cpp` overflows the MSVC object format, fixed with `-DCMAKE_CXX_FLAGS=/bigobj`. Produces sd-cli.exe and sd-server.exe.
- Model: Pony Diffusion V6 XL (6.46 GB fp16 safetensors) + SDXL VAE (0.31 GB). Uncensored/permissive SDXL base, CLIP-only (no T5), fully VRAM-resident.

## First image

Prompt: `score_9, score_8_up, a serene mountain lake at sunrise, pine trees, reflection on calm water, highly detailed, sharp focus` (Pony uses score_ quality tags). 1024x1024, 25 steps, euler_a, CFG 7.

Result: results/images/e043-first-image.png, a coherent on-prompt sunrise lake scene. Generation 69.8s (denoise + VAE decode), 76.6s wall including model load reuse.

## Benchmark analysis

**The research prediction held: image generation is GPU-compute-bound, not bandwidth-bound.** Our LLM throughput law does not apply; the 8 GB VRAM fits the whole SDXL denoiser and it runs at full Blackwell compute. 70s for 25 steps at 1024x1024 is normal for a laptop 5060 class GPU. Levers to speed up: fewer steps (SDXL is fine at 20-30), lower resolution (768 or 512 for drafts), or a Turbo model (Z-Image Turbo / SDXL-Turbo, 4-8 steps) from the queue's quality ladder.

## Integration

- sd-server.exe exposes an OpenAI-style /v1/images/generations endpoint. Runs standalone via Start-ImageGen.bat on port 8082.
- Open WebUI wiring: Settings > Images > engine "OpenAI", base URL http://127.0.0.1:8082/v1. Then the image button in chat generates locally.
- VRAM note: the image model holds ~6.5 GB VRAM while loaded, so run image generation and large-LLM chat one at a time (they contend for the 8 GB card). Start-ImageGen.bat is separate from Start-AI-Hub.bat for this reason.

## Next Steps

- Climb the quality ladder (queue Track B): Z-Image Turbo (8-step, fast) and Chroma1-HD (Flux-class, Apache, uncensored-by-design, needs T5 offload to the 48 GB RAM).
- Optional: register sd-server behind llama-swap so image requests auto-swap the GPU like the LLMs do.
- ComfyUI-GGUF remains the quality-ceiling fallback if sd.cpp output disappoints.
