# Image-generation runtimes for our exact rig (Blackwell sm_120, 8GB, Windows, CUDA 13.3)

Sweep: 2026-07-sweep-02-quality | Written: 2026-07-20 | Method: primary sources (project GitHub repos, official docs, maintainer discussion threads, PyTorch issue tracker) cross-checked against 2026 community reports. Every claim is flagged **runnable-today** or **needs-work** for our specific machine. Where an AI-generated search snippet was wrong I say so and give the primary source.

## TL;DR recommendation

**Primary: ComfyUI.** It is the only runtime in this list with genuine day-1 Blackwell support, the most active development, the best 8GB VRAM tooling (auto weight-offloading + `--lowvram` + GGUF nodes + fp8), the widest model coverage (SDXL, Flux, Qwen-Image, video), and a real HTTP API we can wire behind the hub. On Windows it installs as a portable folder with no PyTorch surgery. Source: [ComfyUI Blackwell support thread #6643](https://github.com/Comfy-Org/ComfyUI/discussions/6643), [blog.comfy.org](https://blog.comfy.org/p/how-to-get-comfyui-running-on-your).

**Ecosystem-native runner-up worth piloting first for hub integration: stable-diffusion.cpp.** It is the literal llama.cpp sibling (same ggml + CUDA backend, same GGUF format, builds the same way we already build b10068), and its `sd-server` exposes an **OpenAI-compatible `/v1/images/generations` endpoint plus an A1111-compatible SDAPI layer**, so it drops behind llama-swap exactly like our LLMs. It is slower and thinner on features than the PyTorch stack, so it is the "cleanest fit" pick rather than the "best pictures" pick. Source: [sd-server api.md](https://github.com/leejet/stable-diffusion.cpp/blob/master/examples/server/api.md), [repo](https://github.com/leejet/stable-diffusion.cpp).

The rest (SD.Next, Forge/reForge, Fooocus, A1111, InvokeAI) are covered below with honest maintenance status. SD.Next is the strongest of that group and a legitimate alternative primary if you want a batteries-included GUI + REST API in one install.

## 0. The one fact that governs everything: sm_120 and PyTorch

Our GPU is **RTX 5060 Laptop = Blackwell consumer = CUDA compute capability 12.0 = `sm_120`**. This is the single gate every PyTorch-based runtime (all of them except stable-diffusion.cpp) has to pass.

**Correction of a bad search result up front:** one AI snippet during this sweep claimed "RTX 5060 is Blackwell, compute capability 9.0, set `CMAKE_CUDA_ARCHITECTURES=90`." That is wrong and would produce a binary that does not run on our card. `sm_90` is Hopper (H100). Consumer Blackwell is `sm_120`. We already build llama.cpp with `-DCMAKE_CUDA_ARCHITECTURES=120`; stable-diffusion.cpp uses the same value. Do not copy the `90` from that snippet.

State of PyTorch + Blackwell as of mid-2026 (primary source: [pytorch/pytorch #159207](https://github.com/pytorch/pytorch/issues/159207), [#164342](https://github.com/pytorch/pytorch/issues/164342), and the [andreaskuhr RTX 50xx guide](https://andreaskuhr.com/en/fooocus-forgewebui-automatic1111-nvidia-rtx-50xx-graphics-card.html)):

1. **sm_120 kernels first shipped in stable PyTorch 2.7.0 (CUDA 12.8 / `cu128` wheels), early 2025.** Anything on PyTorch older than 2.7 or built for CUDA < 12.8 will throw `CUDA error: no kernel image is available for execution on the device` on our card. That is the error every "my 5060 doesn't work" post is hitting.
2. **By mid-2026 the wheel matrix goes up to `cu130` and `cu132`.** The andreaskuhr guide notes `cu132` (CUDA 13.2) wheels available as of May 2026.
3. **Our CUDA 13.3 driver is fine.** The driver CUDA version (13.3) only needs to be **>=** the toolkit a wheel was built against. A `cu128`, `cu129`, or `cu130` PyTorch wheel runs correctly under our 13.3 driver via backward compatibility. We do NOT need a `cu133` wheel and none exists. Do not downgrade the driver.

**Practical consequence for runtime choice:** a runtime is "good on Blackwell" if it either (a) ships/pins PyTorch >= 2.7 `cu128+` by default, or (b) auto-installs the right torch, or (c) sidesteps PyTorch entirely. The three winners on that test are **ComfyUI** (ships it), **SD.Next** (auto-installs it), and **stable-diffusion.cpp** (no PyTorch at all). The three that make you do manual torch surgery are **Forge, Fooocus, and A1111**.

## 1. Comparison matrix

| Runtime | Blackwell/sm_120 status | PyTorch dep | GUI | API for hub | 8GB VRAM tooling | Windows setup | Maintained (2026) |
|---|---|---|---|---|---|---|---|
| **ComfyUI** | Day-1, ships cu128+ | Yes (bundled in portable) | Node graph (browser) | Native HTTP `/prompt` + WS; OpenAI-images via shim | Best: auto-offload, `--lowvram`/`--novram`, GGUF nodes, fp8, tiled VAE | Portable 7z, ~1-click | **Very active** |
| **SD.Next** | Works; auto-installs correct torch | Yes (auto-managed) | Rich web UI (2 modes) | **Built-in FastAPI REST + A1111 SDAPI** | `medvram`/`lowvram`, offload, quant | `git clone` + `webui.bat` | **Very active** (vladmandic) |
| **stable-diffusion.cpp** | Build with `CUDA_ARCHITECTURES=120`; no torch | **None** (pure C/C++ ggml) | Minimal embedded web UI (Apr 2026) + 3rd-party | **`sd-server`: OpenAI `/v1/images/generations` + A1111 SDAPI** | Flash-attn, VAE tiling, GGUF Q4-Q8, `--vae-on-cpu`, `--clip-on-cpu` | Build from source (like our llama.cpp) or prebuilt binary | Active |
| **InvokeAI** | Fixed in recent 5.x (torch cu128) | Yes | Polished web UI + canvas | REST API (OpenAPI) | **Low-VRAM mode default** (partial loading, 3GB working set) | 1-click installer | Active |
| **Forge** (lllyasviel) | Works only after manual torch upgrade to cu128 | Yes (stale default) | A1111-style web UI | A1111 SDAPI | Built for low VRAM (its whole point) | `git clone` + manual pip torch swap | **Paused / stale** |
| **reForge** | Same as Forge, needs manual torch | Yes (stale) | A1111-style | A1111 SDAPI | Low-VRAM focused | Manual torch swap | **Development stopped** |
| **Fooocus** | Needs manual torch upgrade | Yes (old default) | Simplified "just prompt" UI | Limited/none native | Auto low-VRAM (SDXL only) | Manual `python_embeded` pip swap | **Effectively unmaintained**; use RuinedFooocus fork |
| **Automatic1111** | Needs the prebuilt "blackwell" 7z or manual torch | Yes (oldest defaults) | The original web UI | A1111 SDAPI (the reference) | `--medvram`/`--lowvram`/`--xformers` | Prebuilt `sd.webui-*-blackwell.7z` or manual | **Slow / stale** |

Sources for the maintenance column: [andreaskuhr guide](https://andreaskuhr.com/en/fooocus-forgewebui-automatic1111-nvidia-rtx-50xx-graphics-card.html) ("Fooocus, Forge WebUI and Automatic1111 have not been updated for months... Reforge had stopped development"), [Forge Blackwell issue #2775](https://github.com/lllyasviel/stable-diffusion-webui-forge/issues/2775), [A1111 Blackwell discussion #16818](https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/16818).

## 2. Per-runtime deep dive

### 2.1 ComfyUI — the workhorse (PRIMARY)

**Blackwell:** Comfy-Org states "day 1 support for the new 50 series Blackwell GPUs" and maintains a dedicated support thread. The Windows portable now bundles a PyTorch build with sm_120 kernels; the hard rule is **PyTorch with CUDA >= 12.8, nothing lower works** ([blog.comfy.org](https://blog.comfy.org/p/how-to-get-comfyui-running-on-your), [discussion #6643](https://github.com/Comfy-Org/ComfyUI/discussions/6643)). **Runnable-today** on our rig.

**Setup on Windows:** download the portable 7z, extract, run the `run_nvidia_gpu.bat`. No system Python, no manual CUDA. The one gotcha the community documents: **custom nodes list `torch` in their `requirements.txt` and pip will happily replace your good nightly/cu128 build with a stale stable one.** Strip `torch` from custom-node requirements before installing them, and never install `xformers` on Blackwell (it force-downgrades torch) ([lilting.ch Blackwell guide](https://lilting.ch/en/articles/comfyui-blackwell-gpu-compatibility)).

**8GB VRAM (this is where it shines):** ComfyUI has the deepest low-VRAM toolkit of any runtime here.
- `--lowvram` does sequential/partial loading so the model never needs to fit fully in VRAM (~20-30% speed penalty). `--novram` for extreme cases; `--medvram` intermediate.
- **GGUF nodes** (city96's `ComfyUI-GGUF`) let us run Flux at Q4_K_S / Q5_K_S, i.e. the exact quantization family we already use for LLMs, which is what brings 12B Flux into 8GB.
- `fp8_e4m3fn` weight_dtype in the Load Diffusion Model node for fp8 safetensors; use the **fp8 T5-XXL text encoder (~4.6GB)** instead of fp16 (~9.2GB) to free 4-5GB.
- Tiled VAE decode + `--lowvram` stack together (different bottlenecks). Sources: [Local AI Master low-VRAM Flux](https://localaimaster.com/blog/run-flux-on-low-vram-gpu), [ComfyLab reduce-VRAM guide](https://comfylab.dev/blog/guides-pro/reduce-vram-usage-comfyui/), [AI Pixel Guide 8GB settings](https://aipixelguide.com/en/guides/8gb-vram-comfyui-settings/).

**API / hub wiring:** ComfyUI is headless by default; the browser UI is just one client. Start with `--listen 127.0.0.1 --port <p> --disable-auto-launch`. Submit a workflow (API-JSON form) via `POST /prompt`, poll `/history/{id}`, pull the image from `/view`; live progress on a WebSocket. This is not an OpenAI-images endpoint out of the box, so behind our hub it needs a **thin shim** that translates `/v1/images/generations` -> `/prompt` (a ~40-line FastAPI proxy, or an off-the-shelf wrapper like SaladTechnologies/comfyui-api). Sources: [ComfyUI server docs](https://docs.comfy.org/development/comfyui-server/comms_overview), [Runflow API guide](https://www.runflow.io/blog/comfyui-api-developer-guide), [9elements hosting guide](https://9elements.com/blog/hosting-a-comfyui-workflow-via-api/).

**Why primary:** best Blackwell support, best 8GB tooling, widest model zoo, most active, and the whole ecosystem targets it first. The only knock for us is that hub integration needs a small shim.

### 2.2 stable-diffusion.cpp — the ecosystem-native pick (RUNNER-UP, pilot first for the hub)

**Why it fits us specifically:** it is the diffusion sibling of llama.cpp, "pure C/C++ implementation based on ggml," backends **CUDA, Vulkan, Metal, OpenCL, SYCL, and CPU (AVX/AVX2/AVX512)**, weight formats **`.ckpt`/`.safetensors`/`.gguf`**. It has **no PyTorch dependency at all**, so the entire sm_120 headache disappears; it inherits the same ggml CUDA backend our LLMs already run on. Models: SD1.x/2.x/SDXL, SD3/3.5, Flux.1/2, Qwen-Image, Z-Image, plus video (Wan, LTX, Hunyuan). Source: [repo README](https://github.com/leejet/stable-diffusion.cpp).

**Blackwell build:** build from source exactly like we build b10068: `cmake .. -DSD_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120`, `cmake --build . --config Release` ([build.md](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/build.md)). Point `CUDAToolkit_ROOT` at our CUDA install. (Ignore the search snippet that said `=90`; see Section 0.) There are also prebuilt Windows CUDA binaries on the releases page if we want to skip compiling. **Runnable-today**, and the build is a known quantity for us.

**8GB VRAM:** Flash Attention flag for lower attention memory, **VAE tiling** to cap decode memory, GGUF quantization (Q4_0/Q4_K/Q8_0) to shrink weights, and `--vae-on-cpu` / `--clip-on-cpu` to push the VAE and text encoder to system RAM (we have 48GB) and keep the 8GB for the diffusion transformer. This offload-to-RAM story is directly analogous to how we split KV/weights on the LLM side. Sources: [repo README](https://github.com/leejet/stable-diffusion.cpp), [DeepWiki build page](https://deepwiki.com/leejet/stable-diffusion.cpp/4-building-and-installation).

**API / hub wiring (its killer feature for us):** `sd-server` exposes three API families — **OpenAI-compatible (`POST /v1/images/generations`, `/v1/images/edits`, `GET /v1/models`), A1111-style SDAPI, and its native SDCPP API** — built on httplib with an async job manager. That OpenAI-images endpoint means it drops behind llama-swap the same way our LLM binaries do: llama-swap launches `sd-server` on a port and proxies to it, and any OpenAI-images client (including Open WebUI's image feature) just works. This is the lowest-friction path to "image generation lives in the hub." Sources: [sd-server api.md](https://github.com/leejet/stable-diffusion.cpp/blob/master/examples/server/api.md), [Server API DeepWiki](https://deepwiki.com/leejet/stable-diffusion.cpp/5.2-server-api).

**Honest limits:** it trails the PyTorch stack on raw speed, sampler/scheduler breadth, and the long tail of community LoRAs/ControlNets/workflows. The embedded web UI (added ~2026/04/11) is minimal. So it is the "clean integration + tiny footprint" answer, not the "maximum quality/features" answer. For our budget-and-ecosystem ethos it is arguably the most on-brand runtime, which is why it is the first thing to pilot.

### 2.3 SD.Next — strongest of the GUI-first pack (legit alternative primary)

**Blackwell:** SD.Next auto-manages its own torch install and by 2026 pulls a Blackwell-capable build; it is listed among the tools that "already work" on RTX 50xx without the manual surgery Forge/Fooocus/A1111 need ([andreaskuhr guide](https://andreaskuhr.com/en/fooocus-forgewebui-automatic1111-nvidia-rtx-50xx-graphics-card.html)). Backends span CUDA, ROCm, IPEX, DirectML, OpenVINO, ZLUDA. **Runnable-today.**

**Models & GUI:** SD1.5, SDXL, SD3, Flux (including 2026's Flux.2-Klein 4B/9B), PixArt, Cascade, 50+ types. Two UI modes (Standard and the newer "Modern" UI). Source: [SD.Next docs](https://vladmandic.github.io/sdnext-docs/), [DeepWiki](https://deepwiki.com/vladmandic/sdnext), [changelog](https://vladmandic.github.io/sdnext-docs/CHANGELOG/).

**API / hub:** ships a **FastAPI REST API covering all functionality plus A1111-compatible endpoints** — the best built-in API of the GUI-first group, so it wires behind the hub with less custom glue than ComfyUI (A1111 SDAPI is a well-known target). 8GB handled via `medvram`/`lowvram`/offload and quantization.

**Setup:** `git clone` + `webui.bat` on Windows; the installer resolves torch for you. Slightly more moving parts than a portable folder but well documented. **Verdict:** if you want one install that is GUI + REST API + broad models with minimal integration work, SD.Next is the pick; ComfyUI still edges it on 8GB tooling depth and raw ecosystem gravity.

### 2.4 InvokeAI — most polished GUI, good 8GB defaults

**Blackwell:** the RTX 5090 "no kernel image" bug ([issue #7683](https://github.com/invoke-ai/InvokeAI/issues/7683), opened Feb 2025) is **closed**; recent 5.x Community Edition ships/pulls torch cu128 and is listed among tools that work on 50xx. Treat as **runnable-today but verify the installer grabs cu128+** on first run.

**8GB:** **Low-VRAM mode is on by default** via `enable_partial_loading: true` in `invokeai.yaml`, reserving ~3GB working memory; runs SDXL comfortably on 8GB and Flux with aggressive quantization. Source: [InvokeAI low-VRAM docs](https://invoke.ai/configuration/low-vram-mode/), [issue #7683](https://github.com/invoke-ai/InvokeAI/issues/7683).

**GUI/API:** the most designer-friendly UI here (unified canvas, layers, regional control) plus a documented REST/OpenAPI. Good if a human will sit and iterate on images; the node/workflow ceiling is lower than ComfyUI. 1-click Windows installer.

### 2.5 Forge / reForge — low-VRAM specialists, but stale

Forge (lllyasviel) was purpose-built to run heavy models on small VRAM, which sounds perfect for 8GB — but it is **paused/stale and does not run on Blackwell out of the box.** You must open its embedded Python and manually `pip install` a cu128+ torch to clear the sm_120 error ([Forge issue #2775](https://github.com/lllyasviel/stable-diffusion-webui-forge/issues/2775), [discussion #2608](https://github.com/lllyasviel/stable-diffusion-webui-forge/discussions/2608), [andreaskuhr guide](https://andreaskuhr.com/en/fooocus-forgewebui-automatic1111-nvidia-rtx-50xx-graphics-card.html)). **reForge development has stopped** entirely. Exposes A1111 SDAPI. Verdict: only if you have an existing Forge workflow you refuse to leave; otherwise its low-VRAM advantage is now matched by ComfyUI's offloading without the manual torch surgery or the abandonment risk.

### 2.6 Fooocus — simplest UX, effectively unmaintained

Fooocus is the "type a prompt, get a good SDXL image" runtime with automatic low-VRAM handling. But it is **not updated for months** and needs the same manual torch upgrade in `python_embeded` to run on Blackwell ([andreaskuhr guide](https://andreaskuhr.com/en/fooocus-forgewebui-automatic1111-nvidia-rtx-50xx-graphics-card.html)). It is SDXL-centric (no modern Flux/Qwen-Image path) and has no real API for hub wiring. If you specifically want the Fooocus UX, use the more current **RuinedFooocus** fork. Verdict: not a fit for a hub-integrated, current-models setup.

### 2.7 Automatic1111 — the reference, now the slowest to adopt

A1111 defined the web UI and the SDAPI everyone else clones, but it has the **oldest default torch and is the slowest to get onto new hardware.** The community path on Blackwell is either a prebuilt `sd.webui-<ver>-blackwell.7z` bundle or a manual torch swap ([A1111 Blackwell discussion #16818](https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/16818), [andreaskuhr guide](https://andreaskuhr.com/en/fooocus-forgewebui-automatic1111-nvidia-rtx-50xx-graphics-card.html)). It has `--medvram`/`--lowvram`/`--xformers` for 8GB and the canonical SDAPI. Verdict: fine as a familiar SDAPI reference, but on Blackwell in 2026 you are choosing the highest-maintenance option for the least active project. Skip in favor of ComfyUI or SD.Next.

## 3. Wiring image-gen behind our llama-swap hub

llama-swap launches a process that serves HTTP on a port and proxies requests to it, swapping the active backend on demand. Two clean patterns for images:

1. **OpenAI-images native (least glue):** run **stable-diffusion.cpp `sd-server`** as a hub entry. Its `/v1/images/generations` is already OpenAI-shaped, so Open WebUI's image button and any OpenAI client hit it through the hub with zero translation. This is the fastest thing to stand up and the most consistent with how our LLMs are served.
2. **Full-power backend + shim:** run **ComfyUI headless** as a hub entry and put a ~40-line FastAPI shim in front that maps `/v1/images/generations` -> ComfyUI `/prompt` + `/history` + `/view`. More work, but you get ComfyUI's model zoo and 8GB tooling. SD.Next is the middle option: its built-in FastAPI/A1111 API needs only path routing, no image-payload translation.

Because image models are large and 8GB is tight, keep image and LLM backends as **separate hub entries that never co-reside in VRAM** — let llama-swap unload the LLM before loading the diffusion model. Do not try to hold a 30B MoE and Flux resident at once.

## 4. 8GB VRAM playbook (applies to whichever runtime wins)

- **Prefer GGUF-quantized Flux (Q4_K_S/Q5_K_S)** over fp16; this is the same quant family we use for LLMs and is what makes 12B Flux fit 8GB. fp8 is a fixed fallback when GGUF nodes aren't available.
- **Always use the fp8 (or GGUF) T5-XXL text encoder**, not fp16 — frees 4-5GB by itself.
- **Push VAE and text encoder to CPU/RAM** (we have 48GB): `--vae-on-cpu`/`--clip-on-cpu` in sd.cpp, offload toggles in ComfyUI/SD.Next. Keep the 8GB for the diffusion transformer.
- **Tiled VAE decode + low-VRAM/offload mode** stack together; use both.
- **SDXL comfort zone is 768x768 batch 1, no hi-res fix**; 1024x1024 is ~4x the memory and is where 8GB starts thrashing. Upscale as a separate pass. Sources: [Local AI Master](https://localaimaster.com/blog/run-flux-on-low-vram-gpu), [ComfyLab](https://comfylab.dev/blog/guides-pro/reduce-vram-usage-comfyui/), [Apatero GGUF 8GB guide](https://apatero.com/blog/flux-gguf-quantization-8gb-vram-guide-2026).
- **Throughput note (our MEASURED LAW analogue):** diffusion time-to-image on 8GB is dominated by step count x model size x offload penalty, not by anything a runtime flag can cheat. Quality-per-second here means "fewest steps and smallest quant that still hits the quality bar," measured on our own eval, same discipline as the sampler sweep.

## 5. Candidate experiments for our rig

**E-IMG-1 — sd.cpp `sd-server` in the hub (OpenAI-images, ecosystem-native).**
Build stable-diffusion.cpp from source with `-DSD_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120` (or grab the prebuilt CUDA Windows binary), pull an SDXL GGUF (Q8_0) and a Flux.1 GGUF (Q4_K_S), add `sd-server` as a llama-swap entry, and drive it from Open WebUI's image button via `/v1/images/generations`. Measure sec/image and VRAM at 768x768 and 1024x1024. **Falsification target:** does the C++/ggml path actually run acceptably on sm_120 without PyTorch, and is quality good enough to not need the PyTorch stack? This is the cheapest, most on-brand experiment and validates the whole hub-integration thesis.

**E-IMG-2 — ComfyUI portable Flux-GGUF on 8GB.**
Install the Windows portable, add `ComfyUI-GGUF`, run Flux.1-dev Q4_K_S / Q5_K_S with the fp8 T5 encoder, `--lowvram`, and tiled VAE. Record sec/image, VRAM headroom, and quality vs sd.cpp's Flux at matched steps. **Falsification target:** how much quality/speed does the full PyTorch stack buy over sd.cpp for the same model, and is the 8GB experience actually smooth day-to-day? Decides whether ComfyUI earns the extra integration shim.

**E-IMG-3 — Blackwell smoke test across the PyTorch trio (Comfy vs SD.Next vs InvokeAI).**
Install all three, confirm each pulls a torch >= 2.7 `cu128+` and clears the `no kernel image` error on our card, then run one identical SDXL prompt on each. **Falsification target:** verify the "day-1 / auto-installs / fixed-in-5.x" claims are true on our exact CUDA 13.3 driver, and time the from-zero-to-first-image setup for each. Produces the definitive "which install is actually painless on our machine" answer.

**E-IMG-4 — Quality-per-second step/quant sweep (the MEASURED-LAW analogue for images).**
On the winning runtime, sweep steps (e.g. 8/12/20/28) x quant (Q4_K_S/Q5_K_S/Q8_0/fp8) x scheduler on a fixed prompt set, scoring images on a small rubric. **Falsification target:** find the lowest steps + smallest quant that still passes the quality bar, i.e. maximize quality-per-wall-clock-second at fixed VRAM. Directly mirrors the sampler sweep's framing and gives us defensible defaults for the hub launcher.

**E-IMG-5 (optional) — ComfyUI headless + OpenAI-images shim.**
If E-IMG-2 wins on quality, stand up the ~40-line FastAPI shim mapping `/v1/images/generations` -> ComfyUI `/prompt`/`/history`/`/view`, register it as a hub entry, and confirm Open WebUI drives ComfyUI transparently. **Falsification target:** is the shim maintenance burden worth ComfyUI's quality edge over the zero-glue sd.cpp path from E-IMG-1? This is the head-to-head that picks the permanent hub image backend.

## Sources

- ComfyUI Blackwell thread: https://github.com/Comfy-Org/ComfyUI/discussions/6643
- ComfyUI blog (50-series setup): https://blog.comfy.org/p/how-to-get-comfyui-running-on-your
- ComfyUI Blackwell pitfalls: https://lilting.ch/en/articles/comfyui-blackwell-gpu-compatibility
- ComfyUI server/API docs: https://docs.comfy.org/development/comfyui-server/comms_overview
- ComfyUI API developer guide: https://www.runflow.io/blog/comfyui-api-developer-guide
- stable-diffusion.cpp repo: https://github.com/leejet/stable-diffusion.cpp
- sd-server API: https://github.com/leejet/stable-diffusion.cpp/blob/master/examples/server/api.md
- sd.cpp build docs: https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/build.md
- sd.cpp DeepWiki (build + server): https://deepwiki.com/leejet/stable-diffusion.cpp/4-building-and-installation , https://deepwiki.com/leejet/stable-diffusion.cpp/5.2-server-api
- SD.Next docs / changelog: https://vladmandic.github.io/sdnext-docs/ , https://vladmandic.github.io/sdnext-docs/CHANGELOG/
- SD.Next DeepWiki: https://deepwiki.com/vladmandic/sdnext
- InvokeAI low-VRAM: https://invoke.ai/configuration/low-vram-mode/
- InvokeAI Blackwell issue: https://github.com/invoke-ai/InvokeAI/issues/7683
- Forge Blackwell issue/discussion: https://github.com/lllyasviel/stable-diffusion-webui-forge/issues/2775 , https://github.com/lllyasviel/stable-diffusion-webui-forge/discussions/2608
- A1111 Blackwell discussion: https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/16818
- Fooocus/Forge/A1111 RTX 50xx guide: https://andreaskuhr.com/en/fooocus-forgewebui-automatic1111-nvidia-rtx-50xx-graphics-card.html
- PyTorch sm_120 issues: https://github.com/pytorch/pytorch/issues/159207 , https://github.com/pytorch/pytorch/issues/164342
- 8GB VRAM / GGUF Flux guides: https://localaimaster.com/blog/run-flux-on-low-vram-gpu , https://comfylab.dev/blog/guides-pro/reduce-vram-usage-comfyui/ , https://aipixelguide.com/en/guides/8gb-vram-comfyui-settings/ , https://apatero.com/blog/flux-gguf-quantization-8gb-vram-guide-2026

## Feasibility verdicts

Adversarial review against our exact constraints (RTX 5060 Laptop 8GB / 48GB RAM / AVX2, no AVX-512/AMX / Windows 11 / single llama-swap hub / CUDA 13.3 driver), runnable-today only, and honest per-token-speed accounting. Verified 2026-07-20 against primary sources (sd.cpp `api.md` and README, andreaskuhr RTX 50xx guide, reForge discussion #354, PyTorch sm_120 issues).

**Governing per-token-speed finding (applies to all):** image generation costs our LLM stack **zero per-token throughput**. Section 3's design is sound and honest: image and LLM backends are separate hub entries that never co-reside in VRAM; llama-swap unloads the LLM before loading the diffusion model. No candidate secretly needs two models resident, and none needs retraining. The only real cost is a one-time model-swap latency at switch time (wall-clock, not t/s), plus a per-encode CPU offload penalty for the text encoder. Correctly framed under our MEASURED LAW.

### Runtimes

| Candidate | Verdict | One-line reason |
|---|---|---|
| **ComfyUI** | **GO** | Day-1 sm_120, ships cu128 in portable (no torch surgery), deepest 8GB toolkit (lowvram + GGUF + fp8 + CPU offload), fits our rig today; only cost is a ~40-line OpenAI-images shim for the hub. |
| **stable-diffusion.cpp** | **GO** | Verified: no PyTorch (sm_120 headache gone), builds like b10068 with `-DCMAKE_CUDA_ARCHITECTURES=120`, native `/v1/images/generations` drops behind llama-swap with zero glue, `--vae-on-cpu`/`--clip-on-cpu` offload to our 48GB. The on-brand, lowest-friction pick. |
| **SD.Next** | **GO** | Confirmed "already compatible" on RTX 50xx (auto-installs cu128 torch), built-in FastAPI + A1111 SDAPI, medvram/lowvram/quant for 8GB. Genuinely feasible; just redundant given two strong picks and slightly more moving parts than a portable folder. |
| **InvokeAI** | **MAYBE** | Feasible (5.x pulls cu128, low-VRAM default) but must verify the installer grabs cu128+ on first run on our 13.3 driver; API is REST/OpenAPI not OpenAI-images (needs a shim too), and it is a human-in-the-loop GUI. Redundant with ComfyUI/SD.Next. |
| **Forge / reForge** | **KILL** | Runs on Blackwell only after manual embedded-python torch surgery; reForge development ceased (discussion #354), Forge paused. Its low-VRAM edge is now matched by ComfyUI offload without the surgery or abandonment risk. |
| **Fooocus** | **KILL** | Effectively unmaintained (vanilla paused mid-2024), needs manual `python_embeded` torch swap, SDXL-only (no Flux/Qwen path), no real hub API. Does not fit a current-models, hub-integrated setup. |
| **Automatic1111** | **KILL** | Oldest default torch, slowest to Blackwell (prebuilt `-blackwell.7z` or manual swap), least-active of the group. Highest maintenance for no gain over ComfyUI/SD.Next; keep only as an SDAPI reference. |

### Candidate experiments

| Experiment | Verdict | One-line reason |
|---|---|---|
| **E-IMG-1** — sd.cpp `sd-server` in hub | **GO** | Start here. Every claim verified; cheapest, most on-brand, native OpenAI-images endpoint means true zero-glue hub integration. Validates the whole thesis. |
| **E-IMG-2** — ComfyUI Flux-GGUF on 8GB | **GO** | Flux Q4_K_S + fp8 T5 + `--lowvram` + tiled VAE genuinely fits 8GB (tight but real); runnable today. The quality/features benchmark against sd.cpp. |
| **E-IMG-3** — Blackwell smoke test trio | **MAYBE** | All three are feasible, but at ~3 MB/s installing three separate PyTorch/CUDA stacks (~2.5GB torch wheel each) plus models is many hours. Trim to ComfyUI + SD.Next and drop redundant InvokeAI, or explicitly budget the download time. |
| **E-IMG-4** — quality-per-second step/quant sweep | **GO** | The correct MEASURED-LAW analogue for images (fewest steps + smallest quant that clears the bar); runs on the winning runtime, gives defensible hub defaults. |
| **E-IMG-5** — ComfyUI headless + OpenAI shim | **MAYBE** | Feasible (~40-line FastAPI shim, or off-the-shelf wrapper), but conditional on E-IMG-2 actually beating sd.cpp on quality. Do only if the quality gap justifies the standing shim over the zero-glue sd.cpp path. |

**Effort-honesty note:** the doc is honest about the sm_120 gate, the "12B Flux into 8GB" tightness, and the ComfyUI shim size. The one soft spot is download cost on our ~3 MB/s link: a Flux Q4 GGUF (~6-7GB), a cu128 torch wheel (~2.5GB), and the ComfyUI portable (~1.5GB) are each meaningful waits, and E-IMG-3 multiplies that by three. Flag download-hours in the experiment plan.
