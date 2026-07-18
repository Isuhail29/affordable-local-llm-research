# Experiment 013: MoE Experts on CPU (Qwen3-30B-A3B)

Date: 2026-07-18
Status: Complete

## Goal

Test the project's central thesis on its most promising target: run a 30B-class MoE model on 8 GB VRAM at usable speed by keeping expert tensors in system RAM (only 8 of 128 experts per layer are touched per token) and everything dense (attention, KV cache, embeddings) on GPU.

## Background

Qwen3-30B-A3B-Instruct-2507: 48 layers, 128 experts/layer with 8 active, ~30.5B total / ~3.3B active params, GQA 32/4. Q4_K_M GGUF: 18.55 GB, 3.7x our VRAM, but active expert weights are only ~1.1 GB/token. llama.cpp's `-ncmoe N` keeps expert tensors of the first N layers in CPU RAM. Fiddler (research-papers/fiddler.md) predicted a ~30-38 t/s ceiling for our RAM bandwidth. Benchmark protocol per E002 (mmap, warm cache, memory state noted).

## Hypothesis (pre-registered)

1. All experts on CPU (-ngl 99 -ncmoe 48): 15-30 t/s decode, under 4 GB VRAM.
2. Roughly linear gains as expert layers move to GPU; best fit around ncmoe 36-40.
3. Prefill in 100-1000 t/s (wide; unknown whether PCIe streaming applies).
4. Naive -ngl split worse than ncmoe 48 at equal or higher VRAM.

## Implementation

- llama.cpp b10064 CUDA, -t 8, -r 3, mmap; model freshly downloaded (page cache warm except where noted)
- Sweep -ncmoe 48/44/40, then 36; naive -ngl 14; pure CPU (CUDA_VISIBLE_DEVICES=-1)
- VRAM observed via nvidia-smi during runs and a llama-cli load probe; idle baseline 1.36 GB
- Raw data: benchmarks/2026-07-18_e013-*.json

## Actual Result

| Config | VRAM above idle | pp512 t/s | tg128 t/s |
|---|---|---|---|
| Pure CPU (GPU hidden) | 0 | 65.3 ± 2.0 | 20.16 ± 0.16 |
| Naive -ngl 14 (no MoE awareness) | ~5.4 GB | 339.8 ± 73.9 | 27.82 ± 0.56 |
| -ngl 99 -ncmoe 48 (all experts CPU) | ~1.9 GB est | 326.4 ± 36.0 | 29.63 ± 0.33 |
| -ncmoe 44 | ~3.3 GB est | 371.9 ± 40.7 | 30.33 ± 0.65 |
| -ncmoe 40 | ~4.6 GB est | 404.5 ± 60.5 | 35.19 ± 1.27 |
| **-ncmoe 36 (best fit)** | **~6.3 GB measured** | **466.2 ± 22.7** | **38.53 ± 0.55** |

Reference points: Qwen3-8B dense on the same machine: 62.2 t/s (full GPU), 10.1 t/s (CPU).

## Benchmark analysis

**The thesis holds. A 30B-class instruct model decodes at 38.5 t/s on an 8 GB laptop GPU.** That is 3.8x the 8B dense CPU rate, and only 1.6x slower than the 8B dense running entirely in VRAM, from a model 3.7x larger than VRAM.

**H1 confirmed at its optimistic edge** (29.6 vs predicted 15-30). **H2 confirmed**: 29.6 → 30.3 → 35.2 → 38.5 as expert layers shift to GPU. **H3 confirmed** (326-466 t/s). **H4 confirmed**: naive -ngl 14 gets 27.8 t/s while spending ~5.4 GB; expert-aware splitting beats it by 27% at comparable VRAM (35.2) or matches it with a third of the VRAM (29.6 at ~1.9 GB).

**MoE flips the CPU value proposition.** Pure CPU runs this 30B at 20.2 t/s, double the 8B dense (10.1), because sparsity cuts per-token bytes from 5.0 GB to ~1.6 GB. Sparse-activation models are simply the correct architecture for budget hardware.

**New quantitative finding: the expert-scatter penalty.** Back-solving the CPU expert portion of ncmoe 48 (~32 ms/token for ~1.1 GB) gives ~34 GB/s effective, versus 50.8 GB/s for dense streaming: scattered expert reads waste roughly a third of the achievable RAM bandwidth. That gap is a concrete optimization target for E021/E022 (expert-aware layout, prefetch by router output).

## Lessons Learned

1. **-ncmoe is the single most valuable flag for budget MoE inference.** It should be the default recommendation for any model larger than VRAM whose architecture is MoE.
2. **GPU-resident attention is worth ~50% even with zero experts offloaded** (20.2 → 29.6): KV cache and attention math benefit disproportionately from VRAM.
3. **Prefill scales with attention placement too** (65 → 326 just from GPU attention), but stays an order of magnitude below dense-8B full-GPU prefill (2186). Long-prompt use cases pay the MoE-on-CPU tax mostly at prefill.
4. **Warm-cache benchmark runs are fast; instrument accordingly.** Two VRAM polls missed their windows before landing the method (poll during load, not after; verify with a held-open llama-cli load).
5. **The b10064 llama-cli has an interactive TUI that hangs headless probes**; use llama-bench or redirect-aware tooling for automation.

## Possible Improvements

- Measure VRAM precisely per config (llama-server --log-verbose or repeated load probes) instead of estimates for the middle rungs.
- Quality validation: perplexity or a small eval set to confirm the 2507 instruct quants behave (quant quality was not this experiment's subject).
- Longer contexts: KV growth will squeeze the expert-layer budget; the ncmoe sweet spot shifts with context length.
- Sustained thermal run (E032) before quoting these numbers publicly.

## Next Steps

- E014: speculative decoding on top of the ncmoe 36 config (Qwen3-0.6B draft): can we push 38.5 toward 50+?
- E021/E022: attack the expert-scatter penalty (34 vs 51 GB/s): expert-contiguous layout, router-driven prefetch, hot-expert pinning in VRAM.
- E010 generalization: apply -ot tensor-placement lessons to dense models (attention-on-GPU splits).
- Write the public-facing summary once E032 confirms sustained rates.
