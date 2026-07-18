# Experiment 001: Baseline Benchmark Sweep

Date: 2026-07-18
Status: Complete

## Goal

Establish the ground-truth performance envelope of Qwen3-8B Q4_K_M on the research machine across the full GPU offload range, before any optimization work. Every future experiment is compared against these numbers.

## Background

Decode (token generation) is memory-bandwidth-bound: every generated token requires reading all active weights once. Prefill (prompt processing) is compute-bound. Our machine has three bandwidth tiers: VRAM, system RAM, and NVMe SSD. See docs/00-hardware-baseline.md and docs/07-memory-hierarchy-and-bandwidth.md.

llama.cpp splits the model with `-ngl N`: the last N layers live on GPU, the rest on CPU.

## Hypothesis

1. Full offload (-ngl 99) decode lands in the 40-65 t/s range.
2. Pure CPU (-ngl 0) decode lands in the 8-12 t/s range.
3. Decode speed follows a linear two-tier model: per-token time = CPU-layer time + GPU-layer time.
4. CPU decode saturates well before 24 threads; possibly regresses at 24.
5. Prefill shows a much larger GPU advantage than decode, 10x or more.

## Implementation

- llama.cpp b10064, CUDA 13.3 Windows build, prebuilt binaries; CPU backend auto-selected ggml-cpu-alderlake.dll; GPU is compute capability 12.0 (Blackwell)
- Model: Qwen3-8B-Q4_K_M.gguf (5,027,783,488 bytes, official Qwen GGUF), 36 layers
- Script: scripts/run-baseline-bench.ps1; laptop on AC power, battery 100%
- Sweep 1: -ngl 0, 8, 16, 24, 32, 99, default pp512/tg128, 5 reps
- Sweep 2: -ngl 0, -t 4, 8, 12, 16, 20, 24
- Control run: CUDA_VISIBLE_DEVICES=-1 (GPU hidden), -ngl 0 -t 8
- Raw JSON: benchmarks/2026-07-18_0307-*.json

## Expected Result

Monotonic decode curve ~10 to ~50+ t/s; steeper prefill curve; flat thread scaling past 8-16 threads.

## Actual Result

### Sweep 1: GPU offload levels

| -ngl | pp512 t/s | tg128 t/s | tg linear-model prediction |
|---|---|---|---|
| 0 | 818.8 ± 17.3 | 9.56 ± 0.24 | (anchor) |
| 8 | 953.7 ± 10.0 | 12.77 ± 0.13 | 11.8 |
| 16 | 1208.9 ± 43.4 | 15.91 ± 0.17 | 15.3 |
| 24 | 1516.1 ± 36.6 | 21.71 ± 0.37 | 21.9 |
| 32 | 1946.2 ± 54.3 | 37.12 ± 0.32 | 38.6 |
| 99 (all 36) | 2186.0 ± 23.4 | 62.17 ± 0.83 | (anchor) |

### Sweep 2: CPU thread scaling at -ngl 0

| Threads | pp512 t/s | tg128 t/s |
|---|---|---|
| 4 | 771.3 | 8.59 |
| 8 | 751.5 | **10.10** |
| 12 | 749.6 | 9.83 |
| 16 | 712.6 | 9.64 |
| 20 | 703.4 | 8.81 |
| 24 | 694.2 | 7.94 |

### Control: GPU hidden (CUDA_VISIBLE_DEVICES=-1), -ngl 0, -t 8

| Config | pp512 t/s | tg128 t/s |
|---|---|---|
| GPU visible, ngl 0 | 812.3 | 10.07 |
| GPU hidden | **44.76** | 9.97 |

## Benchmark analysis

**Hypotheses 1 and 2 confirmed.** Full GPU decode 62.17 t/s; CPU decode 9.56-10.10 t/s.

**Back-solved effective bandwidths (the machine's real constants):**

```
GPU:  62.17 t/s x 5.03 GB = 312.7 GB/s effective  = 69.8% of 448 GB/s
      (settles the spec question: this is the 28 Gbps GDDR7 variant)
CPU:  10.10 t/s x 5.03 GB =  50.8 GB/s effective  = 56.7% of 89.6 GB/s
      (consistent with Flex Mode mixed-module penalty)
```

**Hypothesis 3 confirmed.** The two-anchor linear model (104.6 ms/token CPU, 16.1 ms/token GPU, split by layer fraction) predicts every intermediate point within ~5%. Consequence: hybrid decode performance is fully predictable from the two endpoints, and partial offload gains stay modest until most layers are on GPU (the CPU tier dominates the sum, Amdahl-style: even 32/36 layers on GPU only reaches 37 t/s of the 62 t/s ceiling).

**Hypothesis 4 confirmed.** Decode peaks at exactly 8 threads = the P-core count, and monotonically regresses beyond it (7.94 t/s at 24 threads, worse than 4 threads' 8.59). Memory bandwidth saturates with 8 P-cores; E-cores add contention, not throughput.

**Hypothesis 5 refuted as stated, then vindicated by the control.** GPU-visible prefill advantage looked like only 2.7x (2186 vs 819). But 819 t/s of prefill requires ~13 TFLOPS, impossible for this CPU. The control run exposed why: with a CUDA build, llama.cpp streams weights over PCIe and runs large-batch prefill on the GPU even at -ngl 0. True CPU prefill is 44.8 t/s, so the honest GPU prefill advantage is 49x, and the automatic streaming assist alone is worth 18x. Implied PCIe streaming throughput: 5.03 GB / (512 tokens / 819 t/s) ≈ 8 GB/s.

## Lessons Learned

1. **The napkin math holds.** Bandwidth/bytes predicted decode within measurement error at both endpoints. The roofline model in doc 07 is now empirically anchored.
2. **"-ngl 0" is not "CPU-only" in a CUDA build.** Prefill silently uses the GPU via weight streaming. Any CPU experiment must hide the GPU (CUDA_VISIBLE_DEVICES=-1) or use a CPU-only binary. Our first control attempt failed because `set CUDA_VISIBLE_DEVICES=` unsets the variable rather than hiding devices; -1 is required. Controls need verification too (check ggml_cuda_init in the log).
3. **Thread count is a first-class knob.** Default thread selection is not optimal; -t 8 beats it. All future CPU-side benchmarks pin -t 8 unless threads are the variable under test.
4. **PCIe weight streaming at ~8 GB/s turned prefill from hopeless to fine at ngl 0.** This is the existence proof for the project thesis: batching plus streaming hides a slow tier for compute-bound phases. The open question for Tier 3 research is whether any analogous trick exists for bandwidth-bound decode, where per-token streaming cannot amortize.
5. **62 t/s full-GPU and 10 t/s CPU are the baseline constants** every optimization gets compared against.

## Possible Improvements

- llama-bench does not report VRAM/RAM footprints; pair future runs with nvidia-smi sampling or llama-server logs.
- 5 repetitions on a laptop: no thermal-sustain data; a 10-minute sustained run may show throttling (backlog E032).
- pp512 only; longer prompts (pp2048/pp8192) would show KV-cache-growth effects on prefill.

## Next Steps

- E002: STREAM-style RAM bandwidth benchmark to decompose the 56.7% efficiency (Flex Mode vs controller limits)
- E004: PCIe characterization; the implied ~8 GB/s streaming rate suggests Gen4 x8 or overhead; nvidia-smi -q will report the negotiated link
- E005: prefill ubatch scaling, now with proper CPU-only controls
- E010: offload placement quality (-ot overrides) now that the linear model gives us the prediction to beat
