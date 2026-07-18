# 00. Hardware Baseline: The Research Machine

Everything in this project is bounded by the physical limits of this specific machine. This document records what we measured, what the spec sheets claim, and the napkin math that predicts inference performance before we run a single benchmark. Later experiments will confirm or refute these predictions.

## Measured configuration (2026-07-18)

| Component | Detail | Source |
|---|---|---|
| CPU | Intel Core i7-14650HX, 16 cores / 24 threads (8 P-cores with HT + 8 E-cores) | Win32_Processor |
| RAM | 48 GB DDR5-5600, dual channel, mixed modules (16 GB Samsung + 32 GB Micron) | Win32_PhysicalMemory |
| GPU | NVIDIA GeForce RTX 5060 Laptop, 8151 MiB VRAM, driver 591.91 | nvidia-smi |
| iGPU | Intel UHD Graphics (shares system RAM) | Win32_VideoController |
| SSD | WD PC SN5000S 1 TB NVMe (PCIe 4.0), 416 GB free | Get-PhysicalDisk |
| OS | Windows 11 Home 10.0.26200 | System |
| Runtime | llama.cpp b10064, CUDA 13.3 Windows build | GitHub releases |

## The memory hierarchy in bandwidth terms

Token generation reads essentially every active model weight once per token. That makes decode speed a bandwidth problem, and this table is the single most important table in the project:

| Tier | Capacity | Theoretical bandwidth | Realistic effective | Notes |
|---|---|---|---|---|
| GPU VRAM (GDDR7) | 8 GB | 448 GB/s (28 Gbps confirmed by E001) | **312.7 GB/s measured** | 62.17 t/s x 5.03 GB, Experiment 001 |
| System RAM (DDR5-5600 x2) | 48 GB | 89.6 GB/s | **50.8 GB/s measured** | 10.10 t/s x 5.03 GB at 8 threads, E001 |
| PCIe link (GPU<->RAM) | n/a | 32-64 GB/s (Gen4/5 x8/x16) | ~12-25 GB/s | To be measured, laptop wiring unknown |
| NVMe SSD | 1 TB | ~5.2 GB/s seq read (spec) | ~3-5 GB/s | Random reads much worse |

The RAM math: DDR5-5600 moves 5600 million transfers/sec, each transfer is 8 bytes per channel, and we have 2 channels: 5600 x 8 x 2 = 89.6 GB/s. The mixed 16+32 GB modules may cost some efficiency in flex mode; a memory benchmark will tell us the real number.

## Napkin-math performance predictions

For Qwen3-8B at Q4_K_M (5.03 GB of weights, all touched for every generated token):

```
tokens/sec ceiling ≈ effective bandwidth / bytes per token

Full GPU offload:  ~270-330 GB/s / 5.03 GB  ≈  54-65 tokens/sec
Pure CPU:          ~60 GB/s  / 5.03 GB  ≈  12 tokens/sec
Weights on SSD:    ~5 GB/s   / 5.03 GB  ≈  1 token/sec
```

Three predictions fall out of this before we benchmark anything:

1. **Full GPU offload should land somewhere near 40-65 t/s.** If it is far below, something other than bandwidth is the bottleneck (kernel efficiency, sampling overhead, thermals).
2. **CPU-only should land near 8-12 t/s.** The 24 threads mostly do not help beyond saturating memory bandwidth; we expect diminishing returns past ~8-16 threads. This is a testable hypothesis.
3. **Anything that forces per-token SSD reads collapses performance by ~50x.** Naive "just mmap a huge model and let it page" is not a strategy, it is a failure mode. Hiding or avoiding that 5 GB/s cliff is the core research challenge of this project.

Prompt processing (prefill) is different: it is compute-bound because many tokens are processed per weight load. GPU prefill should be dramatically faster than CPU prefill, likely by 10-30x.

## The interesting constraint: 8 GB VRAM

Qwen3-8B Q4_K_M (5.03 GB) plus KV cache and compute buffers fits fully in 8 GB VRAM. That is deliberate for the baseline: it gives us a clean "best case" reference point.

The research program then works downward and upward from there:

- **Downward:** artificially restrict offload (-ngl sweeps) to simulate 4 GB and 6 GB GPUs and study exactly how hybrid CPU/GPU execution degrades.
- **Upward:** move to Qwen3-14B, 32B, and MoE models (Qwen3-30B-A3B) that cannot fit, where offloading strategy, expert placement, and streaming become decisive.

## Measured results (Experiment 001, 2026-07-18)

- [x] VRAM effective bandwidth: **312.7 GB/s** (69.8% of the 448 GB/s spec; the GPU is the 28 Gbps GDDR7 variant)
- [x] RAM effective bandwidth during inference: **50.8 GB/s** (56.7% of theoretical; Flex Mode penalty suspected, needs E002 to decompose)
- [x] Decode baselines: **62.17 t/s** full GPU, **10.10 t/s** CPU at 8 threads
- [x] CPU thread optimum: **8 threads** (the P-core count); 24 threads is *slower* than 4
- [x] Prefill: 2186 t/s full GPU, **44.8 t/s true CPU-only**; llama.cpp's automatic PCIe weight streaming gives 819 t/s even at -ngl 0 (~8 GB/s implied PCIe rate)

## Measured results (Experiment 002, 2026-07-18)

- [x] Practical RAM copy ceiling: **55.6 GB/s** (12-process STREAM; single core gets 28.6 GB/s, so multithreading is mandatory)
- [x] E001's 56.7% "efficiency" fully explained: 89.6 theoretical → 55.6 practical platform ceiling → 50.8 during inference (91% extraction; llama.cpp is fine, the platform is the limit)
- [x] **Placement lottery: ±21% CPU decode swing** (8.33-10.10 t/s) purely from where Windows physically places the weights; malloc'd weights tend to land in the single-channel Flex Mode region (41.9 GB/s = 93.5% of that region's ceiling). Protocol: always mmap, run twice, record memory state
- [x] PCIe link: Gen5 x8 negotiated max (idles at Gen1); E001's ~8 GB/s streaming rate remains unexplained → E004

## Measured results (Experiment 014, 2026-07-18)

- [x] Speculative decoding (0.6B draft): net LOSS on both models here; MoE verification reads the union of experts per batch (~6.4x bytes), prose collapses to 0.36x. ngram speculation: neutral, free, only fires on repeated text
- [x] MoE thread optimum: **-t 12** (32.7 t/s vs 30.7 at -t 8); dense optimum stays 8. Scattered expert reads are latency-bound and want more threads
- [x] Page-cache residency is worth 1.5-3x on the 30B and is fragile; a 3 s sequential pre-read (copy /b model NUL) restores it. Built into the launchers
- [x] Config traps: KV-quant with -fa auto halved server speed; llama-server defaults to 4 slots (-np 1 for benchmarks); b10064 needs --spec-type to actually enable speculation; laptop GPU clocks need ~10-20 s of load to ramp

## Field + sustained observations (E032 partial, 2026-07-18)

- [x] Real-user session (browser + apps open, RAM ~90%): 30B dropped to 12.2 t/s, GPU 10-20% util (expected for CPU-bound MoE), GPU 80 C from shared heatsink. Root cause: page-cache eviction under RAM pressure
- [x] Controlled sustained run with --mlock (model pinned, 17.5 GB working set): 33.3 then 32.6 t/s across ~1,700 tokens, GPU 84 C, SM clocks steady at 2805 MHz, no throttle observed yet. --mlock now in the 30B launcher
- [ ] Full E032: 10+ minute continuous load with temperature/clock logging

## Open measurement tasks
- [ ] PCIe negotiated link and transfer rate (E004): E001 implies ~8 GB/s streaming
- [ ] SSD sequential and random read (E003/E020)
- [ ] Sustained vs burst clocks under 10-minute load (E032, laptop thermals)
- [ ] Power draw under CPU vs GPU inference if measurable (E032)
