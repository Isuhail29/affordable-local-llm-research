# Experiment 040: 64k Context on the Flagship (Queue R1)

Date: 2026-07-19
Status: In progress
Source: research-sweeps/2026-07-sweep-01/QUEUE.md R1 (kv-cache survey C1, GO)

## Goal

16x our shipped context window (4k-8k today) on the existing 30B flagship with zero downloads: breakthrough class (c) if it holds. The model is natively 262k; the only wall was KV memory in 8 GB VRAM.

## Arithmetic (pre-registered)

KV at 65536 tokens, q8_0, GQA 4 KV heads x 128 dim x 48 layers x 2 (K+V) x ~1.07 B/elem ≈ 3.2 GiB. All experts on CPU (--cpu-moe) frees the VRAM the ncmoe-40 config spends on expert layers: attention/shared ~2.2 GB + KV 3.2 + compute ~0.7 ≈ 6.1 GB + idle ≈ within 8. The known E014 trap (-fa auto silently disabling with quantized KV, halving speed) is controlled by explicit -fa on and log verification.

## Hypothesis

1. Loads and serves at -c 65536 without OOM or silent system-RAM spill (nvidia-smi under 8,000 MiB).
2. Decode at 60k depth >= 15 t/s (KV reads add ~3.2 GB/token... no: per token attention reads KV once: +3.2 GB/token GPU-side at 313 GB/s ≈ +10 ms... corrected estimate: GPU KV read ~10 ms/token at full depth on top of ~32 ms CPU expert time -> ~24 t/s at depth 0 falling toward ~15-20 at 60k).
3. Paired PPL (f16 vs q8_0 KV, same text) delta under 1%.
4. Prefill of 60k tokens takes minutes (CPU-expert-bound): documented honestly as the usability limit, motivating R4 (KV persistence) next.

## Method

- Server: official b10064, -ngl 99 --cpu-moe -fa on -ctk q8_0 -ctv q8_0 -c 65536 --mlock -t 12 -np 1
- Long document: our own docs corpus tiled to ~240 KB (~60k tokens), sliced at 25/50/75/100%
- Per depth: one request (slice + question, max_tokens 128), record prompt t/s, decode t/s, VRAM, GPU temp
- PPL pair afterwards (server down): llama-perplexity -ctk/-ctv f16 vs q8_0, chunks 8, hybrid config
- Condition note: R11's 21.3 GB download trickles at ~3 MB/s in the background (negligible CPU, minor disk); recorded per protocol

## Actual Result

### The hunt: KV dtype, not window size, governs everything

| Config | Depth | Decode t/s | VRAM |
|---|---|---|---|
| q8_0 KV, -c 65536, --cpu-moe | 16.9k / 33.3k / 50.9k | 18.2 / 13.9 / 10.5 | 4.5 GB flat (KV on host) |
| q8_0 KV, -c 49152 | 33.3k / 46.4k | 13.7 / 11.0 | 3.7 GB flat (KV on host) |
| q8_0 KV, -c 32768, ncmoe 44 | 16.9k / 31.1k | 19.5 / 14.3 | 4.4 GB (KV on host) |
| **f16 KV, -c 32768** | 16.9k / 31.1k | **24.4 / 21.7** | 4.3 GB |
| **f16 KV, -c 65536 (WINNER)** | **46.4k / 62.5k** | **20.2 / 18.2** | **7.4 GB** |

The 67k probe overflowed the window (the corpus tokenized at ~3.66 chars/token vs the 4.0 estimate); 62.5k is the deepest measured point.

### Success criterion: MET

Pre-registered bar: >= 15 t/s at full 64k depth. Measured: **18.24 t/s at 62,548 tokens**, GPU 87 C stable, f16 KV so the PPL-delta condition is moot (reference dtype). Breakthrough class (c): 16x context, zero downloads.

## Benchmark analysis

**The governing discovery: quantized KV cache is a placement trap on this build.** With -ctk/-ctv q8_0, the KV lands in host RAM at every window size and every expert-override style, and decode decays ~1.2 µs per token of depth (PCIe-streamed attention). With f16 KV the depth slope collapses ~3x and the 64k config runs 7.4 GB VRAM. This retroactively unifies E014's "KV-quant halves 8B server speed" trap: same disease, one mechanism, now measured across two experiments. Protocol law: **never quantize KV on this build; f16 KV is strictly better wherever it fits, and it fits to 64k.**

Honest limits: prefill is the usability tax (62k tokens ~ 6 minutes, CPU-expert-bound), which is exactly queue item R4 (KV persistence: save the prefilled state to NVMe, restore in seconds). Depth still costs ~25-45% decode vs short context (24 -> 18 t/s from 17k to 62k). GPU pinned at 86-87 C throughout (E032 soak regime).

## Lessons Learned

1. The community default advice ("quantize KV for long context") is exactly backwards on this stack: the quantized cache bought nothing (VRAM stayed low because it never went there) and cost 30-50% decode at depth.
2. Depth probes with progressive slices + server prefix caching made a 5-config matrix affordable in one evening.
3. E014's config trap needed two experiments and a unifying mechanism to become a law; single-experiment "explanations" (fa auto) were incomplete.

## Next Steps

- Shipped: Start-30B-AI-64K.bat (winner config).
- R4 (KV persistence) is now the highest-leverage usability follow-up: 6-minute prefill -> seconds.
- R8's KV-quant quality ladder is REFRAMED: on this build it is a speed question, not a quality question; deprioritized.
- Report the q8-KV placement behavior upstream-adjacent (in OUR repo; llama.cpp policy prohibits agent submissions).
