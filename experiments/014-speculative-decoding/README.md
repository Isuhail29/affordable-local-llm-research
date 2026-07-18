# Experiment 014: Speculative Decoding (Draft Model)

Date: 2026-07-18
Status: Complete

## Goal

Measure whether a tiny draft model (Qwen3-0.6B) speeds up (a) dense Qwen3-8B and (b) MoE Qwen3-30B-A3B with experts in RAM, and answer the pre-registered theory question: does batch-verification survive when the verifier's bottleneck is CPU-resident MoE experts?

## Hypothesis (pre-registered)

1. Dense 8B: net decode speedup 1.4-1.9x.
2. MoE 30B (ncmoe 40): between 0.8x and 1.3x, because a batch of 8 drafted tokens touches the union of their experts (~51 of 128 per layer, ~6.4x single-token bytes), destroying the amortization that powers speculative decoding.
3. Structured outputs accept more drafts than prose.
4. Draft length optimum lower for MoE than dense.

## Implementation

llama-server b10064, scripts/spec-decode-test.ps1 harness (health-gated, global warmup request, VRAM and clock capture), 3 standard prompts (story/code/facts) at temperature 0, max_tokens 256, run twice, second run recorded. Raw data: benchmarks/e014-*.jsonl.

Two harness-level traps were found and fixed mid-experiment (see Lessons); final clean protocol: fp16 KV cache, -np 1, explicit -t, model file pre-warmed into page cache (copy /b to NUL) before every server start, and crucially **--spec-type draft-simple**, because in b10064 `-md` alone loads the draft but speculation stays OFF (default --spec-type none).

## Actual Result (clean runs, tg t/s, story/code/facts)

| Config | story | code | facts | Verdict |
|---|---|---|---|---|
| 30B MoE baseline (-t 12, warm) | 33.5 | 33.3 | 32.9 | reference |
| 30B + draft loaded but inert (no --spec-type) | 32.5 | 33.4 | 32.2 | placebo, as expected |
| **30B + draft-simple n-max 8** | **11.9** | **30.8** | **17.2** | **0.36-0.93x, net loss** |
| 30B + ngram-simple (free, no draft) | 32.8 | 32.1 | 33.7 | 1.0x, harmless |
| 8B dense ngl 34 baseline | 34.5 | 43.5 | 42.1 | reference |
| 8B + draft-simple n-max 8 | 30.7 | 44.1 | 31.2 | 0.74-1.0x, net loss |

## Benchmark analysis

**H2 confirmed beyond its pessimistic end.** On the MoE, real speculation is a large net loss. Mechanism as pre-registered: batch verification reads the union of experts (up to ~6.4x bytes per pass), so rejected drafts are catastrophically expensive; prose (low acceptance) collapses to 11.9 t/s.

**H1 refuted on this hardware.** Even on the dense model, drafting was flat-to-negative. The literature's 1.5-2x assumes datacenter GPUs with headroom. On a TDP-shared laptop, the draft's GPU passes run at throttled clocks (observed ~1.1 GHz SM during 8B hybrid runs) and partial offload makes batch verification pay a CPU compute cost. Caveat: the dense test ran at -ngl 34 (VRAM forced), and this build's server reports no acceptance stats, so acceptance could not be decomposed from overhead.

**ngram speculation is free and safe but needs repeated text to fire**; our fresh-generation prompts gave it nothing. Candidate for document-editing workloads only.

## Byproduct findings (worth more than the headline)

1. **KV-quant config trap:** --cache-type-k/v q8_0 with default -fa auto halved 8B server throughput (21-34 vs 62 t/s). Never quantize KV without verifying flash attention actually engaged.
2. **llama-server defaults 4 slots**; -np 1 for single-user benchmarking.
3. **MoE thread optimum is 12, not 8** (32.7 vs 30.7 t/s warm): scattered expert reads are latency-bound and want more outstanding requests than dense streaming. Dense optimum remains 8.
4. **Page-cache residency is worth 1.5-3x on MoE** and is fragile: after our ballast experiments partially evicted the model file, the same benchmark gave 20.8 ± 7.7 instead of 30.7 ± 4.3. A 3-second sequential pre-read (copy /b model NUL) restores it. Now part of the launcher.
5. **Laptop GPU clocks ramp over ~10-20 s of load**; first-request numbers are garbage. Harness warms up globally; order effects masquerade as content effects otherwise.
6. **Unreproduced anomaly, documented honestly:** early misconfigured runs (inert draft, uncontrolled cache) showed 55-60 t/s on code/facts for the 30B, far above the clean 33. Suspected degenerate repetitive outputs routing to few experts (extreme locality streams at the 51 GB/s RAM ceiling). The harness now records output text so any recurrence can be diagnosed. If real, controlled expert locality is an optimization direction (ties to E021/E022).

## Lessons Learned

- Flags that load a feature do not necessarily enable it. Verify activation with a measurable signature (VRAM delta, log line), not flag presence.
- A placebo arm (draft loaded but inert) turned out scientifically useful: it cleanly bounded run-to-run noise at ~±1 t/s.
- Record model outputs alongside timings; two anomalies were undiagnosable because only numbers were saved.

## Possible Improvements

- Retest dense speculation at full offload with a Q4 draft (fits VRAM) and on AC-cooled conditions.
- Sweep --spec-draft-n-max 2-4 (large n-max maximizes the MoE union penalty; tiny drafts might squeak out a win on code).
- Parse verbose server logs for acceptance rates.

## Next Steps

- Adopt: -t 12, pre-warm, fp16 KV, -np 1 in the chat launchers (done).
- The speed frontier on this machine is now: expert-scatter bandwidth (E021/E022 layout and prefetch work) and sustained thermals (E032), not speculation.
