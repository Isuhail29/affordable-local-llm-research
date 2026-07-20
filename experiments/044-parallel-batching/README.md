# Experiment 044: The Parallel-Batching Gate (Queue B1)

Date: 2026-07-20
Status: In progress
Source: research-sweeps/2026-07-sweep-02-quality/QUEUE.md B1 (self-consistency survey, GO)

## Goal

Decide whether the entire token-spending quality lane (self-consistency, best-of-N, majority voting) is affordable on this rig. Those techniques generate K samples per question. If K samples cost K x wall clock, they are prohibitively slow at our fixed ~33 t/s. If llama.cpp's continuous batching amortizes the weight reads across concurrent sequences, K samples could cost far less than K x, and the whole lane opens.

## Why this is genuinely uncertain here

Decode is memory-bandwidth-bound (E001/E027): one token-step reads the active weights once. With K concurrent sequences, a single weight read can in principle serve all K, which is why batching is nearly free on datacenter GPUs.

But our flagship is MoE with experts on CPU, and E014 taught us the countervailing mechanism: K concurrent sequences route to the UNION of their experts, not the same 8. Expected distinct experts per layer at K sequences = 128 x (1 - (1 - 8/128)^K), so K=4 touches ~29 experts instead of 8. The union grows sublinearly, so some amortization should survive, but this is exactly the effect that made speculative decoding a net loss. This experiment measures which force wins.

## Method

- Model: Qwen3-30B-A3B-Instruct-2507 (the daily driver; non-reasoning so token counts are exact)
- Server: llama-server on :8080, `-ngl 99 --n-cpu-moe 40 -fa on -c 8192 --mlock -t 12 -np K`, restarted per K
- K in {1, 2, 4, 8}; each run fires K concurrent requests
- Every request generates EXACTLY 200 tokens (`"ignore_eos": true, "max_tokens": 200`) so all comparisons are apples-to-apples
- Metric: wall clock from first send to last completion; aggregate tokens/sec = (K x 200) / wall
- Warm cache, hub stopped, A/B/A note: K=1 is re-run at the end to bound thermal drift (E032 law)

## Hypothesis (pre-registered)

1. Aggregate throughput rises substantially with K but sublinearly: at K=4, aggregate >= 2x the K=1 rate.
2. Per-request t/s falls as K rises (each sequence gets a share), while aggregate rises.
3. Diminishing returns by K=8 as the expert union approaches saturation.
4. **The gate:** at K=4, cost multiplier (wall_4 / wall_1) <= 2.5. That means 4 independent samples for <= 2.5x the time of one, which green-lights self-consistency (E045). If the multiplier is >= 3.5 (near-linear), batching gives us almost nothing and the entire token-spend lane is dead on this rig; we would close it and say so.

## Actual Result

Every request generated exactly 200 tokens (ignore_eos), server restarted per K, warm cache.

| K | Wall (s) | Aggregate t/s | Per-request t/s | Cost multiplier (wall_K / wall_1) |
|---|---|---|---|---|
| 1 (first) | 6.18 | 32.38 | 34.89 | 1.00 |
| 2 | 9.33 | 42.85 | 22.65 | 1.51 |
| 4 | 14.32 | 55.88 | 14.87 | **2.32** |
| 8 | 22.27 | 71.84 | 9.40 | 3.60 |
| 1 (reflank) | 6.15 | 32.52 | 34.35 | 1.00 |

## Benchmark analysis

**H4 (the gate) PASSED: 2.32 <= 2.5.** Four independent samples cost 2.32x the wall clock of one, not 4x. Marginal cost of each extra sample is ~44% of a solo sample at K=4 and ~37% at K=8. Sequential 4 samples would be 24.7s; batched is 14.3s, a 42% saving. **Self-consistency is affordable on this rig; E045 is green-lit.**

**H1 FALSIFIED, and the miss is informative.** I predicted aggregate >= 2x at K=4; measured 1.73x (55.88 vs 32.38). Aggregate scales almost exactly as K^0.38: each doubling of K buys a consistent ~1.30x, never the ~2x a pure bandwidth-amortization model predicts.

**Why batching under-delivers here, and it ties to our own prior work.** A pure memory-bound model says K sequences share one weight read per step, so aggregate should approach K x. Two measured effects cap it:
1. **Expert union (E014's mechanism).** K sequences route to the union of their experts: 128 x (1 - (1 - 8/128)^K) gives ~8, 15.5, 29.2, 51.6 distinct experts for K = 1, 2, 4, 8. More distinct experts per step means more bytes and more matmuls.
2. **The expert path is kernel-compute-bound, not purely bandwidth-bound (E027).** Batching amortizes weight READS but not the per-token expert COMPUTE, and E027 measured the CPU expert path as ALU-bound. So the lever batching pulls is the one that is not our binding constraint, which is precisely why the observed 1.30x-per-doubling falls short of the union model's own prediction.

H2 confirmed (per-request t/s falls 34.9 -> 9.4 as aggregate rises 32 -> 72). H3 partially confirmed: returns per sample keep shrinking, but smoothly (constant ~1.30x per doubling), with no saturation cliff by K=8.

**Thermal control worked perfectly:** the K=1 reflank (6.15s) matched the opening K=1 (6.18s) to within 0.5%, so none of the above is drift. The whole sweep was short enough to stay off E032's soak curve.

## Lessons Learned

1. Batching is a real but modest lever on CPU-expert MoE: budget ~K^0.4 aggregate scaling, not linear. Useful planning constant for every future multi-sample design.
2. The same expert-union mechanism now explains three separate results (E014 speculative loss, E027 attribution, E044 batching ceiling). It is the defining constraint of this architecture on this rig.
3. Short experiments dodge the thermal confound entirely; the reflank costs one extra minute and converts "probably not drift" into "measured, 0.5%".

## Next Steps

- **E045 (green-lit): self-consistency quality-per-second frontier.** Run K=4 majority voting on a hard-for-our-model reasoning slice at the measured 2.32x cost, and test whether accuracy gains justify it. Success bar: >= +5 accuracy points at <= 2.5x wall clock. Note the survey's warning (arXiv 2511.00751) that self-consistency gains have collapsed on modern models, so a null result is a real possibility and worth publishing.
- Practical note for the hub: models are pinned at `-np 1`. If E045 succeeds, the self-consistency route should run its own server with `-np 4` rather than changing the interactive defaults (per-request latency drops from 35 to 15 t/s, which would feel worse in chat).
