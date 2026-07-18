# Experiment 023: Active-Expert Reduction (top-k override)

Date: 2026-07-19
Status: Complete

## Goal

Quantify the speed/quality tradeoff of reducing Qwen3-30B-A3B's active experts per token below the trained top-8, via llama.cpp's metadata override. Motivation: E021 proved the expert-read path is bandwidth-capped at ~30-34 GB/s and no configuration knob fixes it; reducing bytes per token is the one remaining non-source lever (idea sourced from the KTransformers research, notes/internals-ktransformers.md).

## Background

Each MoE layer routes every token to its top-k experts by router score (trained with k=8). `--override-kv qwen3moe.expert_used_count=int:N` changes k at load time. Expert bytes per token scale linearly with k, so k=6 cuts expert reads 25%, k=4 cuts 50%. Router scores are ordered: the experts dropped are the LOWEST-scoring ones, so damage should be sublinear at first.

## Hypothesis

1. Speed scales close to the byte reduction: k=6 gives +15-25% decode.
2. Quality degrades convexly: k=6 mild, k=4 severe.

## Implementation

- Speed: spec-decode-test.ps1 harness, standard config (-ngl 99 --n-cpu-moe 40 -c 8192 -np 1 -t 12 --mlock), E002/E021 protocol (warm cache), story/code/facts prompts, second runs recorded
- Quality: llama-perplexity on datasets/ppl-text.txt (52 KB of our own technical docs; same text across all k, so the PAIRED comparison is meaningful even though absolute PPL on this corpus is nonstandard)
- Reference: 30b-nodraft-warm (k=8): 33.5 / 33.3 / 32.9 t/s

## Actual Result

| k | tg t/s (story/code/facts) | PPL | PPL delta vs k=8 |
|---|---|---|---|
| 8 | 33.5 / 33.3 / 32.9 | 6.6166 ± 0.212 | reference |
| 6 | **41.6 / 40.6 / 37.8** | 6.7750 ± 0.219 | **+2.4%** |
| 4 | (not speed-tested) | 7.7971 ± 0.257 | +17.8% |

## Benchmark analysis

**Both hypotheses confirmed.** k=6 delivers +21% decode speed for +2.4% perplexity; k=4 breaks the model (+17.8% PPL) and is rejected. The +21% is close to the +25% byte-reduction ceiling, which is yet another independent confirmation that decode is purely bandwidth-bound: remove a quarter of the bytes, get almost a quarter more speed.

The convex damage curve matches MoE intuition: router scores are concentrated, so experts ranked 7-8 contribute marginal refinement while 5-6 still carry real signal.

Sanity checks: k=6 outputs on story/code/facts prompts are coherent and correct at temp 0 (recorded in benchmarks/e014-30b-top6.jsonl output_head fields).

## Lessons Learned

1. When a system is proven bandwidth-bound, byte reduction is the most reliable speed lever: it converted at 84% efficiency (21/25) with zero engineering.
2. Same-text paired perplexity is a cheap, defensible way to compare config variants; absolute PPL on a custom corpus means little, the DELTA means a lot.
3. Trained hyperparameters (top-8) are not sacred at inference time, but the exploitable slack is one step deep here (6 yes, 4 no).

## Possible Improvements

- Proper eval beyond perplexity (MMLU-style subset or task-based checks) before recommending turbo mode for serious work.
- Test k=7 (predicted ~+11% speed, near-zero quality cost) if a finer point on the curve is wanted.
- Combine k=6 with a future kernel fix from the source-build track: gains should stack multiplicatively (~33 x 1.21 x kernel-gain).

## Next Steps

- Shipped as Start-30B-AI-Turbo.bat (user-facing opt-in; full-quality top-8 stays the default launcher).
- Remaining speed frontier: the ggml mul_mat_id source patch track (notes/expert-scatter-roadmap.md).
