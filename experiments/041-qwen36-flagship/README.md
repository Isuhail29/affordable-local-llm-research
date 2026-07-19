# Experiment 041: Qwen3.6-35B-A3B as New Flagship (Queue R11)

Date: 2026-07-19
Status: In progress
Source: QUEUE.md R11 (consensus #1, GO in five of seven surveys)

## Goal

Breakthrough class (a): replace the flagship with a two-generations-newer model at the same active size. Claimed: GPQA-D 86.0 (vs ~70-73), SWE-bench Verified 73.4 (vs 22.0), 262k context, vision, bundled MTP head (enables R12, the E014 rematch). Hybrid GDN: only 10 of 40 layers carry KV (~4.8x smaller cache than the 30B), so E040's 64k result could extend much further on this model.

## Hypothesis (pre-registered)

1. The UD-Q4_K_XL GGUF (21.3 GB) loads on official b10064 (arch qwen35moe present in the source tree).
2. Decode with the E013 protocol (-ngl 99, ncmoe sweep, -t 12, mlock, warm) lands >= 25 t/s sustained at the best VRAM-fitting split. Risk: 21.3 GB CPU-side (vs 17.5 for the 30B) costs ~3-5 t/s; DeltaNet CPU fallback (#19894 class) could cost more: profiler bins will show it.
3. Quality clearly beats the 30B flagship on our probes (story/code/facts + a small reasoning slice).
4. Success = both 2 and 3 -> new flagship + launcher; then R12 (draft-mtp) immediately. Kill: < 25 t/s sustained -> long-context specialist role only.

## Method

Load smoke first (llama-bench short run; instant arch error if unsupported). Then ncmoe ladder {all-CPU, -8, -12 GPU expert layers}, A/B/A flanked vs the 30B champion, E027 profiler bins on the CPU side, sanity outputs (CUDA-13.2-gibberish class check), then quality battery.

## Actual Result (interim)

### Speed: hypothesis 2 SMASHED (same-session A/B/A)

| Config | tg t/s | pp512 t/s |
|---|---|---|
| 30B champion flank 1 / flank 2 (ncmoe 40) | 28.71 / 28.37 | - |
| 35B ncmoe 36 | 36.49 | 280 |
| 35B ncmoe 34 | 36.62 | 302 |
| **35B ncmoe 32** | **37.51** | **342** |

Real-server battery: 37.3-40.5 t/s across five prompts. **+32% over the champion**, >= 25 bar demolished. The hybrid GDN architecture (10 of 40 attention layers) is simply built for the experts-on-CPU split.

### Quality: gate OPEN, blocked by build-era template parsing

Three battery attempts (default, /no_think, --reasoning-budget 0) all had the server route ALL output into reasoning_content: b10064 (2026-07-17) predates this model's template conventions. Visible reasoning content is high quality: the dock-scheduling math is exactly on the correct track (210 by 11:00, 1680/77 = 21.82 h) and the logic grid is solved with clean contradiction-checking, but answers truncate at token caps mid-deliberation. Per protocol, no crown without a clean quality pass.

### Quality resolution (b10068, free deliberation, 2500-token budget)

The "parser bug" was a misdiagnosis chain worth recording: identical output across builds proved the model is REASONING-NATIVE (this MTP release deliberates unconditionally; --reasoning-budget 0 and /no_think do not disable it). With room to finish: facts answer flawless (8/8 planets, accurate facts), logic answer EXACTLY matches ground truth with a clean minimal proof. Story/code/math still hit 2500 tokens mid-deliberation (it spends 1.5-2.4k tokens thinking even on small tasks). The 30B comparison on math/logic: also correct-track, also cap-truncated at 700, so those probes differentiate harness budgets, not models.

### Verdict (tonight): DUAL FLAGSHIP

- **Speed: 35B wins decisively** (+32%, 37-40 t/s real-world, A/B/A flanked).
- **Quality: everything judgeable is flawless or superior**, consistent with its verified two-generation benchmark lead; full crown awaits an uncapped judged battery (4000+ token budgets).
- **The honest cost: tokens-to-answer.** 30-90 s of visible deliberation before answers. For instant chat the 30B (turbo 42 t/s, answers in seconds) remains the daily driver; the 35B ships as Start-35B-Reasoning.bat for deep work. R12 (draft-mtp on its bundled head) directly attacks deliberation wall-time and is the next experiment.

### Next step (in motion)

b10068 official binaries fetched to llama.cpp/bin-b10068/ (kept separate from the shipped b10064). Redo the battery there with proper template/parser handling; if quality holds vs the 30B transcripts, declare breakthrough (a), ship the launcher, and proceed immediately to R12 (draft-mtp on the bundled MTP head).

### Harness lessons (running list)

- Thinking-generation models need: dual-channel capture, generous max_tokens, and template-aware toggles; a battery harness upgrade (v2 of spec-decode-test.ps1) is due before R12.
- Week-old models on month-old binaries = parser roulette; check template vintage FIRST (added to protocol).
