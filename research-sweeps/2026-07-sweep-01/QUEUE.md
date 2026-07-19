# QUEUE.md: ranked experiment queue, sweep 2026-07-sweep-01

Compiled 2026-07-19 from the seven domain surveys in this folder (moe-inference-advances, new-models-for-our-class, speculative-decoding-frontier, quantization-frontier, kv-cache-long-context, llamacpp-upstream-movement, hardware-and-platform-tricks). Note: an eighth survey, alt-runtimes-offload.md, was planned but is absent from the folder; alt-runtime coverage below leans on the ik_llama.cpp / KTransformers / PowerInfer material embedded in the MoE and quantization surveys. If that survey lands later, re-check this queue against it.

Only GO and strong-MAYBE candidates appear. Ranked by (breakthrough probability x class) / effort. Breakthrough classes per project definition: (a) model upgrade, (b) speed >= 20% at equal quality on the flagship, (c) capability previously impossible here.

**Run-order note (wall clock, not rank):** start the R11 download (22.9 GB, ~2.2 h at 3 MB/s) before touching anything else. R1-R10 need zero or tiny downloads and run while it trickles. R12 costs nothing once R11 is on disk.

**Housekeeping (unranked, fold in when convenient):** grab b10068 binaries only if DFlash + quantized KV enters play (PR #25823 rotation fix); rebase the E027 profiler patch and re-verify our -ot 3D-repack 0xC0000005 repro on b10068; watch DSpark (PR #25173), the sched sync-reduction re-land (#20793 revert), Q2_0 MoE quants (#24448), Kimi-Linear (PR #17592), per-head KV quant (#21385).

---

## This week (hours-class)

### R1. 64k context on the current flagship (q8_0 KV)
**Class:** capability (16x context). **Effort:** hours. **Verdict source:** kv-cache C1 (GO).
Zero download. The 30B flagship is natively 262k; the only obstacle was KV memory, and the arithmetic says 64k fits at q8_0 (3.19 GiB KV inside the ~5 GB VRAM budget with all experts on CPU). Measure VRAM fit, decode t/s at depth 0/16k/32k/48k/60k, paired PPL f16-vs-q8_0 KV, and re-test the E014 `-fa auto` slowdown. Keep `-ub` moderate; `-c 49152` fallback ready.
- **First step:** `llama-server -m models\Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf -ngl 99 --cpu-moe -fa on -ctk q8_0 -ctv q8_0 -c 65536 --mlock -t 12`
- **Success:** loads and sustains >= 15 t/s at full 64k depth with q8_0 PPL delta < 1% vs f16. **Kill:** OOM at 49152 fallback too, or depth collapse below 10 t/s.

### R2. Qwen3.5-9B fully GPU-resident (speed + quality falsification, first vision)
**Class:** model/speed hybrid, plus capability (vision). **Effort:** hours. **Verdict source:** new-models C1 (GO).
6.17 GB download (~35 min), the cheapest real breakthrough shot in the sweep. Hybrid GDN + sparse MoE, 262k context, vision, claimed GPQA-D 81.7. Fits ENTIRELY in 8 GB VRAM: no RAM wall, no CPU-GPU TDP sharing. A/B/A decode vs the 30B flagship at 0/8k/32k plus our quality probes; vendor-benchmark inflation is the thing under test. Plan Q4_K_S or short context first pass, VRAM headroom is thin with mmproj.
- **First step:** download bartowski `Qwen_Qwen3.5-9B-GGUF` Q4_K_M (6.17 GB) + mmproj from HF; verify arch loads on b10064.
- **Success:** >= 60 t/s GPU-resident AND quality comparable to the flagship on our battery (daily-driver redefinition + first local vision). **Kill:** quality clearly below flagship class; keep as vision utility only.

### R3. Expert-scattering measurement s (Phase-0 referee for all spec decoding)
**Class:** incremental, but it gates R5/R12/R17 and retro-explains E014 quantitatively (publishable alone). **Effort:** hours. **Verdict source:** speculative C1 (GO).
Zero download. Patch the instrumented build (~50 lines at the expert-selection site the E023 override already touches) to log per-layer router top-k IDs across decode on the 30B. Compute U(2..8) and scattering s on chat, code-gen, and code-rewrite workloads. Plug into the section-7 cost model and pre-register predicted speedups for MTP, EAGLE-3, and ngram-mod before running any of them.
- **First step:** add router top-k ID logging to the instrumented build; run 3 workload traces on `models\Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf`.
- **Success:** s measured with tight error on 3 workloads; predictions registered. s <= 0.5 green-lights the spec lane, s >= 0.7 pre-kills EAGLE-3 (R17) and caps MTP expectations.

### R4. Cross-session KV persistence (slot save/restore + --cache-ram)
**Class:** capability (instant-resume 32k document sessions; strictly TTFT, not decode). **Effort:** hours. **Verdict source:** kv-cache C3 (GO).
Zero download. Prefill is our real long-context pain (CPU-expert-bound, minutes for 32k). NVMe at ~5 GB/s restores a 3 GiB slot in seconds. Measure cold re-prefill vs `--cache-ram` in-session restore vs disk restore after full restart, at 16k and 32k; note `--cache-ram` defaults to 8192 MiB so it may already be silently active in our launchers, making log verification mandatory. Wire the winner into Start-30B-AI.bat as a resume path.
- **First step:** relaunch the Start-30B-AI.bat config with `--slot-save-path` set, build a 16k context, `POST /slots/0?action=save`, restart, `action=restore`, time it.
- **Success:** >= 5x faster resume than re-prefill with verified log engagement. **Kill:** restore fails or engagement never fires on our MoE hybrid config.

### R5. ngram-mod in the long-draft saturation regime
**Class:** speed (workload-conditional). **Effort:** hours. **Verdict source:** speculative C3 (GO).
Zero download. E014 killed short-draft ngram-simple; ngram-mod is a different algorithm run in a different regime: 48-64-token drafts where the expert union saturates at E_total (verify cost capped at ~16 token-equivalents on the 30B) and PR-reported acceptance hits 0.7-0.9 on repetitive text. Test on code rewrite/refactor, agentic edit loops, summarize-with-quotes; then the never-measured combined stack `draft-mtp,ngram-mod` once R11/R12 land. Explicitly test the CRLF/LF reset trap on Windows.
- **First step:** `--spec-type ngram-mod --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64` A/B/A on a code-rewrite workload against the 30B baseline.
- **Success:** >= 20% net decode on repetition-heavy workloads with zero regression on chat (honest headline reports the workload mix). **Kill:** no workload class clears +10%.

### R6. Hybrid prefill economics: GGML_OP_OFFLOAD_MIN_BATCH x -b/-ub sweep
**Class:** incremental (but gates all long-context usability, including R1 and R11). **Effort:** hours. **Verdict source:** llamacpp C4 (GO).
Zero download. We have never measured or tuned prefill on the flagship; the offload guides say batch size is THE knob for CPU+GPU MoE. Sweep `GGML_OP_OFFLOAD_MIN_BATCH` {16, 32, 64, 128} x `-b/-ub` {512, 2048, 4096}, measure pp at 2k/8k/32k, A/B/A flanked. Publishable as a tuning map for this rig class.
- **First step:** `$env:GGML_OP_OFFLOAD_MIN_BATCH='16'; llama-bench -m models\Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf ...` pp512/pp2048/pp8192 grid.
- **Success:** any cell beats default prefill by >= 20% at 8k+; new launcher defaults. **Kill:** flat map, document and close.

### R7. Scheduler control sweep: --prio, --cpu-mask, --cpu-strict, core parking
**Class:** incremental + protocol value (error-bar tightening sharpens every later experiment). **Effort:** hours. **Verdict source:** hardware C2 (GO).
Zero download. All flags already in b10064, never set by us. Grid {prio 0 vs 2} x {default vs P-core mask + --cpu-strict 1} x {Balanced vs High Performance + CPMINCORES=100}, A/B/A, then a 10-run variance measurement of the best cell vs baseline. Use `--prio 2`, not 3 (realtime starves the system at -t 12). Expect 0-8% mean; the real prize is cutting the E032 +-10% cross-session error bar.
- **First step:** `llama-bench -m models\Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf -t 12 --prio 2` vs prio 0, flanked.
- **Success:** error bar tightened to +-5% or better (protocol law update), any mean gain is bonus. **Kill:** variance unchanged and gain < 3%.

### R8. KV-quant quality ladder and the K/V asymmetry test
**Class:** incremental (tunes and de-risks R1, R11). **Effort:** hours. **Verdict source:** kv-cache C4 (GO).
Zero download. Sources conflict on whether K or V tolerates 4-bit worse (KIVI theory vs measured blog tables); nobody has published the ladder for an MoE on consumer hardware. Paired PPL on ppl-text.txt for {f16/f16, q8/q8, q8/q4, q4/q8, q4/q4, q5_1 mids} on Qwen3-8B and the 30B, plus decode t/s at 16k/32k to test the "q4_0 KV up to 92% slower at long context" claim on Blackwell.
- **First step:** paired PPL run, flagship, `-ctk q8_0 -ctv q4_0` vs `-ctk q4_0 -ctv q8_0`, -fa on.
- **Success:** publishable asymmetry table; picks the R1 dtype config. Cannot fail as an experiment, only as a hypothesis.

### R9. Flex-mode RAM topography + stick-pull falsification
**Class:** incremental (diagnostic that bounds the only remaining bandwidth lever). **Effort:** hours. **Verdict source:** hardware C1 (GO).
~3 MB download (Intel MLC). Our mismatched 16+32 sticks run in Flex Mode: first 32 GB interleaved dual-channel, last 16 GB single-channel at ~half bandwidth. Nobody has published inference numbers for this topology, and E021-E028 never controlled physical placement. MLC bandwidth-vs-footprint sweep 4 to 44 GB looking for the cliff, then A/B/A the flagship on the 32 GB stick alone vs both. Check BIOS for XMP controls while the case is open (expect locked; closes that thread).
- **First step:** download Intel MLC, run the buffer-size sweep as admin.
- **Success:** measured answer to "is flex mode throttling decode, and what would a matched kit buy". Purchase question opens only at > 15% measured deficit AND post-crisis RAM prices.

### R10. iGPU kill-test: UHD as a Vulkan device beside CUDA
**Class:** incremental (predicted negative, closes the "why not the iGPU?" question with our own numbers). **Effort:** hours. **Verdict source:** hardware C4 (GO).
~50-100 MB download. Our 16-EU UHD shares the DDR5 bus and package power with the P-cores; the 32-EU UHD 770 already loses to a 12900K everywhere. Drop ggml-vulkan.dll from the b10064 Vulkan zip beside the CUDA build (dynamic backend loading), confirm the UHD enumerates, then (a) a few expert layers on Vulkan-UHD via -ot vs --n-cpu-moe, (b) pp on UHD vs CPU. E014-style pre-registered negative.
- **First step:** download the b10064 Vulkan zip, copy ggml-vulkan.dll into the CUDA folder, `llama-server --list-devices`.
- **Success:** the question is closed either way; a surprise win would be a genuine find. Lowest priority of the week; run as filler.

---

## Next (days-class)

### R11. Qwen3.6-35B-A3B as the new flagship (the headline arc, part 1)
**Class:** model upgrade, plus capability (long context via hybrid KV, vision). **Effort:** days. **Verdict source:** GO in five of seven surveys, the consensus #1.
Download unsloth/Qwen3.6-35B-A3B-MTP-GGUF UD-Q4_K_XL (22.9 GB, ~2.2 h; UD-IQ4_XS 18.2 GB fallback). Two generations newer than our flagship at the same A3B active size: GPQA-D 86.0 vs ~70-73, SWE-bench Verified 73.4 vs 22, 262k native context, multimodal, MTP head bundled. Hybrid GDN layout means only 10 of 40 layers carry KV (~20 KiB/token, 4.8x less), so 64k-128k inside 8 GB VRAM becomes plausible. Reproduce the E013 methodology: `-ngl 99`, sweep `--n-cpu-moe`, -t 12, mlock, A/B/A vs the 30B, PPL + quality battery, long-context probe at 32k/64k. Pre-flights: arch string `qwen35moe` confirmed present in b10064 source, but verify the GGUF loads before benchmarking; sanity-check first outputs (CUDA 13.2 gibberish report; we run 13.3); watch for DeltaNet CPU fallback stealing expert threads with the E027 profiler (#19894 class).
- **First step:** start the 22.9 GB download NOW (day 1, before everything else): `huggingface-cli download unsloth/Qwen3.6-35B-A3B-MTP-GGUF --include "*UD-Q4_K_XL*"`.
- **Success:** >= 25 t/s sustained AND clear quality win on our battery = breakthrough (a). **Kill:** < 25 t/s sustained; demote to long-context specialist and keep the 30B as flagship.

### R12. draft-mtp self-speculation on the new flagship (the E014 rematch, part 2)
**Class:** speed upgrade. **Effort:** hours once R11 exists (zero extra download). **Verdict source:** GO in five surveys.
The one spec-decode door E014 left open. MTP inverts both E014 killers: the draft head is bundled (no second model competing for TDP) and n-max 1-3 keeps the verify expert-union small. Public numbers: 1.17x on RTX PRO 6000, ~1.85x on a dense-27B 3090 run, +27.5% vLLM MTP k=1 on the same 3090 where all 19 classic configs lost. Nobody anywhere has published MTP + experts-on-CPU numbers; either sign is a first. A/B/A on three workload classes, acceptance from the new per-position metrics (#24536), 30+ min soak (early-build llama-server MTP crashes reported), expert-union size instrumented via the E027 profiler. Cross-check the result against R3's pre-registered prediction.
- **First step:** on the R11 config: `--spec-type draft-mtp --spec-draft-n-max 2 -fa on -np 1`, sweep n-max 1-3.
- **Success:** >= 20% net sustained decode at identical output quality on at least one workload class = breakthrough (b). **Kill:** net loss at every n-max on all workloads; publish the union-cost explanation and extend the E014 law to native heads.

### R13. GLM-4.7-Flash as coding/agentic flagship
**Class:** model upgrade (domain-specific). **Effort:** days. **Verdict source:** new-models C4 (GO).
17.5 GB download (~1.7 h). 30B-A3.6B reasoning MoE built for local deployment: SWE-bench Verified 59.2 vs our flagship's 22.0, GPQA 75.2. Same --n-cpu-moe pattern; A3.6B may cost a few t/s. Needs `--jinja`, glm47 tool parser, repeat penalty off, and post-Jan-21 quants (sigmoid scoring fix). Measure tokens-to-answer, not just t/s, since it is reasoning-first. Positions as coding specialist next to R11's generalist; run after R11 so the comparison is against the winner.
- **First step:** download unsloth/GLM-4.7-Flash-GGUF UD-Q4_K_XL (17.5 GB).
- **Success:** >= 25 t/s AND clearly better SWE-style task probes than both Qwen flagships. **Kill:** speed below bar or no coding-quality edge over R11.

### R14. Adaptive threshold routing vs static top-6
**Class:** speed-quality frontier (extends E023, our strongest shipped result). **Effort:** days. **Verdict source:** moe C4 (GO, "cheapest real candidate in the sweep" among source changes).
Zero download. The 2025-2026 literature (Ada-K, Alloc-MoE, dynamic top-p, ik's shipped `-ser`) consistently finds adaptive expert count dominates static: easy tokens spend 4 experts, hard tokens keep 8. Patch our instrumented build to skip experts below a fraction of the top-1 router weight (floor 4, cap 8) at the exact site the E023 expert_used_count override touches; sweep the threshold; plot the speed-PPL frontier against static top-6 (+21%/-2.4% PPL) and top-4 (rejected). Re-run the static override on Qwen3.6's 256-expert top-8 once R11 lands.
- **First step:** implement the threshold patch in the instrumented build; PPL harness on datasets/ppl-text.txt.
- **Success:** a threshold point strictly dominates the static frontier (more speed at equal PPL or better PPL at equal speed). **Kill:** no point beats static top-6; close the lane, static is good enough.

### R15. ik_llama.cpp stage 1: CPU-only bake-off
**Class:** speed (fork lever on our proven bottleneck). **Effort:** days. **Verdict source:** moe C3 / quantization C4 (strong MAYBE, kill gate enforced).
No download. ik targets exactly the deficit E025/E026 proved (mul_mat_id kernel-bound, no sgemm path): fused MoE (`-fmoe`), runtime repack (`-rtr`), shipped adaptive expert reduction (`-ser`), iqk kernels with author-claimed 1.06-2.1x TG on AVX2. Honest prior: headline gains concentrate in IQK-format quants and prompt processing, so the Q4_K_M/AVX2 TG gain may sit near the kill line. Build CPU-only with the E026 cmake+ninja environment, prefer clang-cl (our measured 10% MSVC penalty), strip the AVX-512 flags. A/B/A vs mainline pure-CPU 20.14 t/s on the 30B, each flag individually. Stage 2 (CUDA hybrid vs the 31-42 t/s flagship number) only if stage 1 clears AND the CUDA toolkit gets installed.
- **First step:** `git clone https://github.com/ikawrakow/ik_llama.cpp` and configure a clang-cl CPU-only build.
- **Success:** >= +15% TG over mainline pure-CPU = proceed to stage 2. **Kill:** <= +10%; fork overhead not worth carrying, document and close.

### R16. Nemotron 3 Nano Omni: local ears and eyes (gated)
**Class:** capability (audio+vision input, previously impossible here). **Effort:** days, gated on a zero-cost precheck. **Verdict source:** new-models C5 (MAYBE, upgrade to GO if the gate passes).
Omni-modal 30B-A3B (text, image, video, audio in), Mamba2-hybrid so CPU-cheap outside experts. The mmproj files in the GGUF repo are strong evidence for image input; AUDIO via mtmd on b10064 is unverified anywhere and is the whole differentiator vs R2's vision.
- **First step (zero download):** verify b10064 mtmd can load this arch's audio projector (read mtmd docs/source, smoke-test a tiny mtmd model). Only then download Q4_K_M 23.9 GB + mmproj (~2.3 h).
- **Success:** usable image+audio chat at >= 20 t/s text decode = breakthrough (c). **Kill:** audio projector unsupported; skip entirely, R2 covers vision for 6 GB instead of 25.5 GB.

### R17. EAGLE-3 head for Qwen3-30B-A3B (gated on R3)
**Class:** speed (long shot, high falsification value). **Effort:** days. **Verdict source:** speculative C4 (MAYBE, gate resolved in its favor on the converter question).
Run ONLY if R3 measures s <= ~0.55; the PR's own MoE datapoint is 1.06x on homogeneous GPU, and our GPU-draft/CPU-verify split is the one topology where that could move. Converter verified to need only config/tokenizer (no 61 GB target download). ~1-2 GB head + conversion.
- **First step:** open Tengyunw/qwen3_30b_moe_eagle3 config.json and confirm it matches the converter's auto-detection (draft_vocab_size + 1 layer); pre-register the prediction from R3's s.
- **Success:** net decode gain vs baseline where the upstream prior says 1.06x. **Kill:** R3 shows s > 0.55 (never download), or measured net loss confirming the prior.

---

## Ambitious (week+)

### R18. Qwen3.5-122B-A10B UD-IQ2_XXS: 100B-class in 48 GB
**Class:** capability (the 70B+-class quality swing). **Effort:** days to a week. **Verdict source:** quantization C3 (MAYBE, adversarially corrected).
36.6 GB download (~3.5 h). The first 100B-class model with published mainline GGUFs that physically fit our 48+8 GB. Split: routed experts CPU (mlock, -t 12), attention + DeltaNet + shared expert + KV on GPU. Honest framing from the adversarial review: issue #19480 (this architecture family's CPU path reading 3-5x excess bytes) is still OPEN and the 7.74 t/s datapoint is post-fix, so the LIKELY outcome is quantifying and publishing the deficit with our instrumented build, with >= 10 t/s as upside, not baseline. Either result is publishable. GLM-4.5-Air UD-IQ1_M is the fallback control ("is it the arch or the size") since GLM4MOE uses standard attention.
- **First step:** run the RAM budget on paper (36.6 GB weights minus GPU-resident tensors + Windows baseline + KV under ~44 GB), then download unsloth/Qwen3.5-122B-A10B-GGUF UD-IQ2_XXS.
- **Success:** >= 10 t/s sustained with quality clearly above the 35B class = breakthrough (c). **Kill:** deficit confirmed at < 5 t/s; publish the bytes-per-token measurement and close until upstream fixes the arch.

### R19. ik_llama.cpp stage 2 + KT trellis quant bake-off
**Class:** speed + format frontier. **Effort:** week+. **Gated on:** R15 clearing its bar AND the pending CUDA toolkit admin install.
The only route to QTIP-class quality in our RAM-bound regime (mainline closed the door, PR #19726 rejected). Test (a) the 30B flagship requantized to IQ*_K/KT vs mainline Q4_K_M at equal footprint, and (b) if R18 succeeded, ubergarm smol-IQ2_KS (35.3 GiB = 37.9 GB, note the GiB/GB trap; IQ2_KL at 46.5 GB does NOT fit) vs unsloth UD-IQ2_XXS. Mind ik's own caveat: repacked quants lack CUDA kernels in hybrid mode.
- **First step:** confirm R15 result; install CUDA toolkit; rebuild ik with CUDA, AVX-512 flags stripped.
- **Success:** fork quants beat mainline at equal GB on PPL AND t/s by enough to justify carrying a fork as the daily stack. **Kill:** parity or worse at equal footprint.

### R20. Qwen3-Coder-Next 80B-A3B Q3_K_M stretch
**Class:** capability (80B-class agentic coder on a laptop). **Effort:** week+ including stability work. **Verdict source:** new-models stretch (MAYBE). Run only if R11/R13 disappoint on coding.
38.3 GB download (~3.5 h). A3B active at Q3 should decode 20-30 t/s, but 48 GB RAM is genuinely tight: ~35 GB CPU-resident weights + ~5-6 GB Windows baseline + KV lands at ~43-45 GB, mlock must be relaxed, and paging noise collides with our +-10% error bar (R7's tightening helps here). Same #19480 architecture-family risk as R18.
- **First step:** paper RAM math against the R7-tightened error bar; check R18's deficit result first if available (same arch family).
- **Success:** >= 20 t/s sustained, stable over a 30-min soak, coding quality above GLM-4.7-Flash = breakthrough (c). **Kill:** paging instability or the arch deficit; document and wait for upstream.

---

## Amendments (post-sweep, from verified tips)

- **R21 (added 2026-07-19, from Kimi K3 verification): Kimi-Linear-48B-A3B head-to-head.** Moonshot's MIT-licensed preview of K3's exact KDA attention, mainline llama.cpp support since Feb 2026, IQ4_XS 26.5 GB fits our RAM, same A3B active class as our winners. The experiment: KDA vs Qwen3.6's GDN in the same class on the same rig, including long-context KV economics. Source: research-sweeps/2026-07-tip-kimi-k3/.
- **Dated task 2026-07-27: K3 weights land.** Verify actual license text, read the technical report (KDA + Attention Residuals), scan for smaller family members. K3 itself (2.8T) can never run here; its architecture lineage is the prize.

## Skip list: killed ideas we never re-litigate

1. **Classic separate-draft and short-ngram speculative decoding on A3B MoE** (draft-simple, ngram-simple, ngram-cache, short-draft ngram-mod, any vocab-matched small draft model). Killed twice: our E014, and thc1006's independent 19-config RTX 3090 study (all net-negative even at 100% acceptance, same expert-union mechanism). The only doors still open are native heads (R12, R17) and the long-draft saturation regime (R5).
2. **Expert offload/prefetch/caching systems** (DALI, KTransformers, HOBBIT, ProMoE, ExpertFlow, MoE-Infinity, Fiddler). They attack PCIe expert-fetch on 24 GB rigs. Our experts already live permanently in RAM at RAM bandwidth, and E021 measured routing locality worth ~5% here. There is no headroom for prediction or caching on this topology.
3. **CoX-MoE**: requires Intel AMX. The 14650HX has none. Dead on arrival.
4. **CPU kernel micro-optimization** (repack variants, two-row kernels, sgemm for mul_mat_id): our own E021-E028 closed it (uniform 37-42 GB/s extraction, repack tie, sgemm inapplicable), and upstream git log confirms zero x86 MoE kernel movement since. Do not reopen.
5. **Qwen3-Next-80B-A3B (base)**: Q4 does not fit 48 GB; Q3 fits but the measured CPU-path deficit (7.7 t/s on comparable bandwidth, issue #19480) kills the bar. Superseded by Qwen3.6-35B and Coder-Next anyway.
6. **Dense >= 14B as flagship** (Qwen3.6-27B etc.): 16-17 GB of dense weights through a ~40 GB/s pipe is ~3 t/s; no speculative multiplier reaches 25 t/s. The roundups recommending these assume 24 GB cards.
7. **gpt-oss-120B**: 61-63 GB against our 56 GB combined ceiling; NVMe paging with random expert access collapses decode.
8. **NPU**: Raptor Lake HX has no NPU, verified against Intel ARK and launch coverage. Closed forever.
9. **SnapKV / H2O / KVzip / Ada-KV eviction**: papers and PyTorch serving code only; zero runnable Windows llama.cpp implementations. Watch list (KVzip port would jump straight to candidate), not work.
10. **Layer-skip self-speculation** (SWIFT, LayerSkip, Kangaroo): no llama.cpp implementation exists, and MTP is the shipped successor with better acceptance.
11. **Trellis/QTIP quants in mainline**: the door was explicitly closed upstream (PR #19726 rejected 2026-02-23). Fork-only, covered by R19; stop waiting for mainline.
12. **VPTQ / AQLM / QuIP# / HIGGS / SINQ**: no GGUF path, research CUDA kernels only, or checkpoints exceed 8 GB VRAM. Paper tier for this rig.
13. **Matched RAM kit / XMP sticks**: ~$800+ for 2x32 at 2026 crisis prices, and the BIOS is almost certainly frequency-locked. The purchase question opens only if R9 measures a > 15% flex-mode deficit AND prices normalize.
14. **Mistral Small 4 (119B), Llama 4 Scout (109B), GLM-4.6/4.7 full (355B), GLM-4.5-Air as primary**: do not fit, or strictly superseded. Air survives only as R18's architecture control.

### Deprioritized, not killed (each has a re-open trigger)

- **gpt-oss-20B**: clean feasibility (kv-cache survey GO) but two surveys agree it is our current flagship's speed class without being clearly smarter. Trigger: a long-reasoning + 131k-context niche that Qwen3.6 cannot cover.
- **DFlash on dense Qwen3-8B**: 8.08x code-gen headline is real, but incremental class, no prebuilt 8B head GGUF exists, and the conversion path is unproven. Trigger: conversion proven cheap, or R12 lands and the z-lab Qwen3.6-35B DFlash head (292 MB GGUF exists) makes a DFlash-vs-MTP head-to-head a one-afternoon extension.
- **Windows large pages patch**: honest expectation 0-7% decode, below the current +-10% error bar. Trigger: R7 tightens the bar below +-5%; then the publishable-null and upstream-PR value returns.
- **REAP-pruned checkpoints**: RAM headroom enabler, not decode speed (active set unchanged); the available GGUF is a coder-family cross-comparison confound. Trigger: R11 or R18 hit real RAM-pressure incidents (the 12 t/s page-eviction class).
- **EXL3 on the 8 GB card**: v1.0.0 Windows wheels exist but sm_120 support is unconfirmed and the fit math caps us at a 14B at ~3.5 bpw with no verified pre-made quants. Idle-time probe only.
