# The Story So Far: Chasing 2x That Mostly Wasn't There

*A plain-language account of this project's first research arc: ten experiments, five falsified theories, three instruments built, one upstream bug found, one genuine breakthrough shipped, and a mystery that dissolved under honest measurement.*

## The question

Can an ordinary laptop (8 GB of GPU memory, 48 GB of RAM) run AI models far larger than conventional wisdom allows, at speeds people would actually use? And can we understand PRECISELY why it runs at the speed it does, rather than settling for folklore?

## Act I: Learn the physics first

Before optimizing anything we measured the machine (E001, E002). The single governing fact of local AI: generating each word requires reading essentially the entire model from memory once. Speed is memory bandwidth divided by model size, full stop. Our napkin predictions landed within measurement error of reality: 62 tokens/sec for an 8B model in GPU memory, 10 on CPU.

Two early surprises set the tone. First, "CPU-only" prefill secretly used the GPU (llama.cpp streams weights across the bus for batch work): our first lesson in verifying what code actually does rather than what flags claim. Second, Windows physically places identical programs in faster or slower RAM regions run to run, a silent ±21% lottery caused by our mismatched RAM sticks.

## Act II: The breakthrough that actually shipped

Mixture-of-Experts models changed everything (E013). A 30-billion-parameter model that only consults ~3B parameters per word can keep its bulky "experts" in RAM while the always-active parts live in GPU memory. Result: an 18.6 GB model running at 30-38 tokens/sec on hardware that "cannot" run it, three times faster than an 8B dense model on the same CPU.

We then tried to make it faster. Speculative decoding, the community's favorite trick, LOST on this machine in both tested forms (E014): we predicted the MoE failure mechanism in writing beforehand (verifying 8 guessed words touches the union of their experts, destroying the trick's economics). Reducing active experts from 8 to 6 (E023) delivered +21% speed for a measured 2.4% quality cost: shipped as Turbo mode, ~42 tokens/sec. Field debugging (a 12 t/s complaint) traced to RAM pressure evicting model pages; the fix (mlock plus cache warming) went into the launchers.

## Act III: The hunt for the missing bandwidth

Our measurements suggested expert reads extracted only ~30 GB/s from RAM that synthetic tests showed could deliver 60. Half the bandwidth, missing. Worth up to +50% speed if found. We hunted it through five experiments:

- **E021**: Scattered reads? Innocent: random 0.25 MB reads run at full sequential speed. (Bonus: first-touch page mapping runs at 15-17 GB/s, exactly explaining the field slowdown.)
- **E025**: Thread stragglers stalling 144 sync points per token? We patched work-stealing back in, built llama.cpp from source to test it: worth at most 5%. Falsified by its own pre-registered threshold. The control run exposed a 10% compiler effect that redirected the whole investigation toward kernels.
- **E026**: The optimized math engine the dense path enjoys? It turned out to support neither our number format nor single-word generation, anywhere. The celebrated "repack" fast kernel? Already silently active in half our runs, and switching it off changed nothing. We also found and documented a genuine llama.cpp crash bug in this version's expert repacking.
- **E027**: We stopped guessing and built a profiler into the engine. The attribution shocked us: the deficit was NOT expert-specific. The model's ordinary attention math ran at the same depressed rate. And comparing against older sessions exposed our own error: the laptop's whole performance level shifts with heat, contaminating every cross-session comparison we had made.
- **E032** quantified it: 10 minutes of sustained load costs 8.9%, gracefully. Cross-session comparisons carry a ±10% error bar. Sustained real-world speed: ~31 t/s.
- **E028**: We hand-wrote an AVX2 kernel processing two weight rows per call, mathematically verified identical output, and measured: nothing (the flanking control caught a 13% thermal sag that a naive A/B would have blamed on the kernel). The autopsy then found the final piece: the "slow" down-expert tensors were stored in a fatter number format (Q6_K) than our arithmetic assumed. Corrected, they extract BETTER per byte than their siblings. The famous penalty was a units error.

**The verdict**: under matched conditions with correct accounting, every matmul in the MoE extracts a uniform 37-42 GB/s. The dramatic 2x mystery was manufactured from thermal drift, a placement lottery, and one wrong byte-size assumption, leaving a real residue of maybe 5-15% in glue and burst structure. llama.cpp's CPU path is close to optimal on this hardware. The honest ceiling is roughly where we already run.

## What we actually shipped

- **Start-30B-AI.bat**: 30B-class model, ~31-34 t/s, one double-click
- **Start-8B-AI.bat**: small fast model, ~50+ t/s
- **Start-30B-AI-Turbo.bat**: +21% for -2.4% quality
- A tuned recipe transferable to any similar machine: expert-aware placement, 12 threads, mlock, cache pre-warming, matched-conditions benchmarking

## What we learned about doing science

1. Pre-registered predictions with falsification thresholds turned five wrong theories into cheap, fast deaths instead of weeks of misdirected engineering.
2. Instruments beat inference chains: the in-engine profiler settled in one hour what three experiments of theorizing could not.
3. Controls are not bureaucracy: the A/B/A flanking runs and placebo arms rescued correct conclusions from thermal drift three separate times.
4. Verify engagement, verify units, verify thermal state. Our three biggest wrong turns were a feature silently on, a feature silently off, and a byte-size assumption.
5. Negative results compound: the complete map of what does NOT limit this machine is precisely what makes the remaining numbers trustworthy.

## What remains open

The software frontier on this machine is thin; the real levers are now architectural: matched RAM sticks (removes the placement lottery), more RAM bandwidth or VRAM (changes the physics), next-generation sparse models (change the bytes). The methods in this repository transfer to all of them.

Everything here (raw data, patches, instruments, failed theories included) is reproducible from this repository. The failures are documented as carefully as the successes, because they are most of what we know.
