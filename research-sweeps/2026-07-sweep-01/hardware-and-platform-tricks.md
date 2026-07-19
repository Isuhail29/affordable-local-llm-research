# Hardware and Platform Tricks: Survey (2026-07 sweep-01)

Domain: platform-level levers we have never exercised on the rig (i7-14650HX, 48 GB DDR5-5600 mismatched 16+32, RTX 5060 Laptop 8 GB, Windows 11 Home, llama.cpp b10064). Scope: RAM topology and matched kits, XMP headroom, Windows large pages, iGPU as a second compute device, NPU status, process priority and core parking.

Bottom line up front: this domain contains no likely breakthrough, but it contains one free experiment that bounds the only real bandwidth lever we have left (flex-mode RAM topology), one cheap kill-test that closes the iGPU question permanently, and two low-cost software levers (large pages, scheduler control) worth a combined 0-10% plus variance reduction. The NPU thread is closed: Raptor Lake HX has no NPU, verified.

---

## 1. Our mismatched 16+32 GB sticks run in Flex Mode, and nobody has published inference numbers for it

What Flex Mode is: with a 16 GB and a 32 GB SODIMM, Intel's memory controller interleaves the first 16 GB of each stick as true dual-channel (32 GB interleaved region) and leaves the remaining 16 GB of the 32 GB stick as single-channel. Dual-channel region first, single-channel region only after; the single-channel tail runs at roughly half bandwidth ([Tom's Hardware forum explainer](https://forums.tomshardware.com/threads/flex-mode-vs-dual-channel-gaming.3629690/), [Crucial DDR5 overview](https://www.crucial.com/articles/about-memory/everything-about-ddr5-ram)).

Why this matters for us specifically: our measured ~60 GB/s read ceiling and the uniform 37-42 GB/s the CPU matmuls extract (E021-E028) were measured with the model comfortably inside total RAM, but we have never controlled or measured WHERE physically the ~18.6 GB of Qwen3-30B expert weights land. If Windows places part of the working set in the single-channel tail, decode is silently throttled and the throttling would look exactly like our "uniform 37-42 GB/s" plateau. If the working set sits fully in the interleaved region, a matched kit buys little.

What published data exists: essentially nothing for LLM inference on flex-mode laptops. The general CPU-inference literature confirms decode t/s scales near-linearly with bandwidth: DDR5-4800 to DDR5-6000 gave +20-23% generation speed in maximsaplin's testing ([DDR5 Speed, CPU and LLM Inference](https://dev.to/maximsaplin/ddr5-speed-and-llm-inference-3cdn)); Hardware Corner's bandwidth-to-t/s model says the same ([Memory Bandwidth and Tokens per Second](https://www.hardware-corner.net/memory-bandwidth-llm-speed/)); Johannes Gaessler's llama.cpp performance notes rank bandwidth as the dominant CPU-side variable ([llamacpp_performance](https://johannesgaessler.github.io/llamacpp_performance)). The flex-mode-specific measurement is a genuine gap we can fill and publish.

Dual-rank sidebar: DDR5 benefits only marginally from a second rank per channel because 32 bank groups already extract most parallelism; igor'sLAB measured dual-rank DDR5 slightly LOSING read bandwidth vs single-rank on Alder Lake while gaining a little on writes ([Single vs dual-rank DDR4 vs DDR5 on Alder Lake](https://www.igorslab.de/en/single-vs-dual-rank-ddr4-vs-ddr5-on-alder-lake/2/), [Overclock.net dual-rank interleaving thread](https://www.overclock.net/threads/dual-rank-interleaving.1814226/)). So the case for a matched kit rests on eliminating the flex-mode single-channel tail, not on rank interleaving. Do not buy a kit for rank interleaving.

The 2026 price problem: DDR5 prices are up 400%+ since late 2025. A single Crucial 32 GB DDR5-5600 SODIMM tracked at ~$418 in July 2026 (peak $632 in May, low $282 in Dec 2025) ([Pangoly price history](https://pangoly.com/en/price-history/crucial-32gb-ddr5-5600mhz-sodimm), [Tom's Hardware RAM price index](https://www.tomshardware.com/pc-components/ram/ram-price-index-2026-lowest-price-on-ddr5-and-ddr4-memory-of-all-capacities)). A matched 2x32 kit is ~$800+ right now. Conclusion: measure first, buy never unless the free experiment proves a big deficit AND prices normalize.

Free measurement path (no purchase):
- **Intel Memory Latency Checker (MLC)**: free, ~2-3 MB download from Intel, measures bandwidth as a function of buffer size and access pattern. Sweep buffer footprints from 4 GB to 44 GB; a flex-mode topology shows a bandwidth cliff once the footprint spills past the interleaved region.
- **Stick-pull A/B**: run the flagship benchmark with the 32 GB stick alone (pure single-channel, model still fits) vs both sticks (flex). The delta calibrates how much of current decode is bandwidth-limited by topology vs by the CPU's ~65-70% extraction efficiency. Zero dollars, ~2 screws.
- While inside BIOS for the stick-pull: check whether the laptop exposes XMP/memory frequency controls at all (see section 2).

## 2. XMP / RAM overclock headroom: almost certainly locked, verify in 10 minutes

The i7-14650HX officially supports DDR5-5600 max ([Intel ARK spec page](https://www.intel.com/content/www/us/en/products/sku/235996/intel-core-i7-processor-14650hx-30m-cache-up-to-5-20-ghz/specifications.html)). XMP 3.0 SODIMMs up to 6400 MT/s exist (Kingston Fury Impact, with a "Plug N Play" auto-overclock mode for locked BIOSes) ([Kingston Fury Impact DDR5](https://www.kingston.com/en/memory/gaming/kingston-fury-impact-ddr5-memory)), but consumer gaming laptop BIOSes overwhelmingly lock memory frequency to the JEDEC profile of the CPU spec; only a few enthusiast models (MSI Titan class) expose XMP. 5600 to 6400 would be +14% theoretical bandwidth, which at our bandwidth-bound decode margin would be a real gain, but it requires (a) an unlocked BIOS and (b) buying new sticks at 2026 crisis prices. Verdict: check the BIOS during the stick-pull experiment; expect to close this thread as "locked, not available". No standalone experiment.

## 3. iGPU as a second device: evidence says it hurts, and our iGPU is half the size of the one that already loses

Our iGPU: Intel UHD Graphics for 14th Gen, **16 EUs**, 1.6 GHz max ([Intel ARK](https://www.intel.com/content/www/us/en/products/sku/235996/intel-core-i7-processor-14650hx-30m-cache-up-to-5-20-ghz/specifications.html)). That is roughly 0.4 TFLOPS FP32 with no XMX/matrix units, and it shares the same DDR5 bus and the same package power budget as the CPU cores (the exact TDP-sharing mechanism that killed speculative decoding in E014).

Directly relevant evidence:
- A dedicated build-matrix benchmark of llama.cpp on **UHD 770 (32 EU desktop, twice our EU count)** concluded: CPU beats UHD 770 Vulkan on everything; 32 EU, no matrix cores, not competitive with a 12900K on AVX2 ([codeandcodes/intel-llamacpp-bench](https://github.com/codeandcodes/intel-llamacpp-bench)). It also flagged IQ2-family quants regressing catastrophically on this iGPU class (0.31 t/s decode).
- In the llama.cpp Vulkan performance megathread, even a 96 EU Iris Xe (i7-1185G7) managed only 5.9-8.3 t/s tg128 on 7B Q4_0, and the CPU-only path beat the iGPU on decode in the same user's runs ([llama.cpp discussion #10879](https://github.com/ggml-org/llama.cpp/discussions/10879)).
- TechHara's CPU vs iGPU comparison shows iGPU wins on prompt processing only when the iGPU is large (AMD APU class) and still loses decode ([Llama.cpp Benchmark: CPU vs iGPU](https://medium.com/@techhara/llama-cpp-benchmark-cpu-vs-igpu-93b3cc40ece5)).

Mechanically, running experts on the iGPU cannot add bandwidth: Vulkan on an iGPU reads the same DDR5 the CPU reads, minus driver overhead, minus package power stolen from the P-cores. The only theoretical win would be if the iGPU extracted bandwidth MORE efficiently than the CPU's 37-42 GB/s, which the UHD 770 data contradicts.

Plumbing (if we run the kill-test): official Windows releases ship with dynamic backend loading since [PR #13220](https://app.semanticdiff.com/gh/ggerganov/llama.cpp/pull/13220/overview) ([build docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)); dropping ggml-vulkan.dll from the b10064 Vulkan zip (~50-100 MB download, ~1 minute at our 3 MB/s) into the CUDA build folder makes both backends enumerate, then `--list-devices` and `--device`/`-ot` place tensors explicitly. Verify the UHD device appears (Vulkan will also enumerate the RTX 5060; select the UHD by index). SYCL is the alternative backend for Intel iGPUs but requires the oneAPI runtime and shows no evidence of beating Vulkan on a 16 EU part.

Verdict: predicted net loss on decode and on pp (RTX 5060 already does pp). Worth a 2-3 hour falsification run only because it closes the "why don't you use your iGPU?" question permanently with our own numbers, in E014 style.

## 4. NPU on Raptor Lake HX: does not exist. Thread closed.

Raptor Lake Refresh (all 14th gen HX including the 14650HX) has **no NPU**; PCWorld states the 14th-gen Raptor Lake Refresh line "does away with the AI NPU" ([PCWorld](https://www.pcworld.com/article/2103293/intels-14th-gen-core-chips-hit-6ghz-but-performance-stalls.html)); the HX release coverage confirms no NPU in the HX series ([TechPowerUp](https://www.techpowerup.com/317474/intel-releases-14th-gen-core-hx-raptor-lake-refresh-mobile-processors), [Tom's Hardware](https://www.tomshardware.com/pc-components/cpus/intel-unleashes-14th-gen-raptor-lake-refresh-hx-series-laptops-cpus-refreshed-chips-with-up-to-24-cores-58-ghz-boost-clock-and-192gb-ddr5-support)). Intel's first integrated NPU shipped with Meteor Lake (Core Ultra, Dec 2023) ([Geeks3D launch coverage](https://www.geeks3d.com/20231215/intel-core-ultra-meteor-lake-processors-launched-with-ai-boost-npu-and-arc-gpu-for-ultra-thin-notebooks/)). Some 13th-gen mobile parts carried a discrete-ish "VPU" precursor, not the HX line ([wccftech](https://wccftech.com/intel-ai-boosting-vpu-debuts-in-13th-gen-raptor-lake-mobile-full-integration-in-14th-gen-meteor-lake-cpus/)). The legacy GNA block, where present on Raptor Lake SKUs, is a low-power audio-DSP-class accelerator, deprecated in OpenVINO, and cannot run transformer workloads. No experiment possible. Closed.

## 5. Windows large pages: not supported by llama.cpp at all, and the mmap path CANNOT support them, but our --no-mmap path can be patched

Verified in our local b10064 source: zero hits for MEM_LARGE_PAGES / SeLockMemoryPrivilege anywhere in the tree. The Windows mmap path (src/llama-mmap.cpp:550-575) uses CreateFileMapping + MapViewOfFile + PrefetchVirtualMemory, and Windows only allows large pages for pagefile-backed sections (SEC_LARGE_PAGES), never for file-backed mappings ([Microsoft Large-Page Support docs](https://learn.microsoft.com/en-us/windows/win32/memory/large-page-support)). So large pages on Windows require the --no-mmap allocation path: patch the CPU backend buffer allocation to try VirtualAlloc(MEM_RESERVE|MEM_COMMIT|MEM_LARGE_PAGES) with graceful fallback.

Evidence for gains:
- The upstream hugepage request ([llama.cpp issue #12444](https://github.com/ggml-org/llama.cpp/issues/12444)) claims 10x but for LOAD TIME of a 377 GB model on Linux hugetlbfs; it is stale, Linux-only, and says nothing about decode throughput.
- The real decode-side mechanism is TLB pressure: 18.6 GB of weights at 4 KB pages is ~4.9M pages streamed every token, far beyond the ~2K dTLB entries; 2 MB pages cut page-walk traffic by 512x. Microsoft documents the translation-buffer efficiency win ([large-page docs](https://learn.microsoft.com/en-us/windows/win32/memory/large-page-support)); XMRig (memory-hard, random access) treats huge pages as mandatory for full performance ([XMRig hugepages docs](https://xmrig.com/docs/miner/hugepages)). But LLM decode streams sequentially, and hardware prefetchers plus page-walk caches hide most sequential TLB miss cost. Honest expectation: 0-7% decode, possibly ~0.
- Windows 11 Home wrinkle: SeLockMemoryPrivilege ("Lock pages in memory") has no gpedit UI on Home; grant it via ntrights.exe or a small LsaAddAccountRights PowerShell script, then sign out/in ([MahdyTech large pages guide](https://mahdytech.com/large-pages-how-when/)). Large-page allocations also demand contiguous physical memory: allocate right after reboot or the 18.6 GB alloc will fail with fragmentation.
- Side effects to log: large pages are non-pageable (replaces our mlock), and --no-mmap cold load rereads 18.6 GB from NVMe (~4 s at 5 GB/s) instead of hitting the warm page cache; irrelevant for a persistent server, must be excluded from timing windows.

## 6. Process priority, thread affinity, core parking: the flags already exist in b10064, the published evidence is old and pre-dates the threadpool

What b10064 already ships (verified in our source): `--prio` (-1 low to 3 realtime, maps to SetPriorityClass + SetThreadPriority on Windows; common/common.cpp:235-271), `--prio-batch`, `--cpu-mask`, `--cpu-strict`, `--poll`, and llama-bench takes `--prio` directly (tools/llama-bench). We have never set any of them; everything to date ran at NORMAL priority with Windows free to schedule.

Evidence:
- The famous "3x from P-cores only" result is from 12th gen in March 2023, before the ggml threadpool and hybrid-core detection landed ([llama.cpp discussion #572](https://github.com/ggml-org/llama.cpp/discussions/572)); a companion Windows issue showed E-cores staying 60-70% loaded even with P-core affinity set ([issue #842](https://github.com/ggml-org/llama.cpp/issues/842)). Modern llama.cpp auto-detects hybrid topology, so most of that 3x is already banked; our tuned -t 12 MoE / -t 8 dense settings (protocol law) are the survivors of that era.
- Current optimization guides still recommend P-core pinning plus high priority on hybrid Windows machines but report the residual gain as single-digit ([carteakey.dev local LLM optimization guide](https://carteakey.dev/blog/local-inference/local-llm-optimization/)).
- The variance angle matters more for us: E032 established a +-10% cross-session error bar with thermal drift as a known confound. Scheduler wander (threads bouncing between P and E cores, background tasks preempting at NORMAL priority, core parking waking/sleeping E-cores) is a plausible second contributor that `--prio 2 --cpu-strict 1 --cpu-mask` plus `powercfg` core-parking pinning (CPMINCORES 100) would suppress. Tightening the error bar is itself protocol value: it lowers the detection threshold for every future experiment.

Core parking specifics: Windows 11 parks E-cores aggressively on Balanced power plans; `powercfg -attributes SUB_PROCESSOR CPMINCORES -ATTRIB_HIDE` exposes the minimum-unparked-cores knob; High Performance / Ultimate plans disable parking. All free, all reversible, all measurable with our A/B/A flanking protocol.

---

## Candidate experiments for our rig

### C1. Flex-mode bandwidth topography and stick-pull falsification (FREE, the priority)
Map bandwidth vs footprint with Intel MLC (~3 MB download), then A/B/A the flagship Qwen3-30B config in three RAM topologies: 16+32 flex (current), 32 GB stick alone (pure single-channel), and if the BIOS allows, swapped slot order. Deliverable: a measured answer to "is flex mode throttling our decode, and what would a matched kit actually buy?" Also inspect BIOS for XMP controls while the case is open (closes section 2). Decision gate: only if the data shows the working set spilling into the single-channel tail at a >15% decode cost does the matched-kit purchase question even open (and at ~$800 for 2x32 in the 2026 price surge, probably stays closed until prices normalize). Effort: hours. Class: incremental (diagnostic that could unlock a future speed upgrade).

### C2. Scheduler control sweep: --prio, --cpu-mask, --cpu-strict, core parking (FREE)
Grid: {prio 0 vs 2} x {default threads vs P-core-masked --cpu-strict 1} x {Balanced vs High Performance + CPMINCORES=100}, on the flagship MoE config, -t 12, A/B/A flanked, plus a 10-run variance measurement on the best cell vs baseline. Two prizes: any mean gain (expect 0-8%), and a tighter cross-session error bar than E032's +-10%, which sharpens every future experiment. Effort: hours. Class: incremental.

### C3. Windows large pages patch for the --no-mmap weights path (uses our instrumented build)
Grant SeLockMemoryPrivilege on Windows Home via LsaAddAccountRights script, patch CPU-backend buffer allocation to VirtualAlloc MEM_LARGE_PAGES with fallback, verify engagement by logging actual page size, run flanked decode benchmarks fresh after reboot. First published Windows large-page decode numbers for llama.cpp (upstream has nothing; the Linux issue is stale and load-time-only). Expect 0-7%; even a clean null is publishable and PR-able upstream. Effort: days (patch + privilege plumbing + fragmentation-controlled runs). Class: incremental.

### C4. iGPU kill-test: UHD (16 EU) as a Vulkan device beside CUDA (predicted negative, closes the question)
Drop ggml-vulkan.dll from the b10064 Vulkan zip (~50-100 MB, ~1 min download) into the official CUDA build, confirm the UHD enumerates via --list-devices, then measure: (a) a few MoE expert layers on Vulkan-UHD via -ot vs --n-cpu-moe baseline, (b) pp on UHD vs CPU for a CPU-only config. Prediction from UHD 770 evidence (32 EU, loses to a 12900K everywhere) and TDP-sharing (E014): net loss on both. Publishing the falsification with bandwidth accounting permanently answers "why not use the iGPU?". Effort: hours. Class: incremental.

---

## Feasibility verdicts (adversarial review, 2026-07-19)

Verified against the local b10064 source tree and official CUDA bin folder, not just the cited links.

- **C1 Flex-mode topography + stick-pull: GO.** Free, fits trivially (Qwen3-30B Q4_K_M at 18.6 GB still fits in the 32 GB stick alone), Intel MLC is a real free Windows tool (needs admin, loads a driver), no overlap with E021-E028 which never controlled physical page placement. Caveats: Windows gives no user control over physical placement so the MLC cliff is statistical not deterministic, and the stick-pull means opening the laptop; effort "hours" is honest.
- **C2 Scheduler control sweep: GO.** All flags confirmed present in b10064 source (arg.cpp:1358-1429, SetPriorityClass mapping in common.cpp:244-248, llama-bench takes --prio); powercfg CPMINCORES works on Windows 11 Home; zero download; not a duplicate (protocol laws fixed thread count only, never priority/affinity/parking). Use --prio 2 not 3 (realtime needs admin and can starve the system at -t 12). The variance-tightening prize alone justifies it; run this first since it sharpens every later experiment.
- **C3 Windows large pages patch: MAYBE.** Premise verified: zero large-page code in the tree, mmap path (llama-mmap.cpp:543-575) is file-backed so cannot take large pages, --no-mmap path is patchable in our instrumented build, and the Home-edition privilege grant via LsaAddAccountRights is real. But the honest expected gain (0-7%, likely near 0 for sequential streaming with page-walk caches) sits below the E032 +-10% cross-session error bar, allocation of 18.6 GB in 2 MB pages will fail on fragmented memory (reboot-fresh runs only), and effort is days. Do it only after C2 tightens the error bar; the publishable-null and upstream PR angle is the main value.
- **C4 iGPU kill-test: GO.** Plumbing verified locally: official b10064 build ships split backend DLLs (GGML_BACKEND_DL on) and ggml-backend-reg.cpp scans for ggml-*.dll at runtime, so dropping ggml-vulkan.dll from the same-tag Vulkan zip (~50-100 MB, under a minute at 3 MB/s) will enumerate the UHD; --list-devices/--device/-ot all present in source. Not a duplicate of E014 (spec decode). Predicted negative, but it is cheap (hours), E014-style falsification is on-brand, and it closes the "why not the iGPU?" question with our own numbers. Low priority behind C1/C2.
- **Sections 2 (XMP) and 4 (NPU): correctly not candidates.** XMP check is folded into C1's BIOS visit at zero marginal cost; the NPU thread is genuinely closed (Raptor Lake HX has none).

## Sources
- https://forums.tomshardware.com/threads/flex-mode-vs-dual-channel-gaming.3629690/
- https://www.crucial.com/articles/about-memory/everything-about-ddr5-ram
- https://dev.to/maximsaplin/ddr5-speed-and-llm-inference-3cdn
- https://www.hardware-corner.net/memory-bandwidth-llm-speed/
- https://johannesgaessler.github.io/llamacpp_performance
- https://www.igorslab.de/en/single-vs-dual-rank-ddr4-vs-ddr5-on-alder-lake/2/
- https://www.overclock.net/threads/dual-rank-interleaving.1814226/
- https://pangoly.com/en/price-history/crucial-32gb-ddr5-5600mhz-sodimm
- https://www.tomshardware.com/pc-components/ram/ram-price-index-2026-lowest-price-on-ddr5-and-ddr4-memory-of-all-capacities
- https://www.intel.com/content/www/us/en/products/sku/235996/intel-core-i7-processor-14650hx-30m-cache-up-to-5-20-ghz/specifications.html
- https://www.kingston.com/en/memory/gaming/kingston-fury-impact-ddr5-memory
- https://github.com/codeandcodes/intel-llamacpp-bench
- https://github.com/ggml-org/llama.cpp/discussions/10879
- https://medium.com/@techhara/llama-cpp-benchmark-cpu-vs-igpu-93b3cc40ece5
- https://app.semanticdiff.com/gh/ggerganov/llama.cpp/pull/13220/overview
- https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md
- https://www.pcworld.com/article/2103293/intels-14th-gen-core-chips-hit-6ghz-but-performance-stalls.html
- https://www.techpowerup.com/317474/intel-releases-14th-gen-core-hx-raptor-lake-refresh-mobile-processors
- https://www.tomshardware.com/pc-components/cpus/intel-unleashes-14th-gen-raptor-lake-refresh-hx-series-laptops-cpus-refreshed-chips-with-up-to-24-cores-58-ghz-boost-clock-and-192gb-ddr5-support
- https://www.geeks3d.com/20231215/intel-core-ultra-meteor-lake-processors-launched-with-ai-boost-npu-and-arc-gpu-for-ultra-thin-notebooks/
- https://wccftech.com/intel-ai-boosting-vpu-debuts-in-13th-gen-raptor-lake-mobile-full-integration-in-14th-gen-meteor-lake-cpus/
- https://learn.microsoft.com/en-us/windows/win32/memory/large-page-support
- https://github.com/ggml-org/llama.cpp/issues/12444
- https://xmrig.com/docs/miner/hugepages
- https://mahdytech.com/large-pages-how-when/
- https://github.com/ggml-org/llama.cpp/discussions/572
- https://github.com/ggml-org/llama.cpp/issues/842
- https://carteakey.dev/blog/local-inference/local-llm-optimization/
