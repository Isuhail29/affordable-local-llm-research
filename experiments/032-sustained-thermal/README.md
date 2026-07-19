# Experiment 032: Sustained Thermal Characterization

Date: 2026-07-19
Status: In progress

## Goal

Quantify how sustained load shifts this laptop's inference performance level, putting an error bar on every cross-session comparison in this project (E027 exposed the confound: the dense control extracted 47.5 GB/s in a cool session, 39.9 in a hot one).

## Method

Flagship chat config (official b10064 binary, 30B, -ngl 99 --n-cpu-moe 40 -c 8192 -np 1 -t 12 --mlock), scripts/thermal-test.ps1: identical 600-token generations (temp 0, same prompt) back-to-back for 10+ minutes; per request log tokens/sec; between requests sample GPU temp, GPU SM/mem clocks (nvidia-smi) and CPU effective frequency (typeperf % Processor Performance). Machine state at start: already warm from a long research session (worst-case realistic baseline; a cold-start run is a follow-up).

## Hypothesis (pre-registered)

1. Decode declines 5-15% from the first generation to minute 10, then plateaus (thermal soak, not runaway).
2. GPU stays under 90 C; no hard throttle events (SM clock cliffs).
3. The measured plateau-vs-start delta approximates the cross-session error bar and is at least half of the historical 47.5-vs-39.9 dense discrepancy (~19%); if the delta is under 5%, thermal drift cannot explain E027's cross-session gap and something else (placement, background load) contaminated the old comparisons.

## Actual Result

31 identical 600-token generations over 10 minutes (benchmarks/e032-thermal.jsonl):

- Decode: first-3 average **33.76 t/s**, last-3 average **30.76 t/s**: **-8.9% soak**, gradual and plateauing, no cliffs
- GPU: 77 C at start, 85-87 C plateau (max 87); SM clocks sag from ~2800 to ~2200-2570 MHz
- CPU: effective frequency stable ~140-156% of base throughout (no CPU throttle)

## Benchmark analysis

**H1 confirmed** (-8.9%, inside the predicted 5-15%, plateau behavior). **H2 confirmed** (87 C max, graceful clock sag, no hard throttle). **H3 resolved**: thermal soak explains roughly half of the historical 19% cross-session dense-extraction discrepancy; the remainder matches E002's physical-placement lottery. The project's cross-session error bar is now quantified: **treat any cross-session difference under ~10% as noise** unless intra-run controls say otherwise.

Method note: the machine started warm from a long research session; a cold-start version of this run would show a larger first-minutes advantage. For the user-facing claim: sustained real-world chat speed on the flagship config is **~31 t/s**, and the first minutes after startup run ~34.

## Lessons Learned

1. This laptop loses ~9% under sustained MoE load and stabilizes; it never falls off a cliff. The 33-42 t/s headline numbers are honest for interactive use; ~31 is the marathon number.
2. GPU temp is the visible symptom but CPU frequency held steady: the sag is GPU-side clock management, relevant because attention lives there in the flagship config.
3. Every earlier cross-session comparison now carries a quantified error bar instead of an unknown one.

## Next Steps

- Optional cold-start variant for the full curve.
- E028 benchmarking uses A/B/A flanking specifically because of this result.
