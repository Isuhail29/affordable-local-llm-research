# Plain-Language Glossary

Every term used in this project, explained simply. Read top to bottom once; after that, use it as a lookup.

## Part 1: The machine (hardware)

**CPU (processor).** The general-purpose brain of your laptop. Good at everything, best at nothing. Yours is an Intel i7-14650HX.

**Core / Thread.** A core is one worker inside the CPU; yours has 16. A thread is one task a worker handles; yours can juggle 24. More workers only help until they crowd the same doorway (see bandwidth).

**P-cores and E-cores.** Your 16 cores are two types: 8 Performance cores (strong, fast) and 8 Efficiency cores (weaker, power-saving). We proved the 8 P-cores alone give the best AI speed; adding E-cores actually slows things down because they fight over memory.

**GPU (graphics card).** A chip with thousands of tiny workers, built for doing the same simple math on huge piles of numbers at once. Perfect for AI. Yours is an NVIDIA RTX 5060 Laptop.

**VRAM.** The GPU's private, super-fast memory. Yours is 8 GB. This is the "small desk": whatever fits on it runs fast.

**RAM (system memory).** The computer's main memory, 48 GB in your laptop. Bigger than VRAM but about 6x slower for this work. The "bookshelf".

**DDR5, dual channel.** DDR5 is the type of RAM you have. "Dual channel" means data flows through two pipes at once instead of one, doubling speed. Your sticks are mismatched sizes (16 GB + 32 GB), so part of your memory only gets one pipe. That's the "slow corner" we found.

**SSD / NVMe.** Your storage drive (1 TB), where files live permanently. NVMe is the fast type of SSD. Still about 10x slower than RAM. The "basement archive".

**Memory bandwidth (GB/s).** How many gigabytes per second can be read from a memory. THE most important number in this whole project, because writing each word requires reading the whole model once. Your numbers: VRAM ~313 GB/s, RAM ~51 GB/s, SSD ~5 GB/s.

**PCIe.** The highway connecting the GPU to the rest of the computer. Data crossing it pays a toll in time. Yours is "Gen5 x8".

**Page cache.** Windows keeps recently-read files in spare RAM so the next read is instant. First model load is slow (comes from SSD), later loads are fast (comes from cache). That's a "cold" vs "warm" start.

**mmap (memory mapping).** A way of opening a big file where the computer pretends the file is already in memory and fetches pages only when touched. It's how llama.cpp opens models, and we proved it's the fastest option.

**Thermal throttling.** When a laptop gets hot, it deliberately slows down to cool off. Why our speed numbers need a long "sustained" test before we brag about them.

## Part 2: The AI model itself

**Model.** The AI brain: one big file full of numbers. Not a program, just numbers, like a frozen brain snapshot.

**Weights / Parameters.** Those numbers themselves. "8B" means 8 billion of them, "30B" means 30 billion. More parameters usually means smarter but bigger and slower.

**Open-source / open-weights model.** A model whose file anyone can download and run privately, free. Qwen (made by Alibaba) is ours; others include Llama (Meta) and Mistral.

**Token.** A word-piece, the unit AI reads and writes. "understanding" might be 2 tokens. Rule of thumb: 100 tokens is about 75 English words.

**Tokens per second (t/s).** Our speedometer. 25 t/s is about 20 words per second, faster than you read.

**Inference.** Fancy word for "running the model to get answers" (as opposed to training it, which builds the file in the first place, and needs a datacenter).

**Transformer.** The blueprint/architecture almost all modern AI models follow. Explained gently in docs/01.

**Attention.** The part of the transformer where the model looks back at earlier words to decide what matters for the next word. Used for every single token, so we always keep it on the fast GPU.

**KV cache.** The model's short-term memory of your conversation, so it doesn't re-read everything from scratch for each new word. Grows as the chat gets longer, and takes up VRAM.

**Context length.** How much conversation the model can hold in its head at once, measured in tokens. We set yours to 8,192 tokens, roughly 20 pages of text.

**Quantization.** Compressing the model by rounding its numbers to use fewer bits, like saving a photo at lower quality: file shrinks a lot, quality drops a little. "Q4_K_M" in our filenames means a good 4-bit compression recipe. It's why an 8B model is 5 GB instead of 16 GB.

**GGUF.** The file format llama.cpp models come in (like .mp4 is for videos). One file, ready to run.

**Mixture of Experts (MoE).** A model design where each layer has many small specialist brains ("experts", 128 per layer in our 30B model) but only a few (8) answer for each token. So a 30B model only "uses" about 3B per word. This sparsity is the loophole that lets big models run on small hardware.

**Active parameters.** The parameters actually used per token in an MoE model. Our 30B has ~3.3B active: reads like a small model, thinks closer to a big one.

**Dense model.** The opposite of MoE: every parameter used for every token. Our 8B is dense.

## Part 3: Running the AI (software)

**llama.cpp.** The free, open-source program that runs GGUF models. The engine of this entire project.

**llama-cli.** llama.cpp's chat-in-a-terminal tool.

**llama-server.** llama.cpp's tool that gives you a chat webpage in your browser (what your Start-30B-AI.bat uses).

**llama-bench.** llama.cpp's stopwatch tool. Runs the model in standard test patterns and reports tokens per second.

**Offloading / -ngl.** Choosing how many of the model's layers to place on the GPU ("offload" them) versus leaving on CPU. "-ngl 99" means "put everything possible on the GPU".

**--n-cpu-moe (-ncmoe).** The star flag of this project: put the MoE experts (the huge part) in RAM while everything else goes to the GPU. "--n-cpu-moe 40" means experts of 40 layers stay in RAM.

**Prefill (prompt processing, pp).** Phase 1 of every answer: the model reads your question. Measured as pp512 in tests (reading 512 tokens). Can be done in big batches, so it's fast.

**Decode (generation, tg).** Phase 2: writing the answer word by word. Measured as tg128 (writing 128 tokens). This is the slow, bandwidth-limited phase everything in this project revolves around.

**Batch.** Doing many tokens' math in one pass. Possible during prefill (you gave it many words at once), impossible during decode (it must write one word to know the next).

**Speculative decoding.** A planned trick: a tiny "helper" model guesses several words ahead, the big model checks the guesses in one batch. Right guesses = several words for the price of one. Our E014.

## Part 4: Doing science (experiment terms)

**Benchmark.** A standardized stopwatch test, so numbers are comparable across days and settings.

**Baseline.** The "before" measurement everything else is compared to. Ours: 62 t/s (8B on GPU), 10 t/s (8B on CPU).

**Hypothesis / pre-registering.** Writing down what we EXPECT before running the test, so we can't fool ourselves afterward. Every experiment file has a Hypothesis section written first.

**Control run.** A test where the thing being studied is switched off, to prove your measurement means what you think. Example: hiding the GPU entirely to learn the GPU was secretly helping "CPU-only" runs.

**Stddev (standard deviation).** The plus-or-minus wobble across repeated runs. Small stddev = trustworthy number.

**STREAM test.** A classic benchmark that measures raw RAM speed by copying big blocks of data. We used it to find your RAM's true ceiling (55.6 GB/s).

**Roofline / napkin math.** Quick arithmetic that predicts the best possible speed before testing: speed ceiling = bandwidth / bytes needed per token. It predicted our results within a few percent.

**Perplexity.** A score for how well a model predicts text; used to check quantization didn't damage quality. Lower is better. Not measured yet, on our list.

**Hugging Face.** The website where open models are downloaded from, like an app store for AI brains.

**E-numbers (E001, E013...).** Our own experiment numbering. E001 = baseline stopwatch tests, E002 = RAM deep-dive, E013 = the MoE breakthrough. The full menu lives in notes/research-questions.md.
