"""Expert-scatter access-pattern microbenchmark (E021 Part B).

Replays the memory-access pattern of MoE expert reads WITHOUT any compute:
N workers mmap the (page-cache-warm) model file and read random-offset
chunks of size S, forcing the reads with a numpy sum. Sweeping S from
0.25 MB to 64 MB plus a sequential reference separates access-pattern
physics from llama.cpp implementation overhead.

Usage: python scatter-bench.py <file> [workers]   (default workers: 12)
"""
import mmap
import multiprocessing as mp
import os
import random
import sys
import time

import numpy as np

CHUNK_SIZES = [16 * 1024, 32 * 1024, 64 * 1024, 128 * 1024, 256 * 1024, 1024 * 1024, 4 * 1024 * 1024]
BYTES_PER_WORKER = 1536 * 1024 * 1024  # ~1.5 GiB read per worker per config
ALIGN = 4096


def worker(path, chunk, barrier, out, idx, seq):
    rng = random.Random(1234 + idx)
    f = open(path, "rb")
    size = os.path.getsize(path)
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    n_chunks = max(1, BYTES_PER_WORKER // chunk)
    span = size - chunk - ALIGN
    if seq:
        # each worker sweeps its own contiguous slice, forward order
        base = (size // 12) * idx
        base -= base % ALIGN
        offsets = [min(base + i * chunk, span) - (min(base + i * chunk, span) % ALIGN) for i in range(n_chunks)]
    else:
        offsets = [rng.randrange(0, span // ALIGN) * ALIGN for _ in range(n_chunks)]
    # pass 1: populate this process's page tables (soft faults happen here, untimed)
    for o in offsets:
        np.frombuffer(mm, dtype=np.int64, count=chunk // 8, offset=o).sum()
    total = 0
    barrier.wait()
    # pass 2: timed, warm page tables = steady-state access-pattern bandwidth
    t0 = time.perf_counter()
    for o in offsets:
        total += int(np.frombuffer(mm, dtype=np.int64, count=chunk // 8, offset=o).sum())
    dt = time.perf_counter() - t0
    out[idx] = (n_chunks * chunk) / dt / 1e9
    # no explicit mm.close(): lingering numpy views make it raise; process exit cleans up


def run(path, chunk, n_workers, seq):
    barrier = mp.Barrier(n_workers)
    out = mp.Array("d", n_workers)
    procs = [mp.Process(target=worker, args=(path, chunk, barrier, out, i, seq)) for i in range(n_workers)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    return sum(out)


if __name__ == "__main__":
    path = sys.argv[1]
    n_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    print(f"file: {path} ({os.path.getsize(path)/1e9:.1f} GB), workers: {n_workers}")
    for chunk in CHUNK_SIZES:
        gbps = run(path, chunk, n_workers, seq=False)
        print(f"random  {chunk/1024/1024:6.2f} MB chunks: {gbps:6.1f} GB/s aggregate")
    gbps = run(path, 16 * 1024 * 1024, n_workers, seq=True)
    print(f"sequential 16 MB sweep:      {gbps:6.1f} GB/s aggregate")
