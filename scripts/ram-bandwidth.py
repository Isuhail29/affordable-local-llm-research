"""Multiprocess STREAM-style memory bandwidth benchmark (copy kernel).

Each worker allocates two 1 GiB arrays and repeatedly copies one into the
other with numpy (memcpy under the hood, single thread per worker).
Reported bandwidth uses the STREAM copy convention: read + write, so
2 bytes of traffic per element byte per iteration.

Usage: python ram-bandwidth.py [worker_counts...]   (default: 1 2 4 8 12)
"""
import multiprocessing as mp
import sys
import time

import numpy as np

ARRAY_BYTES = 1 * 1024**3  # 1 GiB per array, 2 GiB resident per worker
ITERS = 8


def worker(barrier, out, idx):
    a = np.ones(ARRAY_BYTES // 8, dtype=np.float64)
    b = np.empty_like(a)
    b[:] = a  # warm-up copy, faults every page in
    barrier.wait()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        b[:] = a
    dt = time.perf_counter() - t0
    out[idx] = (2 * ARRAY_BYTES * ITERS) / dt / 1e9  # GB/s, read+write


if __name__ == "__main__":
    counts = [int(x) for x in sys.argv[1:]] or [1, 2, 4, 8, 12]
    for n in counts:
        barrier = mp.Barrier(n)
        out = mp.Array("d", n)
        procs = [mp.Process(target=worker, args=(barrier, out, i)) for i in range(n)]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        per = ", ".join(f"{v:.1f}" for v in out)
        print(f"{n:2d} workers: {sum(out):6.1f} GB/s aggregate  (per-worker: {per})")
