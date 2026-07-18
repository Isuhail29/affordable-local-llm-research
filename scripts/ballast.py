"""Allocate and touch N GiB of RAM, then hold it until killed.

Usage: python ballast.py [gib]   (default 28)
"""
import sys
import time

import numpy as np

gib = int(sys.argv[1]) if len(sys.argv) > 1 else 28
a = np.ones(gib * 1024**3 // 8, dtype=np.float64)  # fill touches every page
print(f"ballast resident: {gib} GiB", flush=True)
time.sleep(3600)
