#!/usr/bin/env python3
"""kernel_math.py — CLI wrapper around ``ckt math`` for zero-hand-math tuning.

Usage:
    python3 tools/kernel_math.py occupancy --threads 256 --regs 128 --smem 12288
    python3 tools/kernel_math.py waves --ctas 200 --threads 256 --regs 128
    python3 tools/kernel_math.py roofline --bytes 44269568 --flops 536870912 --latency-us 613.5 --dtype int8
    python3 tools/kernel_math.py workload --m 2 --n 4096 --k 16384 --weight-dtype iq3_s --act-dtype q8_1 --latency-us 613.5
    python3 tools/kernel_math.py smem --rows 64 --cols 64 --elem-bytes 2 --stages 2
    python3 tools/kernel_math.py mlp --warps-per-sm 16 --load-width-bytes 16
    python3 tools/kernel_math.py ab --base 613.5,614.1,612.9 --cand 480.2,481.0,479.8
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ckt.cli import main  # noqa: E402


if __name__ == "__main__":
    argv = sys.argv[1:]
    if not argv:
        argv = ["math", "--help"]
    elif argv[0] != "math":
        argv = ["math", *argv]
    raise SystemExit(main(argv))
