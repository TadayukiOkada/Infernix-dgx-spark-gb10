#!/usr/bin/env python3
"""Run a hardware-constrained launch-parameter search and print its summary.

Thin wrapper over `ckt.cli search`. Example:

    python3 tools/run_search.py --exe /tmp/ckt_kernels_build/ckt_kernels \
        --workload reduce_block --problem-size 4194304 \
        --strategy model-guided --output results/search-run

Candidates are correctness-gated via --verify before any benchmarking, and the
noise floor is reported alongside every speedup claim.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ckt.cli import main  # noqa: E402


def run() -> int:
    args = ["search"] + [a for a in sys.argv[1:] if a != "--run-search"]
    return main(args)


if __name__ == "__main__":
    raise SystemExit(run())
