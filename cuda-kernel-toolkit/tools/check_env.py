#!/usr/bin/env python3
"""One-shot toolkit health check: toolchain, GPU properties, cache directory.

Usage:
    python3 tools/check_env.py [--config configs/gb10_default.yaml]

Exit codes mirror ckt.cli: 0 ok, 2 usage/validation, 3 missing ncu,
4 ToolError, 130 interrupted.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ckt.cli import main  # noqa: E402


def run() -> int:
    args = ["doctor"] + [a for a in sys.argv[1:] if a != "--check-env"]
    return main(args)


if __name__ == "__main__":
    raise SystemExit(run())
