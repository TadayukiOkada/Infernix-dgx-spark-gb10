#!/usr/bin/env python3
"""Full analysis pipeline producing markdown/HTML/JSON reports.

Thin wrapper over `ckt.cli analyze` that pins output names so downstream
tooling can find them deterministically:

    python3 tools/make_report.py --config configs/gb10_default.yaml \
        [--results-dir results/gb10-sample]

Produces <results-dir>/analysis.{json,md,html}. Re-running with identical
inputs hits the metric cache and completes quickly.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ckt.cli import main  # noqa: E402


def run() -> int:
    args = ["analyze"] + [a for a in sys.argv[1:] if a != "--make-report"]
    return main(args)


if __name__ == "__main__":
    raise SystemExit(run())
