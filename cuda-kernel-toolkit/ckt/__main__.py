"""Allow invoking the toolkit via ``python3 -m ckt`` as well as ``python3 -m ckt.cli``."""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
