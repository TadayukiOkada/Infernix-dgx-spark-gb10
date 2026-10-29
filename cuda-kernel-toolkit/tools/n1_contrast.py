#!/usr/bin/env python3
"""n1_contrast.py — summarise NCU raw CSV exports by kernel group.

Usage:
    python3 tools/n1_contrast.py [ncu_export.csv ...] [--filter SUBSTR]
"""

from __future__ import annotations

import argparse
import collections
import csv
import sys
from pathlib import Path

ORDER = [
    ("gpu__time_duration.sum", "dur_ns"),
    ("smsp__issue_active.avg.pct_of_peak_sustained_active", "issue%"),
    ("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed", "tensor%"),
    ("sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_elapsed", "alu%"),
    ("sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_elapsed", "fma%"),
    ("smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio", "maththr"),
    ("smsp__average_warps_issue_stalled_wait_per_issue_active.ratio", "wait"),
    ("smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio", "barrier"),
    ("smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio", "long_sb"),
    ("sm__warps_active.avg.pct_of_peak_sustained_active", "warps%"),
]


def short(n: str) -> str:
    if "gate_up_tile" in n:
        return "gate_up"
    if "down_resident" in n:
        return "down_res"
    return n.split("(")[0][:24]


def num(v: str) -> float | None:
    try:
        return float(v.replace(",", ""))
    except Exception:
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Summarise NCU CSV exports across kernel instances")
    ap.add_argument("csv_files", nargs="*", default=["/tmp/ncu_n1.csv"], help="NCU CSV file(s)")
    ap.add_argument("--filter", default="", help="substring filter on kernel name")
    args = ap.parse_args(argv)

    for csv_path in args.csv_files:
        p = Path(csv_path)
        if not p.exists():
            print(f"error: CSV file not found: {p}", file=sys.stderr)
            return 2
        with p.open() as f:
            rows = [row for row in csv.reader(f) if row and not row[0].startswith("==")]
        if not rows:
            continue
        h = rows[0]
        ki = h.index("Kernel Name")
        mn = h.index("Metric Name")
        mv = h.index("Metric Value")
        idc = h.index("ID")

        inst: dict[str, dict[str, str]] = collections.OrderedDict()
        for x in rows[1:]:
            if len(x) <= mv or not x[ki]:
                continue
            if args.filter and args.filter not in x[ki]:
                continue
            key = f"{short(x[ki])}#{x[idc]}"
            inst.setdefault(key, {})[x[mn]] = x[mv]

        groups: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
        for k, d in inst.items():
            groups[k.split("#")[0]].append(d)

        if len(args.csv_files) > 1:
            print(f"=== {p} ===")
        lbls = [l for _, l in ORDER]
        print(f"{'kernel':<24}" + "".join(f"{l:>10}" for l in lbls))
        for name, g_list in groups.items():
            row = f"{name:<24}"
            for mkey, _ in ORDER:
                vals = [num(d.get(mkey, "")) for d in g_list]
                vals_f = [v for v in vals if v is not None]
                row += f"{(sum(vals_f) / len(vals_f)):>10.2f}" if vals_f else f"{'NA':>10}"
            print(row)
        print("\ninstances:", {n: len(g) for n, g in groups.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
