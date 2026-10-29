#!/usr/bin/env python3
"""decode_hotspots.py — rank decode kernels from an nsys node-mode SQLite export.

Node-mode tracing (--cuda-graph-trace=node) emits a CUPTI_ACTIVITY_KIND_KERNEL row for every kernel
*inside* a CUDA-graph replay, so the plain kernel table is complete for the decode path. The
graph-trace table is only needed to account for gaps (a sync's duration equals an overlapping graph
row). Usage:

    python3 tools/decode_hotspots.py profiles/nsys/<name>.sqlite [--top N] [--since US]

Totals are summed over the whole captured window. A per-launch mean and the launch count are what
identify a "suspicious" kernel: a large share with few launches is latency/serialisation-bound,
while a large share with many short launches is per-launch underfill. Registers/shared are reported
so the same table feeds the occupancy reasoning without a second tool.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict


def fetch(con: sqlite3.Connection):
    cur = con.cursor()
    # Kernel rows, keyed on name + geometry so distinct launches of one template split apart.
    rows = cur.execute(
        """
        SELECT s.value AS name,
               COUNT(*)                         AS calls,
               SUM(k.end - k.start)             AS ns,
               k.gridX, k.gridY, k.gridZ, k.blockX,
               k.registersPerThread             AS regs,
               (k.dynamicSharedMemory + k.staticSharedMemory) AS smem
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.demangledName = s.id
        GROUP BY s.value, k.gridX, k.gridY, k.gridZ, k.blockX, k.registersPerThread,
                 k.dynamicSharedMemory, k.staticSharedMemory
        """
    )
    rows = cur.fetchall()
    total_busy = sum(r[2] for r in rows)
    # Aggregate by demangled base name for the headline share.
    by_name: dict[str, dict] = defaultdict(lambda: {"ns": 0, "calls": 0})
    for name, calls, ns, *_rest in rows:
        by_name[name]["ns"] += ns
        by_name[name]["calls"] += calls
    return rows, by_name, total_busy


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--grid-detail", action="store_true",
                    help="split rows by launch geometry instead of folding by name")
    args = ap.parse_args(argv)

    con = sqlite3.connect(args.sqlite)
    rows, by_name, total_busy = fetch(con)
    if total_busy == 0:
        print("no kernel rows: was the trace captured with --cuda-graph-trace=node?", file=sys.stderr)
        return 1

    print(f"# total busy kernel time: {total_busy / 1e6:.3f} ms over "
          f"{sum(v['calls'] for v in by_name.values())} launches, "
          f"{len(by_name)} distinct kernels\n")

    if args.grid_detail:
        print(f"{'share%':>7} {'total_ms':>9} {'calls':>7} {'mean_us':>9} "
              f"{'grid':>16} {'blk':>5} {'reg':>4} {'smem':>7}  name")
        for name, calls, ns, gx, gy, gz, bx, regs, smem in sorted(rows, key=lambda r: -r[2])[: args.top]:
            share = 100.0 * ns / total_busy
            mean_us = ns / calls / 1e3
            grid = f"{gx}x{gy}x{gz}"
            short = name if len(name) <= 70 else name[:67] + "..."
            print(f"{share:7.2f} {ns/1e6:9.3f} {calls:7d} {mean_us:9.2f} "
                  f"{grid:>16} {bx:5d} {regs:>4} {smem:>7}  {short}")
    else:
        print(f"{'share%':>7} {'total_ms':>9} {'calls':>7} {'mean_us':>9}  name")
        for name, v in sorted(by_name.items(), key=lambda kv: -kv[1]["ns"])[: args.top]:
            share = 100.0 * v["ns"] / total_busy
            mean_us = v["ns"] / v["calls"] / 1e3
            short = name if len(name) <= 90 else name[:87] + "..."
            print(f"{share:7.2f} {v['ns']/1e6:9.3f} {v['calls']:7d} {mean_us:9.2f}  {short}")
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
