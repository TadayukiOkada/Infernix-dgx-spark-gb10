#!/usr/bin/env python3
"""Rank kernels in an NSYS SQLite export and report per-round GPU occupancy.

Two things this reports that a plain kernel list does not:

* GPU busy fraction split by phase. A whole-capture average is dominated by the
  one-time weight upload (tens of GB of CUPTI_ACTIVITY_KIND_MEMCPY), which makes
  decode look like it has huge idle gaps. Splitting the kernel timeline at gaps
  larger than --gap-us separates model load from generation rounds, so a round's
  real busy fraction can be read. Busy above 100 % means streams overlap.
* Shared-memory and register columns per launch geometry, which is what bounds
  CTAs per SM (65,536 registers and 101,376 B of shared memory on GB10).

Usage: rank_kernels.py TRACE.sqlite [--limit N] [--gap-us US] [--top-kernels N]
"""

import argparse
import sqlite3
import sys


def med(values):
    if not values:
        return float("nan")
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--limit", type=int, default=25, help="rows in the kernel table")
    ap.add_argument("--top-kernels", type=int, default=6, help="phases to report")
    ap.add_argument("--gap-us", type=int, default=1000, help="gap that splits phases")
    args = ap.parse_args()

    con = sqlite3.connect(args.trace)
    cur = con.cursor()

    total = cur.execute(
        "SELECT COUNT(*), COALESCE(SUM(end - start), 0) FROM CUPTI_ACTIVITY_KIND_KERNEL"
    ).fetchone()
    if not total[0]:
        print("no kernel activity in trace")
        return 1
    print(f"kernels: {total[0]}  total kernel time: {total[1] / 1e6:.1f} ms")

    try:
        mc = cur.execute(
            "SELECT COUNT(*), COALESCE(SUM(end - start), 0), COALESCE(SUM(bytes), 0) "
            "FROM CUPTI_ACTIVITY_KIND_MEMCPY"
        ).fetchone()
        print(f"memcpy: {mc[0]} copies, {mc[1] / 1e6:.1f} ms, {mc[2] / 2**30:.1f} GiB")
    except sqlite3.OperationalError:
        pass

    # Phase split: any gap larger than the threshold starts a new phase.
    spans = cur.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start").fetchall()
    phases = []
    cur_start, prev_end = spans[0][0], spans[0][1]
    for s, e in spans[1:]:
        if s - prev_end > args.gap_us * 1000:
            phases.append((cur_start, prev_end))
            cur_start = s
        prev_end = max(prev_end, e)
    phases.append((cur_start, prev_end))

    print(f"\nphases (split at gaps > {args.gap_us} us):")
    for a, b in phases:
        span = b - a
        cnt, ksum = cur.execute(
            "SELECT COUNT(*), COALESCE(SUM(end - start), 0) FROM CUPTI_ACTIVITY_KIND_KERNEL "
            "WHERE start >= ? AND end <= ?",
            (a, b),
        ).fetchone()
        print(f"  span {span / 1e6:8.1f} ms  kernels {ksum / 1e6:8.1f} ms  "
              f"{100.0 * ksum / span:6.1f}% busy  n={cnt}")

    print(f"\ntop kernels by total time (limit {args.limit}):")
    header = f"{'total_ms':>9} {'calls':>6} {'avg_us':>9} {'grid':>13} {'blk':>5} {'reg':>4} {'smem':>7}"
    print(header)
    rows = cur.execute(
        """
        SELECT s.value, COUNT(*), SUM(k.end - k.start), AVG(k.end - k.start),
               k.gridX, k.gridY, k.gridZ, k.blockX, k.registersPerThread,
               COALESCE(k.dynamicSharedMemory, 0) + COALESCE(k.staticSharedMemory, 0)
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.demangledName = s.id
        GROUP BY s.value, k.gridX, k.gridY, k.gridZ, k.blockX, k.registersPerThread
        ORDER BY SUM(k.end - k.start) DESC
        LIMIT ?
        """,
        (args.limit,),
    ).fetchall()
    for name, calls, tsum, tavg, gx, gy, gz, bx, regs, smem in rows:
        grid = f"({gx},{gy},{gz})" if gz and gz > 1 else f"({gx},{gy})"
        print(f"{tsum / 1e6:9.2f} {calls:6d} {tavg / 1e3:9.2f} {grid:>13} "
              f"{bx:5d} {regs or 0:4d} {smem:7d}  {name[:96]}")

    # Per-kernel median latency: the mean hides bimodal tail waves.
    print(f"\nmedian per-call latency for the top {args.top_kernels} kernels:")
    for name, calls, tsum, _tavg, gx, gy, _gz, bx, _regs, _smem in rows[: args.top_kernels]:
        durs = sorted(r[0] for r in cur.execute(
            """
            SELECT k.end - k.start FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON k.demangledName = s.id
            WHERE s.value = ? AND k.gridX = ? AND k.gridY = ?
            """,
            (name, gx, gy),
        ))
        if not durs:
            continue
        p95 = durs[min(len(durs) - 1, int(round(0.95 * (len(durs) - 1))))]
        print(f"  med {med(durs) / 1e3:8.2f} us  p95 {p95 / 1e3:8.2f} us  n={len(durs)}  {name[:80]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
