#!/usr/bin/env python3
"""Print per-launch counters from an NCU raw-page CSV import.

Usage: ncu_report.py REPORT.csv [--grep PATTERN] [--metrics M1,M2,...]

Columns come from the header row; unknown metrics print as 'n/a'.
"""
import argparse
import csv


DEFAULT_METRICS = [
    "gpu__time_duration.sum",
    "launch__registers_per_thread",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "smsp__inst_executed.sum",
    "lts__t_sector_hit_rate.pct",
    "l1tex__t_sector_hit_rate.pct",
    "smsp__sass_thread_inst_executed_op_integer_pred_on.sum",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_wait_per_warp_active.pct",
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--grep", default=None)
    ap.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    args = ap.parse_args()

    rows = list(csv.reader(open(args.path)))
    hdr = rows[0]
    want = args.metrics.split(",")
    idx = {m: hdr.index(m) if m in hdr else None for m in want}
    name_i = hdr.index("Kernel Name") if "Kernel Name" in hdr else None
    grid_i = hdr.index("Grid Size") if "Grid Size" in hdr else None
    blk_i = hdr.index("Block Size") if "Block Size" in hdr else None

    fmt = "{:<34} {:>10} {:>5} {:>9} {:>9} {:>8} {:>8}"
    short = lambda s: s[:33].replace("void unnamed>::", "")
    header = ("kernel", *[f"{m.split('.')[-1][:9]:>10}" for m in want])
    print(f"{'':<36}", *[f"{m.split('.')[-1][:10]:>10}" for m in want])
    for r in rows[1:]:
        if not any(r):
            continue
        name = r[name_i] if name_i is not None and len(r) > name_i else ""
        if args.grep and args.grep not in name:
            continue
        vals = []
        for m in want:
            i = idx[m]
            v = r[i] if i is not None and len(r) > i and r[i] != "" else "n/a"
            try:
                fv = float(v)
                v = f"{fv:.2f}" if "." in str(fv) else f"{int(round(fv)):d}"
            except ValueError:
                pass
            vals.append(f"{v:>10}")
        extra = (r[grid_i] + " x " + r[blk_i]) if grid_i is not None and len(r) > grid_i else ""
        print(short(name), extra, *vals)


if __name__ == "__main__":
    main()
