#!/usr/bin/env python3
"""parse_ncu.py — turn an `ncu --csv` dump into a per-kernel roofline table + JSONL.

The profiler CSV interleaves with the target application's own stdout, so we
locate the header row (starts with "ID") and only keep data rows whose column
count matches. Each kernel launch becomes one record keyed by (Kernel Name,
Grid Size, Block Size); every requested metric becomes a field.

Usage:
    python3 parse_ncu.py PROFILE.csv [--jsonl OUT.jsonl] [--min-duration-us N]
    ... | python3 parse_ncu.py -            # read from stdin

Exit code is non-zero if no kernel records were found (useful in CI).
"""
import sys, csv, json, argparse, re

HEADER_FIRST = "ID"

def find_header(lines):
    for i, ln in enumerate(lines):
        if ln.startswith('"ID"') or ln.startswith('ID,'):
            return i
    return -1

def to_num(v):
    try:
        return float(v.replace(",", ""))
    except (ValueError, AttributeError):
        return None

def parse(text):
    lines = text.splitlines()
    h = find_header(lines)
    if h < 0:
        return []
    reader = csv.reader(lines[h:])
    rows = list(reader)
    header = rows[0]
    ncols = len(header)
    idx = {name: j for j, name in enumerate(header)}
    def col(row, name):
        j = idx.get(name)
        return row[j] if j is not None and j < len(row) else ""

    # Group metrics by launch identity.
    launches = {}
    order = []
    for row in rows[1:]:
        if len(row) != ncols:
            continue  # skip app-stdout noise and malformed lines
        kid = (col(row, "Kernel Name"), col(row, "Grid Size"), col(row, "Block Size"))
        if kid not in launches:
            launches[kid] = {"kernel": kid[0], "grid": kid[1], "block": kid[2],
                             "metrics": {}, "launches": 0}
            order.append(kid)
        rec = launches[kid]
        name = col(row, "Metric Name")
        val = col(row, "Metric Value")
        unit = col(row, "Metric Unit")
        num = to_num(val)
        rec["metrics"][name] = {"value": num if num is not None else val, "unit": unit}
        rec["launches"] += 1
    return [launches[k] for k in order]

def mval(rec, *names):
    for n in names:
        if n in rec["metrics"]:
            v = rec["metrics"][n]["value"]
            if isinstance(v, (int, float)):
                return v
    return None

def classify(rec):
    """Heuristic roofline classification from SOL percentages."""
    comp = mval(rec, "sm__throughput.avg.pct_of_peak_sustained_elapsed")
    mem  = mval(rec, "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
                     "lts__throughput.avg.pct_of_peak_sustained_elapsed")
    tensor = mval(rec, "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
                       "sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_elapsed")
    waves = mval(rec, "launch__waves_per_multiprocessor")
    if tensor is not None and tensor >= 80:
        kind = "TENSOR-SATURATED"
    elif tensor is not None and tensor < 60 and comp is not None and comp > 15:
        kind = "TENSOR-UNDERUSED"   # e.g. legacy CUTLASS tile on Blackwell
    elif mem is not None and comp is not None and mem > comp:
        kind = "MEMORY-BOUND"
    elif comp is not None and comp >= 60:
        kind = "COMPUTE-BOUND"
    elif (waves is not None and waves < 1.0) or (comp is not None and comp < 15 and (mem is None or mem < 15)):
        kind = "LAUNCH/LATENCY-BOUND"
    else:
        kind = "MIXED/LOW-UTIL"
    return kind

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="ncu --csv output (or '-' for stdin)")
    ap.add_argument("--jsonl", default=None, help="write per-kernel records here")
    ap.add_argument("--min-duration-us", type=float, default=0.0,
                    help="drop kernels shorter than this (noise filter)")
    args = ap.parse_args()

    if args.csv == "-":
        text = sys.stdin.read()
    else:
        with open(args.csv) as f:
            text = f.read()

    recs = parse(text)
    recs = [r for r in recs
            if (mval(r, "gpu__time_duration.sum") or 0) / 1e3 >= args.min_duration_us
            or args.min_duration_us <= 0]

    if not recs:
        print("no kernel records found in CSV", file=sys.stderr)
        sys.exit(1)

    def dur_us(r): return (mval(r, "gpu__time_duration.sum") or 0) / 1e3

    recs.sort(key=lambda r: -dur_us(r))

    cols = [
        ("dur_us",     lambda r: dur_us(r)),
        ("SM%",        lambda r: mval(r, "sm__throughput.avg.pct_of_peak_sustained_elapsed")),
        ("MEM%",       lambda r: mval(r, "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed")),
        ("L2%",        lambda r: mval(r, "lts__throughput.avg.pct_of_peak_sustained_elapsed")),
        ("L1%",        lambda r: mval(r, "l1tex__throughput.avg.pct_of_peak_sustained_elapsed")),
        ("TENSOR%",    lambda r: mval(r, "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
                                          "sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_elapsed")),
        ("OCC%",       lambda r: mval(r, "sm__warps_active.avg.pct_of_peak_sustained_active")),
        ("regs/thr",   lambda r: mval(r, "launch__registers_per_thread")),
        ("waves/SM",   lambda r: mval(r, "launch__waves_per_multiprocessor")),
    ]

    def fmt(x):
        if x is None: return "-"
        if isinstance(x, float): return f"{x:.1f}"
        return str(x)

    hdr = ["kernel", "grid", "block", "class"] + [c[0] for c in cols]
    widths = [max(len(hdr[i]), 12) for i in range(len(hdr))]
    def row(vals):
        return "  ".join(str(v)[:w].ljust(w) for v, w in zip(vals, widths))
    print(row(hdr))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))
    out_jsonl = []
    for r in recs:
        vals = [r["kernel"][:40], r["grid"], r["block"], classify(r)] + [fmt(fn(r)) for _, fn in cols]
        print(row(vals))
        full = {"kernel": r["kernel"], "grid": r["grid"], "block": r["block"],
                "launches_seen": r["launches"], "class": classify(r),
                "metrics": {k: v["value"] for k, v in r["metrics"].items()}}
        out_jsonl.append(full)

    if args.jsonl:
        with open(args.jsonl, "w") as f:
            for o in out_jsonl:
                f.write(json.dumps(o) + "\n")
        print(f"\n[parse_ncu] wrote {len(out_jsonl)} kernel records -> {args.jsonl}")

if __name__ == "__main__":
    main()
