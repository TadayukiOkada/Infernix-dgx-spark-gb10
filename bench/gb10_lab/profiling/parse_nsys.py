#!/usr/bin/env python3
"""parse_nsys.py — quantify GPU busy/idle and launch overhead from an nsys sqlite export.

Answers the question that nvidia-smi power cannot: *why* is utilization low? It
computes, from the actual kernel timeline:
  * total span, total GPU-busy time, and the idle GAP time between launches
  * number of launches and mean gap size (launch-overhead signature)
  * per-kernel instance count and total time
  * CUDA API (launch/sync/memcpy) time as seen by the host
A large gap fraction with tiny per-kernel durations == launch/serialization bound.

Usage:
    python3 parse_nsys.py PROFILE.sqlite [--jsonl OUT.jsonl] [--top N]
"""
import sys, json, sqlite3, argparse

def table_exists(c, name):
    return c.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                     (name,)).fetchone() is not None

def resolve_names(c, ids):
    """Map StringIds id -> text when a StringIds table exists."""
    out = {}
    if table_exists(c, "StringIds") and ids:
        q = ",".join(str(int(i)) for i in ids if i is not None)
        if q:
            for rid, val in c.execute(f"SELECT id,value FROM StringIds WHERE id IN ({q})"):
                out[rid] = val
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite", help="exported .sqlite from nsys_profile.sh")
    ap.add_argument("--jsonl", default=None)
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    c = sqlite3.connect(args.sqlite)
    if not table_exists(c, "CUPTI_ACTIVITY_KIND_KERNEL"):
        print("no CUPTI_ACTIVITY_KIND_KERNEL table — was the app traced with -t cuda?", file=sys.stderr)
        sys.exit(1)

    has_str = table_exists(c, "StringIds")
    # Pull kernels with resolved names.
    if has_str:
        krows = c.execute("""
            SELECT k.start,k.end,k.gridX,k.gridY,k.gridZ,k.blockX,k.blockY,k.blockZ,
                   k.registersPerThread,COALESCE(s.value,k.shortName) AS name
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            LEFT JOIN StringIds s ON s.id=k.demangledName
            ORDER BY k.start""").fetchall()
    else:
        krows = c.execute("""
            SELECT start,end,gridX,gridY,gridZ,blockX,blockY,blockZ,registersPerThread,shortName
            FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start""").fetchall()

    if not krows:
        print("no kernel rows found", file=sys.stderr); sys.exit(1)

    # Merge overlapping/parallel intervals to get true busy time (union of intervals).
    merged = []
    for r in krows:
        s, e = r[0], r[1]
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    t0 = krows[0][0]; t1 = max(r[1] for r in krows)
    span_ns = t1 - t0
    busy_ns = sum(e - s for s, e in merged)
    gap_ns = span_ns - busy_ns

    # Inter-launch gaps (end of one merged interval to start of next).
    gaps = [merged[i+1][0] - merged[i][1] for i in range(len(merged)-1)]
    n_launches = len(krows)
    mean_gap = (sum(gaps)/len(gaps)) if gaps else 0.0
    max_gap = max(gaps) if gaps else 0.0

    # Per-kernel aggregation by name.
    per = {}
    for r in krows:
        name = r[9] or "(unknown)"
        d = r[1]-r[0]
        rec = per.setdefault(name, {"name": name, "instances": 0, "total_ns": 0,
                                    "min_ns": d, "max_ns": d, "grid": r[2], "block": r[5]})
        rec["instances"] += 1; rec["total_ns"] += d
        rec["min_ns"] = min(rec["min_ns"], d); rec["max_ns"] = max(rec["max_ns"], d)
    kernels = sorted(per.values(), key=lambda x: -x["total_ns"])

    # Host-side CUDA API time (launches, syncs, memcpy calls).
    api_summary = []
    if table_exists(c, "CUPTI_ACTIVITY_KIND_RUNTIME"):
        api_rows = c.execute("""
            SELECT COALESCE(s.value,'api'), COUNT(*), SUM(r.end-r.start)
            FROM CUPTI_ACTIVITY_KIND_RUNTIME r
            LEFT JOIN StringIds s ON s.id=r.nameId
            GROUP BY 1 ORDER BY 3 DESC LIMIT 8""").fetchall()
        api_summary = [{"api": a, "calls": n, "total_us": round(t/1e3, 1)} for a, n, t in api_rows if a]

    mem = []
    if table_exists(c, "CUPTI_ACTIVITY_KIND_MEMCPY"):
        mrows = c.execute("""SELECT COUNT(*), SUM(end-start), SUM(bytes)
                            FROM CUPTI_ACTIVITY_KIND_MEMCPY""").fetchall()[0]
        if mrows and mrows[0]:
            mem = [{"count": mrows[0], "total_us": round(mrows[1]/1e3, 1), "bytes": mrows[2]}]

    busy_pct = 100.0 * busy_ns / span_ns if span_ns else 0.0
    gap_pct = 100.0 * gap_ns / span_ns if span_ns else 0.0

    print(f"=== nsys timeline analysis ({args.sqlite}) ===")
    print(f"span          : {span_ns/1e6:.3f} ms   ({n_launches} kernel launches)")
    print(f"GPU busy      : {busy_ns/1e6:.3f} ms   ({busy_pct:.1f}% of span)")
    print(f"GPU idle/gap  : {gap_ns/1e6:.3f} ms   ({gap_pct:.1f}% of span)")
    print(f"mean gap      : {mean_gap/1e3:.2f} us   max gap: {max_gap/1e3:.2f} us")
    verdict = ("LAUNCH/SERIALIZATION-BOUND (large idle gaps)"
               if gap_pct > 40 else
               "GPU mostly busy (bottleneck is inside kernels, not launch gaps)")
    print(f"verdict       : {verdict}")
    if mem:
        print(f"memcpy        : {mem[0]['count']} copies, {mem[0]['total_us']} us, {mem[0]['bytes']} bytes")
    print(f"\ntop kernels by total GPU time:")
    print(f"{'name':44} {'inst':>5} {'tot_us':>9} {'avg_us':>8} {'min_us':>8} {'max_us':>8}")
    for k in kernels[:args.top]:
        avg = k['total_ns']/k['instances']/1e3
        print(f"{k['name'][:44]:44} {k['instances']:5d} {k['total_ns']/1e3:9.1f} {avg:8.1f} {k['min_ns']/1e3:8.1f} {k['max_ns']/1e3:8.1f}")
    if api_summary:
        print(f"\ntop host CUDA API calls:")
        for a in api_summary:
            print(f"  {a['api']:38} {a['calls']:6d} calls  {a['total_us']:9.1f} us")

    if args.jsonl:
        out = {"span_ms": span_ns/1e6, "busy_ms": busy_ns/1e6, "gap_ms": gap_ns/1e6,
               "busy_pct": busy_pct, "gap_pct": gap_pct, "launches": n_launches,
               "mean_gap_us": mean_gap/1e3, "max_gap_us": max_gap/1e3, "verdict": verdict,
               "kernels": kernels, "api": api_summary, "memcpy": mem}
        with open(args.jsonl, "w") as f:
            f.write(json.dumps(out) + "\n")
        print(f"\n[parse_nsys] wrote -> {args.jsonl}")

if __name__ == "__main__":
    main()
