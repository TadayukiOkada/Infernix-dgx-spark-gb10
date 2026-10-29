"""Diagnostic: what default (graph-mode) CUPTI tracing reports for CUDA-graph decode."""
import sqlite3
import statistics

db = sqlite3.connect("decode_tg128.sqlite")
cur = db.cursor()
for table in ("CUPTI_ACTIVITY_KIND_GRAPH_TRACE", "CUPTI_ACTIVITY_KIND_KERNEL", "CUDA_GRAPH_EVENTS"):
    print(table, cur.execute("SELECT COUNT(*) FROM " + table).fetchone()[0])

rows = list(cur.execute("SELECT graphId, start, end FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE ORDER BY start"))
durs = [(end - start) / 1e6 for _, start, end in rows]
print("replays %d  min %.2f  median %.2f  max %.2f ms" % (len(rows), min(durs), statistics.median(durs), max(durs)))
bands = {"<1ms": 0, "1-10ms": 0, "10-20ms": 0, ">20ms": 0}
for d in durs:
    if d < 1:
        bands["<1ms"] += 1
    elif d < 10:
        bands["1-10ms"] += 1
    elif d < 20:
        bands["10-20ms"] += 1
    else:
        bands[">20ms"] += 1
print("duration bands:", bands)

per = {}
for gid, start, end in rows:
    per.setdefault(gid, []).append((start, end))
print("distinct graphIds:", len(per))
for gid, spans in sorted(per.items(), key=lambda kv: -len(kv[1])):
    dd = [(e - s) / 1e6 for s, e in spans]
    print("  graphId %d  n=%d  %.2f-%.2f ms" % (gid, len(spans), min(dd), max(dd)))

inside = 0
for gid, start, end in rows:
    inside += cur.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start>=? AND end<=?",
                          (start, end)).fetchone()[0]
print("kernel records inside any graph replay window:", inside)
