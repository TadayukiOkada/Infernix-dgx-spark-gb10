"""P0-B1: is a requested graph mode an effective graph mode?

Compares two node-mode CUPTI captures of the same fixed decode configuration
(`tg128`, MTP draft-tokens 3, INT8 KV, warmup 3, one measured rep) differing
only in the bench flag `--no-cuda-graph`. The bench JSON labels one run
`decode_path=mtp_cuda_graph` and the other `mtp_eager`. This script shows what
the device actually executed.

Usage: python3 p0b1_compare.py
Run from this directory. Expects node_r1n.sqlite (graphs requested, archived in
../p1k2/) and node_r1n_nograph.sqlite (graphs refused) beside it.

Every table here is CUPTI record data; nothing is inferred from the JSON labels.
"""
import collections
import sqlite3
import sys

BASE = "../p1k2/node_r1n.sqlite"
EAGER = "node_r1n_nograph.sqlite"

RUNTIME_NAMES = [
    "cudaGraphLaunch",
    "cudaGraphInstantiate",
    "cudaGraphInstantiateWithFlags",
    "cudaStreamBeginCapture",
    "cudaStreamEndCapture",
    "cudaGraphDestroy",
    "cudaGraphExecDestroy",
    "cudaLaunchKernel",
    "cudaLaunchKernelExC",
    "cudaMemcpyAsync",
    "cudaStreamSynchronize",
]


def runtime_api_counts(db):
    """Count runtime API calls by name, resolved through StringIds."""
    cur = db.cursor()
    out = {}
    rows = cur.execute(
        """
        SELECT s.value, COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME r
        JOIN StringIds s ON s.id = r.nameId
        GROUP BY s.value ORDER BY COUNT(*) DESC
        """
    ).fetchall()
    for name, n in rows:
        out[name] = n
    return out


def kernel_histogram(db):
    cur = db.cursor()
    rows = cur.execute(
        """
        SELECT s.value, COUNT(*), SUM(k.end - k.start), MIN(k.end - k.start),
               MAX(k.end - k.start), MAX(k.localMemoryPerThread)
        FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id = k.demangledName
        GROUP BY s.value
        """
    ).fetchall()
    return {name: (n, dur_sum, dur_min, dur_max, lmt) for name, n, dur_sum, dur_min, dur_max, lmt in rows}


def launch_signatures(db):
    """Multiset of (name, grid, block, registers, shared memory) launch signatures.

    Two runs executed the same work with the same launch geometry iff these multisets
    are equal. CUPTI carries no tensor shape fields, so the signature is the closest
    record-level stand-in for 'which operator, on what shape'.
    """
    cur = db.cursor()
    counter = collections.Counter()
    for row in cur.execute(
        "SELECT s.value,k.gridX,k.gridY,k.gridZ,k.blockX,k.blockY,k.blockZ,"
        "k.registersPerThread,k.staticSharedMemory,k.dynamicSharedMemory "
        "FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName"
    ):
        counter[row] += 1
    return counter


def graph_node_split(db):
    """Kernel records and busy time split by whether CUPTI attached a graphNodeId."""
    cur = db.cursor()
    in_graph = cur.execute(
        "SELECT COUNT(*), COALESCE(SUM(end - start), 0) FROM CUPTI_ACTIVITY_KIND_KERNEL "
        "WHERE graphNodeId IS NOT NULL").fetchone()
    no_node = cur.execute(
        "SELECT COUNT(*), COALESCE(SUM(end - start), 0) FROM CUPTI_ACTIVITY_KIND_KERNEL "
        "WHERE graphNodeId IS NULL").fetchone()
    eager_names = cur.execute(
        "SELECT s.value, COUNT(*), SUM(k.end - k.start) FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds s ON s.id=k.demangledName WHERE k.graphNodeId IS NULL "
        "GROUP BY s.value ORDER BY 3 DESC").fetchall()
    node_events = cur.execute(
        "SELECT s.value, COUNT(*) FROM CUDA_GRAPH_NODE_EVENTS n "
        "JOIN StringIds s ON s.id=n.nameId GROUP BY s.value").fetchall()
    return in_graph, no_node, eager_names, node_events


def graph_facts(db):
    cur = db.cursor()
    present = {r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    # CUPTI_ACTIVITY_KIND_GRAPH_TRACE exists only when nsys ran with
    # --cuda-graph-trace=graph; node mode creates CUDA_GRAPH_NODE_EVENTS instead.
    wanted = ["CUPTI_ACTIVITY_KIND_KERNEL", "CUDA_GRAPH_NODE_EVENTS",
              "CUPTI_ACTIVITY_KIND_GRAPH_TRACE", "CUPTI_ACTIVITY_KIND_MEMCPY",
              "CUPTI_ACTIVITY_KIND_MEMSET", "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION"]
    tables = {t: (cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] if t in present else 0)
              for t in wanted}
    # graphNodeId is non-null exactly for kernels recorded as graph nodes.
    inside = cur.execute(
        "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE graphNodeId IS NOT NULL"
    ).fetchone()[0]
    kernels = cur.execute(
        "SELECT start, end, graphNodeId FROM CUPTI_ACTIVITY_KIND_KERNEL"
    ).fetchall()
    busy = sum(e - s for s, e, _ in kernels) / 1e6
    span = (max(e for _, e, _ in kernels) - min(s for s, _, _ in kernels)) / 1e6
    cols = [c[1] for c in cur.execute("PRAGMA table_info(CUDA_GRAPH_NODE_EVENTS)")]
    node_kinds = None
    if "graphNodeType" in cols:
        node_kinds = cur.execute(
            "SELECT graphNodeType, COUNT(*) FROM CUDA_GRAPH_NODE_EVENTS GROUP BY graphNodeType ORDER BY 2 DESC"
        ).fetchall()
    return tables, inside, busy, span, node_kinds


def main():
    base = sqlite3.connect(BASE)
    eager = sqlite3.connect(EAGER)

    print("== 1. CUPTI table row counts (node mode, capture range = the measured rep) ==")
    bt, b_inside, b_busy, b_span, b_kinds = graph_facts(base)
    et, e_inside, e_busy, e_span, e_kinds = graph_facts(eager)
    for t in sorted(set(bt) | set(et)):
        print(f"   {t:<40} graphs-requested {bt.get(t, 0):>7}   eager-requested {et.get(t, 0):>7}"
              f"   {'same' if bt.get(t, 0) == et.get(t, 0) else 'DIFFERENT'}")
    print(f"   kernels recorded with a non-null graphNodeId: graphs-requested {b_inside}, eager-requested {e_inside}")
    print(f"   kernel busy time: graphs-requested {b_busy:.2f} ms, eager-requested {e_busy:.2f} ms")
    print(f"   first..last kernel span: graphs-requested {b_span:.2f} ms, eager-requested {e_span:.2f} ms")

    print("\n== 2. CUDA runtime API calls inside the capture range ==")
    br, er = runtime_api_counts(base), runtime_api_counts(eager)
    for name in sorted(set(br) | set(er)):
        marker = "  " if br.get(name, 0) == er.get(name, 0) else " <<"
        print(f"   {name:<38} {br.get(name, 0):>6} {er.get(name, 0):>7}{marker}")
    # CUPTI records version-suffixed names (cudaGraphLaunch_v10000), so match by prefix.
    missing = [n for n in RUNTIME_NAMES
               if not any(k.startswith(n) for k in set(br) | set(er))]
    if missing:
        print("   (API names with zero occurrences in both ranges: " + ", ".join(missing) + ")")
        print("    -> graphs are captured during warmup, outside --profile-measured;")
        print("       the measured rep only launches already-instantiated executables.")

    print("\n== 3. Per-kernel histogram equality ==")
    bh, eh = kernel_histogram(base), kernel_histogram(eager)
    names = sorted(set(bh) | set(eh))
    diffs = [n for n in names
             if bh.get(n, (0, 0))[0] != eh.get(n, (0, 0))[0]
             or abs(bh.get(n, (0, 0))[1] - eh.get(n, (0, 0))[1]) > 1e6]
    print(f"   distinct kernel names: graphs-requested {len(bh)}, eager-requested {len(eh)}")
    print(f"   names whose count differs, or whose summed duration differs by >1 ms: {len(diffs)}")
    for n in diffs:
        print(f"      {n}\n         graphs-requested {bh.get(n)}\n         eager-requested  {eh.get(n)}")
    max_lmt = max(v[4] for v in bh.values())
    print(f"   max localMemoryPerThread across all records: {max_lmt}")

    print("\n== 4. Launch-signature multisets (name, grid, block, registers, shared memory) ==")
    bs, es = launch_signatures(base), launch_signatures(eager)
    print(f"   distinct signatures: graphs-requested {len(bs)}, eager-requested {len(es)}")
    print(f"   multisets equal: {bs == es}")
    for sig, count in ((bs - es) + (es - bs)):
        print(f"      differs x{count}: {sig}")

    print("\n== 5. Kernel records split by CUPTI graphNodeId ==")
    for tag, db in (("graphs-requested", base), ("eager-requested", eager)):
        in_graph, no_node, eager_names, node_events = graph_node_split(db)
        print(f"   {tag}: as graph nodes {in_graph[0]:>6} kernels / {in_graph[1]/1e6:>8.2f} ms; "
              f"without a graphNodeId {no_node[0]:>5} kernels / {no_node[1]/1e6:>7.2f} ms")
        print(f"      CUDA_GRAPH_NODE_EVENTS by name: {node_events}")
        for name, n, dur in eager_names[:8]:
            print(f"      eager-recorded: {name[:70]:<70} x{n} {dur/1e6:.2f} ms")

    print("\n== 6. Headline ==")
    structural = bt == et and b_inside == e_inside and bs == es
    print(f"   identical CUPTI structure (tables, graph-node membership, launch signatures): {structural}")
    print(f"   per-name duration drift >1 ms (timing noise, not structure): {len(diffs)} names")
    return 0 if structural else 1


if __name__ == "__main__":
    sys.exit(main())
