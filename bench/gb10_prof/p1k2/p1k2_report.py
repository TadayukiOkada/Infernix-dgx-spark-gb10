#!/usr/bin/env python3
"""Authoritative P1-K2 report generator.

Single source of truth for the numbers quoted in
docs/experiments/p1k2-route-width-histogram.md. Run against a node-mode nsys
sqlite capture produced with:

  nsys profile -t cuda,nvtx --cuda-graph-trace=node -c cudaProfilerApi \
      --capture-range-end=stop -o OUT --force-overwrite=true --export=sqlite \
      ./build-native/bench/infernix_bench ... -r 1 --profile-measured

Units are explicit everywhere: sqlite kernel timestamps are nanoseconds.
"""
import collections
import re
import statistics
import sys

BW = 273e9  # GB10 advertised system-memory bandwidth, bytes/s


def base(name: str) -> str:
    """Strip return type, namespaces, unnamed scope and argument lists."""
    n = name
    if n.startswith("void "):
        n = n[5:]
    n = re.sub(r"\(.*\)$", "", n)
    n = n.replace("(anonymous namespace)::", "").replace("::", ".")
    n = re.sub(r"<unnamed>::?", "", n)
    return n


def template_of(name: str) -> str:
    """First template argument list, e.g. <1>, <Q4>, <872>."""
    m = re.search(r"<([^<>]+)>", name)
    return m.group(1) if m else ""


def load(path):
    import sqlite3
    cur = sqlite3.connect(path).cursor()
    rows = list(cur.execute(
        "SELECT k.start, k.end, s.value, k.gridX, k.gridY, k.gridZ, k.blockX,"
        " k.staticSharedMemory, k.dynamicSharedMemory, k.registersPerThread,"
        " k.localMemoryPerThread, k.localMemoryTotal"
        " FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id = k.demangledName"
        " ORDER BY k.start"))
    out = []
    for (start, end, name, gx, gy, gz, bx, ss, ds, regs, lmt, lmtot) in rows:
        b = base(name)
        out.append(dict(start=start, end=end, dur=end - start, name=name, base=b,
                        tmpl=template_of(name), grid=(gx, gy, gz), block=bx,
                        static_shared=ss, dynamic_shared=ds, registers=regs,
                        local_per_thread=lmt, local_total=lmtot))
    return out


HEAD = "projection_q8_kernel"
HEAD_Q4 = "projection_q4_kernel"


def is_head(k):
    return HEAD in k["base"] or HEAD_Q4 in k["base"]


def widths(kernels):
    """Decode width from the head launch's dynamic shared memory: columns * K * 2 bytes."""
    hist = collections.Counter()
    for k in kernels:
        if is_head(k):
            hist[k["dynamic_shared"] // 5120] += 1
    return hist


def fam_key(name):
    """Kernel identifier plus its first template argument list, e.g. gate_up_kernel<1>."""
    m = re.search(r"([A-Za-z0-9_]*_kernel)(?:<([^<>]*)>)?", name)
    if not m:
        return name
    return m.group(1) + (f"<{m.group(2)}>" if m.group(2) else "")


def families(kernels):
    agg = collections.defaultdict(lambda: dict(calls=0, total=0, durs=[]))
    for k in kernels:
        a = agg[fam_key(k["name"])]
        a["calls"] += 1
        a["total"] += k["dur"]
        a["durs"].append(k["dur"])
    busy = sum(k["dur"] for k in kernels)
    rows = []
    for name, a in agg.items():
        rows.append((name, a["calls"], a["total"] / 1e6, 100.0 * a["total"] / busy,
                     statistics.median(a["durs"]) / 1e3, min(a["durs"]) / 1e3, max(a["durs"]) / 1e3))
    rows.sort(key=lambda r: -r[2])
    return rows, busy


def head_instances(kernels):
    inst = collections.defaultdict(list)
    for k in kernels:
        if is_head(k):
            inst[(k["base"], k["grid"][0], k["dynamic_shared"], k["registers"],
                  k["local_per_thread"])].append(k["dur"])
    return inst


def head_rows(kernels):
    """Split the trace at every lm_head call and classify each preceding segment.

    A main-model forward contains the model's GDN gating kernels (one per gated
    delta-net layer); a draft-module pass contains none of them. Validated against
    JSON round counts: 12 or more gdn_gating_kernel records in the segment means a
    main forward.
    """
    passes = []
    prev = -1
    for i, k in enumerate(kernels):
        if not is_head(k):
            continue
        seg = kernels[prev + 1:i]
        gating = sum(1 for s in seg if "gdn_gating_kernel" in s["name"])
        moe = sum(1 for s in seg if "gate_up_kernel" in s["name"] or "down_kernel" in s["name"])
        kind = "main" if gating >= 12 else "draft"
        passes.append(dict(kind=kind, head=k, seg_kernels=len(seg),
                           seg_ms=sum(s["dur"] for s in seg) / 1e6, gating=gating, moe=moe))
        prev = i
    return passes


def report(path, label, drafted_per_rep=None):
    ks = load(path)
    busy = sum(k["dur"] for k in ks)
    span = ks[-1]["end"] - ks[0]["start"]
    gaps = [ks[i + 1]["start"] - ks[i]["end"] for i in range(len(ks) - 1)]
    big = [g for g in gaps if g > 20_000]
    print("=" * 78)
    print(f"{label}: {path}")
    print(f"  kernel records      {len(ks)}")
    print(f"  busy (sum of durations)   {busy / 1e6:.2f} ms")
    print(f"  span (first start..last end) {span / 1e6:.2f} ms")
    print(f"  idle inside span       {100.0 * (1 - busy / span):.2f} %")
    print(f"  gaps > 20 us           {len(big)}  total {sum(big) / 1e6:.1f} ms  max {max(big) / 1e3:.1f} us")

    passes = head_rows(ks)
    main = [p for p in passes if p["kind"] == "main"]
    draft = [p for p in passes if p["kind"] == "draft"]
    hist = collections.Counter(p["head"]["dynamic_shared"] // 5120 for p in main)
    draft_hist = collections.Counter(p["head"]["dynamic_shared"] // 5120 for p in draft)
    print(f"  main-model forwards   {len(main)}   widths {dict(sorted(hist.items()))}")
    print(f"  draft-module head calls {len(draft)}  widths {dict(sorted(draft_hist.items()))}")
    if main:
        print(f"  main seg kernels {min(p['seg_kernels'] for p in main)}-"
              f"{max(p['seg_kernels'] for p in main)}, seg ms "
              f"{min(p['seg_ms'] for p in main):.2f}-{max(p['seg_ms'] for p in main):.2f}")
    if draft:
        print(f"  draft seg kernels {min(p['seg_kernels'] for p in draft)}-"
              f"{max(p['seg_kernels'] for p in draft)}, seg ms "
              f"{min(p['seg_ms'] for p in draft):.2f}-{max(p['seg_ms'] for p in draft):.2f}"
              f", gating {set(p['gating'] for p in draft)}, moe {set(p['moe'] for p in draft)}")
    mean_w = sum(t * c for t, c in hist.items()) / max(1, len(main))
    print(f"  mean main verify width {mean_w:.3f}")
    if drafted_per_rep is not None:
        closure = sum((t - 1) * c for t, c in hist.items())
        print(f"  closure sum(T-1)*count = {closure}  vs JSON drafted_tokens {drafted_per_rep}")

    print("\n  head instances (base, grid.x, dynS bytes, regs, local/thread): calls total_ms min/med/max us")
    for key, durs in sorted(head_instances(ks).items()):
        b, gx, ds, regs, lmt = key
        print(f"    {b:<20} grid {gx:>6} dynS {ds:>6} regs {regs:>3} local/thr {lmt:>4}  n={len(durs):>4}"
              f" total {sum(durs) / 1e6:>7.1f} ms  {min(durs) / 1e3:>7.1f} /"
              f" {statistics.median(durs) / 1e3:>7.1f} / {max(durs) / 1e3:>7.1f} us")

    head_total = sum(k["dur"] for k in ks if is_head(k))
    print(f"\n  total head cost {head_total / 1e6:.1f} ms = {100.0 * head_total / busy:.2f} % of busy")

    print("\n  family table (share of busy kernel time)")
    print(f"    {'kernel':<34}{'calls':>7}{'total ms':>10}{'share':>8}{'min us':>9}{'med us':>9}{'max us':>9}")
    rows, _ = families(ks)
    for name, calls, total_ms, share, med, mn, mx in rows[:26]:
        print(f"    {name:<34}{calls:>7}{total_ms:>10.1f}{share:>7.2f}%{mn:>9.1f}{med:>9.1f}{mx:>9.1f}")

    # MoE instance detail: raw specialization + launch geometry.
    print("\n  MoE instances (raw name, grid, dynS, regs): calls avg us")
    moe = collections.defaultdict(list)
    for k in ks:
        if "gate_up_kernel" in k["base"] or "down_kernel" in k["base"]:
            key = (k["name"], k["grid"], k["dynamic_shared"], k["registers"])
            moe[key].append(k["dur"])
    for (name, grid, ds, regs), durs in sorted(moe.items(), key=lambda kv: -len(kv[1])):
        short = re.sub(r"^void |\(.*\)$", "", name)
        short = short.replace("(anonymous namespace)::", "").replace("<unnamed>::", "")
        print(f"    n={len(durs):>4} grid {grid} dynS {ds:>6} regs {regs:>3}"
              f" avg {statistics.mean(durs) / 1e3:>7.1f} us  {short[:60]}")

    # Bandwidth model for the heads actually launched.
    print("\n  head bandwidth model (codes + scales streamed once per call)")
    seen = {}
    for k in ks:
        if is_head(k):
            seen.setdefault((k["base"], k["dynamic_shared"]), []).append(k["dur"])
    for (b, ds), durs in sorted(seen.items()):
        # q8_g32_fp16: 1 byte/code, one FP16 scale per 32 values.
        # q4_g64_fp16: 4 bits/code, one FP16 scale per 64 values.
        # See src/ops/projection_fp32/projection_fp32.cu:258-260 and :320-322.
        rows_n, bits, group = (131072, 4, 64) if HEAD_Q4 in b else (248320, 8, 32)
        codes = rows_n * 2560 * bits / 8
        scales = rows_n * (2560 // group) * 2
        floor = (codes + scales) / BW
        med = statistics.median(durs) / 1e9
        print(f"    {b:<20} rows {rows_n:>7} T={ds // 5120} traffic {(codes + scales) / 1e6:>7.1f} MB"
              f" floor {floor * 1e3:>6.3f} ms  measured {med * 1e3:>6.3f} ms"
              f"  {(codes + scales) / med / 1e9:>6.1f} GB/s = {100 * ((codes + scales) / med) / BW:>5.1f} % of 273")


if __name__ == "__main__":
    report("node_r1n.sqlite", "FULL proposal head (retained comparison config)", 96)
    report("node_r1n_draft.sqlite", "OPTIMIZED proposal head (--lm-head-draft)", 97)
