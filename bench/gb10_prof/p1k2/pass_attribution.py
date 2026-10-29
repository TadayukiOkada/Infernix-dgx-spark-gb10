#!/usr/bin/env python3
"""Attribute kernel families to main-model forwards vs draft-module passes."""
import collections
import re
import sqlite3

HEAD_RE = re.compile(r"projection_q(8|4)_kernel")
FAM_RE = re.compile(r"[A-Za-z0-9_]*_kernel")


def load(path):
    cur = sqlite3.connect(path).cursor()
    return [dict(dur=r[1] - r[0], name=r[2], gx=r[3], regs=r[4]) for r in cur.execute(
        "SELECT k.start, k.end, s.value, k.gridX, k.registersPerThread"
        " FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id = k.demangledName"
        " ORDER BY k.start")]


for db, label in [("node_r1n.sqlite", "full"), ("node_r1n_draft.sqlite", "draft")]:
    K = load(db)
    busy = sum(k["dur"] for k in K)
    passes = []
    prev = -1
    for i, k in enumerate(K):
        if not HEAD_RE.search(k["name"]):
            continue
        seg = K[prev + 1:i]
        gating = sum(1 for s in seg if "gdn_gating_kernel" in s["name"])
        passes.append(("main" if gating >= 12 else "draft", seg + [k]))
        prev = i
    trailing = K[prev + 1:]
    print("==", label, "busy %.1f ms" % (busy / 1e6), "passes", len(passes),
          "trailing kernels after last head", len(trailing))
    for kind in ("main", "draft"):
        agg = collections.defaultdict(lambda: [0, 0])
        total = 0
        for kk, seg in passes:
            if kk != kind:
                continue
            for s in seg:
                f = FAM_RE.search(s["name"]).group(0)
                agg[f][0] += 1
                agg[f][1] += s["dur"]
                total += s["dur"]
        print(f"  {kind}: {total / 1e6:.1f} ms ({100 * total / busy:.2f} % of busy)")
        for f, (n, d) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:12]:
            print(f"     {f:<36}{n:>6} {d / 1e6:>8.1f} ms {100 * d / busy:>6.2f}%")
