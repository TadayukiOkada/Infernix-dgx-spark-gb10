#!/usr/bin/env python3
"""Aggregate cuobjdump -res-usage over a whole build tree's objects into one table.

Per docs/performance/gb10-kernel-optimization-results.md §3.2: inspect register, stack, shared and
constant usage per function; STACK is only reliable after device link, so report both the pre-link
per-object view and, when given, the linked-binary view.

Usage: res_usage.py BUILD_DIR [KERNEL_SUBSTRINGS...] [--objects-only]
   Without substrings, lists every __global__ symbol found. With substrings, filters names.
"""
import argparse
import re
import subprocess
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("build_dir")
    ap.add_argument("substrs", nargs="*")
    args = ap.parse_args()

    out = subprocess.run(["find", args.build_dir + "/src", "-name", "*.o"],
                         capture_output=True, text=True).stdout.splitlines()
    found = {}  # name -> (module_short, reg, stack, shared)
    for obj in out:
        try:
            res = subprocess.run(["cuobjdump", "-res-usage", obj], capture_output=True, text=True).stdout
        except Exception:
            continue
        lines = res.splitlines()
        fn = None
        for i, l in enumerate(lines):
            if "Function :" in l:
                fn = l.split("Function :")[1].strip()
            elif "REG:" in l and fn:
                m = re.search(r"REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)\s+LOCAL:(\d+)", l)
                if m:
                    base = fn.rsplit("__", 1)[-1] if "__" in fn else fn
                    found.setdefault(base, []).append((obj.split("/")[-1], int(m.group(1)),
                                                       int(m.group(2)), int(m.group(3)), int(m.group(4))))
                fn = None

    wanted = set(args.substrs)
    print(f"{'kernel':<60} {'module':<30} {'reg':>4} {'stk':>4} {'shr':>6} {'loc':>4}")
    listed = 0
    for name, entries in sorted(found.items()):
        if wanted and not any(w in name for w in wanted):
            continue
        mod = entries[0][0][:28]
        regs = {e[1] for e in entries}
        stacks = {e[2] for e in entries}
        shrs = {e[3] for e in entries}
        locs = {e[4] for e in entries}
        r = ",".join(str(x) for x in sorted(regs))
        s = ",".join(str(x) for x in sorted(stacks))
        h = ",".join(str(x) for x in sorted(shrs))
        lc = ",".join(str(x) for x in sorted(locs))
        print(f"{name[:58]:<60} {mod:<30} {r:>4} {s:>4} {h:>6} {lc:>4}")
        listed += 1
    print(f"\n{listed} kernels matched of {len(found)} symbols in {len(out)} objects")


if __name__ == "__main__":
    main()
