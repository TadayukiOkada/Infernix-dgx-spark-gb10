#!/usr/bin/python3
"""Rank a route-sweep CSV per token width: production vs best bit-exact alternative."""
import csv
import sys
from collections import defaultdict

path = sys.argv[1]
rows = defaultdict(list)
for r in csv.DictReader(open(path)):
    if r["bit_exact"] != "1":
        continue
    rows[int(r["T"])].append((float(r["median_us"]), r["route"], float(r["gbs"])))

print(f"{'T':>4} {'production_us':>13} {'prod_route':<28} {'best_us':>8} "
      f"{'best_route':<28} {'gain%':>7} {'best_GBs':>9}")
for t in sorted(rows):
    cands = sorted(rows[t])
    prod = [c for c in cands if c[1] == "production"]
    best = cands[0]
    if not prod:
        continue
    p_us, p_route, _ = prod[0]
    gain = (p_us - best[0]) / p_us * 100.0
    star = "  <-- WIN" if gain > 5 and best[1] != "production" else ""
    print(f"{t:>4} {p_us:>13.2f} {p_route:<28} {best[0]:>8.2f} "
          f"{best[1]:<28} {gain:>7.1f} {best[2]:>9.1f}{star}")
