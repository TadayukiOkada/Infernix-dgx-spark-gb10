#!/usr/bin/python3
"""Step 2 scoring: compare each audited route's measured eff_GB/s and TFLOP/s against the
measured GB10 pipe ceilings in docs/gb10_roadmap.md section 5 and flag low candidates.

Decode regime (T=1..8) is scored against the sustained LPDDR5X read ceiling; the prefill regime
(T=512..4096) is scored against the Tensor Core pipe ceiling for the qtype/policy.
"""
import csv
import glob
import os
import sys
from collections import defaultdict

SUSTAINED_READ_GBS = 246.0
# TFLOP/s flags from the roadmap table, per qtype/policy.
TC_FLAG = {
    ("BF16", "a16"): 70.0,
    ("Q8", "a16"): 52.0,
    ("Q5", "a16"): 45.0,
    ("Q6", "a16"): 45.0,
    ("Q4", "a16"): 45.0,
    ("NVFP4", "a16"): 45.0,
    ("NVFP4", "a4"): 90.0,
    ("FP8", "a8"): 75.0,
    ("FP8", "a16"): 52.0,
    ("Q4", "a8"): 65.0,
}
TC_PEAK = {("BF16", "a16"): 125.4, ("Q8", "a16"): 125.4, ("NVFP4", "a16"): 125.4,
           ("NVFP4", "a4"): 501.6, ("FP8", "a8"): 245.9, ("Q4", "a16"): 125.4,
           ("Q5", "a16"): 125.4, ("Q6", "a16"): 125.4, ("FP8", "a16"): 125.4,
           ("Q4", "a8"): 251.2}

SRC = sys.argv[1] if len(sys.argv) > 1 else "/tmp/gb10_audit"
shape_rows = defaultdict(dict)

for path in sorted(glob.glob(os.path.join(SRC, "*.csv"))):
    for r in csv.DictReader(open(path)):
        qtype, pol = r["qtype"], r["policy"].lower()
        key = (qtype, pol, int(r["N"]), int(r["K"]))
        shape_rows[key][int(r["T"])] = r

print(f"{'qtype/pol':<12} {'N':>7} {'K':>6} | decode T1..8 GB/s (flag <170) | "
      f"prefill TFLOP/s T512/T1024/T4096 (flag) | 4096/512 | verdict")
for key in sorted(shape_rows):
    qtype, pol, n, k = key
    rows = shape_rows[key]
    dec = [(t, rows[t]) for t in sorted(rows) if 1 <= t <= 8 and "effective_gbs" in rows[t]]
    wbytes = int(dec[0][1]["weight_bytes"]) if dec else 0
    if not dec:
        continue
    dmin = min(float(r["effective_gbs"]) for _, r in dec)
    dec_flag = dmin < 170.0 and wbytes >= 4 << 20
    pre = {t: rows.get(t) for t in (512, 1024, 4096)}
    tf = {t: (float(r["useful_tflops"]) if r and r["useful_tflops"] else None)
          for t, r in pre.items()}
    thr = TC_FLAG.get((qtype, pol))
    drop = None
    if tf[512] and tf[4096]:
        drop = (tf[4096] - tf[512]) / tf[512] * 100.0
    tf_flag = (thr is not None and any(v is not None and v < thr for v in tf.values())) or \
              (drop is not None and drop < -15.0)
    reasons = []
    if dec_flag:
        reasons.append(f"DECODE {dmin:.0f} GB/s")
    if tf_flag:
        if thr and any(v is not None and v < thr for v in tf.values()):
            lo = min(v for v in tf.values() if v is not None)
            reasons.append(f"PREFILL {lo:.1f} TF/s < {thr:.0f}")
        if drop is not None and drop < -15.0:
            reasons.append(f"L2-THRASH {drop:+.1f}% 512->4096")
    verdict = "LOW " + "; ".join(reasons) if reasons else "ok"
    ts = " ".join(f"{t}:{tf[t]:.1f}" for t in (512, 1024, 4096) if tf[t])
    print(f"{qtype+'/'+pol:<12} {n:>7} {k:>6} | {dmin:>6.1f} min ({'LOW' if dec_flag else 'ok'}) | "
          f"{ts:<40} | "
          f"{('%+.1f%%' % drop) if drop is not None else '-':>8} | {verdict}")
