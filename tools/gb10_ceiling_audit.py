#!/usr/bin/python3
"""Step 2 of docs/gb10_roadmap.md section 5: audit active-model routes against measured
GB10 pipe ceilings and flag low GB/s (decode) / low TFLOP/s (prefill) candidates.

Only shapes bound by the active .infernix artifacts are swept (see tools/gb10_active_inventory.py).
"""
import csv
import os
import subprocess
import sys

BENCH = "./build-native/bench/infernix_linear_bench"
OUT = os.environ.get("AUDIT_OUT", "/tmp/gb10_audit")
FLUSH = "32"
REPEAT = "12"
WARMUP = "2"

# (qtype, policy, N, K, calls per forward pass, label) -- from the verified artifact inventory.
SHAPES = [
    ("Q8",    "a16", 10240,   320, 98, "hc_up"),
    ("Q8",    "a16",   324, 10240, 98, "hc_down"),
    ("Q8",    "a16",  1280,  2560, 49, "shared_gate_up"),
    ("Q8",    "a16",  2560,   640, 49, "shared_down"),
    ("Q8",    "a16", 16384,  2560, 36, "gdn_in_proj"),
    ("Q8",    "a16",  2560,  6144, 36, "gdn_out_proj"),
    ("Q8",    "a16",  2560,  2560,  1, "q8_2560x2560"),
    ("Q8",    "a16", 248320, 2560,  1, "head_q8"),
    ("NVFP4", "a16", 16384,  2560, 36, "nvfp4_gdn_in"),
    ("NVFP4", "a16",  2560,  6144, 36, "nvfp4_gdn_out"),
    ("NVFP4", "a16", 13952,  2560, 12, "nvfp4_13952"),
    ("NVFP4", "a4",  16384,  2560,  0, "nvfp4_a4_gdn_in"),
    ("BF16",  "a16", 10240,   320, 96, "iq3s_hc_up"),
    ("BF16",  "a16",   324, 10240, 96, "iq3s_hc_down"),
    ("BF16",  "a16", 248320, 2560,  1, "head_bf16"),
    ("BF16",  "a16",  1280,  2560, 12, "bf16_1280x2560"),
    ("BF16",  "a16",   513,  2560, 49, "bf16_513x2560"),
    ("FP8",   "a8",  16384,  5120,  0, "qwen35_fp8_a8"),
    ("FP8",   "a8",   5120, 17408,  0, "qwen35_fp8_down"),
]

DECODE_DENSE = "1:16:1"
DECODE_EXTRA = [32, 64]
PREFILL = [512, 1024, 4096]


def run(qtype, policy, n, k, tokens_spec, tag):
    csv_path = f"{OUT}/{tag}.csv"
    if os.path.exists(csv_path):
        return csv_path
    cmd = [BENCH, "--qtype", qtype, "--policy", policy, "--n", str(n), "--k", str(k),
           "--repeat", REPEAT, "--warmup", WARMUP, "--flush-mib", FLUSH,
           "--execution", "graph", "--csv-out", csv_path]
    if ":" in tokens_spec:
        cmd += ["--sweep", tokens_spec]
    else:
        cmd += ["--t", tokens_spec]
    try:
        subprocess.run(cmd, check=False, timeout=1800,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        print(f"!! timeout {tag}", flush=True)
        return None
    return csv_path if os.path.exists(csv_path) else None


def main():
    os.makedirs(OUT, exist_ok=True)
    for qtype, policy, n, k, calls, label in SHAPES:
        specs = [(f"n{n}_k{k}", DECODE_DENSE)]
        specs += [(f"n{n}_k{k}", str(t)) for t in DECODE_EXTRA]
        if (n, k) != (248320, 2560):
            specs += [(f"n{n}_k{k}", str(t)) for t in PREFILL]
        for _, spec in specs:
            tag = f"{qtype}_{policy}_{label}_{spec.replace(':', '_')}"
            print(f"sweep {qtype}/{policy} N={n} K={k} T={spec}", flush=True)
            run(qtype, policy, n, k, spec, tag)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
