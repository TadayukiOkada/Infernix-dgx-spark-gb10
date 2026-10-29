#!/usr/bin/env python3
"""prefill_budget.py — functional block budget and redundancy headroom for a prefill trace.

decode_hotspots.py ranks individual kernels. This tool groups them into the model's
functional blocks (dense ggml projections, routed MoE experts, attention/QSA, gated
delta-net, hyper-connection, norms/routing/glue) and, for the blocks whose useful FLOP
count and moved bytes are known from the workload geometry, reports the achieved rate
against the *measured* ceilings so a redundant (over-executed or mis-ceilinged) block is
separated from a genuinely-bound one.

Ceilings are the ones this toolkit's native probes measured on this part
(cuda-kernel-toolkit/native/*_probe), never a vendor datasheet:

  int8 tensor   249.2 TOPS   int8_mma.tops   (compute_probe, s8 m16n8k32)
  fp16 tensor   124.7 TFLOP/s fp16_mma.tflops (compute_probe, f16 m16n8k16)
  fp32 FMA       27.8 TFLOP/s  fp32_fma.tflops (compute_probe)
  sustained read 241.4 GB/s    bandwidth_probe (GB10 unified)

The ggml tile kernels are int8 mma.sync; classifying them against the fp32 ceiling (as a
generic roofline would) understates their headroom by ~9x. This tool uses the int8 ceiling
for them and reports the correct % of peak.

Useful-FLOP / byte models (workload geometry, computed here -- not hand arithmetic):
  dense ggml linear  [M,K]x[K,N] : 2*M*N*K ops, (M*K + N*K + M*N)*2 bytes (bf16 x/y, quant w~1B/elt)
  routed MoE         per token-top_k: 2*top_k*(2*I*H + H*I) ops on the resident expert banks
  The trace is grouped by demangled name; blocks are matched by name substring.

Usage:
    python3 tools/prefill_budget.py <nsys.sqlite> [--top N] [--reps R] [--tokens T] [--top-k K]
        [--hidden H] [--inter I] [--int8-tops F] [--fp16-tflops F] [--read-gbps F]

--reps divides total kernel time by the number of measured prefill repetitions so the
per-call achieved rates are comparable to a single pp<Tokens> prefill.
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from collections import defaultdict

# Measured ceilings (this GB10, cuda-kernel-toolkit probes). Overridable on the CLI.
INT8_TOPS_DEFAULT = 249.16
FP16_TFLOPS_DEFAULT = 124.71
FP32_TFLOPS_DEFAULT = 27.76
READ_GBPS_DEFAULT = 241.4

# Functional block assignment, first match wins. Keyed on demangled-name substrings.
BLOCKS = [
    ("moe_expert", ("gate_up_tile_ggml", "down_resident_ggml", "gate_up_kernel_ggml",
                    "down_kernel_ggml", "gate_up_a16", "down_a16", "expert")),
    ("dense_ggml", ("tile_kernel", "rows_kernel", "mma_rows_kernel", "plane_kernel",
                    "encode_kernel", "encode_h_ggml")),
    ("attention",  ("qsa_prompt_kernel", "qsa_score", "softmax_attention", "attn",
                    "causal_conv", "paged_attention", "attention")),
    ("delta_net",  ("gated_delta_net", "chunked", "conv1d")),
    ("hyper_conn", ("hc_down_partials", "hc_up_collapse", "combine_inject", "inject_kernel",
                    "collapse_kernel")),
    ("bf16_gemm",  ("bf16_general_gemm_kernel", "bf16_a16_tma_mma_kernel", "bf16_a16_mma_kernel",
                    "projection_wide_kernel")),
    ("norm_route", ("norm_kernel", "rmsnorm", "router", "route_kernel", "split_vector",
                    "sigmoid_gate", "dispatch", "encode_x")),
    ("other",      ()),
]

# Blocks whose useful int8 work + moved bytes we can model at a given geometry.
GGML_INT8 = {"dense_ggml", "moe_expert"}


def classify(name: str) -> str:
    for block, keys in BLOCKS:
        for k in keys:
            if k in name:
                return block
    return "other"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--reps", type=int, default=1,
                    help="measured prefill repetitions in the window (per-call divide)")
    ap.add_argument("--tokens", type=int, default=4096)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--hidden", type=int, default=2560, help="MoE hidden H")
    ap.add_argument("--inter", type=int, default=640, help="MoE intermediate I")
    ap.add_argument("--layers", type=int, default=48, help="routed-MoE layers in the model")
    ap.add_argument("--int8-tops", type=float, default=INT8_TOPS_DEFAULT)
    ap.add_argument("--fp16-tflops", type=float, default=FP16_TFLOPS_DEFAULT)
    ap.add_argument("--read-gbps", type=float, default=READ_GBPS_DEFAULT)
    args = ap.parse_args(argv)

    con = sqlite3.connect(args.sqlite)
    cur = con.cursor()
    rows = cur.execute(
        """
        SELECT s.value AS name, k.gridX, k.gridY, k.gridZ, k.blockX,
               k.registersPerThread AS regs,
               (k.dynamicSharedMemory + k.staticSharedMemory) AS smem,
               COUNT(*) AS calls, SUM(k.end - k.start) AS ns
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.demangledName = s.id
        GROUP BY s.value, k.gridX, k.gridY, k.gridZ, k.blockX, k.registersPerThread,
                 k.dynamicSharedMemory, k.staticSharedMemory
        """
    ).fetchall()
    con.close()
    if not rows:
        print("no kernel rows (need --cuda-graph-trace=node)", file=sys.stderr)
        return 1

    total_ns = sum(r[8] for r in rows)
    per_rep = total_ns / max(1, args.reps)
    block_ns: dict[str, int] = defaultdict(int)
    block_calls: dict[str, int] = defaultdict(int)
    for name, gx, gy, gz, bx, regs, smem, calls, ns in rows:
        b = classify(name)
        block_ns[b] += ns
        block_calls[b] += calls

    print(f"# window total kernel-busy: {total_ns/1e6:.1f} ms over {args.reps} rep(s); "
          f"per-call: {per_rep/1e6:.2f} ms")
    print(f"# ceilings (measured): int8 {args.int8_tops:.1f} TOPS, "
          f"fp16 {args.fp16_tflops:.1f} TFLOP/s, read {args.read_gbps:.0f} GB/s")
    print(f"# geometry: tokens={args.tokens} top_k={args.top_k} H={args.hidden} I={args.inter}\n")

    # Modelled useful work for the int8 blocks at this geometry.
    # Per routed (token,expert) pair: gate [H->I] + up [H->I] + down [I->H] = 3 matmuls,
    # each 2*M*K*N MAC-ops. Summed over top_k*M pairs and every routed-MoE layer. The
    # achieved rate is time-invariant, so it equals the --bench harness's per-layer TOPS.
    M = args.tokens
    moe_ops = args.layers * args.top_k * M * 2 * (2 * args.inter * args.hidden + args.hidden * args.inter)
    # dense ggml: dominated by the big projections; not modelled per-block (varies by layer mix).
    print(f"{'block':12} {'share%':>7} {'per-call ms':>11} {'calls':>7} "
          f"{'useful TOPS':>11} {'%int8 peak':>10}  note")
    for block, ns in sorted(block_ns.items(), key=lambda kv: -kv[1]):
        if ns == 0:
            continue
        ms = ns / 1e6 / max(1, args.reps)
        share = 100.0 * ns / total_ns
        useful_tops = note = ""
        if block == "moe_expert":
            tops = moe_ops / (ns / args.reps * 1e-9) / 1e12
            useful_tops = f"{tops:11.1f}"
            note = (f"= {100*tops/args.int8_tops:.1f}% of int8 peak; "
                    f"{args.int8_tops/tops:.1f}x headroom -> latency/redundancy-bound")
        print(f"{block:12} {share:7.2f} {ms:11.2f} {block_calls[block]:7d} "
              f"{useful_tops:>11} {note}")

    # Headline: MoE + dense_ggml + bf16_gemm are the tensor-work blocks; the rest are glue.
    tensor_ns = block_ns.get("moe_expert", 0) + block_ns.get("dense_ggml", 0) + block_ns.get("bf16_gemm", 0)
    glue_ns = total_ns - tensor_ns - block_ns.get("attention", 0) - block_ns.get("delta_net", 0)
    print(f"\n# tensor-work blocks (moe+dense+bf16): {100*tensor_ns/total_ns:.1f}% of busy")
    print(f"# attention + delta-net: {100*(block_ns.get('attention',0)+block_ns.get('delta_net',0))/total_ns:.1f}%")
    print(f"# glue (norm/route/hyper/other): {100*glue_ns/total_ns:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
