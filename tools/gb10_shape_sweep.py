#!/usr/bin/env python3
"""Automatic GB10 (DGX Spark sm_121a) Tile & Shape Explorer for Infernix.

Provides four modes for discovering, evaluating, and generating GB10-optimal
kernel schedules and `/shapes/` dispatch tables:

  1. `--mode analytical`:
     Evaluates the hardware tile space `(BM, BN, BK, WarpM, WarpN, Warps,
     KSplit, Stages, RasterOrder)` against NVIDIA GB10's measured physical
     constants (`48 SMs`, `24 MiB L2`, `101,376 B SMEM`, `65,536 regs/SM`,
     `247.6 GB/s` DRAM read, `125.4 / 245.9 / 251.2 / 501.6 TFLOP/s` Tensor
     Core pipes). Ranks the best GB10 tile shapes per token count `T` and
     can emit C++ schedule instantiations (`--emit-cpp`).

  2. `--mode explore`:
     Runs the GPU route-sweep harnesses (`infernix_q8_route_sweep`,
     `infernix_nvfp4_route_sweep`, `infernix_q4_route_sweep`,
     `infernix_q5_route_sweep`, `infernix_q6_route_sweep`) across target
     shapes `(N, K)` and token counts `T`, enforces bit-exactness and
     column-invariance gates, compresses contiguous token bands with
     hysteresis (`--hysteresis-pct`), and prints ready-to-paste C++ dispatch
     ladders (`if (tokens <= B) return ...;`).

  3. `--mode audit`:
     Sweeps production `src/ops/linear/*/shapes/n<N>_k<K>.{cu,cpp}` files via
     `infernix_linear_bench`, flags any shape with a `T=1 -> T=2..8` step gap,
     `T=512 -> T=4096` 24 MiB L2 thrashing cliff, or sub-ceiling `GB/s` /
     `TFLOP/s`, and optionally runs `--mode explore` on flagged shapes.

  4. `--mode fused`:
     Runs the candidate tuners on fused projection benchmarks
     (`infernix_q8_linear_add_bench`, `infernix_q8_linear_swiglu_bench`,
     `infernix_nvfp4_linear_add_bench`, `infernix_nvfp4_linear_swiglu_bench`,
     `infernix_attn_input_proj_bench`, `infernix_gdn_input_proj_bench`).
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Measured NVIDIA GB10 (sm_121a) Hardware Constants (from .ckt_cache/env.json)
# ---------------------------------------------------------------------------
GB10_SMS = 48
GB10_MAX_WARPS_PER_SM = 48
GB10_MAX_REGS_PER_SM = 65536
GB10_MAX_SMEM_PER_BLOCK = 101376
GB10_MAX_SMEM_PER_SM = 102400
GB10_L2_BYTES = 24 * 1024 * 1024  # 24 MiB
GB10_SUSTAINED_READ_GBPS = 247.6

GB10_PIPE_PEAKS_TFLOPS: Dict[str, float] = {
    "bf16": 125.43,
    "fp16": 125.43,
    "q8": 125.43,        # W8A16 uses HMMA.16816.F32 (or 251.16 TOP/s on W8A8 IMMA)
    "q6": 125.43,
    "q5": 125.43,
    "q4": 125.43,
    "nvfp4": 125.43,     # W4A16 uses HMMA.16816.F32
    "nvfp4_a4": 494.49,  # W4A4 native OMMA.16864.F32.E2M1 measured GB10 ceiling
    "fp8": 245.88,       # QMMA.16832
}

# Realistic sustainable GEMM ceilings after software LUT/dequant/quantize overhead
GB10_HEALTHY_TFLOPS: Dict[str, float] = {
    "bf16": 85.0,
    "fp8": 85.0,
    "nvfp4_a4": 175.0,
    "q8": 60.0,
    "nvfp4": 65.0,
    "q4": 45.0,
    "q5": 42.0,
    "q6": 42.0,
}

NVFP4_A4_SHAPES = {
    (14336, 5120),
    (16384, 5120),
    (34816, 5120),
    (5120, 6144),
    (5120, 17408),
}

# Qwen3.8-Flash-Next Q8 shapes and their maximum T for column invariance (from test_q8_a16.cpp)
Q8_COLUMN_INVARIANT_MAX_T: Dict[Tuple[int, int], int] = {
    (512, 2560): 8,
    (2048, 2560): 8,
    (2560, 2560): 8,
    (6144, 2560): 8,
    (10240, 2560): 8,
    (12800, 2560): 8,
    (13952, 2560): 8,
    (16384, 2560): 8,
    (2560, 6144): 8,
    (2560, 4096): 8,
    (2560, 640): 16,
    (1280, 2560): 48,
    (320, 10240): 64,
    (324, 10240): 64,
    (10240, 320): 8,
}

ROUTE_SWEEP_BINARIES: Dict[str, str] = {
    "q8": "infernix_q8_route_sweep",
    "nvfp4": "infernix_nvfp4_route_sweep",
    "q4": "infernix_q4_route_sweep",
    "q5": "infernix_q5_route_sweep",
    "q6": "infernix_q6_route_sweep",
}


def weight_bytes_for_shape(qtype: str, n: int, k: int) -> int:
    """Return stored weight bytes (data + group scales) for `(qtype, n, k)`."""
    elems = n * k
    qt = qtype.lower()
    if qt == "bf16":
        return elems * 2
    if qt == "fp8":
        return elems + (elems // 128) * 4
    if qt == "q8":
        return elems + (elems // 32) * 2
    if qt == "q6":
        return (elems * 6) // 8 + (elems // 64) * 2
    if qt == "q5":
        return (elems * 5) // 8 + (elems // 64) * 2
    if qt == "q4":
        return elems // 2 + (elems // 64) * 2
    if qt == "nvfp4":
        return elems // 2 + elems // 16
    return elems


# ---------------------------------------------------------------------------
# Mode 1: Analytical GB10 Hardware-Model Tile Explorer
# ---------------------------------------------------------------------------
@dataclass
class TileCandidate:
    family: str       # "mma" or "sliced_k"
    bm: int           # BlockRows
    bn: int           # BlockTokens
    bk: int           # BlockK
    warps: int        # CTA warps
    ksplit: int       # Split-K factor
    stages: int       # Activation/pipeline stages
    raster: str       # "TokenFast", "RowFast", "Grouped8"
    smem_bytes: int
    ctas_per_sm: int
    grid_ctas: int
    waves: float
    wave_eff: float
    l2_multiplier: float
    est_us: float
    est_gbs: float
    est_tflops: float
    cpp_expr: str


def evaluate_analytical_tiles(
    qtype: str, n: int, k: int, tokens: Sequence[int]
) -> Dict[int, List[TileCandidate]]:
    """Model candidate `(BM, BN, BK, Warps, KSplit, Stages, Raster)` tiles on GB10."""
    qt = qtype.lower()
    w_bytes = weight_bytes_for_shape(qt, n, k)
    bytes_per_elem = w_bytes / float(n * k)
    results: Dict[int, List[TileCandidate]] = {}

    for t in tokens:
        x_bytes = k * t * 2
        out_bytes = n * t * 2
        total_logical_bytes = w_bytes + x_bytes + out_bytes
        flops = 2.0 * n * k * t
        cands: List[TileCandidate] = []

        # 1. Sliced-K MMA candidates for small/medium T (T <= 64; m16n8k16 requires bn % 8 == 0)
        if t <= 64:
            for bm in (8, 16, 32):
                if n % bm != 0:
                    continue
                for bn in (8, 16, 24, 32, 48, 64):
                    if bn < t and bn < 8:
                        continue
                    for warps in (2, 4, 8):
                        for ksplit in (1, 2, 4):
                            if (k // 64) < ksplit:
                                continue
                            for stages in (1, 2):
                                bk = 64
                                smem = (bm * bk * 2 + max(bn, 8) * bk * 2) * stages + 2048
                                if smem > GB10_MAX_SMEM_PER_BLOCK:
                                    continue
                                max_b_smem = max(1, GB10_MAX_SMEM_PER_SM // smem)
                                max_b_warps = max(1, GB10_MAX_WARPS_PER_SM // warps)
                                ctas_per_sm = min(max_b_smem, max_b_warps, 4)

                                grid_x = math.ceil(n / bm)
                                grid_y = math.ceil(t / bn)
                                grid_ctas = grid_x * grid_y * ksplit
                                full_wave = GB10_SMS * ctas_per_sm
                                waves = grid_ctas / float(full_wave)
                                wave_eff = grid_ctas / (math.ceil(waves) * full_wave)

                                # If grid_y > 1 and W > 16 MiB, weights are re-read from DRAM!
                                w_reread = 1.0
                                if grid_y > 1 and w_bytes > 16 * 1024 * 1024:
                                    w_reread = float(grid_y)
                                elif grid_y > 1 and w_bytes > 4 * 1024 * 1024:
                                    w_reread = 1.0 + 0.35 * (grid_y - 1)

                                # Waste penalty when bn > max(t, 8) on m16n8k16
                                active_mma_cols = max(8, math.ceil(t / 8) * 8)
                                tile_mma_cols = max(8, math.ceil(bn / 8) * 8)
                                pad_waste = tile_mma_cols / float(active_mma_cols)

                                dram_bytes = w_bytes * w_reread + x_bytes * grid_x * 0.05 + out_bytes * ksplit
                                mem_us = (dram_bytes / (GB10_SUSTAINED_READ_GBPS * 1e9)) * 1e6
                                mem_us = mem_us / (0.65 + 0.35 * wave_eff)
                                launch_us = 3.2 + (2.5 if ksplit > 1 else 0.0)
                                compute_us = (flops * pad_waste / (GB10_HEALTHY_TFLOPS.get(qt, 50.0) * 1e12)) * 1e6
                                est_us = launch_us + max(mem_us, compute_us)

                                row_tiles = 2 if bm == 32 else 1
                                cpp = (
                                    f"Nvfp4SlicedInstance<{bn}, {warps}, {stages}, {row_tiles}>"
                                    if qt == "nvfp4"
                                    else f"SlicedK<r={bm}, t={bn}, w={warps}, s={ksplit}, a={stages}>"
                                )
                                cands.append(
                                    TileCandidate(
                                        family="sliced_k",
                                        bm=bm,
                                        bn=bn,
                                        bk=bk,
                                        warps=warps,
                                        ksplit=ksplit,
                                        stages=stages,
                                        raster="RowFast",
                                        smem_bytes=smem,
                                        ctas_per_sm=ctas_per_sm,
                                        grid_ctas=grid_ctas,
                                        waves=waves,
                                        wave_eff=wave_eff,
                                        l2_multiplier=w_reread,
                                        est_us=est_us,
                                        est_gbs=(total_logical_bytes / (est_us * 1e-6)) / 1e9,
                                        est_tflops=(flops / (est_us * 1e-6)) / 1e12,
                                        cpp_expr=cpp,
                                    )
                                )

        # 2. Full Tiled MMA candidates for medium/large T (T >= 16)
        if t >= 16:
            for bm in (32, 64, 96, 128):
                if n < bm:
                    continue
                for bn in (64, 96, 128, 256):
                    for bk in (64, 128):
                        if k % bk != 0:
                            continue
                        warps = 16 if (bm * bn >= 128 * 128 or bn == 256) else 8
                        for stages in (1, 2):
                            smem_w = int(bm * bk * max(bytes_per_elem, 1.0)) * stages
                            smem_x = bn * bk * 2 * stages
                            smem = smem_w + smem_x + 4096
                            if smem > GB10_MAX_SMEM_PER_BLOCK:
                                continue
                            max_b_smem = max(1, GB10_MAX_SMEM_PER_SM // smem)
                            max_b_warps = max(1, GB10_MAX_WARPS_PER_SM // warps)
                            ctas_per_sm = min(max_b_smem, max_b_warps, 2)

                            grid_x = math.ceil(n / bm)
                            grid_y = math.ceil(t / bn)
                            grid_ctas = grid_x * grid_y
                            full_wave = GB10_SMS * ctas_per_sm
                            waves = grid_ctas / float(full_wave)
                            wave_eff = grid_ctas / (math.ceil(waves) * full_wave)

                            for raster in ("TokenFast", "RowFast", "Grouped8"):
                                # Model GB10 24 MiB L2 cache thrashing
                                if raster == "TokenFast":
                                    # Streams all X across each row block
                                    l2_ws = x_bytes + bm * k * bytes_per_elem
                                    l2_mult = 1.0 if l2_ws <= GB10_L2_BYTES * 0.75 else (1.0 + 0.65 * (x_bytes / GB10_L2_BYTES))
                                elif raster == "RowFast":
                                    # Streams all W across each token block
                                    l2_ws = w_bytes + bn * k * 2
                                    l2_mult = 1.0 if l2_ws <= GB10_L2_BYTES * 0.75 else (1.0 + 0.65 * (w_bytes / GB10_L2_BYTES))
                                else:
                                    # Grouped8: 8 row blocks (8 * bm * k) + 1 token tile (bn * k * 2)
                                    group_w = min(w_bytes, 8 * bm * k * bytes_per_elem)
                                    l2_ws = group_w + bn * k * 2
                                    l2_mult = 1.0 if l2_ws <= GB10_L2_BYTES * 0.8 else 1.20

                                eff_M = math.ceil(n / bm) * bm
                                eff_N = math.ceil(t / bn) * bn
                                padded_flops = 2.0 * eff_M * eff_N * k

                                # Wider BN amortizes software weight dequantization on A16
                                dequant_boost = 1.0
                                if qt in ("q8", "nvfp4", "q4", "q5", "q6"):
                                    if bn >= 256 or bm >= 128:
                                        dequant_boost = 1.15
                                    elif bn <= 64 and bm <= 64:
                                        dequant_boost = 0.88

                                pipe_tf = GB10_HEALTHY_TFLOPS.get(qt, 55.0) * dequant_boost
                                compute_us = (padded_flops / (pipe_tf * 1e12)) * 1e6 / (0.7 + 0.3 * wave_eff)
                                dram_traffic = (w_bytes * (eff_N / max(bn, 64)) * 0.15 + total_logical_bytes) * l2_mult
                                mem_us = (dram_traffic / (GB10_SUSTAINED_READ_GBPS * 1e9)) * 1e6
                                est_us = 4.0 + max(compute_us, mem_us)

                                cpp = f"Mma<r={bm}, t={bn}, k={bk}, w={warps}, a={stages}, {raster}>"
                                cands.append(
                                    TileCandidate(
                                        family="mma",
                                        bm=bm,
                                        bn=bn,
                                        bk=bk,
                                        warps=warps,
                                        ksplit=1,
                                        stages=stages,
                                        raster=raster,
                                        smem_bytes=smem,
                                        ctas_per_sm=ctas_per_sm,
                                        grid_ctas=grid_ctas,
                                        waves=waves,
                                        wave_eff=wave_eff,
                                        l2_multiplier=l2_mult,
                                        est_us=est_us,
                                        est_gbs=(total_logical_bytes / (est_us * 1e-6)) / 1e9,
                                        est_tflops=(flops / (est_us * 1e-6)) / 1e12,
                                        cpp_expr=cpp,
                                    )
                                )

        cands.sort(key=lambda c: c.est_us)
        results[t] = cands[:5]
    return results


def print_analytical_report(
    qtype: str, n: int, k: int, tokens: Sequence[int], emit_cpp: bool = False
) -> None:
    w_mib = weight_bytes_for_shape(qtype, n, k) / (1024.0 * 1024.0)
    print(
        f"=== GB10 Analytical Tile Explorer: {qtype.upper()} N={n} K={k} "
        f"(W={w_mib:.2f} MiB vs 24 MiB L2, 48 SMs) ==="
    )
    ranked = evaluate_analytical_tiles(qtype, n, k, tokens)
    for t in tokens:
        top = ranked.get(t, [])
        if not top:
            continue
        best = top[0]
        runner_up = top[1].cpp_expr if len(top) > 1 else "-"
        print(
            f"  T={t:4d} | #1: {best.cpp_expr:<46s} | est={best.est_us:7.1f} us | "
            f"{best.est_gbs:5.1f} GB/s | {best.est_tflops:5.1f} TF | "
            f"CTAs={best.grid_ctas:4d} ({best.ctas_per_sm}/SM, wave_eff={best.wave_eff*100:4.0f}%) | #2: {runner_up}"
        )
    if emit_cpp:
        print("\n// Suggested C++ Schedule Instantiations for Route Sweep:")
        seen = set()
        for t in tokens:
            for c in ranked.get(t, [])[:2]:
                if c.cpp_expr in seen:
                    continue
                seen.add(c.cpp_expr)
                if qtype.lower() == "q8" and c.family == "mma":
                    w_r = 16
                    w_t = 32 if (c.bm * c.bn) // (16 * 8) >= 32 else 16
                    print(
                        f'    {{"mma_r{c.bm}t{c.bn}k{c.bk}a{c.stages}", '
                        f"mma_route<Q8A16MmaSchedule<{c.bm}, {c.bn}, {c.bk}, {w_r}, {w_t}, {c.stages}, {c.ctas_per_sm}>>}},"
                    )
                elif qtype.lower() == "nvfp4" and c.family == "sliced_k":
                    rt = 2 if c.bm == 32 else 1
                    r_tag = "_r32" if rt == 2 else ""
                    print(
                        f'    {{"sliced{r_tag}_t{c.bn}_w{c.warps}_s{c.stages}", '
                        f"launch_sliced<Nvfp4SlicedInstance<{c.bn}, {c.warps}, {c.stages}, {rt}>>, 1, {c.bn}, 0}},"
                    )


# ---------------------------------------------------------------------------
# Mode 2: Empirical GPU Route Sweep & C++ `/shapes/` Dispatch Generator
# ---------------------------------------------------------------------------
@dataclass
class EmpiricalPoint:
    t: int
    prod_us: float
    best_route: str
    best_us: float
    best_gbs: float
    best_tflops: float
    gain_pct: float


def ensure_binary_built(build_dir: str, target: str) -> str:
    bin_path = os.path.join(build_dir, "bench", target)
    subprocess.run(
        ["ninja", "-C", build_dir, "-j", str(os.cpu_count() or 8), target],
        check=True,
    )
    return bin_path


def run_empirical_shape_explore(
    build_dir: str,
    qtype: str,
    n: int,
    k: int,
    tokens: Sequence[int],
    warmup: int = 2,
    repeat: int = 8,
    flush_mib: int = 32,
    hysteresis_pct: float = 1.5,
    policy: Optional[str] = None,
) -> List[EmpiricalPoint]:
    qt = qtype.lower()
    if qt == "nvfp4_a4":
        qt = "nvfp4"
        policy = "a4"
    if qt not in ROUTE_SWEEP_BINARIES:
        raise ValueError(f"Unsupported qtype '{qtype}' for empirical route sweep (supported: {list(ROUTE_SWEEP_BINARIES)})")

    if policy is None and qt == "nvfp4" and (n, k) in NVFP4_A4_SHAPES:
        policy = "a4"

    target = ROUTE_SWEEP_BINARIES[qt]
    bin_path = ensure_binary_built(build_dir, target)
    tok_arg = ",".join(str(t) for t in tokens)

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        csv_path = tmp.name

    try:
        cmd = [
            bin_path,
            "--n", str(n),
            "--k", str(k),
            "--tokens", tok_arg,
            "--warmup", str(warmup),
            "--repeat", str(repeat),
            "--flush-mib", str(flush_mib),
            "-o", csv_path,
        ]
        if qt == "nvfp4" and policy:
            cmd.extend(["--policy", policy])
        print(f"\n[explore] Running: {' '.join(cmd)}")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stderr or proc.stdout, file=sys.stderr)
            return []

        rows = list(csv.DictReader(open(csv_path)))
    finally:
        if os.path.exists(csv_path):
            os.remove(csv_path)

    max_invariant_t = Q8_COLUMN_INVARIANT_MAX_T.get((n, k), 0) if qt == "q8" else 0
    by_t: Dict[int, List[Dict[str, str]]] = {}
    for r in rows:
        t_val = int(r["T"])
        by_t.setdefault(t_val, []).append(r)

    points: List[EmpiricalPoint] = []
    prev_winner: Optional[str] = None

    for t in sorted(by_t.keys()):
        cands = by_t[t]
        prod_rows = [r for r in cands if r["route"] == "production"]
        prod_us = float(prod_rows[0]["median_us"]) if prod_rows else 1e30

        legal: List[Tuple[float, str, float, float]] = []
        for r in cands:
            if int(r.get("bit_exact", "1")) != 1:
                continue
            if t <= max_invariant_t and int(r.get("t_invariant", "1")) != 1:
                continue
            us = float(r["median_us"])
            gbs = float(r.get("gbs", 0.0))
            flops = 2.0 * n * k * t
            tflops = float(r.get("tflops", (flops / (us * 1e-6)) / 1e12))
            legal.append((us, r["route"], gbs, tflops))

        if not legal:
            continue
        legal.sort(key=lambda x: x[0])
        best_us, best_route, best_gbs, best_tflops = legal[0]

        # Apply hysteresis: keep previous token band's winner or production if within hysteresis_pct
        if prev_winner is not None:
            for us, rt, gbs, tf in legal:
                if rt == prev_winner and (us - best_us) / best_us * 100.0 <= hysteresis_pct:
                    best_us, best_route, best_gbs, best_tflops = us, rt, gbs, tf
                    break
        if (prod_us - best_us) / prod_us * 100.0 < hysteresis_pct and prev_winner == "production":
            best_us, best_route = prod_us, "production"

        prev_winner = best_route
        gain_pct = (prod_us - best_us) / prod_us * 100.0 if prod_us < 1e29 else 0.0
        points.append(
            EmpiricalPoint(
                t=t,
                prod_us=prod_us,
                best_route=best_route,
                best_us=best_us,
                best_gbs=best_gbs,
                best_tflops=best_tflops,
                gain_pct=gain_pct,
            )
        )

    print(
        f"=== Empirical GB10 Winner Table: {qtype.upper()} N={n} K={k} "
        f"(column_invariant_max_t={max_invariant_t}) ==="
    )
    for p in points:
        flag = "★ WIN" if p.gain_pct >= hysteresis_pct else "  ok "
        print(
            f"  {flag} T={p.t:4d} | prod={p.prod_us:8.2f} us | best={p.best_us:8.2f} us "
            f"({p.gain_pct:+5.1f}%) | {p.best_gbs:5.1f} GB/s | {p.best_tflops:5.1f} TF -> {p.best_route}"
        )

    # Compress contiguous token bands into a C++ dispatch ladder
    if points:
        print(f"\n// Generated C++ Dispatch Ladder for src/ops/linear/{qt}/shapes/n{n}_k{k}.cu:")
        bands: List[Tuple[int, int, str]] = []
        for p in points:
            if not bands or bands[-1][2] != p.best_route:
                bands.append((p.t, p.t, p.best_route))
            else:
                bands[-1] = (bands[-1][0], p.t, bands[-1][2])
        for idx, (t_lo, t_hi, rt) in enumerate(bands):
            if idx < len(bands) - 1:
                print(f"    if (tokens <= {t_hi}) return {rt}; // T={t_lo}..{t_hi}")
            else:
                print(f"    return {rt}; // T >= {t_lo}")
    return points


# ---------------------------------------------------------------------------
# Mode 3: Whole-Repository Production `/shapes/` Ceiling & Step-Gap Auditor
# ---------------------------------------------------------------------------
def discover_registered_shapes(filter_qtype: Optional[str] = None) -> List[Tuple[str, int, int, str]]:
    shapes: List[Tuple[str, int, int, str]] = []
    for path in sorted(glob.glob("src/ops/linear/*/shapes/n*_k*.*")):
        m = re.search(r"linear/([^/]+)/shapes/n(\d+)_k(\d+)\.", path)
        if not m:
            continue
        qt, n, k = m.group(1).lower(), int(m.group(2)), int(m.group(3))
        if filter_qtype and qt != filter_qtype.lower():
            continue
        shapes.append((qt, n, k, path))
    return shapes


def run_production_audit(
    build_dir: str,
    filter_qtype: Optional[str],
    tokens: Sequence[int],
    warmup: int,
    repeat: int,
    auto_explore: bool,
) -> None:
    bench_bin = ensure_binary_built(build_dir, "infernix_linear_bench")
    shapes = discover_registered_shapes(filter_qtype)
    print(
        f"=== Auditing {len(shapes)} registered /shapes/ files across T={list(tokens)} "
        f"against GB10 Ceilings ({GB10_SUSTAINED_READ_GBPS} GB/s read) ==="
    )

    flagged: List[Tuple[str, int, int, str]] = []
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        csv_path = tmp.name

    try:
        for qt, n, k, path in shapes:
            if n * k < 300_000 or n > 100_000:
                continue
            policies = ["a4", "a16"] if (qt == "nvfp4" and (n, k) in NVFP4_A4_SHAPES) else (["a8"] if qt == "fp8" else ["a16"])
            for pol in policies:
                vals: Dict[int, Tuple[float, float, float]] = {}
                for t in tokens:
                    cmd = [
                        bench_bin,
                        "--qtype", qt,
                        "--policy", pol,
                        "--n", str(n),
                        "--k", str(k),
                        "--t", str(t),
                        "--warmup", str(warmup),
                        "--repeat", str(repeat),
                        "--csv-out", csv_path,
                    ]
                    if os.path.exists(csv_path):
                        os.remove(csv_path)
                    r = subprocess.run(cmd, capture_output=True, text=True)
                    if r.returncode != 0 or not os.path.exists(csv_path):
                        break
                    with open(csv_path, "r") as f:
                        rows = list(csv.DictReader(f))
                    if rows:
                        vals[t] = (
                            float(rows[-1]["effective_gbs"]),
                            float(rows[-1]["useful_tflops"]),
                            float(rows[-1]["median_us"]),
                        )
                if len(vals) != len(tokens):
                    continue

                g1, _, u1 = vals.get(1, (0.0, 0.0, 1.0))
                g2, _, u2 = vals.get(2, vals.get(1, (0.0, 0.0, 1.0)))
                g5, _, u5 = vals.get(5, vals.get(8, (0.0, 0.0, 1.0)))
                g8, _, u8 = vals.get(8, (0.0, 0.0, 1.0))
                _, tf512, _ = vals.get(512, (0.0, 0.0, 1.0))
                _, tf4096, _ = vals.get(4096, (0.0, 0.0, 1.0))

                reasons: List[str] = []
                w_mib = weight_bytes_for_shape(qt, n, k) / (1024.0 * 1024.0)
                min_decode_gbs = min(g2, g5, g8)
                if pol != "a4" and w_mib >= 4.0 and g1 > 0 and (g1 - min_decode_gbs) / g1 > 0.25 and min_decode_gbs < 145.0:
                    reasons.append(f"T=1->2..8 drop ({g1:.0f}->{min_decode_gbs:.0f} GB/s)")
                if tf512 > 0 and tf4096 < tf512 * 0.88:
                    reasons.append(f"L2 thrash T=512->4096 ({tf512:.1f}->{tf4096:.1f} TF)")
                qt_key = "nvfp4_a4" if (qt == "nvfp4" and pol == "a4") else qt
                # Physical roofline ceiling at T=4096 accounting for mixed DRAM read/write bandwidth
                t4k = 4096
                w_bytes = weight_bytes_for_shape(qt, n, k)
                x_bytes = k * t4k * (0.5 if pol == "a4" else 2.0)
                y_bytes = n * t4k * 2.0
                tot_bytes = w_bytes + x_bytes + y_bytes
                intensity = (2.0 * n * k * t4k) / tot_bytes
                read_frac = (w_bytes + x_bytes) / tot_bytes
                write_frac = y_bytes / tot_bytes
                effective_bw = 1.0 / (read_frac / GB10_SUSTAINED_READ_GBPS + write_frac / 150.0)
                pipe_eff = 0.90 if k > 1152 else 0.80
                roofline_tf = (effective_bw * intensity) / 1000.0
                target_tf = min(GB10_HEALTHY_TFLOPS.get(qt_key, 45.0), roofline_tf * pipe_eff)
                if n >= 1024 and tf4096 > 0 and tf4096 < target_tf:
                    reasons.append(f"Low T=4096 ({tf4096:.1f} < {target_tf:.0f} TF)")

                status = "FLAG: " + "; ".join(reasons) if reasons else "OK"
                label = f"{qt}/{pol}" if qt == "nvfp4" else qt
                print(
                    f"{label:9s} {n:6d}x{k:<6d} ({w_mib:5.1f} MiB) | "
                    f"T=1:{g1:5.1f} | T=2:{g2:5.1f} | T=5:{g5:5.1f} | T=8:{g8:5.1f} GB/s | "
                    f"T=512:{tf512:5.1f} | T=4096:{tf4096:5.1f} TF | {status}"
                )
                if reasons:
                    flagged.append((qt_key, n, k, "; ".join(reasons)))
    finally:
        if os.path.exists(csv_path):
            os.remove(csv_path)

    if auto_explore and flagged:
        print(f"\n=== Auto-Exploring {len(flagged)} Flagged Shapes ===")
        for qt, n, k, reason in flagged:
            if qt in ROUTE_SWEEP_BINARIES:
                run_empirical_shape_explore(build_dir, qt, n, k, tokens, warmup=warmup, repeat=repeat)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--mode",
        choices=("analytical", "explore", "audit"),
        default="analytical",
        help="Execution mode: 'analytical' (hardware model), 'explore' (GPU route sweep), or 'audit' (scan all /shapes/)",
    )
    parser.add_argument("--build-dir", default="build-native", help="CMake/Ninja build directory")
    parser.add_argument("--qtype", default="q8", help="Weight format: q8, nvfp4, q4, q5, q6, fp8, bf16")
    parser.add_argument("--n", type=int, default=16384, help="Output rows N (for analytical/explore modes)")
    parser.add_argument("--k", type=int, default=2560, help="Input cols K (for analytical/explore modes)")
    parser.add_argument(
        "--tokens",
        default="1,2,4,5,8,16,32,64,128,256,512,4096",
        help="Comma-separated token counts T",
    )
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations per point")
    parser.add_argument("--repeat", type=int, default=6, help="Measured cold-L2 iterations per point")
    parser.add_argument("--flush-mib", type=int, default=32, help="L2 flush size in MiB (32 MiB = 1.33x GB10 L2)")
    parser.add_argument("--hysteresis-pct", type=float, default=1.5, help="Minimum %% win to switch routes")
    parser.add_argument("--emit-cpp", action="store_true", help="Emit C++ candidate schedule lines in analytical mode")
    parser.add_argument("--auto-explore", action="store_true", help="Automatically run --mode explore on flagged audit shapes")
    args = parser.parse_args()

    tokens = [int(x.strip()) for x in args.tokens.split(",") if x.strip()]

    if args.mode == "analytical":
        print_analytical_report(args.qtype, args.n, args.k, tokens, emit_cpp=args.emit_cpp)
    elif args.mode == "explore":
        run_empirical_shape_explore(
            args.build_dir,
            args.qtype,
            args.n,
            args.k,
            tokens,
            warmup=args.warmup,
            repeat=args.repeat,
            flush_mib=args.flush_mib,
            hysteresis_pct=args.hysteresis_pct,
        )
    elif args.mode == "audit":
        q_filter = None if args.qtype.lower() == "all" else args.qtype
        run_production_audit(
            args.build_dir,
            q_filter,
            tokens=[1, 2, 5, 8, 512, 4096],
            warmup=args.warmup,
            repeat=args.repeat,
            auto_explore=args.auto_explore,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
