"""kernel_math — exact analytical calculators for CUDA kernel tuning on GB10.

A core rule of ``cuda-kernel-toolkit`` is that **an LLM agent or human engineer
must never guess or perform hand arithmetic** when sizing tiles, checking
register/shared-memory step thresholds, computing wave quantization, evaluating
GGML quantized byte traffic, diagnosing bank conflicts, or judging A/B speedups.

Every function in this module takes explicit parameters (plus ``DeviceInfo``)
and returns a deterministic, JSON-serialisable dictionary with all intermediate
and final numbers pre-calculated.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from .environment import DeviceInfo
from .performance_model import (
    ceil_div,
    occupancy_model,
    roofline_report,
    round_up,
    wave_analysis,
)
from .util import mean, percentile, stdev


# --------------------------------------------------------------------------- #
# Quantized & standard tensor format specifications (block_elements, bytes)
# --------------------------------------------------------------------------- #

FORMAT_SPECS: dict[str, tuple[int, int, str]] = {
    # (elements_per_block, bytes_per_block, description)
    "fp32": (1, 4, "IEEE 754 float32 (4 B/elem)"),
    "f32": (1, 4, "IEEE 754 float32 (4 B/elem)"),
    "fp16": (1, 2, "IEEE 754 float16 (2 B/elem)"),
    "f16": (1, 2, "IEEE 754 float16 (2 B/elem)"),
    "bf16": (1, 2, "bfloat16 (2 B/elem)"),
    "fp8": (1, 1, "FP8 E4M3/E5M2 (1 B/elem)"),
    "int8": (1, 1, "signed 8-bit integer (1 B/elem)"),
    "i8": (1, 1, "signed 8-bit integer (1 B/elem)"),
    "int32": (1, 4, "signed 32-bit integer (4 B/elem)"),
    "i32": (1, 4, "signed 32-bit integer (4 B/elem)"),
    "int4": (2, 1, "packed 4-bit integer (0.5 B/elem)"),
    "mxfp4": (32, 17, "OCP MXFP4 (32 elems + 1 E8M0 scale = 17 B / 32 elems, 4.25 bpw)"),
    "nvfp4": (16, 9, "NVIDIA NVFP4 (16 elems + 1 UE4M3 scale = 9 B / 16 elems, 4.50 bpw)"),
    "fp8_e4m3": (1, 1, "FP8 E4M3 (1 B/elem)"),
    "fp8_e5m2": (1, 1, "FP8 E5M2 (1 B/elem)"),
    # Infernix native group-quantized formats
    "q4": (64, 34, "Infernix Q4_G64_FP16 (64 elems + fp16 scale = 34 B / 64 elems, 4.25 bpw)"),
    "q4_g64": (64, 34, "Infernix Q4_G64_FP16 (64 elems + fp16 scale = 34 B / 64 elems, 4.25 bpw)"),
    "q5": (64, 42, "Infernix Q5_G64_FP16 (64 elems + fp16 scale = 42 B / 64 elems, 5.25 bpw)"),
    "q5_g64": (64, 42, "Infernix Q5_G64_FP16 (64 elems + fp16 scale = 42 B / 64 elems, 5.25 bpw)"),
    "q6": (64, 50, "Infernix Q6_G64_FP16 (64 elems + fp16 scale = 50 B / 64 elems, 6.25 bpw)"),
    "q6_g64": (64, 50, "Infernix Q6_G64_FP16 (64 elems + fp16 scale = 50 B / 64 elems, 6.25 bpw)"),
    "q8": (32, 34, "Infernix Q8_G32_FP16 (32 elems + fp16 scale = 34 B / 32 elems, 8.50 bpw)"),
    "q8_g32": (32, 34, "Infernix Q8_G32_FP16 (32 elems + fp16 scale = 34 B / 32 elems, 8.50 bpw)"),
    # GGML legacy & K-quants / I-quants
    "q4_0": (32, 18, "GGML Q4_0 (32 elems, fp16 scale = 18 B / 32 elems, 4.50 bpw)"),
    "q4_1": (32, 20, "GGML Q4_1 (32 elems, fp16 scale+min = 20 B / 32 elems, 5.00 bpw)"),
    "q5_0": (32, 22, "GGML Q5_0 (32 elems, fp16 scale + qh = 22 B / 32 elems, 5.50 bpw)"),
    "q5_1": (32, 24, "GGML Q5_1 (32 elems, fp16 scale+min + qh = 24 B / 32 elems, 6.00 bpw)"),
    "q8_0": (32, 34, "GGML Q8_0 (32 elems, fp16 scale = 34 B / 32 elems, 8.50 bpw)"),
    "q8_1": (32, 36, "GGML Q8_1 (32 elems, fp16 scale+sum = 36 B / 32 elems, 9.00 bpw)"),
    "q2_k": (256, 84, "GGML Q2_K (256 super-block = 84 B, 2.625 bpw)"),
    "q3_k": (256, 110, "GGML Q3_K (256 super-block = 110 B, 3.4375 bpw)"),
    "q4_k": (256, 144, "GGML Q4_K (256 super-block = 144 B, 4.50 bpw)"),
    "q5_k": (256, 176, "GGML Q5_K (256 super-block = 176 B, 5.50 bpw)"),
    "q6_k": (256, 210, "GGML Q6_K (256 super-block = 210 B, 6.5625 bpw)"),
    "q8_k": (256, 292, "GGML Q8_K (256 super-block = 292 B, 9.125 bpw)"),
    "iq1_s": (256, 50, "GGML IQ1_S (256 super-block = 50 B, 1.5625 bpw)"),
    "iq1_m": (256, 56, "GGML IQ1_M (256 super-block = 56 B, 1.75 bpw)"),
    "iq2_xxs": (256, 66, "GGML IQ2_XXS (256 super-block = 66 B, 2.0625 bpw)"),
    "iq2_xs": (256, 74, "GGML IQ2_XS (256 super-block = 74 B, 2.3125 bpw)"),
    "iq2_s": (256, 82, "GGML IQ2_S (256 super-block = 82 B, 2.5625 bpw)"),
    "iq3_xxs": (256, 98, "GGML IQ3_XXS (256 super-block = 98 B, 3.0625 bpw)"),
    "iq3_s": (256, 110, "GGML IQ3_S (256 super-block = 110 B, 3.4375 bpw)"),
    "iq4_nl": (32, 18, "GGML IQ4_NL (32 block = 18 B, 4.50 bpw)"),
    "iq4_xs": (256, 136, "GGML IQ4_XS (256 super-block = 136 B, 4.25 bpw)"),
}


def tensor_bytes(num_elements: int, dtype: str) -> dict[str, Any]:
    """Calculate exact byte footprint for ``num_elements`` in ``dtype``."""
    key = dtype.strip().lower()
    if key not in FORMAT_SPECS:
        raise ValueError(
            f"unknown dtype {dtype!r}; supported formats: {', '.join(sorted(FORMAT_SPECS))}"
        )
    block_elems, block_bytes, desc = FORMAT_SPECS[key]
    num_blocks = ceil_div(num_elements, block_elems)
    total_bytes = num_blocks * block_bytes
    bits_per_elem = (8.0 * block_bytes) / block_elems
    return {
        "dtype": key,
        "description": desc,
        "elements": int(num_elements),
        "block_elements": block_elems,
        "block_bytes": block_bytes,
        "num_blocks": num_blocks,
        "bytes": int(total_bytes),
        "megabytes": round(total_bytes / 1e6, 4),
        "mebibytes": round(total_bytes / (1024.0 * 1024.0), 4),
        "bits_per_element": round(bits_per_elem, 4),
    }


# --------------------------------------------------------------------------- #
# 1. Occupancy, register step table, SMEM step table, and block-size sweep
# --------------------------------------------------------------------------- #


def calculate_occupancy_tables(
    dev: DeviceInfo,
    *,
    threads_per_block: int,
    registers_per_thread: int,
    shared_mem_per_block: int = 0,
) -> dict[str, Any]:
    """Compute exact occupancy, register/SMEM step tables, and block-size sweep."""
    cur = occupancy_model(
        dev,
        threads_per_block=threads_per_block,
        registers_per_thread=registers_per_thread,
        shared_mem_per_block=shared_mem_per_block,
    )
    sweep_threads = (32, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768, 896, 1024)
    block_sweep: list[dict[str, Any]] = []
    for t in sweep_threads:
        if t > dev.max_threads_per_block:
            continue
        om = occupancy_model(
            dev,
            threads_per_block=t,
            registers_per_thread=registers_per_thread,
            shared_mem_per_block=shared_mem_per_block,
        )
        block_sweep.append(
            {
                "threads_per_block": t,
                "warps_per_block": om.warps_per_block,
                "active_blocks_per_sm": om.active_blocks_per_sm,
                "active_warps_per_sm": om.active_warps_per_sm,
                "occupancy_pct": round(om.occupancy_pct, 2),
                "limiting_factor": om.limiting_factor,
                "registers_allocated_per_block": om.registers_allocated_per_block,
                "unused_registers_per_sm": om.unused_registers_per_sm,
                "max_regs_to_gain_one_block": om.max_regs_per_thread_to_gain_one_block,
                "free_reg_headroom_per_thread": om.free_reg_headroom_per_thread,
            }
        )
    hw_ranges = {
        "threads_per_block": {
            "min": 32,
            "max": dev.max_threads_per_block,
            "step": dev.warp_size,
            "max_for_100pct_occupancy": dev.max_threads_per_multiprocessor // 2,
            "notes": (
                f"Hardware accepts 1..{dev.max_threads_per_block} (use warp multiples 32..{dev.max_threads_per_block} step 32 = 1..32 warps/CTA). "
                f"Because SM limit is {dev.max_threads_per_multiprocessor} threads/SM ({dev.max_warps_per_multiprocessor} warps/SM), "
                "threads=1024 caps residency at 1 block/SM (66.7% max occ); threads in {64, 128, 192, 256, 384, 512, 768} can reach 100% occ."
            ),
        },
        "registers_per_thread": {
            "min": 1,
            "max": dev.max_regs_per_thread_hw,
            "step": dev.reg_alloc_step,
            "per_sm_total": dev.regs_per_multiprocessor,
            "notes": (
                f"Allocated in {dev.reg_alloc_unit_warp}-reg warp chunks (step of "
                f"{dev.reg_alloc_step} regs/thread). "
                f"At 100% occupancy ({dev.max_threads_per_multiprocessor} threads/SM), max is {dev.regs_per_multiprocessor // dev.max_threads_per_multiprocessor} regs/thread."
            ),
        },
        "shared_mem_per_block_bytes": {
            "min": 0,
            "max_default": dev.shared_mem_per_block,
            "max_optin": dev.shared_mem_per_block_optin,
            "per_sm_total": dev.shared_mem_per_multiprocessor,
            "step": dev.smem_alloc_unit_bytes,
            "driver_reserved_per_block": dev.reserved_shared_mem_per_block,
        },
        "blocks_per_sm": {
            "min": 1,
            "max": dev.max_blocks_per_multiprocessor,
        },
    }
    return {
        "device": dev.name,
        "arch": dev.arch,
        "sms": dev.multi_processor_count,
        "regs_per_sm": dev.regs_per_multiprocessor,
        "smem_per_sm": dev.shared_mem_per_multiprocessor,
        "max_warps_per_sm": dev.max_warps_per_multiprocessor,
        "hardware_parameter_ranges": hw_ranges,
        "current": cur.to_dict(),
        "register_step_table": cur.register_step_table,
        "smem_step_table": cur.smem_step_table,
        "block_size_sweep": block_sweep,
    }


# --------------------------------------------------------------------------- #
# 2. Wave quantization and persistent grid sizing
# --------------------------------------------------------------------------- #


def calculate_wave_tables(
    dev: DeviceInfo,
    *,
    total_ctas: int,
    threads_per_block: int = 256,
    registers_per_thread: int = 64,
    shared_mem_per_block: int = 0,
    total_items: int | None = None,
) -> dict[str, Any]:
    """Compute exact wave quantization and wave-aligned grid options."""
    occ = occupancy_model(
        dev,
        threads_per_block=threads_per_block,
        registers_per_thread=registers_per_thread,
        shared_mem_per_block=shared_mem_per_block,
    )
    wv = wave_analysis(dev, occ, total_ctas)
    wave_cap = wv.blocks_per_wave
    options: list[dict[str, Any]] = []
    if wave_cap > 0:
        cand_waves = sorted(
            {
                1,
                2,
                3,
                4,
                6,
                8,
                max(1, wv.full_waves),
                max(1, wv.full_waves + 1),
            }
        )
        for k in cand_waves:
            ctas = k * wave_cap
            row: dict[str, Any] = {
                "waves": k,
                "total_ctas": ctas,
                "total_threads": ctas * threads_per_block,
                "delta_ctas_from_current": ctas - total_ctas,
            }
            if total_items is not None and total_items > 0:
                row["items_per_cta"] = ceil_div(total_items, ctas)
                row["items_per_thread"] = ceil_div(total_items, ctas * threads_per_block)
            options.append(row)
    return {
        "device": dev.name,
        "sms": dev.multi_processor_count,
        "active_blocks_per_sm": occ.active_blocks_per_sm,
        "occupancy_pct": round(occ.occupancy_pct, 2),
        "wave_analysis": wv.to_dict(),
        "wave_aligned_grids": options,
    }


# --------------------------------------------------------------------------- #
# 3. Roofline & Linear/Quantized Workload Calculator
# --------------------------------------------------------------------------- #


def calculate_roofline_math(
    dev: DeviceInfo,
    *,
    bytes_moved: float,
    flops: float,
    latency_us: float | None = None,
    dtype: str = "fp32",
) -> dict[str, Any]:
    """Compute complete roofline bounds across all device ceilings."""
    rof = roofline_report(
        bytes_moved=bytes_moved,
        flops=flops,
        latency_us=latency_us,
        dev=dev,
        dtype=dtype,
    )
    d = rof.to_dict()
    bw, _ = dev.peak_bandwidth_gbps()
    ceilings_table: list[dict[str, Any]] = []
    for dt_name, dp_obj in sorted(dev.dtype_peaks.items()):
        tflops = getattr(dp_obj, "tflops", None) if not isinstance(dp_obj, dict) else dp_obj.get("tflops")
        src = getattr(dp_obj, "source", "unknown") if not isinstance(dp_obj, dict) else dp_obj.get("source", "unknown")
        if not tflops:
            continue
        comp_us = (flops / (float(tflops) * 1e12)) * 1e6 if flops else None
        ridge = (float(tflops) * 1e12) / (bw * 1e9) if bw else None
        mem_us = rof.memory_bound_latency_us
        lim_us = max(x for x in (mem_us, comp_us) if x is not None) if (mem_us or comp_us) else None
        regime = (
            ("memory-bound" if (rof.arithmetic_intensity or 0) < ridge else "compute-bound")
            if (ridge is not None and rof.arithmetic_intensity is not None)
            else "unknown"
        )
        ceilings_table.append(
            {
                "dtype": dt_name,
                "ceiling_tflops": round(float(tflops), 3),
                "source": src,
                "ridge_point_flops_per_byte": round(ridge, 2) if ridge else None,
                "compute_bound_us": round(comp_us, 4) if comp_us else None,
                "roofline_limit_us": round(lim_us, 4) if lim_us else None,
                "regime": regime,
                "compute_utilization_pct": (
                    round(100.0 * rof.effective_tflops / float(tflops), 2)
                    if rof.effective_tflops
                    else None
                ),
                "max_speedup_vs_latency": (
                    round(latency_us / lim_us, 3)
                    if (latency_us and lim_us and lim_us > 0)
                    else None
                ),
            }
        )
    d["ceilings_comparison"] = ceilings_table
    return d


def calculate_workload_math(
    dev: DeviceInfo,
    *,
    m: int,
    n: int,
    k: int,
    weight_dtype: str,
    act_dtype: str = "fp32",
    out_dtype: str = "fp32",
    latency_us: float | None = None,
    compute_dtype: str = "fp32",
) -> dict[str, Any]:
    """Calculate exact byte volume, FLOPs, and roofline bounds for a [M, N, K] GEMV/GEMM.

    Follows Infernix convention where weight matrix has shape [N, K] (N output
    rows, K reduction columns), activation input has shape [M, K] (M batch/tokens
    or columns), and output has shape [M, N].
    """
    w_info = tensor_bytes(n * k, weight_dtype)
    a_info = tensor_bytes(m * k, act_dtype)
    o_info = tensor_bytes(m * n, out_dtype)

    bytes_unique = w_info["bytes"] + a_info["bytes"] + o_info["bytes"]
    bytes_unamortized_m = m * w_info["bytes"] + a_info["bytes"] + o_info["bytes"]
    flops = 2.0 * float(m) * float(n) * float(k)

    rof_unique = calculate_roofline_math(
        dev,
        bytes_moved=float(bytes_unique),
        flops=flops,
        latency_us=latency_us,
        dtype=compute_dtype,
    )
    rof_unamortized = calculate_roofline_math(
        dev,
        bytes_moved=float(bytes_unamortized_m),
        flops=flops,
        latency_us=latency_us,
        dtype=compute_dtype,
    )
    return {
        "shape": {"M": m, "N": n, "K": k},
        "weights": w_info,
        "activations": a_info,
        "outputs": o_info,
        "flops": int(flops),
        "gflops": round(flops / 1e9, 4),
        "bytes_moved_unique": int(bytes_unique),
        "bytes_moved_unamortized_m": int(bytes_unamortized_m),
        "bytes_saved_by_m_reuse": int(bytes_unamortized_m - bytes_unique),
        "m_reuse_traffic_reduction_pct": round(
            100.0 * (bytes_unamortized_m - bytes_unique) / max(1, bytes_unamortized_m), 2
        ),
        "roofline_unique_weights": rof_unique,
        "roofline_unamortized_weights": rof_unamortized,
    }


# --------------------------------------------------------------------------- #
# 4. Shared-memory footprint, bank-conflict & padding calculator
# --------------------------------------------------------------------------- #


def calculate_smem_layout(
    dev: DeviceInfo,
    *,
    rows: int,
    cols: int,
    elem_bytes: float,
    stages: int = 1,
    other_bytes: int = 0,
    threads_per_block: int = 256,
    registers_per_thread: int = 64,
) -> dict[str, Any]:
    """Calculate exact shared-memory footprint, bank conflicts, and optimal padding."""
    row_bytes = int(round(cols * elem_bytes))
    tile_bytes_per_stage = rows * row_bytes
    total_user_smem = tile_bytes_per_stage * stages + other_bytes
    reserve = dev.reserved_shared_mem_per_block
    alloc_unit = dev.smem_alloc_unit_bytes
    allocated_smem = round_up(total_user_smem + reserve, alloc_unit) if total_user_smem > 0 else 0

    occ = occupancy_model(
        dev,
        threads_per_block=threads_per_block,
        registers_per_thread=registers_per_thread,
        shared_mem_per_block=total_user_smem,
    )

    # Bank conflict analysis for column-major / strided access across rows:
    # thread l accesses row l, same column -> address stride between adjacent
    # threads is row_bytes.  Shared memory has 32 banks of 4 bytes (128 B total).
    stride_words = row_bytes // 4 if row_bytes >= 4 else 1
    conflict_way = math.gcd(stride_words, 32) if row_bytes % 4 == 0 else 1

    # Find smallest scalar padding (4B bank coprime with 32 -> odd number of 4B words)
    scalar_pad = 0
    for pad in range(0, 64):
        rb = int(round((cols + pad) * elem_bytes))
        if rb > 0 and rb % 4 == 0 and math.gcd(rb // 4, 32) == 1:
            scalar_pad = pad
            break
    scalar_row_bytes = int(round((cols + scalar_pad) * elem_bytes))
    scalar_user_smem = rows * scalar_row_bytes * stages + other_bytes
    scalar_occ = occupancy_model(
        dev,
        threads_per_block=threads_per_block,
        registers_per_thread=registers_per_thread,
        shared_mem_per_block=scalar_user_smem,
    )

    # Find smallest 16B-aligned padding (for float4 / LDS.128 / cp.async 16B)
    # where stride in 16B units is odd (i.e. (rb // 4) % 8 == 4, minimizing 128B wavefront conflicts
    # or shifting by 4 banks = 16 bytes per row).
    vec16_pad = 0
    for pad in range(0, 128):
        rb = int(round((cols + pad) * elem_bytes))
        if rb > 0 and rb % 16 == 0 and ((rb // 4) % 8 == 4):
            vec16_pad = pad
            break
    vec16_row_bytes = int(round((cols + vec16_pad) * elem_bytes))
    vec16_user_smem = rows * vec16_row_bytes * stages + other_bytes
    vec16_occ = occupancy_model(
        dev,
        threads_per_block=threads_per_block,
        registers_per_thread=registers_per_thread,
        shared_mem_per_block=vec16_user_smem,
    )

    return {
        "tile_shape": {"rows": rows, "cols": cols, "elem_bytes": elem_bytes, "stages": stages, "other_bytes": other_bytes},
        "unpadded": {
            "row_bytes": row_bytes,
            "tile_bytes_per_stage": tile_bytes_per_stage,
            "total_user_smem_bytes": total_user_smem,
            "reserved_smem_bytes": reserve,
            "allocated_smem_bytes_per_block": allocated_smem,
            "requires_optin_attribute": bool(total_user_smem > dev.shared_mem_per_block),
            "exceeds_device_max_smem": bool(allocated_smem > dev.shared_mem_per_multiprocessor),
            "blocks_by_shared": occ.blocks_by_shared,
            "active_blocks_per_sm": occ.active_blocks_per_sm,
            "occupancy_pct": round(occ.occupancy_pct, 2),
            "limiting_factor": occ.limiting_factor,
            "free_smem_headroom_bytes": occ.free_smem_headroom_bytes,
            "stride_4b_banks": stride_words,
            "column_access_bank_conflict_way": conflict_way,
        },
        "scalar_4b_conflict_free_padding": {
            "pad_elements": scalar_pad,
            "padded_cols": cols + scalar_pad,
            "row_bytes": scalar_row_bytes,
            "stride_4b_banks": scalar_row_bytes // 4,
            "bank_conflict_way": math.gcd(scalar_row_bytes // 4, 32),
            "total_user_smem_bytes": scalar_user_smem,
            "allocated_smem_bytes_per_block": scalar_occ.smem_allocated_per_block,
            "active_blocks_per_sm": scalar_occ.active_blocks_per_sm,
            "occupancy_pct": round(scalar_occ.occupancy_pct, 2),
        },
        "aligned_16b_padding": {
            "pad_elements": vec16_pad,
            "padded_cols": cols + vec16_pad,
            "row_bytes": vec16_row_bytes,
            "stride_4b_banks": vec16_row_bytes // 4,
            "total_user_smem_bytes": vec16_user_smem,
            "allocated_smem_bytes_per_block": vec16_occ.smem_allocated_per_block,
            "active_blocks_per_sm": vec16_occ.active_blocks_per_sm,
            "occupancy_pct": round(vec16_occ.occupancy_pct, 2),
            "note": "Maintains 16-byte row alignment for cp.async.cg / LDS.128 while offsetting adjacent rows by 4 banks (16 B). For zero bank conflicts with 128-bit accesses, prefer XOR swizzling: col ^ (row & 7).",
        },
    }


# --------------------------------------------------------------------------- #
# 5. Little's Law / Memory-Level Parallelism (MLP) Calculator
# --------------------------------------------------------------------------- #


def calculate_mlp(
    dev: DeviceInfo,
    *,
    active_warps_per_sm: int,
    load_width_bytes: int = 4,
    mem_latency_ns: float = 250.0,
) -> dict[str, Any]:
    """Calculate Little's Law memory-level parallelism required to saturate DRAM."""
    bw_gbps, bw_src = dev.peak_bandwidth_gbps()
    bw = bw_gbps or 241.4
    sms = max(1, dev.multi_processor_count or 48)
    warp_size = dev.warp_size or 32

    bytes_in_flight_gpu = bw * mem_latency_ns  # GB/s * ns = bytes
    bytes_in_flight_sm = bytes_in_flight_gpu / sms

    width_table: list[dict[str, Any]] = []
    for w in (1, 2, 4, 8, 16):
        b_per_warp_inst = warp_size * w
        b_at_1_inst = max(1, active_warps_per_sm) * b_per_warp_inst
        loads_needed = math.ceil(bytes_in_flight_sm / b_at_1_inst)
        width_table.append(
            {
                "load_width_bytes": w,
                "sass_instruction": f"LDG.E.{w * 8}" if w != 4 else "LDG.E (32-bit)",
                "bytes_per_warp_request": b_per_warp_inst,
                "bytes_in_flight_at_1_req_per_warp": b_at_1_inst,
                "independent_loads_per_thread_needed": loads_needed,
                "register_ footprint_per_thread_for_loads": loads_needed * max(1, w // 4),
            }
        )

    cur_b_warp = warp_size * load_width_bytes
    cur_b_sm = max(1, active_warps_per_sm) * cur_b_warp
    cur_loads_needed = math.ceil(bytes_in_flight_sm / cur_b_sm)

    return {
        "device": dev.name,
        "sms": sms,
        "bandwidth_ceiling_gbps": round(bw, 2),
        "bandwidth_source": bw_src,
        "assumed_mem_latency_ns": mem_latency_ns,
        "bytes_in_flight_needed_gpu": round(bytes_in_flight_gpu, 1),
        "bytes_in_flight_needed_per_sm": round(bytes_in_flight_sm, 1),
        "active_warps_per_sm": active_warps_per_sm,
        "current_load_width_bytes": load_width_bytes,
        "current_bytes_in_flight_at_1_load_per_warp": cur_b_sm,
        "independent_loads_per_thread_needed": cur_loads_needed,
        "load_width_comparison": width_table,
    }


# --------------------------------------------------------------------------- #
# 6. A/B Statistical & End-to-End Speedup Calculator
# --------------------------------------------------------------------------- #


def calculate_ab_stats(
    base_samples: Sequence[float],
    cand_samples: Sequence[float],
    *,
    launches_per_token: int | None = None,
    baseline_token_ms: float | None = None,
) -> dict[str, Any]:
    """Compute exact A/B timing statistics, noise floor, SNR, and Amdahl token impact."""
    if not base_samples or not cand_samples:
        raise ValueError("base_samples and cand_samples must both be non-empty")

    def _stats(xs: Sequence[float]) -> dict[str, float]:
        med = percentile(xs, 50)
        mn = mean(xs)
        sd = stdev(xs)
        iq = percentile(xs, 75) - percentile(xs, 25)
        cv_pct = (100.0 * sd / mn) if mn > 0 else 0.0
        iqr_pct = (100.0 * iq / med) if med > 0 else 0.0
        return {
            "n": len(xs),
            "median_us": round(med, 4),
            "mean_us": round(mn, 4),
            "stdev_us": round(sd, 4),
            "p10_us": round(percentile(xs, 10), 4),
            "p90_us": round(percentile(xs, 90), 4),
            "min_us": round(min(xs), 4),
            "max_us": round(max(xs), 4),
            "iqr_us": round(iq, 4),
            "cv_pct": round(cv_pct, 3),
            "iqr_over_median_pct": round(iqr_pct, 3),
        }

    st_a = _stats(base_samples)
    st_b = _stats(cand_samples)
    med_a = st_a["median_us"]
    med_b = st_b["median_us"]
    delta_us = med_b - med_a
    pct_change = (100.0 * delta_us / med_a) if med_a > 0 else 0.0
    speedup = (med_a / med_b) if med_b > 0 else 0.0

    # Use IQR/median when n >= 4 and a single outlier inflates parametric CV (> 2.5x IQR%),
    # since delta_us is computed on medians rather than un-trimmed means.
    disp_a = (
        st_a["iqr_over_median_pct"]
        if (st_a["n"] >= 4 and st_a["iqr_over_median_pct"] > 0 and st_a["cv_pct"] > 2.5 * st_a["iqr_over_median_pct"])
        else max(st_a["cv_pct"], st_a["iqr_over_median_pct"])
    )
    disp_b = (
        st_b["iqr_over_median_pct"]
        if (st_b["n"] >= 4 and st_b["iqr_over_median_pct"] > 0 and st_b["cv_pct"] > 2.5 * st_b["iqr_over_median_pct"])
        else max(st_b["cv_pct"], st_b["iqr_over_median_pct"])
    )
    noise_floor_pct = max(disp_a, disp_b)
    snr = (abs(pct_change) / noise_floor_pct) if noise_floor_pct > 0 else float("inf")
    # Interleaved quantile overlap check
    non_overlapping = (st_b["p90_us"] < st_a["p10_us"]) or (st_a["p90_us"] < st_b["p10_us"])

    paired_wins_b = None
    if len(base_samples) == len(cand_samples):
        paired_wins_b = sum(1 for a, b in zip(base_samples, cand_samples) if b < a)

    if abs(pct_change) <= max(noise_floor_pct, 0.5) and not (non_overlapping and abs(pct_change) >= 1.5):
        verdict = "WITHIN_NOISE"
    elif delta_us < 0 and (snr >= 1.5 or non_overlapping):
        verdict = "MEANINGFUL_GAIN"
    elif delta_us > 0 and (snr >= 1.5 or non_overlapping):
        verdict = "REGRESSION"
    else:
        verdict = "MARGINAL_OR_NOISY"

    out: dict[str, Any] = {
        "baseline": st_a,
        "candidate": st_b,
        "delta_us": round(delta_us, 4),
        "pct_change": round(pct_change, 3),
        "speedup": round(speedup, 4),
        "noise_floor_pct": round(noise_floor_pct, 3),
        "signal_to_noise_ratio": round(snr, 2) if math.isfinite(snr) else "inf",
        "p10_p90_non_overlapping": non_overlapping,
        "paired_b_faster_rounds": f"{paired_wins_b}/{len(base_samples)}" if paired_wins_b is not None else None,
        "verdict": verdict,
    }

    if launches_per_token is not None and launches_per_token > 0:
        saved_us_per_token = -delta_us * launches_per_token
        saved_ms_per_token = saved_us_per_token / 1000.0
        e2e: dict[str, Any] = {
            "launches_per_token": launches_per_token,
            "baseline_stage_ms_per_token": round((med_a * launches_per_token) / 1000.0, 4),
            "candidate_stage_ms_per_token": round((med_b * launches_per_token) / 1000.0, 4),
            "saved_ms_per_token": round(saved_ms_per_token, 4),
        }
        if baseline_token_ms is not None and baseline_token_ms > 0:
            new_token_ms = max(1e-6, baseline_token_ms - saved_ms_per_token)
            base_tok_s = 1000.0 / baseline_token_ms
            new_tok_s = 1000.0 / new_token_ms
            e2e["baseline_token_ms"] = round(baseline_token_ms, 4)
            e2e["projected_candidate_token_ms"] = round(new_token_ms, 4)
            e2e["baseline_tok_per_s"] = round(base_tok_s, 3)
            e2e["projected_candidate_tok_per_s"] = round(new_tok_s, 3)
            e2e["projected_e2e_speedup"] = round(baseline_token_ms / new_token_ms, 4)
            e2e["projected_e2e_pct_gain"] = round(100.0 * (new_tok_s - base_tok_s) / base_tok_s, 3)
        out["end_to_end_projection"] = e2e

    return out
