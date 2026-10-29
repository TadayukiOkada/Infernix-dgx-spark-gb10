"""performance_model — analytical modelling and bottleneck classification.

Three responsibilities, kept strictly separate so that no modelled number is
ever mistaken for a measured one:

1. **Occupancy prediction** from *compiler-reported* resources (registers,
   shared memory, block size) using the target's real allocation granularity.
   This is arithmetic on measured inputs, so it is reported as ``estimated``.
2. **Roofline / limit analysis** relating measured latency to the memory and
   compute ceilings.  The memory ceiling is the *measured* bandwidth when
   available; deriving it from bus width is refused on unified-memory parts.
3. **Bottleneck classification** into the nine required categories from NCU
   counters plus SASS evidence.  Every classification carries the counters that
   support it and the experiment that would confirm or refute it.

Nothing here performs latency prediction from first principles; it bounds and
classifies.  Speculative "predicted speedup" numbers are not produced.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .environment import DeviceInfo
from .profiler import KernelMetricResult
from .util import LOG, percentile


def ceil_div(a: int, b: int) -> int:
    if b <= 0:
        return 0
    return -(-int(a) // int(b))


def round_up(v: int, m: int) -> int:
    if m <= 0:
        return int(v)
    return ceil_div(int(v), m) * m


# --------------------------------------------------------------------------- #
# Occupancy
# --------------------------------------------------------------------------- #


@dataclass
class OccupancyModel:
    threads_per_block: int = 0
    registers_per_thread: int = 0
    shared_mem_per_block: int = 0
    warps_per_block: int = 0
    blocks_by_threads: int = 0
    blocks_by_warps: int = 0
    blocks_by_registers: int = 0
    blocks_by_shared: int = 0
    blocks_by_hw_limit: int = 0
    active_blocks_per_sm: int = 0
    active_warps_per_sm: int = 0
    max_warps_per_sm: int = 0
    occupancy_pct: float = 0.0
    limiting_factor: str = ""
    registers_allocated_per_block: int = 0
    smem_allocated_per_block: int = 0
    unused_registers_per_sm: int = 0
    max_regs_per_thread_to_gain_one_block: int | None = None
    max_regs_before_dropping_block: int | None = None
    free_reg_headroom_per_thread: int | None = None
    free_smem_headroom_bytes: int | None = None
    register_step_table: list[dict[str, Any]] = field(default_factory=list)
    smem_step_table: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "threads_per_block": self.threads_per_block,
            "registers_per_thread": self.registers_per_thread,
            "shared_mem_per_block": self.shared_mem_per_block,
            "warps_per_block": self.warps_per_block,
            "blocks_by_threads": self.blocks_by_threads,
            "blocks_by_warps": self.blocks_by_warps,
            "blocks_by_registers": self.blocks_by_registers,
            "blocks_by_shared": self.blocks_by_shared,
            "blocks_by_hw_limit": self.blocks_by_hw_limit,
            "active_blocks_per_sm": self.active_blocks_per_sm,
            "active_warps_per_sm": self.active_warps_per_sm,
            "max_warps_per_sm": self.max_warps_per_sm,
            "occupancy_pct": round(self.occupancy_pct, 2),
            "limiting_factor": self.limiting_factor,
            "registers_allocated_per_block": self.registers_allocated_per_block,
            "smem_allocated_per_block": self.smem_allocated_per_block,
            "unused_registers_per_sm": self.unused_registers_per_sm,
            "max_regs_per_thread_to_gain_one_block": self.max_regs_per_thread_to_gain_one_block,
            "max_regs_before_dropping_block": self.max_regs_before_dropping_block,
            "free_reg_headroom_per_thread": self.free_reg_headroom_per_thread,
            "free_smem_headroom_bytes": self.free_smem_headroom_bytes,
            "register_step_table": self.register_step_table,
            "smem_step_table": self.smem_step_table,
            "notes": self.notes,
        }

    @property
    def label(self) -> str:
        return "estimated"


def occupancy_model(
    dev: DeviceInfo,
    *,
    threads_per_block: int,
    registers_per_thread: int,
    shared_mem_per_block: int = 0,
    reserved_shared_mem_bytes: int | None = None,
) -> OccupancyModel:
    """Predict resident blocks per SM from compiler-reported resources.

    Register allocation is rounded to the hardware's per-warp allocation unit
    before the block's footprint is computed; ignoring that granularity is the
    single most common source of bogus occupancy claims.
    """
    m = OccupancyModel(
        threads_per_block=threads_per_block,
        registers_per_thread=registers_per_thread,
        shared_mem_per_block=shared_mem_per_block,
    )
    if threads_per_block <= 0:
        m.notes.append("threads_per_block must be positive")
        return m
    warp_size = dev.warp_size or 32
    m.warps_per_block = ceil_div(threads_per_block, warp_size)
    m.max_warps_per_sm = dev.max_warps_per_multiprocessor

    # 1. Thread / warp / block-count limits.
    m.blocks_by_threads = dev.max_threads_per_multiprocessor // threads_per_block
    m.blocks_by_warps = m.max_warps_per_sm // m.warps_per_block
    m.blocks_by_hw_limit = min(m.blocks_by_threads, m.blocks_by_warps, dev.max_blocks_per_multiprocessor)

    # 2. Register limit, using the real per-warp allocation granularity.
    if registers_per_thread > dev.max_regs_per_thread_hw:
        m.notes.append(
            f"registers_per_thread ({registers_per_thread}) exceeds the hardware maximum "
            f"({dev.max_regs_per_thread_hw}); the block cannot be resident"
        )
        m.blocks_by_registers = 0
    elif registers_per_thread <= 0:
        # Unknown register count: assume the compiler's default ceiling would
        # not limit residency, but say so rather than pretending to know.
        m.blocks_by_registers = dev.max_blocks_per_multiprocessor
        m.notes.append("register count unavailable; register residency limit not evaluated")
    else:
        regs_per_warp = round_up(registers_per_thread * warp_size, dev.reg_alloc_unit_warp)
        m.registers_allocated_per_block = regs_per_warp * m.warps_per_block
        m.blocks_by_registers = max(
            0, dev.regs_per_multiprocessor // m.registers_allocated_per_block
        )

    # 3. Shared memory limit, including the per-block system reservation.
    reserve = dev.reserved_shared_mem_per_block if reserved_shared_mem_bytes is None else reserved_shared_mem_bytes
    if shared_mem_per_block > 0:
        m.smem_allocated_per_block = round_up(shared_mem_per_block + reserve, dev.smem_alloc_unit_bytes)
        if m.smem_allocated_per_block > dev.shared_mem_per_multiprocessor:
            m.notes.append(
                f"shared memory per block ({m.smem_allocated_per_block} B including the "
                f"{reserve} B reserved region) exceeds the {dev.shared_mem_per_multiprocessor} B "
                "per-SM capacity; the block cannot be resident"
            )
            m.blocks_by_shared = 0
        else:
            m.blocks_by_shared = dev.shared_mem_per_multiprocessor // m.smem_allocated_per_block
    else:
        m.blocks_by_shared = dev.max_blocks_per_multiprocessor

    m.active_blocks_per_sm = max(
        0, min(m.blocks_by_hw_limit, m.blocks_by_registers, m.blocks_by_shared)
    )
    m.active_warps_per_sm = m.active_blocks_per_sm * m.warps_per_block
    if m.max_warps_per_sm:
        m.occupancy_pct = 100.0 * m.active_warps_per_sm / m.max_warps_per_sm
    if m.active_blocks_per_sm > 0 and m.registers_allocated_per_block > 0:
        m.unused_registers_per_sm = max(
            0, dev.regs_per_multiprocessor - m.active_blocks_per_sm * m.registers_allocated_per_block
        )

    limits = {
        "threads/warps/block-count": m.blocks_by_hw_limit,
        "registers": m.blocks_by_registers,
        "shared memory": m.blocks_by_shared,
    }
    m.limiting_factor = min(limits, key=lambda k: limits[k])

    # Pre-calculate exact register step table so agents never do hand math.
    max_blk_non_reg = max(0, min(m.blocks_by_hw_limit, m.blocks_by_shared))
    for blk in range(1, max_blk_non_reg + 1):
        r_blk = dev.regs_per_multiprocessor // blk
        r_warp = (r_blk // m.warps_per_block) // dev.reg_alloc_unit_warp * dev.reg_alloc_unit_warp
        max_r = min(dev.max_regs_per_thread_hw, r_warp // warp_size)
        if max_r <= 0:
            break
        occ_b = 100.0 * (blk * m.warps_per_block) / m.max_warps_per_sm if m.max_warps_per_sm else 0.0
        m.register_step_table.append(
            {
                "blocks_per_sm": blk,
                "warps_per_sm": blk * m.warps_per_block,
                "occupancy_pct": round(occ_b, 2),
                "max_regs_per_thread": int(max_r),
                "allocated_regs_per_block": int(r_warp * m.warps_per_block),
            }
        )

    # Pre-calculate exact shared-memory step table.
    max_blk_non_smem = max(0, min(m.blocks_by_hw_limit, m.blocks_by_registers))
    for blk in range(1, max_blk_non_smem + 1):
        alloc_cap = (dev.shared_mem_per_multiprocessor // blk) // dev.smem_alloc_unit_bytes * dev.smem_alloc_unit_bytes
        user_cap = max(0, alloc_cap - reserve)
        occ_b = 100.0 * (blk * m.warps_per_block) / m.max_warps_per_sm if m.max_warps_per_sm else 0.0
        m.smem_step_table.append(
            {
                "blocks_per_sm": blk,
                "warps_per_sm": blk * m.warps_per_block,
                "occupancy_pct": round(occ_b, 2),
                "max_user_smem_bytes": int( min(user_cap, dev.shared_mem_per_block_optin or user_cap) ),
                "allocated_smem_bytes_per_block": int(alloc_cap),
                "requires_optin": bool(user_cap > dev.shared_mem_per_block),
            }
        )

    if m.active_blocks_per_sm > 0:
        for row in m.register_step_table:
            if row["blocks_per_sm"] == m.active_blocks_per_sm:
                m.max_regs_before_dropping_block = row["max_regs_per_thread"]
                if registers_per_thread > 0:
                    m.free_reg_headroom_per_thread = max(0, row["max_regs_per_thread"] - registers_per_thread)
        for row in m.smem_step_table:
            if row["blocks_per_sm"] == m.active_blocks_per_sm:
                m.free_smem_headroom_bytes = max(0, row["max_user_smem_bytes"] - shared_mem_per_block)

    if m.limiting_factor == "registers" and registers_per_thread > 0:
        # Exactly how far registers must fall to admit one more resident block.
        target = m.active_blocks_per_sm + 1
        if target <= min(m.blocks_by_hw_limit, m.blocks_by_shared):
            regs_per_block = dev.regs_per_multiprocessor // target
            regs_per_warp = (regs_per_block // m.warps_per_block) // dev.reg_alloc_unit_warp * dev.reg_alloc_unit_warp
            candidate = regs_per_warp // warp_size
            if candidate < registers_per_thread:
                m.max_regs_per_thread_to_gain_one_block = candidate
            else:
                m.notes.append(
                    "register granularity prevents another resident block even at a lower "
                    "per-thread register count"
                )
    return m


# --------------------------------------------------------------------------- #
# Roofline / limit analysis
# --------------------------------------------------------------------------- #


@dataclass
class RooflineReport:
    bytes_moved: float | None = None
    flops: float | None = None
    latency_us: float | None = None
    arithmetic_intensity: float | None = None
    effective_bandwidth_gbps: float | None = None
    effective_tflops: float | None = None
    achieved_l2_bytes_per_s: float | None = None
    achieved_global_load_sectors: float | None = None
    achieved_global_store_sectors: float | None = None
    achieved_sectors_per_request: float | None = None
    achieved_l2_hit_rate_pct: float | None = None

    # Only populated where an independent measurement or a reliable spec exists.
    bandwidth_ceiling_gbps: float | None = None
    bandwidth_ceiling_source: str = "unknown"
    compute_ceiling_tflops: float | None = None
    compute_ceiling_dtype: str = "fp32"
    compute_ceiling_source: str = "unknown"
    ridge_point_flops_per_byte: float | None = None
    regime: str = "unknown"

    memory_utilization_pct: float | None = None
    compute_utilization_pct: float | None = None
    memory_bound_latency_us: float | None = None
    compute_bound_latency_us: float | None = None
    limit_us: float | None = None
    headroom_us: float | None = None
    max_theoretical_speedup: float | None = None

    working_set_vs_l2_ratio: float | None = None
    l2_residency_regime: str = "unknown"
    l2_traffic_amplification: float | None = None
    all_compute_bounds_us: dict[str, float] = field(default_factory=dict)

    provenance: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items()}
        for k in (
            "arithmetic_intensity",
            "effective_bandwidth_gbps",
            "effective_tflops",
            "working_set_vs_l2_ratio",
            "l2_traffic_amplification",
            "max_theoretical_speedup",
            "headroom_us",
        ):
            if d.get(k) is not None:
                d[k] = round(d[k], 4)
        for k in ("memory_utilization_pct", "compute_utilization_pct"):
            if d.get(k) is not None:
                d[k] = round(d[k], 2)
        return d


def _peak_tflops(dp: Any) -> tuple[float | None, str]:
    if dp is None:
        return None, "unknown"
    if isinstance(dp, dict):
        return dp.get("tflops"), str(dp.get("source", "unknown"))
    return getattr(dp, "tflops", None), str(getattr(dp, "source", "unknown"))


def roofline_report(
    *,
    bytes_moved: float | None,
    flops: float | None,
    latency_us: float | None,
    dev: DeviceInfo,
    dtype: str = "fp32",
    ncu: KernelMetricResult | None = None,
    is_integer_workload: bool = False,
    max_load_width_bytes: int = 4,
    instruction_profile: Any | None = None,
) -> RooflineReport:
    """Relate measured latency to the memory and compute ceilings.

    The bandwidth ceiling comes from a *measurement* (``bandwidth_probe``) or, on
    architectures where the bus is dedicated, from the memory clock and bus
    width.  On unified-memory parts the bus-width formula is refused because it
    overshoots by more than 2x.
    """
    r = RooflineReport(bytes_moved=bytes_moved, flops=flops, latency_us=latency_us)
    r.provenance["latency_us"] = "measured (Stage A baseline)"
    if bytes_moved:
        r.provenance["bytes_moved"] = "declared by the harness workload definition"
    if flops:
        r.provenance["flops"] = "declared by the harness workload definition"

    if latency_us and latency_us > 0:
        t_s = latency_us * 1e-6
        if bytes_moved:
            r.effective_bandwidth_gbps = (bytes_moved / 1e9) / t_s
        if flops:
            r.effective_tflops = (flops / 1e12) / t_s
        if bytes_moved and flops:
            r.arithmetic_intensity = flops / bytes_moved

    # L2 cache working-set residency analysis.
    if bytes_moved and dev.l2_cache_size and dev.l2_cache_size > 0:
        r.working_set_vs_l2_ratio = bytes_moved / float(dev.l2_cache_size)
        r.provenance["working_set_vs_l2_ratio"] = "declared bytes_moved / device l2_cache_size"
        if r.working_set_vs_l2_ratio <= 0.75:
            r.l2_residency_regime = "l2-resident"
            r.notes.append(
                f"Working set ({bytes_moved / 1e6:.2f} MB) is {r.working_set_vs_l2_ratio * 100.0:.1f}% of the "
                f"{dev.l2_cache_size / (1024 * 1024):.0f} MB L2 cache: fits comfortably in L2 if reused."
            )
        elif r.working_set_vs_l2_ratio <= 1.50:
            r.l2_residency_regime = "l2-borderline"
            r.notes.append(
                f"Working set ({bytes_moved / 1e6:.2f} MB) is {r.working_set_vs_l2_ratio:.2f}x the "
                f"{dev.l2_cache_size / (1024 * 1024):.0f} MB L2 cache: borderline L2 residency; "
                "tiling or L2 persistence hints can prevent L2 capacity thrashing."
            )
        else:
            r.l2_residency_regime = "dram-streaming"
            r.notes.append(
                f"Working set ({bytes_moved / 1e6:.2f} MB) is {r.working_set_vs_l2_ratio:.2f}x the "
                f"{dev.l2_cache_size / (1024 * 1024):.0f} MB L2 cache: cold streaming from memory "
                "unless activation/tile subsets are reused in L2/SMEM."
            )

    # ---- ceilings -------------------------------------------------------- #
    bw, bw_src = dev.peak_bandwidth_gbps()
    if bw is not None:
        r.bandwidth_ceiling_gbps = bw
        r.bandwidth_ceiling_source = bw_src
        if bw_src == "measured":
            r.provenance["bandwidth_ceiling_gbps"] = "measured by bandwidth_probe (Stage 0)"
        elif bw_src == "derived-from-bus-width":
            r.provenance["bandwidth_ceiling_gbps"] = (
                "derived from memory bus width and memory clock; dedicated-bus architectures only"
            )
    else:
        r.notes.append(
            "No bandwidth ceiling available. Run the bandwidth probe (`ckt env --measure-bandwidth`) "
            "before drawing utilisation conclusions; a unified-memory part must never use the "
            "bus-width formula."
        )
        r.provenance["bandwidth_ceiling_gbps"] = "unavailable"

    # Auto-detect Tensor Core ceiling from SASS instruction profile when dtype was left at default fp32.
    effective_dtype = dtype
    if instruction_profile is not None and dtype == "fp32":
        fp4_tc = int(getattr(instruction_profile, "fp4_tensor_core_ops", 0) or 0)
        fp8_tc = int(getattr(instruction_profile, "fp8_tensor_core_ops", 0) or 0)
        int_tc = int(getattr(instruction_profile, "int_tensor_core_ops", 0) or 0)
        flt_tc = int(getattr(instruction_profile, "float_tensor_core_ops", 0) or 0)
        any_tc = int(getattr(instruction_profile, "tensor_core_ops", 0) or 0)
        if fp4_tc > 0 and ("fp4" in dev.dtype_peaks or "nvfp4" in dev.dtype_peaks):
            effective_dtype = "fp4" if "fp4" in dev.dtype_peaks else "nvfp4"
            r.notes.append(
                f"SASS emits {fp4_tc} FP4 Tensor Core (OMMA) instructions; using measured "
                f"{effective_dtype} Tensor Core ceiling instead of scalar FP32 FMA."
            )
        elif fp8_tc > 0 and ("fp8" in dev.dtype_peaks or "mxfp8" in dev.dtype_peaks):
            effective_dtype = "fp8" if "fp8" in dev.dtype_peaks else "mxfp8"
            r.notes.append(
                f"SASS emits {fp8_tc} FP8/FP6 Tensor Core (QMMA) instructions; using measured "
                f"{effective_dtype} Tensor Core ceiling instead of scalar FP32 FMA."
            )
        elif int_tc > 0 and "int8" in dev.dtype_peaks:
            effective_dtype = "int8"
            r.notes.append(
                f"SASS emits {int_tc} integer Tensor Core (IMMA) instructions; using measured "
                "int8_mma Tensor Core ceiling instead of scalar FP32 FMA."
            )
        elif (flt_tc > 0 or any_tc > 0) and "fp16" in dev.dtype_peaks:
            effective_dtype = "fp16"
            r.notes.append(
                f"SASS emits {flt_tc or any_tc} float Tensor Core (HMMA/MMA) instructions; using measured "
                "fp16_mma Tensor Core ceiling instead of scalar FP32 FMA."
            )

    r.compute_ceiling_dtype = effective_dtype
    if flops:
        for dt_name, dp_obj in dev.dtype_peaks.items():
            tf, _ = _peak_tflops(dp_obj)
            if tf and tf > 0:
                r.all_compute_bounds_us[dt_name] = round((flops / (tf * 1e12)) * 1e6, 4)

    if is_integer_workload:
        r.compute_ceiling_source = "not modelled for integer workloads"
        r.notes.append("Compute ceiling not modelled for integer workloads; classify from counters.")
    else:
        dp = dev.dtype_peaks.get(effective_dtype) or dev.dtype_peaks.get(dtype) or dev.dtype_peaks.get("fp32")
        tf, src = _peak_tflops(dp)
        if tf is not None:
            r.compute_ceiling_tflops = float(tf)
            r.compute_ceiling_source = src
            r.provenance["compute_ceiling_tflops"] = (
                f"measured by compute_probe for {effective_dtype} ({src})" if src == "measured"
                else f"specification table for {effective_dtype} on {dev.arch} ({src})"
            )
            if src != "measured":
                r.notes.append(
                    f"Compute ceiling for {effective_dtype} is a vendor figure ({src}), not a "
                    "measurement; utilisation against it is indicative only. Run the compute "
                    "probe for a measured ceiling."
                )
        else:
            r.notes.append(f"No compute ceiling known for dtype '{effective_dtype}' on {dev.arch}.")

    if r.compute_ceiling_tflops and r.bandwidth_ceiling_gbps:
        r.ridge_point_flops_per_byte = (r.compute_ceiling_tflops * 1e12) / (r.bandwidth_ceiling_gbps * 1e9)
        r.provenance["ridge_point_flops_per_byte"] = "derived from the two ceilings above"
        if r.arithmetic_intensity is not None:
            r.regime = (
                "memory-bound"
                if r.arithmetic_intensity < r.ridge_point_flops_per_byte
                else "compute-bound"
            )
        if bytes_moved:
            r.memory_bound_latency_us = (bytes_moved / (r.bandwidth_ceiling_gbps * 1e9)) * 1e6
        if flops:
            r.compute_bound_latency_us = (flops / (r.compute_ceiling_tflops * 1e12)) * 1e6
        cands = [x for x in (r.memory_bound_latency_us, r.compute_bound_latency_us) if x]
        if cands:
            r.limit_us = max(cands)
            r.provenance["limit_us"] = "max of the modelled memory and compute bound latencies"
            if latency_us and latency_us > 0 and r.limit_us > 0:
                r.headroom_us = max(0.0, latency_us - r.limit_us)
                r.max_theoretical_speedup = latency_us / r.limit_us

    if r.effective_bandwidth_gbps and r.bandwidth_ceiling_gbps:
        r.memory_utilization_pct = 100.0 * r.effective_bandwidth_gbps / r.bandwidth_ceiling_gbps
    if r.effective_tflops and r.compute_ceiling_tflops:
        r.compute_utilization_pct = 100.0 * r.effective_tflops / r.compute_ceiling_tflops

    # ---- measured memory behaviour from NCU ------------------------------ #
    if ncu:
        def g(*names: str) -> float | None:
            for n in names:
                if n in ncu.metrics:
                    return ncu.metrics[n]
            return None

        if r.latency_us:
            lts_bytes = g("lts__t_bytes.sum")
            if lts_bytes:
                r.achieved_l2_bytes_per_s = (lts_bytes / 1e9) / (r.latency_us * 1e-6)
                r.provenance["achieved_l2_bytes_per_s"] = "measured (lts__t_bytes.sum / measured latency)"
        r.achieved_global_load_sectors = g("l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum")
        r.achieved_global_store_sectors = g("l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum")
        spr = g("l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio")
        r.achieved_sectors_per_request = spr
        r.achieved_l2_hit_rate_pct = g("lts__t_sector_hit_rate.pct")

        if r.achieved_sectors_per_request is not None:
            r.provenance["achieved_sectors_per_request"] = "measured (NCU l1tex sectors per request)"
            if r.achieved_sectors_per_request > 20.0:
                r.notes.append(
                    f"Global loads average {r.achieved_sectors_per_request:.2f} sectors per request; "
                    "each request is pulling more than one full 128 B cache line with unused "
                    "portions, which inflates memory traffic. Inspect thread-to-address mapping."
                )
            else:
                r.notes.append(
                    f"Global loads average {r.achieved_sectors_per_request:.2f} sectors per request "
                    "(one full 128 B line per request): no partial-line waste observed."
                )
        if bytes_moved and r.achieved_l2_bytes_per_s and r.latency_us:
            physical_bytes = r.achieved_l2_bytes_per_s * 1e9 * (r.latency_us * 1e-6)
            ratio = physical_bytes / bytes_moved
            r.l2_traffic_amplification = ratio
            if ratio > 1.35:
                r.notes.append(
                    f"L2 traffic ({physical_bytes / 1e6:.1f} MB) exceeds the workload's declared "
                    f"bytes ({bytes_moved / 1e6:.1f} MB) by {ratio:.2f}x. Redundant or uncoalesced "
                    "traffic is being issued; investigate the access pattern before tuning "
                    "occupancy."
                )
            else:
                r.notes.append(
                    f"L2 traffic is {ratio:.2f}x the declared bytes, consistent with a single pass "
                    "over the declared data."
                )
    if r.memory_utilization_pct is not None and r.memory_utilization_pct > 100.0:
        r.notes.append(
            f"Effective bandwidth ({r.effective_bandwidth_gbps:.1f} GB/s) exceeds the measured "
            f"DRAM ceiling ({r.bandwidth_ceiling_gbps:.1f} GB/s) because part of the traffic is "
            "served from L2. The DRAM ceiling is a single-direction measurement, so the "
            "utilisation figure is not a hard bound for mixed read/write kernels."
        )
    return r


# --------------------------------------------------------------------------- #
# Bottleneck classification
# --------------------------------------------------------------------------- #

BOTTLENECK_CATEGORIES = [
    "launch-overhead",
    "compute-bound",
    "memory-bandwidth-bound",
    "memory-latency-bound",
    "occupancy-limited",
    "instruction-latency-bound",
    "synchronization-bound",
    "control-flow-bound",
    "mixed-or-inconclusive",
]


@dataclass
class Classification:
    category: str = "mixed-or-inconclusive"
    confidence: str = "low"
    evidence: list[str] = field(default_factory=list)
    counter_summary: dict[str, float] = field(default_factory=dict)
    stall_breakdown_pct: dict[str, float] = field(default_factory=dict)
    corroborating_sass: list[str] = field(default_factory=list)
    confirming_experiment: str = ""
    alternatives: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "counter_summary": {k: round(v, 3) for k, v in self.counter_summary.items()},
            "stall_breakdown_pct": {k: round(v, 2) for k, v in self.stall_breakdown_pct.items()},
            "corroborating_sass": self.corroborating_sass,
            "confirming_experiment": self.confirming_experiment,
            "alternatives": self.alternatives,
        }


def classifier_context(
    *,
    dev: DeviceInfo,
    instruction_profile: Any | None = None,
    occupancy: OccupancyModel | None = None,
    launch_latency_us: float | None = None,
    kernel_duration_us: float | None = None,
) -> dict[str, Any]:
    return {
        "dev": dev,
        "ip": instruction_profile,
        "occ": occupancy,
        "launch_us": launch_latency_us,
        "dur_us": kernel_duration_us,
    }


def classify_bottleneck(
    result: KernelMetricResult,
    *,
    dev: DeviceInfo,
    instruction_profile: Any | None = None,
    occupancy: OccupancyModel | None = None,
    launch_latency_us: float | None = None,
    kernel_duration_us: float | None = None,
    roofline: RooflineReport | None = None,
) -> Classification:
    """Classify the dominant limiter from counters plus SASS evidence.

    The classifier is intentionally conservative: when the counters do not single
    out a limiter, it reports ``mixed-or-inconclusive`` with the experiment that
    would disambiguate, rather than forcing a category.  No category is assigned
    from the mere presence of global-memory instructions.
    """
    c = Classification()
    M = result.metrics

    def g(*names: str) -> float | None:
        for n in names:
            if n in M:
                return M[n]
        return None

    comp = g("sm__throughput.avg.pct_of_peak_sustained_elapsed")
    mem = g(
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    )
    l2 = g("lts__throughput.avg.pct_of_peak_sustained_elapsed")
    issue = g("smsp__issue_active.avg.pct_of_peak_sustained_active")
    warps = g("sm__warps_active.avg.pct_of_peak_sustained_active")
    tensor = g("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed")
    inst_exec = g("sm__inst_executed.avg.pct_of_peak_sustained_elapsed")
    l1tex = g("l1tex__throughput.avg.pct_of_peak_sustained_elapsed")
    smem = g("l1tex__data_pipe_lsu_wavefronts_mem_shared.avg.pct_of_peak_sustained_elapsed")
    lsu = g("l1tex__lsu_writeback_active.avg.pct_of_peak_sustained_elapsed")
    fma = g("sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed")
    alu = g("sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_elapsed")
    fp64 = g("sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_elapsed")
    xu = g("sm__pipe_xu_cycles_active.avg.pct_of_peak_sustained_elapsed")

    stalls: dict[str, float] = {}
    for cat in ("long_scoreboard", "short_scoreboard", "wait", "barrier", "math_pipe_throttle",
                "lg_throttle", "mio_throttle", "not_selected", "no_instruction",
                "branch_resolving", "drain", "membar"):
        v = g(f"smsp__average_warps_issue_stalled_{cat}_per_issue_active.ratio")
        if v is not None:
            stalls[cat] = v

    for k, v in (
        ("sm__throughput.pct", comp),
        ("gpu__compute_memory_throughput.pct", mem),
        ("lts__throughput.pct", l2),
        ("l1tex__throughput.pct", l1tex),
        ("shared_mem_wavefronts.pct", smem),
        ("fma_pipe.pct", fma),
        ("alu_pipe.pct", alu),
        ("fp64_pipe.pct", fp64),
        ("xu_pipe.pct", xu),
        ("inst_executed.pct", inst_exec),
        ("issue_active.pct", issue),
        ("warps_active.pct", warps),
        ("tensor_active.pct", tensor),
    ):
        if v is not None:
            c.counter_summary[k] = v
    total_stall = sum(stalls.values())
    for k, v in sorted(stalls.items(), key=lambda kv: -kv[1]):
        c.counter_summary[f"stall_{k}"] = v
        if total_stall > 0:
            c.stall_breakdown_pct[k] = 100.0 * v / total_stall

    dur = kernel_duration_us if kernel_duration_us is not None else result.duration_us

    # --- SASS corroboration --------------------------------------------- #
    ip = instruction_profile
    if ip is not None:
        try:
            if getattr(ip, "has_local_memory", False):
                c.corroborating_sass.append(
                    "SASS contains local-memory accesses (STL/LDL): register spilling is present"
                )
            mw = getattr(ip, "max_load_width", 0)
            if mw and mw <= 4:
                c.corroborating_sass.append(
                    f"Widest global load is {mw} B; narrower than 16 B vector accesses"
                )
            if getattr(ip, "tensor_core_ops", 0) == 0 and getattr(ip, "global_loads", 0) > 0:
                c.corroborating_sass.append("No tensor-core instructions emitted")
            if getattr(ip, "async_copies", 0) == 0 and getattr(ip, "global_loads", 0) > 0:
                c.corroborating_sass.append("No asynchronous copies (cp.async/TMA) present")
            if getattr(ip, "fp64_ops", 0) > 0:
                c.corroborating_sass.append(
                    f"SASS contains {getattr(ip, 'fp64_ops', 0)} FP64 instruction(s) (67:1 penalty on GB10)"
                )
        except Exception:  # pragma: no cover - profile objects are duck-typed
            pass

    occ = occupancy

    # --- launch overhead -------------------------------------------------- #
    if launch_latency_us is not None and dur is not None and launch_latency_us > 0.35 * max(dur, 1e-9) and dur < 20.0:
        c.category = "launch-overhead"
        c.confidence = "medium"
        c.evidence.append(
            f"Kernel duration ({dur:.3f} us) is comparable to the measured launch overhead "
            f"({launch_latency_us:.3f} us); the grid is too small to amortise launch cost"
        )
        c.confirming_experiment = (
            "Batch iterations inside one launch and re-measure; if per-iteration cost drops "
            "sharply, launch overhead is the limiter."
        )
        return c

    stalled_any = bool(stalls)
    top_stall = max(stalls, key=lambda k: stalls[k]) if stalls else None
    top_stall_frac = None
    if top_stall is not None:
        top_stall_frac = stalls[top_stall] / total_stall if total_stall > 0 else None

    def _fmt(v):
        return "?" if v is None else f"{v:.1f}"

    arith_busy = any(
        p is not None and p >= 65.0
        for p in (fma, alu, fp64, xu, tensor)
    )
    l1tex_busy = l1tex is not None and l1tex >= 65.0
    shared_busy = smem is not None and smem >= 65.0
    occ_pct = getattr(occ, "occupancy_pct", None) if occ else None
    measured_warps = warps

    fma_gap = (fma is not None and inst_exec is not None and inst_exec - fma >= 20.0)
    serial_latency = (
        comp is not None and comp >= 65.0
        and not l1tex_busy
        and issue is not None and issue >= 80.0
        and inst_exec is not None and inst_exec >= 60.0
        and (measured_warps is not None and measured_warps < 70.0)
        and (top_stall == "wait" or top_stall == "not_selected")
        and (fma_gap or occ_pct is not None and occ_pct < 70.0)
    )

    # --- compute-bound ---------------------------------------------------- #
    if comp is not None and comp >= 65.0:
        if serial_latency:
            c.category = "instruction-latency-bound"
            c.confidence = "medium"
            c.evidence.append(
                f"Aggregate SM throughput is {comp:.1f}% of peak but measured occupancy is only "
                f"{_fmt(measured_warps)}% (predicted {occ_pct:.0f}%) and the FMA pipe sits at "
                f"{_fmt(fma)}% while instruction execution is at {_fmt(inst_exec)}%: each warp issues "
                "long dependent chains instead of many independent operations, so the pipes look "
                "full without doing proportional work"
            )
            c.confirming_experiment = (
                "Raise instruction-level parallelism per thread (more accumulators, unrolled "
                "interleaved chains) or raise occupancy so independent warps overlap the latency; "
                "re-measure whether FMA-pipe activity rises without adding instructions."
            )
            return c
        if l1tex_busy and not arith_busy:
            c.category = "memory-bandwidth-bound"
            c.confidence = "high" if shared_busy else "medium"
            c.evidence.append(
                f"SM throughput {_fmt(comp)}% of peak but no arithmetic pipe is busy "
                f"(FMA {_fmt(fma)}%, ALU {_fmt(alu)}%, FP64 {_fmt(fp64)}%) while L1TEX is at "
                f"{_fmt(l1tex)}% and shared-memory wavefronts at {_fmt(smem)}%: the limiter is "
                "the on-chip memory path, not arithmetic"
            )
            c.confirming_experiment = (
                "Reduce shared-memory round trips (register blocking) or remove bank "
                "conflicts; this kind of kernel does not improve with lower precision or "
                "fewer arithmetic operations."
            )
            return c
        if tensor is not None and tensor >= 30.0:
            c.category = "compute-bound"
            c.evidence.append(
                f"SM throughput {comp:.1f}% of peak with tensor-pipe activity {tensor:.1f}%: "
                "the tensor pipeline is the saturated resource"
            )
        else:
            c.category = "compute-bound"
            c.evidence.append(f"SM throughput {comp:.1f}% of peak indicates saturated execution pipes")
        if top_stall == "math_pipe_throttle":
            c.evidence.append(
                f"Dominant stall is math_pipe_throttle ({stalls['math_pipe_throttle']:.2f} warps/issue): "
                "warps are ready but the arithmetic pipe is busy"
            )
        c.confidence = "high"
        c.confirming_experiment = (
            "Reduce arithmetic work (wider vector types, fewer index computations, or lower "
            "precision) and re-measure; a compute-bound kernel scales with arithmetic reduction."
        )
        return c

    # --- memory bandwidth vs latency -------------------------------------- #
    # On unified-memory architectures (such as GB10 LPDDR5X), NCU's
    # gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed is normalised
    # against the theoretical controller bus rate (~512 GB/s), whereas measured
    # sustained LPDDR5X read bandwidth is ~241.4 GB/s (~47% of theoretical).
    # Therefore, a kernel achieving >= 70% of the *measured* bandwidth ceiling
    # (which shows up as ~35-50% in NCU's counter) is genuinely DRAM-bandwidth-bound.
    measured_bw_high = (
        roofline is not None
        and roofline.bandwidth_ceiling_source == "measured"
        and roofline.memory_utilization_pct is not None
        and roofline.memory_utilization_pct >= 68.0
        and (top_stall in ("long_scoreboard", "lg_throttle", "membar", None) or (mem is not None and mem >= 35.0))
    )
    memory_high = (mem is not None and mem >= 70.0) or (l2 is not None and l2 >= 80.0) or measured_bw_high
    if memory_high:
        c.category = "memory-bandwidth-bound"
        if (mem is not None and mem >= 80.0) or (
            roofline and roofline.memory_utilization_pct and roofline.memory_utilization_pct >= 75.0
        ):
            c.confidence = "high"
        else:
            c.confidence = "medium"
        if roofline and roofline.memory_utilization_pct is not None and roofline.bandwidth_ceiling_source == "measured":
            c.evidence.append(
                f"Effective bandwidth ({roofline.effective_bandwidth_gbps:.1f} GB/s) is "
                f"{roofline.memory_utilization_pct:.1f}% of the *measured* device ceiling "
                f"({roofline.bandwidth_ceiling_gbps:.1f} GB/s)"
            )
            if dev.integrated_uma and mem is not None and mem < 65.0:
                c.evidence.append(
                    f"NCU reports memory throughput at {mem:.1f}% of theoretical bus peak, which "
                    f"corresponds to {roofline.memory_utilization_pct:.1f}% of achievable unified-memory "
                    f"LPDDR5X bandwidth on {dev.name}"
                )
        if mem is not None:
            c.evidence.append(f"NCU memory throughput {mem:.1f}% of peak")
        if l2 is not None:
            c.evidence.append(f"L2 throughput {l2:.1f}% of peak")
        if top_stall in ("long_scoreboard", "lg_throttle"):
            c.evidence.append(
                f"Dominant stall is {top_stall} ({stalls[top_stall]:.2f} warps/issue, "
                f"{c.stall_breakdown_pct.get(top_stall, 0.0):.1f}% of stalls): demand for memory "
                "exceeds what the interface can deliver"
            )
        if roofline and roofline.achieved_sectors_per_request and roofline.achieved_sectors_per_request > 20.0:
            c.evidence.append(
                f"Loads average {roofline.achieved_sectors_per_request:.2f} sectors/request, so part "
                "of the traffic is wasted on partially used cache lines"
            )
        c.confirming_experiment = (
            "Reduce bytes moved (fusion, narrower dtype, L2/SMEM reuse across columns/rows) or "
            "improve coalescing; a bandwidth-bound kernel scales with traffic reduction."
        )
        if (
            not measured_bw_high
            and roofline
            and roofline.memory_utilization_pct is not None
            and roofline.memory_utilization_pct < 60.0
        ):
            c.alternatives.append("memory-latency-bound")
            c.evidence.append(
                "Note: the *measured* effective bandwidth is well below the measured ceiling, so "
                "the NCU memory percentage may be overstated on this unified-memory part."
            )
            c.confidence = "medium"
        return c

    # --- latency-bound & stall-dominated family --------------------------- #
    # Any kernel that is neither compute-pipe-saturated (comp < 65%) nor
    # memory-bandwidth-saturated (mem < 70% and below measured DRAM ceiling)
    # is limited by latency, synchronization, occupancy, or control flow.
    not_saturated = (comp is None or comp < 65.0) and not memory_high
    if not_saturated and stalled_any:
        if top_stall == "barrier":
            c.category = "synchronization-bound"
            c.confidence = "high" if (top_stall_frac and top_stall_frac >= 0.35) else "medium"
            c.evidence.append(
                f"Dominant stall is barrier ({stalls['barrier']:.2f} warps/issue, "
                f"{c.stall_breakdown_pct.get('barrier', 0.0):.1f}% of all stalls): warps idle at "
                "__syncthreads() waiting on the slowest peers while compute "
                f"({_fmt(comp)}%) and memory ({_fmt(mem)}%) remain unsaturated"
            )
            if occ_pct is not None and occ_pct <= 35.0:
                c.evidence.append(
                    f"Occupancy is only {occ_pct:.1f}% ({getattr(occ, 'active_blocks_per_sm', 1)} block/SM), "
                    "so no other CTA on the SM can execute while the single resident block waits at a barrier"
                )
                c.alternatives.append("occupancy-limited")
            c.confirming_experiment = (
                "Reduce the number of __syncthreads() barriers per tile, rebalance work across warps "
                "before the barrier, or lower registers so a second CTA can co-reside on the SM and "
                "issue instructions while the first CTA is at a barrier."
            )
            return c
        if top_stall == "long_scoreboard":
            c.category = "memory-latency-bound"
            c.confidence = "high" if (top_stall_frac and top_stall_frac > 0.4) else "medium"
            c.evidence.append(
                f"Neither compute ({_fmt(comp)}%) nor memory bandwidth ({_fmt(mem)}%) is saturated, "
                f"and the dominant stall is long_scoreboard ({stalls['long_scoreboard']:.2f} warps/issue, "
                f"{c.stall_breakdown_pct.get('long_scoreboard', 0.0):.1f}% of all stalls): warps wait on "
                "global/L2 memory returns without enough in-flight concurrency to hide latency"
            )
            if roofline and roofline.memory_utilization_pct is not None:
                c.evidence.append(
                    f"Effective bandwidth is only {roofline.memory_utilization_pct:.1f}% of the measured "
                    f"{roofline.bandwidth_ceiling_gbps:.1f} GB/s ceiling, confirming memory latency "
                    "rather than bandwidth saturation"
                )
            if occ_pct is not None and occ_pct <= 50.0:
                c.evidence.append(
                    f"Predicted occupancy is {occ_pct:.1f}% ({getattr(occ, 'active_blocks_per_sm', 0)} blocks/SM, "
                    f"limited by {getattr(occ, 'limiting_factor', 'resources')}), restricting warp-level MLP"
                )
            c.confirming_experiment = (
                "Increase in-flight memory requests per thread (unroll and prefetch independent loads "
                "before consuming them, or stage tiles in shared memory) or reduce register pressure to "
                "admit another resident block per SM; latency-bound kernels speed up without reducing bytes moved."
            )
            c.alternatives = ["occupancy-limited", "instruction-latency-bound"]
            return c
        if top_stall == "short_scoreboard":
            c.category = "memory-latency-bound"
            c.confidence = "medium"
            c.evidence.append(
                f"Dominant stall is short_scoreboard ({stalls['short_scoreboard']:.2f} warps/issue, "
                f"{c.stall_breakdown_pct.get('short_scoreboard', 0.0):.1f}% of stalls): warps wait on "
                "shared-memory (LDS/LDSM) or MIO-returned data"
            )
            c.confirming_experiment = (
                "Prefetch shared-memory fragments into registers one iteration ahead (double-buffered "
                "register staging) or eliminate shared-memory bank conflicts; short_scoreboard stalls "
                "are unaffected by global-memory traffic."
            )
            return c
        if top_stall == "wait":
            c.category = "instruction-latency-bound"
            c.confidence = "medium"
            c.evidence.append(
                f"Dominant stall is wait ({stalls['wait']:.2f} warps/issue, "
                f"{c.stall_breakdown_pct.get('wait', 0.0):.1f}% of stalls): warps are stalled on "
                "fixed-latency instruction dependencies, so the kernel lacks instruction-level parallelism"
            )
            c.confirming_experiment = (
                "Increase ILP by processing independent elements per thread (more accumulators, "
                "interleaved dependency chains) and re-measure."
            )
            return c
        if top_stall in ("no_instruction", "branch_resolving"):
            c.category = "control-flow-bound"
            c.confidence = "medium"
            c.evidence.append(
                f"Dominant stall is {top_stall} ({stalls[top_stall]:.2f} warps/issue, "
                f"{c.stall_breakdown_pct.get(top_stall, 0.0):.1f}% of stalls): warps are held up by "
                "instruction fetch or branch resolution rather than by data"
            )
            c.confirming_experiment = (
                "Reduce branch divergence and shrink the unrolled body so it fits the instruction "
                "cache; re-measure and watch the same stall."
            )
            return c
        if top_stall == "not_selected":
            c.category = "occupancy-limited"
            c.confidence = "medium"
            c.evidence.append(
                f"Dominant stall is not_selected ({stalls['not_selected']:.2f} warps/issue): more warps are "
                "resident than the scheduler can issue, so latency is already well hidden and "
                "adding occupancy will not help"
            )
            c.confirming_experiment = (
                "Reduce work per thread or raise per-thread ILP instead of adding blocks; "
                "not_selected means occupancy is not the problem."
            )
            return c
        if top_stall == "math_pipe_throttle":
            c.category = "compute-bound"
            c.confidence = "medium"
            c.evidence.append(
                f"Dominant stall is math_pipe_throttle ({stalls['math_pipe_throttle']:.2f} warps/issue) even "
                "though aggregate utilisation is low: the arithmetic pipe is locally saturated"
            )
            return c
        if top_stall in ("mio_throttle", "lg_throttle"):
            c.category = "memory-latency-bound"
            c.confidence = "medium"
            c.evidence.append(
                f"Dominant stall is {top_stall} ({stalls[top_stall]:.2f} warps/issue): the memory/MIO "
                "issue queue is back-pressured by bursty requests"
            )
            c.confirming_experiment = (
                "Widen narrow loads/stores to 128-bit accesses and space out memory requests across "
                "arithmetic instructions to relieve queue back-pressure."
            )
            return c

    # --- occupancy-limited ------------------------------------------------- #
    if occ is not None and occ.occupancy_pct < 35.0 and stalls and top_stall in ("long_scoreboard", "not_selected", "wait"):
        c.category = "occupancy-limited"
        c.confidence = "medium"
        c.evidence.append(
            f"Predicted occupancy is {occ.occupancy_pct:.1f}% ({occ.active_blocks_per_sm} blocks/SM), "
            f"limited by {occ.limiting_factor}"
        )
        if occ.max_regs_per_thread_to_gain_one_block:
            c.evidence.append(
                f"Registers must fall to <= {occ.max_regs_per_thread_to_gain_one_block} per thread to "
                "admit one more resident block"
            )
        else:
            c.evidence.append(
                "Register granularity blocks any further resident block at a lower register count"
            )
        if ip is not None and getattr(ip, "has_local_memory", False):
            c.evidence.append(
                "SASS shows local-memory traffic, so register pressure has already forced spills: "
                "lowering the register ceiling further will make this worse"
            )
        c.confirming_experiment = (
            "Change only the launch configuration (block size) and re-measure at equal work; "
            "if latency does not improve, occupancy was not the limiter."
        )
        if warps is not None:
            c.evidence.append(f"NCU reports {warps:.1f}% of peak active warps")
        return c

    # --- explicit occupancy evidence from NCU ------------------------------ #
    if not_saturated and warps is not None and warps < 25.0 and not stalled_any:
        c.category = "occupancy-limited"
        c.confidence = "low"
        c.evidence.append(
            f"Only {warps:.1f}% of peak warps are active and no stall breakdown is available to "
            "explain the low utilisation"
        )
        c.confirming_experiment = "Collect the stall breakdown (Stage C) before changing anything."
        return c

    # --- inconclusive ------------------------------------------------------ #
    c.category = "mixed-or-inconclusive"
    c.confidence = "low"
    if comp is not None and mem is not None:
        c.evidence.append(
            f"Neither compute ({comp:.1f}%) nor memory ({mem:.1f}%) is near a ceiling, and the "
            "stall profile does not single out one limiter"
        )
    else:
        c.evidence.append("Insufficient counters to name a dominant limiter")
    if top_stall:
        c.evidence.append(f"Largest stall is {top_stall} ({stalls[top_stall]:.2f})")
    c.confirming_experiment = (
        "Run a Stage C pass with the full stall breakdown and cache sections, then re-classify."
    )
    c.alternatives = [cat for cat in BOTTLENECK_CATEGORIES if cat != "mixed-or-inconclusive"]
    return c


def _fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.1f}"


# --------------------------------------------------------------------------- #
# Launch-configuration / wave analysis
# --------------------------------------------------------------------------- #


@dataclass
class WaveAnalysis:
    total_ctas: int = 0
    blocks_per_wave: int = 0
    waves: float = 0.0
    full_waves: int = 0
    tail_ctas: int = 0
    tail_efficiency_pct: float = 100.0
    quantization_loss_pct: float = 0.0
    max_tail_speedup: float = 1.0
    nearest_lower_wave_ctas: int = 0
    nearest_upper_wave_ctas: int = 0
    wave_multiples: list[int] = field(default_factory=list)
    problem: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["waves"] = round(self.waves, 3)
        d["tail_efficiency_pct"] = round(self.tail_efficiency_pct, 2)
        d["quantization_loss_pct"] = round(self.quantization_loss_pct, 2)
        d["max_tail_speedup"] = round(self.max_tail_speedup, 4)
        return d


def wave_analysis(dev: DeviceInfo, occ: OccupancyModel, total_ctas: int) -> WaveAnalysis:
    """Quantify grid-tail and sub-wave inefficiency.

    Two distinct failure modes are reported separately:

    * **under-fill** — fewer CTAs in total than the device can host at once, so
      some SMs never receive work at all;
    * **tail** — more CTAs than one wave, with the final partial wave leaving a
      fraction of the machine idle.
    """
    w = WaveAnalysis(total_ctas=int(total_ctas))
    w.blocks_per_wave = dev.multi_processor_count * occ.active_blocks_per_sm
    if w.blocks_per_wave <= 0:
        w.problem = "unknown"
        w.notes.append("Resident-block count is zero, so no wave analysis is possible")
        return w
    w.waves = total_ctas / w.blocks_per_wave
    w.full_waves = total_ctas // w.blocks_per_wave
    w.tail_ctas = total_ctas - w.full_waves * w.blocks_per_wave
    w.nearest_lower_wave_ctas = w.full_waves * w.blocks_per_wave
    w.nearest_upper_wave_ctas = (w.full_waves + (1 if w.tail_ctas > 0 else 0)) * w.blocks_per_wave
    w.wave_multiples = [w.blocks_per_wave * k for k in (1, 2, 3, 4, 6, 8)]
    if w.tail_ctas == 0:
        w.tail_efficiency_pct = 100.0
        w.max_tail_speedup = 1.0
    else:
        w.tail_efficiency_pct = 100.0 * w.tail_ctas / w.blocks_per_wave
        if w.waves > 0:
            w.max_tail_speedup = math.ceil(w.waves) / w.waves
    if w.waves < 1.0:
        w.problem = "under-fill"
        w.quantization_loss_pct = 100.0 * (1.0 - w.waves)
        w.notes.append(
            f"Only {total_ctas} CTAs for {w.blocks_per_wave} concurrent slots "
            f"({dev.multi_processor_count} SMs x {occ.active_blocks_per_sm} blocks): "
            f"{100.0 * w.waves:.1f}% of the device is used at most. Increase parallelism "
            "(split the reduction dimension, refine the work partition) rather than tuning "
            "per-thread efficiency."
        )
    elif w.tail_ctas > 0:
        w.problem = "tail"
        # tail_efficiency_pct is a percentage; convert to a fraction before use.
        w.quantization_loss_pct = 100.0 * (1.0 - w.tail_efficiency_pct / 100.0) / max(w.waves, 1e-9)
        w.notes.append(
            f"{w.waves:.2f} waves: {w.full_waves} full wave(s) plus a tail of {w.tail_ctas} of "
            f"{w.blocks_per_wave} slots (max theoretical speedup from removing tail: {w.max_tail_speedup:.3f}x; "
            f"exact wave multiples: {w.nearest_lower_wave_ctas} or {w.nearest_upper_wave_ctas} CTAs)."
        )
        if w.waves < 4.0:
            w.notes.append(
                "With so few waves the tail is a material fraction of runtime; consider a "
                "grid-stride persistent formulation or a CTA count that is a multiple of "
                "the wave size."
            )
    else:
        w.problem = None
        w.notes.append(f"{w.waves:.2f} waves with no partial tail: the grid is well quantised.")
    return w


def estimate_grid_stride_efficiency(dev: DeviceInfo, occ: OccupancyModel, total_ctas: int) -> dict[str, Any]:
    """Compare a one-CTA-per-tile grid against a persistent grid-stride grid."""
    wave = wave_analysis(dev, occ, total_ctas)
    ideal_ctas = wave.full_waves * wave.blocks_per_wave if wave.full_waves else wave.blocks_per_wave
    return {
        "current_ctas": total_ctas,
        "wave_capacity": wave.blocks_per_wave,
        "waves": round(wave.waves, 3),
        "suggested_persistent_ctas": ideal_ctas,
        "rationale": (
            "A persistent grid sized to an exact multiple of the wave capacity removes the "
            "partial-wave tail, at the cost of adding a grid-stride loop."
        ),
        "caveat": "This is a structural suggestion derived from the occupancy model, not a measurement.",
    }


# --------------------------------------------------------------------------- #
# Launch-overhead measurement
# --------------------------------------------------------------------------- #


def measure_launch_overhead(
    *,
    empty_kernel_us: float | None,
    grid_sizes: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Summarise a measured empty-kernel launch cost.

    The value must come from an actual measurement of a kernel that does no work
    (for example ``--workload empty --blocks 384``); this function only packages
    it and states how it may be used.
    """
    if empty_kernel_us is None:
        return {
            "measured": False,
            "note": "no empty-kernel timing supplied; launch overhead is unknown and must not be assumed",
        }
    return {
        "measured": True,
        "launch_overhead_us": empty_kernel_us,
        "grid_sizes": list(grid_sizes) if grid_sizes else None,
        "usage": (
            "Compare a kernel's measured duration against this floor. If they are comparable, "
            "the workload is launch-dominated and the fix is fewer, larger launches rather than "
            "kernel tuning."
        ),
    }
