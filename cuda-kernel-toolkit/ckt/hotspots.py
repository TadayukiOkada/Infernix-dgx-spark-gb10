"""hotspots — NSYS SQLite kernel hotspot and functional-block budget analysis.

Reads ``CUPTI_ACTIVITY_KIND_KERNEL`` from an ``nsys`` SQLite export (captured
with ``--cuda-graph-trace=node``) and enriches every kernel launch configuration
with exact pre-calculated GB10 occupancy, register step thresholds, wave
quantisation, functional-block time shares, NVTX phase decomposition (prefill vs
decode), timeline duty cycle, and binary compiler stack spill cross-referencing.
"""

from __future__ import annotations

import bisect
import re
import sqlite3
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

from .environment import DeviceInfo
from .performance_model import occupancy_model, wave_analysis
from .util import which


FUNCTIONAL_BLOCKS: list[tuple[str, tuple[str, ...]]] = [
    (
        "moe_expert",
        (
            "gate_up_tile_ggml",
            "down_resident_ggml",
            "gate_up_kernel_ggml",
            "down_kernel_ggml",
            "gate_up_a16",
            "down_a16",
            "expert",
        ),
    ),
    (
        "dense_ggml",
        (
            "tile_kernel",
            "rows_kernel",
            "mma_rows_kernel",
            "plane_kernel",
            "encode_kernel",
            "encode_h_ggml",
        ),
    ),
    (
        "attention",
        (
            "qsa_prompt_kernel",
            "qsa_score",
            "softmax_attention",
            "attn",
            "causal_conv",
            "paged_attention",
            "attention",
        ),
    ),
    ("delta_net", ("gated_delta_net", "chunked", "conv1d")),
    (
        "hyper_conn",
        (
            "hc_down_partials",
            "hc_up_collapse",
            "combine_inject",
            "inject_kernel",
            "collapse_kernel",
        ),
    ),
    (
        "bf16_gemm",
        (
            "bf16_general_gemm_kernel",
            "bf16_a16_tma_mma_kernel",
            "bf16_a16_mma_kernel",
            "projection_wide_kernel",
        ),
    ),
    (
        "norm_route",
        (
            "norm_kernel",
            "rmsnorm",
            "router",
            "route_kernel",
            "split_vector",
            "sigmoid_gate",
            "dispatch",
            "encode_x",
        ),
    ),
]


def classify_functional_block(name: str) -> str:
    for block, keys in FUNCTIONAL_BLOCKS:
        for k in keys:
            if k in name:
                return block
    return "other"


def get_binary_spills(exe_path: str | Path) -> dict[str, dict[str, Any]]:
    """Inspect binary with cuobjdump -res-usage and demangle with c++filt to detect stack spills."""
    exe_p = Path(exe_path)
    if not exe_p.exists():
        return {}
    cuobjdump = which("cuobjdump")
    if not cuobjdump:
        return {}
    try:
        res = subprocess.check_output([cuobjdump, "-res-usage", str(exe_p)], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return {}

    sym_res: dict[str, dict[str, Any]] = {}
    cur_sym = ""
    for line in res.splitlines():
        if line.startswith(" Function "):
            cur_sym = line.split()[1].rstrip(":")
        elif "REG:" in line and cur_sym:
            m_st = re.search(r"STACK:(\d+)", line)
            m_rg = re.search(r"REG:(\d+)", line)
            if m_st and int(m_st.group(1)) > 0:
                sym_res[cur_sym] = {
                    "stack": int(m_st.group(1)),
                    "reg": int(m_rg.group(1)) if m_rg else 0,
                }
            cur_sym = ""

    if not sym_res:
        return {}

    clean_syms: list[str] = []
    sym_keys = list(sym_res.keys())
    for s in sym_keys:
        m = re.search(r"(_Z\w+)", s)
        clean_syms.append(m.group(1) if m else s)

    cppfilt = which("c++filt")
    if cppfilt:
        try:
            proc = subprocess.Popen([cppfilt], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
            stdout, _ = proc.communicate("\n".join(clean_syms), timeout=30)
            demangled_list = stdout.splitlines()
            for k, d in zip(sym_keys, demangled_list):
                sym_res[k]["demangled"] = d
        except Exception:
            pass

    return sym_res


def find_spill_for_kernel(cupti_name: str, spills: dict[str, dict[str, Any]]) -> int:
    """Match a CUPTI kernel demangled signature to a binary spill record."""
    if not spills:
        return 0
    m = re.search(r"::(\w+)(?:<([^>]+)>)?\(", cupti_name)
    if not m:
        m = re.search(r"(\w+)(?:<([^>]+)>)?\(", cupti_name)
    if not m:
        return 0
    fn_name = m.group(1)
    tmpl_args = m.group(2) or ""
    tokens = re.findall(r"\b\d+\b", tmpl_args)
    tag = tokens[0] if tokens else None

    for sym, info in spills.items():
        dem = info.get("demangled", sym)
        if f"::{fn_name}<" in dem or f" {fn_name}<" in dem or f"::{fn_name}(" in dem:
            if tag is None:
                return info["stack"]
            if tag in dem:
                return info["stack"]
    return 0


def analyze_nsys_sqlite(
    sqlite_path: str | Path,
    dev: DeviceInfo,
    *,
    top: int = 30,
    reps: int = 1,
    phase: str = "all",
    exe: str | Path | None = None,
) -> dict[str, Any]:
    """Parse an NSYS SQLite database with occupancy, wave math, NVTX phase breakdown, and spills."""
    p = Path(sqlite_path)
    if not p.exists():
        raise FileNotFoundError(f"NSYS SQLite file not found: {p}")

    con = sqlite3.connect(str(p))
    cur = con.cursor()
    try:
        tables = {r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}

        # 1. Timeline bounds and auxiliary stats
        total_wall_ms = 0.0
        sync_wait_ms = 0.0
        sync_calls = 0
        memcpy_ms = 0.0
        memcpy_bytes = 0
        memcpy_calls = 0

        if "CUPTI_ACTIVITY_KIND_KERNEL" in tables:
            bounds = cur.execute("SELECT MIN(start), MAX(end) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()
            if bounds and bounds[0] is not None and bounds[1] is not None:
                total_wall_ms = (bounds[1] - bounds[0]) / 1e6

        if "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION" in tables:
            s_row = cur.execute("SELECT COUNT(*), SUM(end - start)/1e6 FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION").fetchone()
            if s_row and s_row[0]:
                sync_calls = int(s_row[0])
                sync_wait_ms = round(float(s_row[1] or 0), 2)

        if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
            m_row = cur.execute("SELECT COUNT(*), SUM(end - start)/1e6, SUM(bytes) FROM CUPTI_ACTIVITY_KIND_MEMCPY").fetchone()
            if m_row and m_row[0]:
                memcpy_calls = int(m_row[0])
                memcpy_ms = round(float(m_row[1] or 0), 2)
                memcpy_bytes = int(m_row[2] or 0)

        # 1b. CUDA Runtime Host API overhead
        cuda_runtime_hotspots: list[dict[str, Any]] = []
        host_cuda_runtime_ms = 0.0
        if "CUPTI_ACTIVITY_KIND_RUNTIME" in tables and "StringIds" in tables:
            rt_rows = cur.execute(
                """
                SELECT s.value AS name,
                       COUNT(*) AS calls,
                       SUM(r.end - r.start)/1e6 AS total_ms,
                       AVG(r.end - r.start)/1e3 AS mean_us
                FROM CUPTI_ACTIVITY_KIND_RUNTIME r
                JOIN StringIds s ON r.nameId = s.id
                GROUP BY s.value
                ORDER BY total_ms DESC
                LIMIT 15
                """
            ).fetchall()
            for r in rt_rows:
                ms = round(float(r[2]), 3)
                host_cuda_runtime_ms += ms
                cuda_runtime_hotspots.append({
                    "name": r[0],
                    "calls": int(r[1]),
                    "total_ms": ms,
                    "mean_us": round(float(r[3]), 2),
                })

        # 1c. Synchronization wait breakdown by sync type
        sync_breakdown: list[dict[str, Any]] = []
        if "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION" in tables:
            sync_query = """
                SELECT COALESCE(e.name, 'SyncType_' || s.syncType) AS sync_type,
                       COUNT(*) AS calls,
                       SUM(s.end - s.start)/1e6 AS total_ms
                FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION s
                LEFT JOIN ENUM_CUPTI_SYNC_TYPE e ON s.syncType = e.id
                GROUP BY s.syncType
                ORDER BY total_ms DESC
            """ if "ENUM_CUPTI_SYNC_TYPE" in tables else """
                SELECT 'SyncType_' || syncType, COUNT(*), SUM(end - start)/1e6
                FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION
                GROUP BY syncType
                ORDER BY SUM(end - start) DESC
            """
            for row in cur.execute(sync_query).fetchall():
                sync_breakdown.append({
                    "sync_type": row[0],
                    "calls": int(row[1]),
                    "total_ms": round(float(row[2] or 0), 3),
                })

        # 1d. CPU IP sampling & PMU events (when kernel.perf_event_paranoid <= 1)
        cpu_samples_count = 0
        cpu_hotspots: list[dict[str, Any]] = []
        has_masked_kernel_symbols = False

        sample_table = None
        for cand in ("SMP_SAMPLE", "CPU_SAMPLES", "SAMPLE"):
            if cand in tables:
                sample_table = cand
                break

        if sample_table:
            c_row = cur.execute(f"SELECT COUNT(*) FROM {sample_table}").fetchone()
            if c_row and c_row[0]:
                cpu_samples_count = int(c_row[0])

            bt_table = None
            for cand in ("SMP_BACKTRACE", "BACKTRACE", "Backtrace"):
                if cand in tables:
                    bt_table = cand
                    break

            if bt_table and "StringIds" in tables:
                cols = {c[1] for c in cur.execute(f"PRAGMA table_info({bt_table})").fetchall()}
                sym_col = "symbolId" if "symbolId" in cols else "nameId" if "nameId" in cols else None
                if sym_col:
                    s_rows = cur.execute(
                        f"""
                        SELECT s.value, COUNT(*) as samples
                        FROM {sample_table} smp
                        JOIN {bt_table} bt ON smp.backtraceId = bt.id
                        JOIN StringIds s ON bt.{sym_col} = s.id
                        GROUP BY s.value
                        ORDER BY samples DESC
                        LIMIT 20
                        """
                    ).fetchall()
                    for sym, cnt in s_rows:
                        if "00000000" in sym or "[kernel.kallsyms]" in sym:
                            has_masked_kernel_symbols = True
                        cpu_hotspots.append({
                            "symbol": sym,
                            "samples": int(cnt),
                            "share_pct": round(100.0 * cnt / cpu_samples_count, 2) if cpu_samples_count > 0 else 0.0,
                        })

        # 2. NVTX Phase matching
        phase_intervals: list[tuple[int, int, str]] = []
        phase_starts: list[int] = []
        phase_stats: dict[str, dict[str, Any]] = {
            name: {"wall_ms": 0.0, "kernel_ms": 0.0, "calls": 0, "duty_pct": 0.0}
            for name in ["prefill", "decode", "engine.load", "device.wait"]
        }

        if "NVTX_EVENTS" in tables:
            nvtx_rows = cur.execute(
                """
                SELECT COALESCE(n.text, s.value) as name, n.start, n.end
                FROM NVTX_EVENTS n
                LEFT JOIN StringIds s ON n.textId = s.id
                WHERE n.end IS NOT NULL AND COALESCE(n.text, s.value) IN ('prefill', 'decode', 'engine.load', 'device.wait')
                ORDER BY n.start
                """
            ).fetchall()
            for r in nvtx_rows:
                name, st, en = r[0], int(r[1]), int(r[2])
                phase_intervals.append((st, en, name))
                phase_starts.append(st)
                if name in phase_stats:
                    phase_stats[name]["wall_ms"] += (en - st) / 1e6

        # 3. Kernel retrieval (filtered by phase if requested)
        if phase and phase.lower() in ("prefill", "decode"):
            target_phase = phase.lower()
            target_ranges = [iv for iv in phase_intervals if iv[2] == target_phase]
            # Fetch all raw kernels and filter in memory for speed
            all_kernels = cur.execute(
                """
                SELECT s.value AS name,
                       (k.end - k.start) AS ns,
                       k.start, k.end,
                       k.gridX, k.gridY, k.gridZ,
                       k.blockX, k.blockY, k.blockZ,
                       k.registersPerThread AS regs,
                       k.dynamicSharedMemory AS dyn_smem,
                       k.staticSharedMemory AS stat_smem
                FROM CUPTI_ACTIVITY_KIND_KERNEL k
                JOIN StringIds s ON k.demangledName = s.id
                """
            ).fetchall()
            t_starts = [iv[0] for iv in target_ranges]
            t_ends = [iv[1] for iv in target_ranges]

            filtered_map: dict[tuple, list[int]] = defaultdict(lambda: [0, 0, 1<<62, 0])
            for k in all_kernels:
                k_start, k_end, k_dur = int(k[2]), int(k[3]), int(k[1])
                idx = bisect.bisect_right(t_starts, k_start) - 1
                if idx >= 0 and k_start <= t_ends[idx]:
                    key = (k[0], k[4], k[5], k[6], k[7], k[8], k[9], k[10], k[11], k[12])
                    st = filtered_map[key]
                    st[0] += 1
                    st[1] += k_dur
                    if k_dur < st[2]: st[2] = k_dur
                    if k_dur > st[3]: st[3] = k_dur

            rows = [
                (k[0], v[0], v[1], v[2], v[3], k[1], k[2], k[3], k[4], k[5], k[6], k[7], k[8], k[9])
                for k, v in filtered_map.items()
            ]
        else:
            rows = cur.execute(
                """
                SELECT s.value AS name,
                       COUNT(*) AS calls,
                       SUM(k.end - k.start) AS ns,
                       MIN(k.end - k.start) AS min_ns,
                       MAX(k.end - k.start) AS max_ns,
                       k.gridX, k.gridY, k.gridZ,
                       k.blockX, k.blockY, k.blockZ,
                       k.registersPerThread AS regs,
                       k.dynamicSharedMemory AS dyn_smem,
                       k.staticSharedMemory AS stat_smem
                FROM CUPTI_ACTIVITY_KIND_KERNEL k
                JOIN StringIds s ON k.demangledName = s.id
                GROUP BY s.value, k.gridX, k.gridY, k.gridZ,
                         k.blockX, k.blockY, k.blockZ,
                         k.registersPerThread, k.dynamicSharedMemory, k.staticSharedMemory
                """
            ).fetchall()

            # Attribute kernels to phase budgets when computing all
            if phase_starts:
                kernel_timings = cur.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
                for kt in kernel_timings:
                    k_st, k_en = int(kt[0]), int(kt[1])
                    idx = bisect.bisect_right(phase_starts, k_st) - 1
                    if idx >= 0 and phase_intervals[idx][0] <= k_st and k_en <= phase_intervals[idx][1]:
                        p_name = phase_intervals[idx][2]
                        if p_name in phase_stats:
                            phase_stats[p_name]["kernel_ms"] += (k_en - k_st) / 1e6
                            phase_stats[p_name]["calls"] += 1

                for st in phase_stats.values():
                    if st["wall_ms"] > 0:
                        st["duty_pct"] = round(100.0 * st["kernel_ms"] / st["wall_ms"], 1)
                    st["wall_ms"] = round(st["wall_ms"], 2)
                    st["kernel_ms"] = round(st["kernel_ms"], 2)

    finally:
        con.close()

    # 4. Binary compiler stack spills inspection
    binary_spills: dict[str, dict[str, Any]] = {}
    if exe:
        binary_spills = get_binary_spills(exe)
    elif Path("build-native/bench/infernix_bench").exists():
        binary_spills = get_binary_spills("build-native/bench/infernix_bench")

    total_busy_ns = sum(int(r[2]) for r in rows)
    total_calls = sum(int(r[1]) for r in rows)
    reps = max(1, int(reps))

    if total_busy_ns == 0:
        return {
            "sqlite": str(p),
            "total_busy_ms": 0.0,
            "total_calls": 0,
            "error": "No CUPTI_ACTIVITY_KIND_KERNEL rows found for the requested phase.",
            "by_geometry": [],
            "by_name": [],
            "by_functional_block": [],
            "phase_budget": [],
            "spills_detected": [],
        }

    geom_records: list[dict[str, Any]] = []
    by_name_agg: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"ns": 0, "calls": 0, "geometries": 0, "functional_block": "other", "spill_bytes": 0}
    )
    by_block_agg: dict[str, dict[str, Any]] = defaultdict(lambda: {"ns": 0, "calls": 0, "kernels": set()})
    spills_detected_set: set[str] = set()

    for (
        name,
        calls,
        ns,
        min_ns,
        max_ns,
        gx,
        gy,
        gz,
        bx,
        by,
        bz,
        regs,
        dyn_smem,
        stat_smem,
    ) in sorted(rows, key=lambda r: -int(r[2])):
        calls_i = int(calls)
        ns_i = int(ns)
        ctas = int(gx) * int(gy) * int(gz)
        threads = int(bx) * int(by) * int(bz)
        regs_i = int(regs or 0)
        smem_i = int(dyn_smem or 0) + int(stat_smem or 0)
        fblock = classify_functional_block(str(name))

        spill_bytes = find_spill_for_kernel(str(name), binary_spills)
        if spill_bytes > 0:
            spills_detected_set.add(f"{str(name).split('(')[0]} ({spill_bytes} B stack spill)")

        occ = occupancy_model(
            dev,
            threads_per_block=threads,
            registers_per_thread=regs_i,
            shared_mem_per_block=smem_i,
        )
        wv = wave_analysis(dev, occ, ctas)

        share_pct = 100.0 * ns_i / total_busy_ns
        mean_us = (ns_i / calls_i) / 1e3
        geom_records.append(
            {
                "name": str(name),
                "functional_block": fblock,
                "share_pct": round(share_pct, 2),
                "total_ms": round(ns_i / 1e6, 3),
                "per_rep_ms": round((ns_i / 1e6) / reps, 3),
                "calls": calls_i,
                "calls_per_rep": round(calls_i / reps, 2),
                "mean_us": round(mean_us, 2),
                "min_us": round(int(min_ns) / 1e3, 2),
                "max_us": round(int(max_ns) / 1e3, 2),
                "grid": [int(gx), int(gy), int(gz)],
                "total_ctas": ctas,
                "block": [int(bx), int(by), int(bz)],
                "threads_per_block": threads,
                "registers_per_thread": regs_i,
                "shared_mem_bytes": smem_i,
                "active_blocks_per_sm": occ.active_blocks_per_sm,
                "active_warps_per_sm": occ.active_warps_per_sm,
                "occupancy_pct": round(occ.occupancy_pct, 2),
                "limiting_factor": occ.limiting_factor,
                "max_regs_to_gain_one_block": occ.max_regs_per_thread_to_gain_one_block,
                "free_reg_headroom": occ.free_reg_headroom_per_thread,
                "blocks_per_wave": wv.blocks_per_wave,
                "waves": round(wv.waves, 2),
                "tail_efficiency_pct": round(wv.tail_efficiency_pct, 1),
                "wave_problem": wv.problem,
                "spill_bytes": spill_bytes,
                "has_spill": spill_bytes > 0,
            }
        )

        bn = by_name_agg[str(name)]
        bn["ns"] += ns_i
        bn["calls"] += calls_i
        bn["geometries"] += 1
        bn["functional_block"] = fblock
        if spill_bytes > bn["spill_bytes"]:
            bn["spill_bytes"] = spill_bytes

        bb = by_block_agg[fblock]
        bb["ns"] += ns_i
        bb["calls"] += calls_i
        bb["kernels"].add(str(name))

    by_name_list: list[dict[str, Any]] = []
    for name, d in sorted(by_name_agg.items(), key=lambda kv: -kv[1]["ns"])[:top]:
        by_name_list.append(
            {
                "name": name,
                "functional_block": d["functional_block"],
                "share_pct": round(100.0 * d["ns"] / total_busy_ns, 2),
                "total_ms": round(d["ns"] / 1e6, 3),
                "per_rep_ms": round((d["ns"] / 1e6) / reps, 3),
                "calls": d["calls"],
                "mean_us": round((d["ns"] / d["calls"]) / 1e3, 2),
                "distinct_geometries": d["geometries"],
                "spill_bytes": d["spill_bytes"],
                "has_spill": d["spill_bytes"] > 0,
            }
        )

    by_block_list: list[dict[str, Any]] = []
    for fblock, d in sorted(by_block_agg.items(), key=lambda kv: -kv[1]["ns"]):
        by_block_list.append(
            {
                "functional_block": fblock,
                "share_pct": round(100.0 * d["ns"] / total_busy_ns, 2),
                "total_ms": round(d["ns"] / 1e6, 3),
                "per_rep_ms": round((d["ns"] / 1e6) / reps, 3),
                "calls": d["calls"],
                "distinct_kernels": len(d["kernels"]),
            }
        )

    phase_budget_list = [
        {
            "phase": k,
            "wall_ms": v["wall_ms"],
            "kernel_ms": v["kernel_ms"],
            "calls": v["calls"],
            "duty_pct": v["duty_pct"],
        }
        for k, v in phase_stats.items()
        if v["wall_ms"] > 0 or v["calls"] > 0
    ]

    total_busy_ms = round(total_busy_ns / 1e6, 3)
    gpu_duty_cycle_pct = round(100.0 * total_busy_ms / total_wall_ms, 1) if total_wall_ms > 0 else 0.0
    gpu_idle_bubbles_ms = round(max(0.0, total_wall_ms - total_busy_ms), 3) if total_wall_ms > 0 else 0.0

    return {
        "sqlite": str(p),
        "device": dev.name,
        "phase": phase,
        "reps": reps,
        "timeline": {
            "total_wall_ms": round(total_wall_ms, 2),
            "gpu_busy_ms": total_busy_ms,
            "gpu_duty_cycle_pct": gpu_duty_cycle_pct,
            "gpu_idle_bubbles_ms": gpu_idle_bubbles_ms,
            "sync_wait_ms": sync_wait_ms,
            "sync_calls": sync_calls,
            "memcpy_ms": memcpy_ms,
            "memcpy_gib": round(memcpy_bytes / (1024**3), 2),
            "memcpy_calls": memcpy_calls,
            "host_cuda_runtime_ms": round(host_cuda_runtime_ms, 2),
        },
        "phase_budget": phase_budget_list,
        "spills_detected": sorted(spills_detected_set),
        "cuda_runtime_hotspots": cuda_runtime_hotspots,
        "sync_breakdown": sync_breakdown,
        "cpu_profiling": {
            "enabled": cpu_samples_count > 0,
            "samples_count": cpu_samples_count,
            "has_masked_kernel_symbols": has_masked_kernel_symbols,
            "hotspots": cpu_hotspots,
        },
        "total_busy_ms": total_busy_ms,
        "per_rep_busy_ms": round((total_busy_ns / 1e6) / reps, 3),
        "total_calls": total_calls,
        "distinct_kernels": len(by_name_agg),
        "by_functional_block": by_block_list,
        "by_name": by_name_list,
        "by_geometry": geom_records[:top],
    }
