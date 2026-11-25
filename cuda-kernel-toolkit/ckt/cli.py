"""cli — command-line entry point for the toolkit.

Subcommands are organised by how much of the pipeline they run:

    ckt env                 discover hardware and toolchain; measure ceilings
    ckt static              source and binary analysis without running anything
    ckt profile             NCU metric collection for one kernel
    ckt search              launch-parameter search with correctness gating
    ckt analyze             the full pipeline: discover -> static -> profile -> report
    ckt report              render a stored analysis as Markdown/HTML/JSON
    ckt experiments         inspect and manage experiment history
    ckt doctor              verify that the environment can support each stage

Every subcommand validates its inputs before doing any work, so a mistyped
workload name or a missing binary fails immediately rather than after an
expensive build or profile run.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .binary_inspector import BinaryInspector
from .build_manager import BuildConfig, BuildManager
from .cache import Cache
from .correctness import gate_from_verification, verification_from_records
from .environment import Environment
from .experiment_manager import ExperimentManager, ExperimentRecord, identity_from
from .parameter_search import (
    LaunchConfig,
    SearchSpace,
    generate_candidates,
    revalidate,
    run_search,
    summarise_search,
)
from .performance_model import (
    classify_bottleneck,
    occupancy_model,
    roofline_report,
    wave_analysis,
)
from .profiler import Profiler, baseline_timing, ncu_base_name, verify_harness
from .ab_runner import compare_shape_csvs, run_interleaved_ab
from .cliffs import (
    ALL_FAMILIES,
    format_cli_summary,
    merge_reports,
    parse_family_list,
    render_markdown_report,
    run_cliffs_analysis,
)
from .recommendations import recommend
from .reporting import environment_section, report_from_analysis
from .shapes import benchmark_shape, discover_shapes, parse_shape_file
from .source_discovery import build_inventory, discover_kernels
from .autotune.cli import register_autotune_subparsers
from .kt.cli import register_kt_subparsers
from .util import LOG, ToolError, read_json, setup_logging, write_json

DEFAULT_CACHE = ".ckt_cache"
DEFAULT_RESULTS = "results"


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


def _simple_yaml_load(text: str) -> dict[str, Any]:
    """Minimal stdlib parser for flat/1-level YAML configs when PyYAML is absent."""
    result: dict[str, Any] = {}
    current_list_key: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if stripped.startswith("- ") and current_list_key is not None and indent > 0:
            val_s = stripped[2:].strip().strip("'\"")
            result.setdefault(current_list_key, []).append(_coerce_yaml_scalar(val_s))
            continue
        if ":" in stripped and indent == 0:
            k, _, v = stripped.partition(":")
            key = k.strip()
            v = v.strip()
            if not v:
                current_list_key = key
                result[key] = []
            elif v.startswith("[") and v.endswith("]"):
                current_list_key = None
                items = [x.strip().strip("'\"") for x in v[1:-1].split(",") if x.strip()]
                result[key] = [_coerce_yaml_scalar(x) for x in items]
            else:
                current_list_key = None
                result[key] = _coerce_yaml_scalar(v.strip("'\""))
    return result


def _coerce_yaml_scalar(val: str) -> Any:
    if val.lower() in ("true", "yes"):
        return True
    if val.lower() in ("false", "no"):
        return False
    try:
        return int(val)
    except ValueError:
        try:
            return float(val)
        except ValueError:
            return val


def load_config(path: str | Path | None) -> dict[str, Any]:
    """Load a YAML or TOML configuration file.

    Both formats are optional dependencies; when neither parser is present and a
    config file was requested, that is reported as an error rather than ignored.
    """
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"config file not found: {p}")
    text = p.read_text()
    if p.suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore

            data = yaml.safe_load(text) or {}
        except ImportError:
            data = _simple_yaml_load(text)
    elif p.suffix == ".toml":
        try:
            import tomllib
        except ImportError:
            raise SystemExit("TOML config requires Python 3.11+ for tomllib")
        data = tomllib.loads(text)
    elif p.suffix == ".json":
        data = json.loads(text)
    else:
        raise SystemExit(f"unsupported config extension '{p.suffix}'; use .yaml, .toml or .json")
    if not isinstance(data, dict):
        raise SystemExit(f"config file {p} must contain a mapping at the top level")
    return data


def _merged(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(getattr(args, "config", None))
    # Command-line values take precedence over the file, but only when the user
    # actually supplied them (argparse defaults of None mark "not supplied").
    for key in (
        "exe", "workload", "workloads", "problem_size", "iterations", "warmup", "repeats",
        "strategy", "blocks", "threads", "shared_mem", "results_dir",
    ):
        v = getattr(args, key, None)
        if v is not None:
            cfg[key] = v
    return cfg


def _open_env(args: argparse.Namespace) -> Environment:
    cache_dir = Path(getattr(args, "cache_dir", DEFAULT_CACHE))
    cache_dir.mkdir(parents=True, exist_ok=True)
    env_path = cache_dir / "env.json"
    if env_path.exists() and not getattr(args, "refresh", False):
        try:
            raw = read_json(env_path)
            return _env_from_dict(raw)
        except (ValueError, KeyError, TypeError):
            LOG.debug("stored environment unusable; rediscovering")
    return Environment.discover(
        cache_dir,
        device_index=getattr(args, "device", 0),
        rebuild_native=getattr(args, "rebuild", False),
        measure_bandwidth=getattr(args, "measure_bandwidth", True),
        measure_compute=getattr(args, "measure_compute", True),
    )


def _env_from_dict(raw: dict[str, Any]) -> Environment:
    """Rehydrate an Environment from its serialized form."""
    from . import spec as _spec
    from .environment import DeviceInfo, ToolInfo, detect_tools

    env = Environment(
        hostname=raw.get("hostname", ""),
        platform=raw.get("platform", ""),
        python_version=raw.get("python_version", ""),
    )
    dev_raw = raw.get("device") or {}
    dev = DeviceInfo()
    for f in DeviceInfo.__dataclass_fields__:  # type: ignore[attr-defined]
        if f in dev_raw and dev_raw[f] is not None:
            try:
                setattr(dev, f, dev_raw[f])
            except (AttributeError, TypeError):
                pass
    # Rehydrate DTypePeak entries from dicts.
    peaks_raw = dev_raw.get("dtype_peaks") or {}
    dev.dtype_peaks = {}
    for k, v in peaks_raw.items():
        if isinstance(v, _spec.DTypePeak):
            dev.dtype_peaks[k] = v
        elif isinstance(v, dict) and v.get("tflops") is not None:
            dev.dtype_peaks[k] = _spec.DTypePeak(
                str(v.get("dtype", k)),
                float(v["tflops"]),
                str(v.get("source", "measured")),
            )
    # Re-attach the architecture facts, which are objects rather than plain data.
    dev.arch_facts = _spec.arch_facts(dev.cc_major, dev.cc_minor)
    env.bandwidth = raw.get("bandwidth") or {}
    if env.bandwidth:
        from .environment import _to_float
        if getattr(dev, "measured_read_bandwidth_gbps", None) is None:
            dev.measured_read_bandwidth_gbps = _to_float(env.bandwidth.get("read", {}).get("gbps")) or dev.measured_bandwidth_gbps
        if getattr(dev, "measured_write_bandwidth_gbps", None) is None:
            dev.measured_write_bandwidth_gbps = _to_float(env.bandwidth.get("write", {}).get("gbps"))
        if getattr(dev, "measured_copy_bandwidth_gbps", None) is None:
            dev.measured_copy_bandwidth_gbps = _to_float(env.bandwidth.get("copy", {}).get("gbps"))
    env.compute = raw.get("compute") or {}
    if env.compute:
        dev.apply_compute_probe(env.compute)
    env.device = dev
    tools_raw = raw.get("tools") or {}
    if tools_raw:
        for k, tv in tools_raw.items():
            if isinstance(tv, dict):
                env.tools[k] = ToolInfo(
                    name=str(tv.get("name", k)),
                    path=tv.get("path"),
                    version=tv.get("version"),
                    available=bool(tv.get("available")),
                )
    else:
        env.tools = detect_tools()
    env.warnings = list(raw.get("warnings") or [])
    env.profiling = raw.get("profiling") or {}
    return env


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #


def cmd_env(args: argparse.Namespace) -> int:
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    env = Environment.discover(
        cache_dir,
        device_index=args.device,
        rebuild_native=args.rebuild,
        measure_bandwidth=not args.no_measure_bandwidth,
        measure_compute=not args.no_measure_compute,
    )
    env.save(cache_dir / "env.json")
    if args.json:
        print(json.dumps(env.to_dict(), indent=2))
        return 0
    dev = env.device
    bw, bw_src = dev.peak_bandwidth_gbps()
    ct, ct_src = dev.compute_ceiling_tflops
    print(f"device            {dev.name}")
    print(f"compute capability{dev.compute_capability:>8}   arch {dev.arch}")
    print(f"multiprocessors   {dev.multi_processor_count}")
    print(f"max threads/SM    {dev.max_threads_per_multiprocessor}   (warps/SM {dev.max_warps_per_multiprocessor})")
    print(f"registers/SM      {dev.regs_per_multiprocessor}")
    print(f"shared mem/SM     {dev.shared_mem_per_multiprocessor} B   (per block {dev.shared_mem_per_block}, opt-in {dev.shared_mem_per_block_optin})")
    print(f"L2 cache          {dev.l2_cache_size} B")
    print(f"unified memory    {dev.integrated_uma}")
    print(f"bandwidth         {bw:.1f} GB/s  ({bw_src})" if bw else f"bandwidth         unknown ({bw_src})")
    if dev.measured_read_bandwidth_gbps or dev.measured_write_bandwidth_gbps or dev.measured_copy_bandwidth_gbps:
        r_bw = dev.measured_read_bandwidth_gbps or bw
        w_bw = dev.measured_write_bandwidth_gbps
        c_bw = dev.measured_copy_bandwidth_gbps
        if r_bw:
            print(f"  read only       {r_bw:.1f} GB/s  (measured)")
        if w_bw:
            print(f"  write only      {w_bw:.1f} GB/s  (measured)")
        if c_bw:
            print(f"  read + write/copy {c_bw:.1f} GB/s  (measured)")
    if dev.measured_smem_bandwidth_gbps:
        print(f"shared mem bw     {dev.measured_smem_bandwidth_gbps:.1f} GB/s  (measured)")
    if dev.features:
        enabled = [k for k, v in dev.features.items() if v]
        print(f"features          {', '.join(enabled)}")
    print(f"FP32 ceiling      {ct:.2f} TFLOP/s  ({ct_src})" if ct else f"FP32 ceiling      unknown ({ct_src})")
    for dtype, dp in dev.dtype_peaks.items():
        print(f"  {dtype:<14}{dp.tflops:.3f} TFLOP/s  ({dp.source})")
    for n in dev.measured_ratio_notes:
        print(f"  note: {n}")
    if getattr(args, "tools", False) or getattr(args, "profiling", False) or True:
        prof = getattr(env, "profiling", None) or {}
        if prof:
            paranoid = prof.get("perf_event_paranoid")
            kptr = prof.get("kptr_restrict")
            p_desc = "unlocked: CPU PMU hardware counters, CPU IP sampling, and context switches enabled" if (paranoid is not None and paranoid <= 0) else f"locked (current: {paranoid}) -> run: sudo sysctl -w kernel.perf_event_paranoid=-1"
            k_desc = "unlocked: kernel symbol resolution in call stacks enabled" if kptr == 0 else f"restricted (current: {kptr}) -> run: sudo sysctl -w kernel.kptr_restrict=0"
            print("\nprofiling permissions")
            print(f"  perf_event_paranoid {paranoid if paranoid is not None else 'unknown':>2}   ({p_desc})")
            print(f"  kptr_restrict       {kptr if kptr is not None else 'unknown':>2}   ({k_desc})")
            if prof.get("pending_sysctl_reload"):
                print("  ! [NOTE] /etc/sysctl.d configuration differs from live /proc/sys settings; run 'sudo sysctl --system' to apply.")
    if args.tools:
        print("\ntoolchain")
        for name, t in sorted(env.tools.items()):
            status = "ok" if t.available else "MISSING"
            print(f"  {name:<16}{status:<9}{t.version or ''}  {t.path or ''}")
    if env.warnings:
        print("\nwarnings")
        for w in env.warnings:
            print(f"  ! {w}")
    return 0


def _batch_demangle(symbols: list[str]) -> dict[str, str]:
    """Batch-demangle C++ symbols via ``cu++filt`` or ``c++filt`` in a single call."""
    if not symbols:
        return {}
    from .util import which
    filt = which("cu++filt") or which("c++filt")
    if not filt:
        return {s: s for s in symbols}
    import subprocess
    try:
        proc = subprocess.run(
            [filt],
            input="\n".join(symbols),
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        lines = proc.stdout.splitlines()
        if len(lines) == len(symbols):
            return dict(zip(symbols, (ln.strip() for ln in lines)))
    except Exception:
        pass
    return {s: s for s in symbols}


def _infer_symbol_threads_and_min_blocks(
    demangled: str,
    explicit_threads: int | None = None,
) -> tuple[int, int | None, str]:
    """Infer ``(threads_per_block, declared_min_blocks_per_sm, source_kind)`` from a demangled C++ kernel signature."""
    if explicit_threads is not None and explicit_threads > 0:
        return int(explicit_threads), None, "explicit"

    # 1. *SlicedKMmaSchedule<BlockRows, BlockTokens, KWarps, Stages, WCache, ACache, MinBlocksPerSm, ...>
    m = re.search(
        r"SlicedKMmaSchedule<\(int\)(\d+),\s*\(int\)(\d+),\s*\(int\)(\d+),\s*\(int\)(\d+),\s*[^,]+,\s*[^,]+,\s*\(int\)(\d+)",
        demangled,
    )
    if m:
        k_warps = int(m.group(3))
        min_blks = int(m.group(5))
        if 1 <= k_warps <= 32:
            return k_warps * 32, min_blks, "template(SlicedKMmaSchedule)"

    # 2. *MmaSchedule<BlockRows, BlockTokens, BlockK, WarpRows, WarpTokens, ...>
    m = re.search(
        r"MmaSchedule<\(int\)(\d+),\s*\(int\)(\d+),\s*\(int\)(\d+),\s*\(int\)(\d+),\s*\(int\)(\d+)",
        demangled,
    )
    if m:
        b_rows, b_toks, _b_k, w_rows, w_toks = (int(m.group(i)) for i in range(1, 6))
        if w_rows > 0 and w_toks > 0 and b_rows >= w_rows and b_toks >= w_toks:
            warps = (b_rows // w_rows) * (b_toks // w_toks)
            if 1 <= warps <= 32:
                return warps * 32, None, "template(MmaSchedule)"

    # 3. *GemvSchedule<RowsPerBlock, WarpsPerRow, ...>
    m = re.search(r"GemvSchedule<\(int\)(\d+),\s*\(int\)(\d+)", demangled)
    if m:
        rows_blk, warps_row = int(m.group(1)), int(m.group(2))
        warps = rows_blk * warps_row
        if 1 <= warps <= 32:
            return warps * 32, None, "template(GemvSchedule)"

    # 4. *SimtSchedule<BlockRows, BlockTokens, WarpsPerRow, ...>
    m = re.search(r"SimtSchedule<\(int\)(\d+),\s*\(int\)(\d+),\s*\(int\)(\d+)", demangled)
    if m:
        b_rows, _b_toks, warps_row = int(m.group(1)), int(m.group(2)), int(m.group(3))
        warps = b_rows * warps_row
        if 1 <= warps <= 32:
            return warps * 32, None, "template(SimtSchedule)"

    return 256, None, "default(256)"


def cmd_static(args: argparse.Namespace) -> int:
    """Source and binary inspection; runs nothing on the GPU."""
    sources = list(args.sources or [])
    exe = getattr(args, "exe", None) or getattr(args, "exe_pos", None)
    if not sources and not exe:
        print("error: supply --sources and/or --exe (or positional <exe>)", file=sys.stderr)
        return 2

    insp = BinaryInspector(Cache(args.cache_dir))
    binaries: list[Path] = []
    bin_syms: dict[str, list[Any]] = {}
    demangled_map: dict[str, str] = {}
    if exe:
        binaries.append(Path(exe))
        usage = insp.resource_usage(exe)
        demangled_map = _batch_demangle(list(usage.keys()))
        for sym, ru in usage.items():
            base = ncu_base_name(sym)
            bin_syms.setdefault(base, []).append(ru)

    kfilter = getattr(args, "kernel", None)
    spills_only = bool(getattr(args, "spills_only", False))
    raw_threads = getattr(args, "threads", None)
    explicit_threads: int | None = None
    if raw_threads is not None and str(raw_threads).lower() not in ("auto", ""):
        explicit_threads = int(raw_threads)
    dev = _open_env(args).device

    inv: list[Any] = []
    json_out: list[dict[str, Any]] = []
    launch_bounds_warnings: list[dict[str, Any]] = []
    if sources:
        kernels, launches = discover_kernels(sources)
        entries = build_inventory(kernels, [], launches)
        for e in entries:
            if kfilter and kfilter not in e.name:
                continue
            src = e.source
            loc = f"{src.file}:{src.line}" if src else "?"
            regs = sorted({r.reg for r in bin_syms.get(e.name, []) if r.reg})
            if not args.json:
                print(f"  {e.name:<32} {loc}")
                if e.mangled_symbols:
                    print(f"      symbols: {', '.join(s[:70] for s in e.mangled_symbols)}")
                if regs:
                    print(f"      registers/thread: {regs}")
            inv.append(e)
            json_out.append({"name": e.name, "location": loc, "registers": regs})
    for b in binaries:
        listed = {i.name for i in inv}
        extra = sorted(set(bin_syms) - listed)
        for name in extra:
            matching_rus = []
            for ru in bin_syms[name]:
                dem = demangled_map.get(ru.symbol, ru.symbol)
                if kfilter and (kfilter not in name and kfilter not in ru.symbol and kfilter not in dem):
                    continue
                if spills_only and int(ru.stack or 0) == 0 and int(ru.local or 0) == 0:
                    continue
                matching_rus.append((ru, dem))
            if not matching_rus:
                continue
            if not args.json:
                print(f"  {name:<32} ({len(matching_rus)} instantiation(s))")
            for ru, dem in matching_rus:
                threads, decl_min_blks, thr_src = _infer_symbol_threads_and_min_blocks(dem, explicit_threads)
                occ_info = None
                sass_prof = None
                lb_warning = None
                if dev and ru.reg:
                    occ_obj = occupancy_model(
                        dev,
                        threads_per_block=threads,
                        registers_per_thread=int(ru.reg),
                        shared_mem_per_block=int(ru.shared or 0),
                    )
                    occ_info = occ_obj.to_dict()
                    occ_info["threads_source"] = thr_src
                    if decl_min_blks is not None:
                        occ_info["declared_min_blocks_per_sm"] = decl_min_blks
                    if int(ru.stack or 0) > 0 and occ_obj.blocks_by_shared < occ_obj.blocks_by_registers:
                        max_regs_at_smem_cap = int(ru.reg) + int(occ_obj.free_reg_headroom_per_thread or 0)
                        lb_warning = {
                            "symbol": ru.symbol,
                            "demangled": dem,
                            "threads_per_block": threads,
                            "reg": ru.reg,
                            "stack": ru.stack,
                            "shared": ru.shared,
                            "blocks_by_registers": occ_obj.blocks_by_registers,
                            "blocks_by_shared": occ_obj.blocks_by_shared,
                            "recommended_min_blocks_per_sm": occ_obj.blocks_by_shared,
                            "max_regs_without_occupancy_loss": max_regs_at_smem_cap,
                            "insight": (
                                f"OVER-CONSTRAINED LAUNCH BOUNDS (@{threads}t): REG={ru.reg} is capped for "
                                f"{occ_obj.blocks_by_registers} blk/SM, causing STACK={ru.stack}B spills, "
                                f"but SHARED={ru.shared}B limits hardware residency to {occ_obj.blocks_by_shared} blk/SM! "
                                f"Lower MinBlocksPerSm to {occ_obj.blocks_by_shared} (allows up to {max_regs_at_smem_cap} regs/thread "
                                f"with 0% occupancy loss and eliminates local-memory spills)."
                            ),
                        }
                        launch_bounds_warnings.append(lb_warning)
                if getattr(args, "sass", False) and exe:
                    try:
                        dis = insp.disassemble(exe, ru.symbol)
                        sass_prof = dis.profile.to_dict()
                    except ToolError:
                        sass_prof = None
                if not args.json:
                    occ_str = ""
                    if occ_info:
                        occ_str = (
                            f" -> {occ_info['active_blocks_per_sm']} blk/SM "
                            f"({occ_info['occupancy_pct']:.0f}% occ @{threads}t, "
                            f"lim={occ_info['limiting_factor']}, "
                            f"+1blk<={occ_info['max_regs_per_thread_to_gain_one_block']}r, "
                            f"free_reg={occ_info['free_reg_headroom_per_thread']}r)"
                        )
                    dem_short = dem if len(dem) <= 135 else dem[:132] + "..."
                    print(
                        f"      REG={ru.reg:3d} STACK={ru.stack:4d} SHARED={ru.shared:6d} LOCAL={ru.local:3d}"
                        f"{occ_str}\n        {dem_short}"
                    )
                    if lb_warning:
                        print(f"        ! {lb_warning['insight']}")
                    if sass_prof:
                        print(
                            f"        SASS: total={sass_prof['total']} ldg={sass_prof['global_loads']} "
                            f"(max {sass_prof['max_load_width']}B) lds={sass_prof['shared_loads']} "
                            f"tc={sass_prof['tensor_core_ops']} (int={sass_prof['int_tensor_core_ops']}, "
                            f"flt={sass_prof['float_tensor_core_ops']}, fp8={sass_prof['fp8_tensor_core_ops']}, "
                            f"fp4={sass_prof['fp4_tensor_core_ops']}) bar={sass_prof['barriers']} "
                            f"shfl={sass_prof['shuffles']} fp64={sass_prof['fp64_ops']} "
                            f"spills(ldl/stl)={sass_prof['local_loads']}/{sass_prof['local_stores']}"
                        )
                json_out.append(
                    {
                        "name": name,
                        "symbol": ru.symbol,
                        "demangled": dem,
                        "resources": ru.to_dict(),
                        "occupancy_at_threads": occ_info,
                        "launch_bounds_spill_warning": lb_warning,
                        "sass_profile": sass_prof,
                    }
                )
    if args.json:
        print(json.dumps(json_out, indent=2))
        return 0
    print(f"\nkernels matched: {len(json_out)}")
    if launch_bounds_warnings:
        print(f"over-constrained __launch_bounds__ spill warnings: {len(launch_bounds_warnings)}")
    return 0


def cmd_profile(args: argparse.Namespace) -> int:
    cfg = _merged(args)
    exe = cfg.get("exe")
    if not exe:
        print("error: --exe is required", file=sys.stderr)
        return 2
    if not Path(exe).exists():
        print(f"error: executable not found: {exe}", file=sys.stderr)
        return 2
    cache = Cache(args.cache_dir)
    prof = Profiler(cache)
    if not prof.ncu:
        print("error: ncu not found; profiling requires Nsight Compute", file=sys.stderr)
        return 3
    args_list = _harness_args(cfg)
    results = prof.profile(
        exe, args_list,
        kernel_filter=cfg.get("kernel_filter"),
        stage=cfg.get("stage", "B"),
        focus=cfg.get("focus"),
        launch_count=cfg.get("launch_count", 1),
        launch_skip=cfg.get("launch_skip", 2),
        use_cache=not args.no_cache,
    )
    payload = [r.to_dict() for r in results]
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0
    for r in results:
        print(f"kernel: {r.kernel}")
        print(f"  grid {r.grid}  block {r.block}")
        for k, v in sorted(r.metrics.items()):
            print(f"  {k} = {v}")
        if r.unavailable:
            print(f"  unavailable on this target: {', '.join(r.unavailable)}")
    return 0


def _harness_args(cfg: dict[str, Any]) -> list[str]:
    out: list[str] = []
    wl = cfg.get("workload")
    if not wl:
        raise SystemExit("error: 'workload' is required (see --list on the harness)")
    out += ["--workload", str(wl)]
    if cfg.get("problem_size") is not None:
        out += ["--n", str(int(cfg["problem_size"]))]
    for key, flag in (("blocks", "--blocks"), ("threads", "--threads"), ("shared_mem", "--smem")):
        if cfg.get(key) is not None:
            out += [flag, str(int(cfg[key]))]
    out += ["--iters", str(int(cfg.get("iterations", 30)))]
    out += ["--warmup", str(int(cfg.get("warmup", 5)))]
    return out


def _resolve_registers(
    exe: str,
    kernel_name: str,
    cache_dir: Path | str,
    *,
    explicit: Any = None,
) -> int | None:
    """Find the register count for one kernel.

    An explicit value wins.  Otherwise the stored symbol is resolved from the
    binary and its resource usage read with cuobjdump, which is cheaper than an
    NCU replay and carries no instrumentation overhead.  When neither is
    available the result is ``None`` rather than a guess: occupancy predictions
    without a register count are meaningless and would silently mislead every
    conclusion drawn downstream.
    """
    if explicit:
        try:
            return int(explicit)
        except (TypeError, ValueError):
            return None
    insp = BinaryInspector(Cache(cache_dir))
    syms = insp.resolve_symbol(exe, kernel_name)
    if not syms:
        return None
    usage = insp.resource_usage(exe)
    ru = usage.get(syms[0])
    return int(ru.reg) if ru and ru.reg else None


def cmd_search(args: argparse.Namespace) -> int:
    cfg = _merged(args)
    exe = cfg.get("exe")
    if not exe or not Path(exe).exists():
        print(f"error: --exe must name an existing executable (got {exe!r})", file=sys.stderr)
        return 2
    env = _open_env(args)
    dev = env.device
    cache = Cache(args.cache_dir)

    wl = cfg.get("workload")
    if not wl:
        print("error: 'workload' is required", file=sys.stderr)
        return 2
    n = int(cfg.get("problem_size", 1 << 22))
    space = SearchSpace(
        threads=[int(t) for t in cfg.get("threads", [128, 256, 512])],
        sm_multiples=[int(m) for m in cfg.get("sm_multiples", [2, 4, 8, 16, 32])],
        shared_mem=[int(s) for s in cfg.get("shared_mem", [0])],
    )
    regs = _resolve_registers(exe, wl, args.cache_dir, explicit=cfg.get("registers_per_thread"))
    cands, viol = generate_candidates(
        dev, space,
        strategy=cfg.get("strategy", "model-guided"),
        registers_per_thread=regs,
        budget=int(cfg.get("budget", 24)),
    )
    base_cfg = LaunchConfig(
        int(cfg.get("baseline_blocks", dev.multi_processor_count * 8)),
        int(cfg.get("baseline_threads", 256)),
        0,
    )
    res = run_search(
        exe,
        dev=dev, workload=wl,
        problem_size_args=["--workload", str(wl), "--n", str(n)],
        problem_size=n,
        candidates=cands, violations=viol, baseline_config=base_cfg,
        registers_per_thread=regs,
        iterations=int(cfg.get("iterations", 25)),
        warmup=int(cfg.get("warmup", 5)),
        repeats=int(cfg.get("repeats", 1)),
        verify=not args.no_verify,
        budget_seconds=cfg.get("budget_seconds"),
        strategy=cfg.get("strategy", "model-guided"),
    )
    if args.json:
        print(json.dumps(res.to_dict(), indent=2))
        return 0
    for line in summarise_search(res):
        print(line)
    print()
    print(f"{'configuration':<20}{'latency us':>12}{'speedup':>10}{'verified':>10}{'occ %':>8}")
    for c in res.candidates:
        occ = c.occupancy.occupancy_pct if c.occupancy else None
        print(
            f"{c.config.label:<20}"
            f"{(f'{c.latency_us:.2f}' if c.latency_us else 'ERROR'):>12}"
            f"{(f'{c.speedup:.3f}' if c.speedup else '-'):>10}"
            f"{('yes' if c.passed_correctness else 'NO'):>10}"
            f"{(f'{occ:.0f}' if occ is not None else '-'):>8}"
        )
    if args.output:
        write_json(Path(args.output) / "search.json", res.to_dict())
        print(f"\nwritten to {args.output}/search.json")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    """The full pipeline for one or more workloads."""
    cfg = _merged(args)
    exe = cfg.get("exe")
    if not exe or not Path(exe).exists():
        print(f"error: --exe must name an existing executable (got {exe!r})", file=sys.stderr)
        return 2
    env = _open_env(args)
    dev = env.device
    cache = Cache(args.cache_dir)
    prof = Profiler(cache)
    insp = BinaryInspector(cache)

    workloads = cfg.get("workloads") or cfg.get("workload")
    if not workloads:
        print("error: 'workload' or 'workloads' is required", file=sys.stderr)
        return 2
    if isinstance(workloads, str):
        workloads = [workloads]

    n = int(cfg.get("problem_size", 1 << 22))
    iters = int(cfg.get("iterations", 30))
    warmup = int(cfg.get("warmup", 5))
    repeats = int(cfg.get("repeats", 3))
    out_dir = Path(args.output or DEFAULT_RESULTS)
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for wl in workloads:
        LOG.info("analyzing workload %s", wl)
        rec: dict[str, Any] = {"workload": wl, "kernel": wl}
        run_args = ["--workload", wl, "--n", str(n)]
        bt = baseline_timing(exe, run_args, workload=wl,
                             iters=iters, warmup=warmup, repeats=repeats)
        rec["baseline_timing"] = bt.to_dict()
        blocks = int(bt.raw_records.get("config.blocks", dev.multi_processor_count * 8) or dev.multi_processor_count * 8)
        threads = int(bt.raw_records.get("config.threads", 256) or 256)
        smem = int(bt.raw_records.get("config.smem", 0) or 0)
        # Verification is a *separate single launch* on freshly initialised
        # buffers, matching the exact configuration that was timed.  Reusing the
        # timed run's records is wrong for in-place kernels, which would already
        # have accumulated N applications of the update.
        vargs = run_args + ["--blocks", str(blocks), "--threads", str(threads), "--smem", str(smem)]
        vrec = verify_harness(exe, vargs)
        ver = verification_from_records(vrec, workload=wl)
        gate = gate_from_verification(ver)
        rec["correctness"] = ver.to_dict()
        rec["correctness_gate"] = gate.to_dict()
        if not gate.validated:
            LOG.warning(
                "workload %s did not pass verification; its timings are reported but no "
                "performance conclusion may be drawn", wl,
            )
        rec["kernel"] = bt.raw_records.get("kernel", wl)

        # Static / binary analysis
        syms = insp.resolve_symbol(exe, wl) if insp.info.cuobjdump else []
        if syms:
            try:
                d = insp.disassemble(exe, syms[0])
                rec["disassembly"] = d.to_dict(include_instructions=False)
                rec["instruction_profile"] = d.profile.to_dict()
                rec["symbol"] = syms[0]
            except ToolError as exc:
                rec["disassembly_error"] = str(exc)

        regs = None
        ip = rec.get("instruction_profile") or {}
        if rec.get("disassembly"):
            regs = rec["disassembly"].get("registers_used")

        if regs:
            occ = occupancy_model(dev, threads_per_block=threads,
                                  registers_per_thread=int(regs), shared_mem_per_block=smem)
            rec["occupancy"] = occ.to_dict()
            rec["waves"] = wave_analysis(dev, occ, blocks).to_dict()

        # Profiling
        if prof.ncu and not args.no_profile and rec.get("symbol"):
            try:
                pres = prof.profile(
                    exe, run_args + ["--blocks", str(blocks), "--threads", str(threads),
                                     "--smem", str(smem), "--iters", "5", "--warmup", "2"],
                    kernel_filter=wl, symbols=syms, stage=cfg.get("stage", "B"),
                    focus=cfg.get("focus"), use_cache=not args.no_cache,
                )
                if pres:
                    r0 = pres[0]
                    rec["metrics"] = r0.to_dict()
                    # Prefer the NCU-reported register count when available.
                    ncu_regs = r0.metrics.get("launch__registers_per_thread")
                    if ncu_regs and not regs:
                        regs = int(ncu_regs)
                        occ = occupancy_model(dev, threads_per_block=threads,
                                              registers_per_thread=regs, shared_mem_per_block=smem)
                        rec["occupancy"] = occ.to_dict()
                        rec["waves"] = wave_analysis(dev, occ, blocks).to_dict()
            except ToolError as exc:
                rec["profile_error"] = str(exc)

        # Modelling
        try:
            bytes_moved = float(bt.raw_records.get("derived.bytes_moved", 0) or 0)
        except ValueError:
            bytes_moved = 0.0
        latency = bt.stats.median if bt.stats else None
        ip = rec.get("instruction_profile") or {}
        rof = roofline_report(
            bytes_moved=bytes_moved or None,
            flops=_safe_float(bt.raw_records.get("derived.flops")) or None,
            latency_us=latency,
            dev=dev,
            ncu=_result_stub(rec.get("metrics")),
            max_load_width_bytes=int(ip.get("max_load_width") or 4),
            instruction_profile=_ip_stub(ip),
        )
        rec["roofline"] = rof.to_dict()

        if rec.get("metrics"):
            cl = classify_bottleneck(
                _result_stub(rec["metrics"]),
                dev=dev,
                instruction_profile=_ip_stub(ip),
                occupancy=_occ_stub(rec.get("occupancy")),
                kernel_duration_us=latency,
                roofline=rof,
            )
            rec["classification"] = cl.to_dict()
            rs = recommend(
                result=_result_stub(rec["metrics"]),
                dev=dev,
                classification=cl,
                occupancy=_occ_stub(rec.get("occupancy")),
                roofline=rof,
                waves=_wave_stub(rec.get("waves")),
                instruction_profile=_ip_stub(ip),
            )
            rec["recommendations"] = rs.to_dict()
        else:
            rec["classification"] = {"category": "unavailable", "confidence": "low",
                                     "evidence": ["no NCU metrics collected"]}
            rec["recommendations"] = {"recommendations": [], "unavailable_checks": ["no NCU metrics collected"]}

        records.append(rec)

    rep = report_from_analysis(env=env, results=records, title=args.title or "CUDA Kernel Analysis Report")
    written = rep.write(out_dir, stem=args.stem or "analysis", html=not args.no_html)
    print(json.dumps({"written": written, "kernels": len(records)}, indent=2))
    for r in records:
        st = (r.get("correctness") or {}).get("status", "unvalidated")
        cat = (r.get("classification") or {}).get("category", "n/a")
        print(f"  {r['workload']:<20} verify={st:<12} classification={cat}")
    return 0


def _safe_float(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


class _MetricsStub:
    """Minimal adapter so model functions accept a plain dict of metrics."""

    def __init__(self, metrics: dict[str, Any], kernel: str = "", grid=(1, 1, 1), block=(1, 1, 1)):
        self.metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        self.kernel = kernel
        self.grid = grid
        self.block = block
        self.notes: list[str] = []

    @property
    def duration_us(self):
        v = self.metrics.get("gpu__time_duration.sum") or self.metrics.get("gpu__time_duration.avg")
        return v / 1000.0 if v else None

    def get(self, *names):
        for n in names:
            if n in self.metrics:
                return self.metrics[n]
        return None


def _result_stub(d: dict[str, Any] | None) -> _MetricsStub:
    d = d or {}
    return _MetricsStub(d.get("metrics") or {}, kernel=d.get("kernel", ""),
                        grid=tuple(d.get("grid") or (1, 1, 1)), block=tuple(d.get("block") or (1, 1, 1)))


class _DictStub:
    """Adapter exposing a stored dict as the attribute-style object the model
    functions expect.  ``to_dict`` returns the original payload so a stubbed
    value round-trips without loss."""

    _defaults: dict[str, Any] = {}

    def __init__(self, d: dict[str, Any] | None):
        d = d or {}
        self._data = d
        for k, v in {**self._defaults, **d}.items():
            setattr(self, k, v)

    def to_dict(self) -> dict[str, Any]:
        return dict(self._data)


class _OccStub(_DictStub):
    _defaults = {
        "limiting_factor": "",
        "occupancy_pct": 0.0,
        "active_blocks_per_sm": 0,
        "registers_per_thread": 0,
        "threads_per_block": 0,
        "max_regs_per_thread_to_gain_one_block": None,
    }


def _occ_stub(d: dict[str, Any] | None) -> _OccStub | None:
    return _OccStub(d) if d else None


class _IPStub(_DictStub):
    _defaults = {
        "has_local_memory": False,
        "max_load_width": 0,
        "max_shared_load_width": 0,
        "local_loads": 0,
        "local_stores": 0,
        "global_loads": 0,
        "tensor_core_ops": 0,
        "int_tensor_core_ops": 0,
        "float_tensor_core_ops": 0,
        "fp64_ops": 0,
        "shuffles": 0,
        "uniform_ops": 0,
        "async_copies": 0,
    }


def _ip_stub(d: dict[str, Any] | None) -> _IPStub | None:
    return _IPStub(d) if d else None


class _WaveStub(_DictStub):
    pass


def _wave_stub(d: dict[str, Any] | None) -> _WaveStub | None:
    return _WaveStub(d) if d else None


def cmd_math(args: argparse.Namespace) -> int:
    """Exact tuning calculator: occupancy, waves, roofline, workload, smem, mlp, ab."""
    from .kernel_math import (
        calculate_ab_stats,
        calculate_mlp,
        calculate_occupancy_tables,
        calculate_roofline_math,
        calculate_smem_layout,
        calculate_wave_tables,
        calculate_workload_math,
    )

    env = _open_env(args)
    dev = env.device
    topic = args.topic

    if topic == "occupancy":
        raw_t = str(getattr(args, "threads", "auto") or "auto").strip().lower()
        if raw_t in ("all", "sweep", "auto"):
            primary_t = 256
        elif "," in raw_t:
            primary_t = int(raw_t.split(",")[0].strip())
        else:
            primary_t = int(raw_t)
        res = calculate_occupancy_tables(
            dev,
            threads_per_block=primary_t,
            registers_per_thread=args.regs,
            shared_mem_per_block=args.smem,
        )
        if args.json:
            print(json.dumps(res, indent=2))
            return 0
        cur = res["current"]
        hw_r = res["hardware_parameter_ranges"]
        print(f"Device: {res['device']} ({res['arch']}) — {res['sms']} SMs, {res['regs_per_sm']} regs/SM, {res['smem_per_sm']} B smem/SM, {res['max_warps_per_sm']} max warps/SM")
        print(
            f"Ranges: threads={hw_r['threads_per_block']['min']}..{hw_r['threads_per_block']['max']} "
            f"(step {hw_r['threads_per_block']['step']}, max {hw_r['threads_per_block']['max_for_100pct_occupancy']} for 100% occ), "
            f"regs={hw_r['registers_per_thread']['min']}..{hw_r['registers_per_thread']['max']} "
            f"(step {hw_r['registers_per_thread']['step']}), "
            f"smem=0..{hw_r['shared_mem_per_block_bytes']['max_default']} B default / "
            f"{hw_r['shared_mem_per_block_bytes']['max_optin']} B opt-in "
            f"(+{hw_r['shared_mem_per_block_bytes']['driver_reserved_per_block']} B driver reserved/block)"
        )
        print(f"Config: threads={cur['threads_per_block']} ({cur['warps_per_block']} warps), regs/thread={cur['registers_per_thread']}, user_smem={cur['shared_mem_per_block']} B")
        print(f"Result: {cur['active_blocks_per_sm']} active blocks/SM ({cur['active_warps_per_sm']}/{cur['max_warps_per_sm']} warps = {cur['occupancy_pct']:.2f}% occupancy)")
        print(f"Limiter: {cur['limiting_factor']} (by_hw={cur['blocks_by_hw_limit']}, by_regs={cur['blocks_by_registers']}, by_smem={cur['blocks_by_shared']})")
        print(f"Allocated: {cur['registers_allocated_per_block']} regs/block ({cur['unused_registers_per_sm']} unused regs/SM), {cur['smem_allocated_per_block']} B smem/block")
        print(f"Headroom: free_reg_headroom=+{cur['free_reg_headroom_per_thread']} regs/thread (max {cur['max_regs_before_dropping_block']} before dropping block), regs_for_+1_block={cur['max_regs_per_thread_to_gain_one_block']}, free_smem_headroom=+{cur['free_smem_headroom_bytes']} B")
        print("\nRegister Step Table (at current threads & smem):")
        print(f"  {'blocks/SM':>9}  {'warps/SM':>8}  {'occ %':>7}  {'max_regs/thread':>15}  {'alloc_regs/block':>16}")
        for r in res["register_step_table"]:
            print(f"  {r['blocks_per_sm']:>9}  {r['warps_per_sm']:>8}  {r['occupancy_pct']:>7.2f}  {r['max_regs_per_thread']:>15}  {r['allocated_regs_per_block']:>16}")
        print("\nShared-Memory Step Table (at current threads & regs):")
        print(f"  {'blocks/SM':>9}  {'warps/SM':>8}  {'occ %':>7}  {'max_user_smem_B':>15}  {'alloc_smem_B':>12}  {'opt-in':>6}")
        for r in res["smem_step_table"]:
            print(f"  {r['blocks_per_sm']:>9}  {r['warps_per_sm']:>8}  {r['occupancy_pct']:>7.2f}  {r['max_user_smem_bytes']:>15}  {r['allocated_smem_bytes_per_block']:>12}  {str(r['requires_optin']):>6}")
        print("\nBlock-Size Sweep Across GB10 Thread Range (32..1024 threads at current regs & smem):")
        print(f"  {'threads':>7}  {'warps':>5}  {'blocks/SM':>9}  {'warps/SM':>8}  {'occ %':>7}  {'limiter':<24}  {'+1blk<=regs':>11}  {'free_regs':>9}")
        for r in res["block_size_sweep"]:
            print(f"  {r['threads_per_block']:>7}  {r['warps_per_block']:>5}  {r['active_blocks_per_sm']:>9}  {r['active_warps_per_sm']:>8}  {r['occupancy_pct']:>7.2f}  {r['limiting_factor']:<24}  {str(r['max_regs_to_gain_one_block']):>11}  {str(r['free_reg_headroom_per_thread']):>9}")
        return 0

    if topic == "waves":
        res = calculate_wave_tables(
            dev,
            total_ctas=args.ctas,
            threads_per_block=args.threads,
            registers_per_thread=args.regs,
            shared_mem_per_block=args.smem,
            total_items=args.items,
        )
        if args.json:
            print(json.dumps(res, indent=2))
            return 0
        wv = res["wave_analysis"]
        print(f"Device: {res['device']} ({res['sms']} SMs) — {res['active_blocks_per_sm']} blocks/SM ({res['occupancy_pct']:.2f}% occ) -> {wv['blocks_per_wave']} CTAs/wave")
        print(f"Grid:   {wv['total_ctas']} CTAs = {wv['waves']:.3f} waves ({wv['full_waves']} full waves + {wv['tail_ctas']} tail CTAs, tail eff {wv['tail_efficiency_pct']:.1f}%)")
        print(f"Loss:   quantization_loss={wv['quantization_loss_pct']:.2f}%, max_tail_speedup={wv['max_tail_speedup']:.4f}x, problem={wv['problem']}")
        print("\nWave-Aligned Grid Options:")
        print(f"  {'waves':>5}  {'total_ctas':>10}  {'total_threads':>13}  {'delta_ctas':>10}  {'items/cta':>10}  {'items/thread':>12}")
        for g in res["wave_aligned_grids"]:
            print(f"  {g['waves']:>5}  {g['total_ctas']:>10}  {g['total_threads']:>13}  {g['delta_ctas_from_current']:>+10}  {str(g.get('items_per_cta', '-')):>10}  {str(g.get('items_per_thread', '-')):>12}")
        return 0

    if topic == "roofline":
        res = calculate_roofline_math(
            dev,
            bytes_moved=args.bytes,
            flops=args.flops,
            latency_us=args.latency_us,
            dtype=args.dtype,
        )
        if args.json:
            print(json.dumps(res, indent=2))
            return 0
        print(f"Bytes moved:          {res['bytes_moved']:,.0f} B ({res['bytes_moved']/1e6:.3f} MB) — L2 ratio: {res.get('working_set_vs_l2_ratio')}x ({res.get('l2_residency_regime')})")
        print(f"FLOPs:                {res['flops']:,.0f} ({res['flops']/1e9:.3f} GFLOP)")
        print(f"Arithmetic intensity: {res.get('arithmetic_intensity')} FLOP/B")
        print(f"Memory bound latency: {res.get('memory_bound_latency_us'):.3f} us (at {res.get('bandwidth_ceiling_gbps')} GB/s {res.get('bandwidth_ceiling_source')})")
        if res.get("latency_us"):
            print(f"Measured latency:     {res['latency_us']:.3f} us -> Eff BW = {res.get('effective_bandwidth_gbps')} GB/s ({res.get('memory_utilization_pct')}% of ceiling), Eff Compute = {res.get('effective_tflops')} TFLOP/s")
            print(f"Roofline limit ({res['compute_ceiling_dtype']}): {res.get('limit_us'):.3f} us (headroom {res.get('headroom_us')} us, max theoretical speedup {res.get('max_theoretical_speedup')}x)")
        print("\nCeilings Comparison Across DTypes:")
        print(f"  {'dtype':<8}  {'ceiling_tflops':>14}  {'ridge_flop/B':>12}  {'comp_bound_us':>13}  {'limit_us':>10}  {'regime':<14}  {'comp_util%':>10}  {'max_speedup':>11}")
        for c in res.get("ceilings_comparison", []):
            print(f"  {c['dtype']:<8}  {c['ceiling_tflops']:>14.2f}  {str(c['ridge_point_flops_per_byte']):>12}  {str(c['compute_bound_us']):>13}  {str(c['roofline_limit_us']):>10}  {c['regime']:<14}  {str(c['compute_utilization_pct']):>10}  {str(c['max_speedup_vs_latency']):>11}")
        return 0

    if topic == "workload":
        res = calculate_workload_math(
            dev,
            m=args.m,
            n=args.n,
            k=args.k,
            weight_dtype=args.weight_dtype,
            act_dtype=args.act_dtype,
            out_dtype=args.out_dtype,
            latency_us=args.latency_us,
            compute_dtype=args.compute_dtype,
        )
        if args.json:
            print(json.dumps(res, indent=2))
            return 0
        w = res["weights"]
        a = res["activations"]
        o = res["outputs"]
        ru = res["roofline_unique_weights"]
        print(f"Shape: M={args.m}, N={args.n}, K={args.k} -> {res['flops']:,} FLOPs ({res['gflops']} GFLOP)")
        print(f"Weights ({w['dtype']}):     {w['bytes']:,} B ({w['megabytes']} MB, {w['bits_per_element']} bits/elem) — {w['description']}")
        print(f"Activations ({a['dtype']}): {a['bytes']:,} B ({a['megabytes']} MB)")
        print(f"Outputs ({o['dtype']}):     {o['bytes']:,} B ({o['megabytes']} MB)")
        print(f"Total bytes (unique W):      {res['bytes_moved_unique']:,} B ({res['bytes_moved_unique']/1e6:.3f} MB), AI = {ru.get('arithmetic_intensity')} FLOP/B, mem_bound = {ru.get('memory_bound_latency_us'):.3f} us")
        if args.m > 1:
            rm = res["roofline_unamortized_weights"]
            print(f"Total bytes (unamortized M): {res['bytes_moved_unamortized_m']:,} B ({res['bytes_moved_unamortized_m']/1e6:.3f} MB), AI = {rm.get('arithmetic_intensity')} FLOP/B, mem_bound = {rm.get('memory_bound_latency_us'):.3f} us")
            print(f"M-tiling weight reuse saves: {res['bytes_saved_by_m_reuse']:,} B ({res['m_reuse_traffic_reduction_pct']}% traffic reduction)")
        if args.latency_us:
            print(f"At {args.latency_us} us: Eff BW (unique W) = {ru.get('effective_bandwidth_gbps')} GB/s ({ru.get('memory_utilization_pct')}% of ceiling), max speedup vs limit = {ru.get('max_theoretical_speedup')}x")
        return 0

    if topic == "smem":
        res = calculate_smem_layout(
            dev,
            rows=args.rows,
            cols=args.cols,
            elem_bytes=args.elem_bytes,
            stages=args.stages,
            other_bytes=args.other_bytes,
            threads_per_block=args.threads,
            registers_per_thread=args.regs,
        )
        if args.json:
            print(json.dumps(res, indent=2))
            return 0
        u = res["unpadded"]
        sp = res["scalar_4b_conflict_free_padding"]
        vp = res["aligned_16b_padding"]
        print(f"Tile: {args.rows}x{args.cols} x {args.elem_bytes} B x {args.stages} stage(s) + {args.other_bytes} B")
        print(f"Unpadded: user_smem={u['total_user_smem_bytes']} B, alloc_smem={u['allocated_smem_bytes_per_block']} B (reserve={u['reserved_smem_bytes']} B), optin_req={u['requires_optin_attribute']}")
        print(f"          blocks/SM={u['active_blocks_per_sm']} ({u['occupancy_pct']}% occ, limiter={u['limiting_factor']}), column bank conflict={u['column_access_bank_conflict_way']}-way (stride={u['stride_4b_banks']} banks)")
        print(f"Scalar 4B conflict-free pad: +{sp['pad_elements']} elems -> cols={sp['padded_cols']} ({sp['row_bytes']} B/row, {sp['bank_conflict_way']}-way), user_smem={sp['total_user_smem_bytes']} B, blocks/SM={sp['active_blocks_per_sm']} ({sp['occupancy_pct']}% occ)")
        print(f"Aligned 16B pad (LDS.128):   +{vp['pad_elements']} elems -> cols={vp['padded_cols']} ({vp['row_bytes']} B/row), user_smem={vp['total_user_smem_bytes']} B, blocks/SM={vp['active_blocks_per_sm']} ({vp['occupancy_pct']}% occ)")
        return 0

    if topic == "mlp":
        res = calculate_mlp(
            dev,
            active_warps_per_sm=args.warps_per_sm,
            load_width_bytes=args.load_width_bytes,
            mem_latency_ns=args.mem_latency_ns,
        )
        if args.json:
            print(json.dumps(res, indent=2))
            return 0
        print(f"Device: {res['device']} ({res['sms']} SMs, {res['bandwidth_ceiling_gbps']} GB/s) @ {res['assumed_mem_latency_ns']} ns latency")
        print(f"Little's Law: need {res['bytes_in_flight_needed_gpu']:,.0f} B in flight across GPU = {res['bytes_in_flight_needed_per_sm']:,.0f} B/SM")
        print(f"At {res['active_warps_per_sm']} active warps/SM and {res['current_load_width_bytes']} B loads: 1 load/warp = {res['current_bytes_in_flight_at_1_load_per_warp']} B/SM -> need {res['independent_loads_per_thread_needed']} independent load(s)/thread")
        print("\nLoad Width Comparison:")
        print(f"  {'width_B':>7}  {'SASS':<16}  {'B/warp_req':>10}  {'B/SM @1_req':>11}  {'indep_loads/thread':>18}  {'regs/thread':>11}")
        for r in res["load_width_comparison"]:
            print(f"  {r['load_width_bytes']:>7}  {r['sass_instruction']:<16}  {r['bytes_per_warp_request']:>10}  {r['bytes_in_flight_at_1_req_per_warp']:>11}  {r['independent_loads_per_thread_needed']:>18}  {r['register_ footprint_per_thread_for_loads']:>11}")
        return 0

    if topic == "ab":
        base_vals = [float(x) for x in args.base.split(",") if x.strip()]
        cand_vals = [float(x) for x in args.cand.split(",") if x.strip()]
        res = calculate_ab_stats(
            base_vals,
            cand_vals,
            launches_per_token=args.launches_per_token,
            baseline_token_ms=args.token_ms,
        )
        if args.json:
            print(json.dumps(res, indent=2))
            return 0
        b = res["baseline"]
        c = res["candidate"]
        print(f"Baseline:  median={b['median_us']:.3f} us (mean={b['mean_us']:.3f}, iqr={b['iqr_us']:.3f}, cv={b['cv_pct']:.2f}%, n={b['n']})")
        print(f"Candidate: median={c['median_us']:.3f} us (mean={c['mean_us']:.3f}, iqr={c['iqr_us']:.3f}, cv={c['cv_pct']:.2f}%, n={c['n']})")
        print(f"Comparison: delta={res['delta_us']:+.3f} us ({res['pct_change']:+.2f}%), speedup={res['speedup']:.4f}x, noise_floor={res['noise_floor_pct']:.2f}%, SNR={res['signal_to_noise_ratio']}, verdict={res['verdict']}")
        if "end_to_end_projection" in res:
            e2e = res["end_to_end_projection"]
            print(f"End-to-End ({e2e['launches_per_token']} launches/tok): stage {e2e['baseline_stage_ms_per_token']:.3f} -> {e2e['candidate_stage_ms_per_token']:.3f} ms/tok (saved {e2e['saved_ms_per_token']:+.3f} ms/tok)")
            if "projected_candidate_tok_per_s" in e2e:
                print(f"Projected throughput: {e2e['baseline_tok_per_s']:.2f} -> {e2e['projected_candidate_tok_per_s']:.2f} tok/s ({e2e['projected_e2e_pct_gain']:+.2f}%, {e2e['projected_e2e_speedup']:.4f}x)")
        return 0

    return 2


def cmd_compare(args: argparse.Namespace) -> int:
    """Compare two binaries/symbols (SASS + resources + occupancy) or two analysis JSON reports."""
    from .binary_inspector import compare_profiles

    if args.report_a and args.report_b:
        ra = read_json(args.report_a)
        rb = read_json(args.report_b)
        ka = {r.get("workload") or r.get("kernel"): r for r in (ra.get("data", {}).get("kernels") or [])}
        kb = {r.get("workload") or r.get("kernel"): r for r in (rb.get("data", {}).get("kernels") or [])}
        common = sorted(set(ka) & set(kb))
        deltas: list[dict[str, Any]] = []
        for k in common:
            da = ka[k]
            db = kb[k]
            med_a = ((da.get("baseline_timing") or {}).get("stats") or {}).get("median")
            med_b = ((db.get("baseline_timing") or {}).get("stats") or {}).get("median")
            bw_a = (da.get("roofline") or {}).get("effective_bandwidth_gbps")
            bw_b = (db.get("roofline") or {}).get("effective_bandwidth_gbps")
            occ_a = (da.get("occupancy") or {}).get("occupancy_pct")
            occ_b = (db.get("occupancy") or {}).get("occupancy_pct")
            reg_a = (da.get("occupancy") or {}).get("registers_per_thread")
            reg_b = (db.get("occupancy") or {}).get("registers_per_thread")
            cat_a = (da.get("classification") or {}).get("category")
            cat_b = (db.get("classification") or {}).get("category")
            speedup = (med_a / med_b) if (med_a and med_b and med_b > 0) else None
            pct = (100.0 * (med_b - med_a) / med_a) if (med_a and med_b) else None
            deltas.append(
                {
                    "kernel": k,
                    "latency_before_us": med_a,
                    "latency_after_us": med_b,
                    "delta_us": round(med_b - med_a, 4) if (med_a and med_b) else None,
                    "pct_change": round(pct, 2) if pct is not None else None,
                    "speedup": round(speedup, 4) if speedup is not None else None,
                    "bw_before_gbps": bw_a,
                    "bw_after_gbps": bw_b,
                    "regs_before": reg_a,
                    "regs_after": reg_b,
                    "occ_before_pct": occ_a,
                    "occ_after_pct": occ_b,
                    "class_before": cat_a,
                    "class_after": cat_b,
                }
            )
        if args.json:
            print(json.dumps({"comparisons": deltas}, indent=2))
            return 0
        print(f"{'kernel':<24} {'before_us':>10} {'after_us':>10} {'delta%':>8} {'speedup':>8} {'BW_before':>10} {'BW_after':>10} {'regs':>9} {'occ%':>11}")
        for d in deltas:
            pct_str = f"{d['pct_change']:+.2f}%" if d["pct_change"] is not None else "n/a"
            spd_str = f"{d['speedup']:.3f}x" if d["speedup"] is not None else "n/a"
            reg_str = f"{d['regs_before']}->{d['regs_after']}"
            occ_str = f"{d['occ_before_pct']}->{d['occ_after_pct']}"
            print(
                f"{str(d['kernel'])[:24]:<24} "
                f"{str(d['latency_before_us']):>10} "
                f"{str(d['latency_after_us']):>10} "
                f"{pct_str:>8} "
                f"{spd_str:>8} "
                f"{str(d['bw_before_gbps']):>10} "
                f"{str(d['bw_after_gbps']):>10} "
                f"{reg_str:>9} "
                f"{occ_str:>11}"
            )
        return 0

    exe_a = args.exe_a or getattr(args, "exe_a_pos", None)
    exe_b = args.exe_b or getattr(args, "exe_b_pos", None) or exe_a
    if not exe_a:
        print("error: supply --exe-a (and optionally --exe-b) or --report-a/--report-b", file=sys.stderr)
        return 2
    sym_a_q = args.symbol_a or args.symbol or getattr(args, "kernel", None)
    sym_b_q = args.symbol_b or args.symbol or getattr(args, "kernel", None) or sym_a_q
    if not sym_a_q or not sym_b_q:
        print("error: supply --symbol / --kernel or --symbol-a / --symbol-b", file=sys.stderr)
        return 2

    insp = BinaryInspector(Cache(args.cache_dir))
    syms_a = insp.resolve_symbol(exe_a, sym_a_q)
    if not syms_a:
        usage_a = insp.resource_usage(exe_a)
        dem_a = _batch_demangle(list(usage_a.keys()))
        syms_a = [s for s in usage_a.keys() if sym_a_q in s or sym_a_q in dem_a.get(s, "")]
    syms_b = insp.resolve_symbol(exe_b, sym_b_q)
    if not syms_b:
        usage_b = insp.resource_usage(exe_b)
        dem_b = _batch_demangle(list(usage_b.keys()))
        syms_b = [s for s in usage_b.keys() if sym_b_q in s or sym_b_q in dem_b.get(s, "")]
    if not syms_a:
        print(f"error: symbol {sym_a_q!r} not found in {exe_a}", file=sys.stderr)
        return 2
    if not syms_b:
        print(f"error: symbol {sym_b_q!r} not found in {exe_b}", file=sys.stderr)
        return 2

    ua = insp.resource_usage(exe_a).get(syms_a[0])
    ub = insp.resource_usage(exe_b).get(syms_b[0])
    da = insp.disassemble(exe_a, syms_a[0])
    db = insp.disassemble(exe_b, syms_b[0])

    diff = compare_profiles(
        da.profile,
        db.profile,
        regs_a=ua.reg if ua else da.registers_used,
        regs_b=ub.reg if ub else db.registers_used,
        shared_a=ua.shared if ua else None,
        shared_b=ub.shared if ub else None,
        stack_a=ua.stack if ua else None,
        stack_b=ub.stack if ub else None,
    )
    dev = _open_env(args).device
    raw_t = getattr(args, "threads", None)
    explicit_t = int(raw_t) if (raw_t is not None and str(raw_t).lower() not in ("auto", "")) else None
    dem_sym_a = _batch_demangle([syms_a[0]]).get(syms_a[0], syms_a[0])
    dem_sym_b = _batch_demangle([syms_b[0]]).get(syms_b[0], syms_b[0])
    threads_a, _, _ = _infer_symbol_threads_and_min_blocks(dem_sym_a, explicit_t)
    threads_b, _, _ = _infer_symbol_threads_and_min_blocks(dem_sym_b, explicit_t)
    occ_a = occupancy_model(
        dev,
        threads_per_block=threads_a,
        registers_per_thread=int((ua.reg if ua else da.registers_used) or 0),
        shared_mem_per_block=int((ua.shared if ua else 0) or 0),
    )
    occ_b = occupancy_model(
        dev,
        threads_per_block=threads_b,
        registers_per_thread=int((ub.reg if ub else db.registers_used) or 0),
        shared_mem_per_block=int((ub.shared if ub else 0) or 0),
    )
    diff["symbol_a"] = syms_a[0]
    diff["symbol_b"] = syms_b[0]
    diff["threads_a"] = threads_a
    diff["threads_b"] = threads_b
    diff["occupancy_before"] = occ_a.to_dict()
    diff["occupancy_after"] = occ_b.to_dict()

    if args.json:
        print(json.dumps(diff, indent=2))
        return 0

    print(f"Symbol A: {syms_a[0]} ({exe_a})")
    print(f"Symbol B: {syms_b[0]} ({exe_b})")
    print(
        f"Occupancy: {occ_a.active_blocks_per_sm} blk/SM ({occ_a.occupancy_pct:.1f}% @{threads_a}t) "
        f"-> {occ_b.active_blocks_per_sm} blk/SM ({occ_b.occupancy_pct:.1f}% @{threads_b}t)"
    )
    print("\nCompiler Resources:")
    for k, v in diff["resources"].items():
        print(f"  {k:<18} {str(v['before']):>8} -> {str(v['after']):>8}   (delta={v['delta']})")
    print("\nSASS Instruction Mix Delta:")
    print(f"  {'category':<24} {'before':>8} {'after':>8} {'delta':>8} {'pct_change':>10}")
    for k, v in diff["instructions"].items():
        if v["before"] == 0 and v["after"] == 0:
            continue
        print(f"  {k:<24} {v['before']:>8} {v['after']:>8} {v['delta']:>+8} {v['pct_change']:>+9.2f}%")
    return 0


def cmd_hotspots(args: argparse.Namespace) -> int:
    """Analyze an NSYS SQLite export with automatic GB10 occupancy & wave math."""
    from .hotspots import analyze_nsys_sqlite

    env = _open_env(args)
    res = analyze_nsys_sqlite(
        args.sqlite,
        env.device,
        top=args.top,
        reps=args.reps,
        phase=getattr(args, "phase", "all"),
        exe=getattr(args, "exe", None),
    )
    if args.json:
        print(json.dumps(res, indent=2))
        return 0
    if "error" in res:
        print(f"error: {res['error']}", file=sys.stderr)
        return 1

    tl = res.get("timeline", {})
    duty_str = f", GPU duty cycle {tl.get('gpu_duty_cycle_pct')}%" if "gpu_duty_cycle_pct" in tl else ""
    phase_str = f" [Phase: {res['phase'].upper()}]" if res.get("phase") and res["phase"] != "all" else ""
    print(
        f"# NSYS Hotspot & Budget Analysis ({res['device']}){phase_str}\n"
        f"# Total kernel busy time: {res['total_busy_ms']:.3f} ms "
        f"({res['per_rep_busy_ms']:.3f} ms/rep over {res['reps']} rep(s)), "
        f"{res['total_calls']} launches, {res['distinct_kernels']} distinct kernels{duty_str}"
    )

    if tl and "total_wall_ms" in tl and tl["total_wall_ms"] > 0:
        bubbles = tl.get("gpu_idle_bubbles_ms", 0)
        bubble_str = f" | host dispatch idle {bubbles:.2f} ms" if bubbles > 0 else ""
        print(
            f"# Timeline: wall span {tl['total_wall_ms']:.2f} ms | "
            f"sync wait {tl.get('sync_wait_ms', 0):.2f} ms ({tl.get('sync_calls', 0)} events) | "
            f"memcpy {tl.get('memcpy_ms', 0):.2f} ms ({tl.get('memcpy_gib', 0):.2f} GiB in {tl.get('memcpy_calls', 0)} transfers){bubble_str}\n"
        )
    else:
        print()

    if res.get("phase_budget"):
        print("## Inference Phase Budget (from NVTX Ranges)")
        print(f"  {'phase':<14} {'wall_ms':>10} {'kernel_ms':>10} {'gpu_duty%':>10} {'calls':>8}")
        for pb in res["phase_budget"]:
            print(
                f"  {pb['phase']:<14} {pb['wall_ms']:>10.2f} {pb['kernel_ms']:>10.2f} "
                f"{pb['duty_pct']:>9.1f}% {pb['calls']:>8}"
            )
        print()

    print("## Functional Block Budget")
    print(f"  {'block':<14} {'share%':>7} {'total_ms':>10} {'per_rep_ms':>10} {'calls':>7} {'kernels':>8}")
    for b in res["by_functional_block"]:
        print(
            f"  {b['functional_block']:<14} {b['share_pct']:>7.2f} {b['total_ms']:>10.3f} "
            f"{b['per_rep_ms']:>10.3f} {b['calls']:>7} {b['distinct_kernels']:>8}"
        )

    if res.get("cuda_runtime_hotspots"):
        print("\n## Host CUDA Runtime Overhead (Top API Calls)")
        print(f"  {'api_call':<36} {'calls':>8} {'total_ms':>10} {'mean_us':>10}")
        for r in res["cuda_runtime_hotspots"][:8]:
            print(f"  {r['name']:<36} {r['calls']:>8} {r['total_ms']:>10.2f} {r['mean_us']:>10.1f}")

    if res.get("sync_breakdown"):
        print("\n## Synchronization Breakdown")
        print(f"  {'sync_type':<45} {'calls':>8} {'total_ms':>10}")
        for s in res["sync_breakdown"]:
            print(f"  {s['sync_type']:<45} {s['calls']:>8} {s['total_ms']:>10.2f}")

    cpu_prof = res.get("cpu_profiling", {})
    if cpu_prof.get("enabled"):
        print(f"\n## Host CPU Hotspots ({cpu_prof.get('samples_count', 0)} samples via PMU / CPU IP sampling)")
        if cpu_prof.get("has_masked_kernel_symbols"):
            print("  ! [NOTE] Kernel symbols are masked (kptr_restrict > 0). Run 'sudo sysctl -w kernel.kptr_restrict=0' for kernel symbol resolution.")
        print(f"  {'share%':>7} {'samples':>8}  symbol")
        for ch in cpu_prof.get("hotspots", [])[:15]:
            print(f"  {ch['share_pct']:>7.2f} {ch['samples']:>8}  {ch['symbol']}")

    if res.get("spills_detected"):
        print("\n## [ALERT] Compiler Stack Spills Detected in Hotspot Kernels")
        print("  Local memory stack spills cause L1/L2 and DRAM round-trip latency stalls:")
        for sp in res["spills_detected"][:10]:
            print(f"  * {sp}")
        if len(res["spills_detected"]) > 10:
            print(f"  * ... and {len(res['spills_detected']) - 10} more symbols with spills")

    print("\n## Top Kernels by Launch Geometry (with Pre-Calculated GB10 Occupancy & Waves)")
    has_spill_col = any(g.get("spill_bytes", 0) > 0 for g in res["by_geometry"])
    spill_hdr = "   spill" if has_spill_col else ""
    print(
        f"  {'share%':>6} {'total_ms':>9} {'calls':>6} {'mean_us':>8} {'ctas':>6} {'thr':>4} "
        f"{'reg':>4} {'smem':>6} {'blk/SM':>6} {'occ%':>6} {'+1blk<=r':>8} {'waves':>6} {'tail%':>6}{spill_hdr}  name"
    )
    for g in res["by_geometry"]:
        short = g["name"] if len(g["name"]) <= 68 else g["name"][:65] + "..."
        spill_val = f" {g['spill_bytes']:>5} B" if has_spill_col else ""
        if has_spill_col and g.get("spill_bytes", 0) == 0:
            spill_val = f" {'-':>7}"
        print(
            f"  {g['share_pct']:>6.2f} {g['total_ms']:>9.3f} {g['calls']:>6} {g['mean_us']:>8.2f} "
            f"{g['total_ctas']:>6} {g['threads_per_block']:>4} {g['registers_per_thread']:>4} "
            f"{g['shared_mem_bytes']:>6} {g['active_blocks_per_sm']:>6} {g['occupancy_pct']:>6.1f} "
            f"{str(g['max_regs_to_gain_one_block']):>8} {g['waves']:>6.2f} {g['tail_efficiency_pct']:>6.1f}{spill_val}  {short}"
        )

    if getattr(args, "by_name", False) or not res["by_geometry"]:
        print("\n## Top Kernels (Aggregated by Symbol Name)")
        print(f"  {'share%':>6} {'total_ms':>10} {'calls':>7} {'mean_us':>8} {'spill':>7}  name")
        for k in res["by_name"]:
            short = k["name"] if len(k["name"]) <= 80 else k["name"][:77] + "..."
            spill_s = f"{k['spill_bytes']:>5} B" if k.get("spill_bytes", 0) > 0 else f"{'-':>7}"
            print(
                f"  {k['share_pct']:>6.2f} {k['total_ms']:>10.3f} {k['calls']:>7} {k['mean_us']:>8.2f} {spill_s}  {short}"
            )

    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    """Sweep environment variables or problem sizes on a harness with correctness gating."""
    from .kernel_math import calculate_ab_stats

    exe = args.exe
    if not exe or not Path(exe).exists():
        print(f"error: --exe must name an existing executable (got {exe!r})", file=sys.stderr)
        return 2
    wl = args.workload
    n = int(args.problem_size) if args.problem_size is not None else None
    iters = int(args.iterations or 20)
    warmup = int(args.warmup or 5)
    repeats = int(args.repeats or 3)
    extra_args = list(args.extra_arg or [])
    if getattr(args, "ckt", False) and "--ckt" not in extra_args:
        extra_args.insert(0, "--ckt")

    # Parse --env KEY=v1,v2,v3
    env_var = None
    env_vals: list[str] = [""]
    if args.env:
        if "=" not in args.env:
            print("error: --env must have the form VAR=val1,val2,...", file=sys.stderr)
            return 2
        env_var, raw_vals = args.env.split("=", 1)
        env_vals = [v.strip() for v in raw_vals.split(",") if v.strip()]

    results_list: list[dict[str, Any]] = []
    base_samples: list[float] | None = None

    for idx, val in enumerate(env_vals):
        label = f"{env_var}={val}" if env_var else "default"
        old_env = os.environ.get(env_var) if env_var else None
        if env_var:
            os.environ[env_var] = val
        try:
            run_args = extra_args + ["--workload", wl]
            if n is not None:
                run_args += ["--n", str(n)]
            bt = baseline_timing(exe, run_args, workload=wl, iters=iters, warmup=warmup, repeats=repeats)
            vrec = verify_harness(exe, run_args) if not args.no_verify else {"verify.status": "skipped"}
            ver = verification_from_records(vrec, workload=wl)
            gate = gate_from_verification(ver)
        finally:
            if env_var:
                if old_env is None:
                    os.environ.pop(env_var, None)
                else:
                    os.environ[env_var] = old_env

        samples = bt.samples or ([bt.stats.median] if bt.stats else [])
        ab = None
        if base_samples is None and samples:
            base_samples = list(samples)
        elif base_samples and samples:
            ab = calculate_ab_stats(base_samples, samples)

        med = bt.stats.median if bt.stats else None
        bw = bt.derived.get("derived.effective_bandwidth_gbps")
        tf = bt.derived.get("derived.tflops")
        results_list.append(
            {
                "label": label,
                "workload": wl,
                "problem_size": n,
                "median_us": med,
                "p10_us": bt.stats.p10 if bt.stats else None,
                "p90_us": bt.stats.p90 if bt.stats else None,
                "cv": bt.stats.cv if bt.stats else None,
                "effective_bandwidth_gbps": bw,
                "tflops": tf,
                "verified": gate.validated if not args.no_verify else True,
                "verify_detail": ver.detail,
                "ab_vs_first": ab,
            }
        )

    if args.json:
        print(json.dumps({"workload": wl, "sweep": results_list}, indent=2))
        return 0

    print(f"Sweep for workload '{wl}' (n={n}, iters={iters}, repeats={repeats}):")
    print(f"  {'variant':<26} {'median_us':>10} {'eff_BW_GB/s':>12} {'TFLOP/s':>9} {'speedup':>9} {'verdict':<18} {'verified':>8}")
    for r in results_list:
        ab = r.get("ab_vs_first") or {}
        spd = f"{ab['speedup']:.3f}x" if ab.get("speedup") else "1.000x"
        verd = ab.get("verdict", "BASELINE")
        med_str = f"{r['median_us']:.2f}" if r["median_us"] else "ERR"
        bw_str = f"{r['effective_bandwidth_gbps']:.2f}" if r["effective_bandwidth_gbps"] else "-"
        tf_str = f"{r['tflops']:.3f}" if r["tflops"] else "-"
        ver_str = "yes" if r["verified"] else "FAIL"
        print(
            f"  {r['label']:<26} "
            f"{med_str:>10} "
            f"{bw_str:>12} "
            f"{tf_str:>9} "
            f"{spd:>9} "
            f"{verd:<18} "
            f"{ver_str:>8}"
        )
    if args.output:
        out_p = Path(args.output)
        if out_p.suffix == ".json":
            out_p.parent.mkdir(parents=True, exist_ok=True)
            write_json(out_p, {"workload": wl, "sweep": results_list})
        else:
            out_p.mkdir(parents=True, exist_ok=True)
            write_json(out_p / "sweep.json", {"workload": wl, "sweep": results_list})
    return 0


def cmd_experiments(args: argparse.Namespace) -> int:
    m = ExperimentManager(args.dir)
    if args.prune is not None:
        removed = m.prune(keep=args.prune, dry_run=not args.yes)
        for p in removed:
            print(("removed " if args.yes else "would remove ") + p)
        return 0
    recs = m.history()
    if args.json:
        print(json.dumps([r.to_dict() for r in recs], indent=2))
        return 0
    if not recs:
        print(f"no experiments recorded in {args.dir}")
        return 0
    print(f"{'name':<28}{'status':<10}{'outcome':<10}{'created':<21}identity")
    for r in recs:
        print(f"{r.name:<28}{r.status:<10}{r.outcome:<10}{r.created_at:<21}{r.identity.digest()}")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """Check which pipeline stages this environment can support."""
    cache = Cache(args.cache_dir)
    env = _open_env(args)
    problems: list[str] = []

    def check(name: str, ok: bool, detail: str, fatal: bool = False) -> None:
        mark = "ok  " if ok else "FAIL"
        print(f"  [{mark}] {name:<26}{detail}")
        if not ok and fatal:
            problems.append(f"{name}: {detail}")

    print("environment")
    dev = env.device
    check("CUDA device", dev.multi_processor_count > 0, f"{dev.name} ({dev.arch})", fatal=True)
    bw, bw_src = dev.peak_bandwidth_gbps()
    check("measured bandwidth", bw is not None, f"{bw:.1f} GB/s ({bw_src})" if bw else "unavailable — memory utilisation will be unknown")
    ct, ct_src = dev.compute_ceiling_tflops
    check("measured compute ceiling", ct is not None, f"{ct:.2f} TFLOP/s ({ct_src})" if ct else "unavailable — compute utilisation will be unknown")

    print("toolchain")
    for name in ("nvcc", "ncu", "cuobjdump", "nvdisasm", "nsys", "compute-sanitizer"):
        t = env.tools.get(name)
        check(name, bool(t and t.available), (t.version or t.path or "") if t else "not found")

    if not env.tools.get("ncu") or not env.tools["ncu"].available:
        problems.append(
            "ncu is unavailable: profiling, bottleneck classification and recommendations "
            "are all disabled. Static and timing stages still work."
        )
    if not env.tools.get("cuobjdump") or not env.tools["cuobjdump"].available:
        problems.append(
            "cuobjdump is unavailable: SASS inspection and instruction-level evidence are disabled."
        )

    prof = Profiler(cache)
    if prof.ncu:
        print("profiler")
        check("metric catalog", prof.catalog.raw_count > 1000,
              f"{prof.catalog.raw_count} metric names, ncu {prof.catalog.ncu_version}")
        for grp in ("duration", "memory_throughput", "stall_long_scoreboard", "l2_hit_rate", "launch_registers"):
            r = prof.catalog.resolve_group(grp)
            check(f"metrics: {grp}", bool(r), ", ".join(r) if r else "not present on this target")

    print("python")
    check("stdlib only", True, f"python {env.python_version}")
    for mod, why in (("yaml", "YAML config files"), ("tomllib", "TOML config files")):
        try:
            __import__(mod)
            check(f"module: {mod}", True, f"available ({why})")
        except ImportError:
            check(f"module: {mod}", False, f"missing ({why}); use JSON config instead")

    print()
    if problems:
        print(f"{len(problems)} problem(s) found:")
        for p in problems:
            print(f"  ! {p}")
        return 1
    print("no blocking problems found")
    return 0


def cmd_shapes(args: argparse.Namespace) -> int:
    env = _open_env(args)
    dev = env.device
    action = getattr(args, "shapes_action", "list")
    repo_root = getattr(args, "repo_root", ".")

    if action == "list":
        specs = discover_shapes(
            repo_root,
            qtype=getattr(args, "qtype", None),
            n=getattr(args, "n", None),
            k=getattr(args, "k", None),
            dev=dev,
        )
        if getattr(args, "json", False):
            print(json.dumps([s.to_dict() for s in specs], indent=2))
            return 0
        print(f"Found {len(specs)} shape dispatch files in /shapes/ (device: {dev.name}, L2={dev.l2_cache_size // (1024*1024)} MiB)")
        print(f"  {'qtype':<7}{'N':>8}{'K':>8}{'weight_MB':>11}{'L2_regime':<16}{'mem_floor_T1':>14}{'bands':>7}{'origin':>10}  file")
        for s in specs:
            mf = f"{s.memory_floor_t1_us:.2f} us" if s.memory_floor_t1_us else "-"
            rel = Path(s.file_path).name
            print(
                f"  {s.qtype:<7}{s.n:>8}{s.k:>8}{s.weight_mb:>11.2f}  {s.l2_residency_regime:<14}{mf:>14}{len(s.bands):>7}{s.tuning_origin:>10}  {s.qtype}/shapes/{rel}"
            )
        return 0

    if action == "inspect":
        if getattr(args, "file", None):
            spec = parse_shape_file(Path(args.file), dev=dev)
            specs = [spec] if spec else []
        else:
            specs = discover_shapes(
                repo_root,
                qtype=args.qtype,
                n=args.n,
                k=args.k,
                dev=dev,
            )
        if not specs:
            raise SystemExit("no matching shape file found in src/ops/linear/*/shapes/")
        spec = specs[0]
        if getattr(args, "json", False):
            print(json.dumps(spec.to_dict(), indent=2))
            return 0
        print(f"=== Shape Inspection: {spec.qtype.upper()} N={spec.n} K={spec.k} ({spec.file_path}) ===")
        print(
            f"weight: {spec.weight_mb:.2f} MB ({spec.weight_bytes:,} B)  "
            f"L2 regime: {spec.l2_residency_regime}  "
            f"T=1 mem floor: {spec.memory_floor_t1_us:.2f} us  "
            f"tuning origin: {spec.tuning_origin}"
        )
        print(
            f"  {'pol':<5}{'line':>5}  {'condition':<16}{'T_range':<11}{'tile(RxT)':<10}"
            f"{'thr':>5}{'k_sl':>5}  {'CTAs(min..max)':<14}{'wave_eff%':<12}{'floor_us':<14}{'reduction_sig':<28}route"
        )
        for b in spec.bands:
            t_rng = f"{b.min_tokens}..{b.max_tokens if b.max_tokens is not None else 'inf'}"
            tile_s = f"{b.tile_rows or '?'}x{b.tile_tokens or '?'}" if b.tile_rows else "-"
            thr_s = str(b.threads_per_block or "-")
            ksl_s = str(b.k_slices or "-")
            rsig_s = str(b.reduction_signature or "-")
            cta_s = (
                f"{b.ctas_at_min_t}..{b.ctas_at_max_t}"
                if (b.ctas_at_min_t and b.ctas_at_max_t and b.ctas_at_min_t != b.ctas_at_max_t)
                else (str(b.ctas_at_min_t) if b.ctas_at_min_t else "-")
            )
            weff_s = (
                f"{b.wave_eff_at_min_t_pct:.0f}%..{b.wave_eff_at_max_t_pct:.0f}%"
                if (b.wave_eff_at_min_t_pct and b.wave_eff_at_max_t_pct and b.wave_eff_at_min_t_pct != b.wave_eff_at_max_t_pct)
                else (f"{b.wave_eff_at_min_t_pct:.0f}%" if b.wave_eff_at_min_t_pct else "-")
            )
            fl_s = (
                f"{b.roofline_floor_at_min_t_us:.1f}..{b.roofline_floor_at_max_t_us:.1f}"
                if (b.roofline_floor_at_min_t_us and b.roofline_floor_at_max_t_us)
                else (f"{b.roofline_floor_at_min_t_us:.1f}" if b.roofline_floor_at_min_t_us else "-")
            )
            print(
                f"  {b.policy:<5}{b.line_number:>5}  {b.condition:<16}{t_rng:<11}{tile_s:<10}"
                f"{thr_s:>5}{ksl_s:>5}  {cta_s:<14}{weff_s:<12}{fl_s:<14}{rsig_s:<28}{b.route_expr}"
            )
        return 0

    if action == "bench":
        tok_list: list[int] | None = None
        if getattr(args, "tokens", None):
            tok_list = [int(x.strip()) for x in str(args.tokens).split(",") if x.strip()]
        res = benchmark_shape(
            dev=dev,
            qtype=args.qtype,
            n=args.n,
            k=args.k,
            policy=args.policy,
            mode=args.mode,
            tokens=args.t,
            token_list=tok_list,
            max_boundary_tokens=getattr(args, "max_tokens", None),
            sweep=args.sweep,
            execution=args.execution,
            warmup=args.warmup,
            repeat=args.repeat,
            fast=not args.full_flush,
            flush_mib=args.flush_mib,
            repo_root=repo_root,
            bench_exe=args.bench_exe,
            csv_out=args.csv_out,
        )
        if getattr(args, "output", None):
            write_json(args.output, res)
        if getattr(args, "json", False):
            print(json.dumps(res, indent=2))
            return 0

        print(
            f"=== Shape Benchmark: {res['qtype'].upper()} policy={res['policy']} N={res['N']} K={res['K']} "
            f"(mode={res['mode']}, flush={res['flush_mib']} MiB, origin={res['tuning_origin']}) ==="
        )
        if res.get("shape_file"):
            print(f"shape file: {res['shape_file']}")
        print(
            f"  {'T':>5}{'median_us':>11}{'min_us':>10}{'eff_GB/s':>10}{'read_%':>8}"
            f"{'TFLOP/s':>9}{'TC_%':>7}{'step_%':>8}{'CTAs':>6}{'wave_%':>8}  active_route (line)"
        )
        for pt in res["points"]:
            rd_s = f"{pt['sustained_read_pct']:.1f}%" if pt["sustained_read_pct"] is not None else "-"
            tc_s = f"{pt['tensor_peak_pct']:.1f}%" if pt["tensor_peak_pct"] is not None else "-"
            st_s = f"{pt['step_delta_pct']:+.1f}%" if pt["step_delta_pct"] is not None else "-"
            ct_s = str(pt["ctas"]) if pt["ctas"] is not None else "-"
            wv_s = f"{pt['wave_tail_eff_pct']:.0f}%" if pt["wave_tail_eff_pct"] is not None else "-"
            marker = " *SWITCH*" if pt.get("route_changed") else ""
            print(
                f"  {pt['T']:>5}{pt['median_us']:>11.2f}{pt['min_us']:>10.2f}{pt['effective_gbs']:>10.1f}"
                f"{rd_s:>8}{pt['useful_tflops']:>9.2f}{tc_s:>7}{st_s:>8}{ct_s:>6}{wv_s:>8}  "
                f"{pt['active_route']} (L{pt['shape_line']}){marker}"
            )
        if res["cliffs"]:
            print(f"\nDetected {len(res['cliffs'])} route-transition / step cliff(s):")
            for c in res["cliffs"]:
                print(f"  ! {c['insight']}")
        else:
            print("\nNo sharp route-transition cliffs (>= +15%) detected across probed boundaries.")
        return 0

    raise SystemExit(f"unknown shapes action: {action}")


def cmd_ab(args: argparse.Namespace) -> int:
    if args.csv_a and args.csv_b:
        res = compare_shape_csvs(
            args.csv_a,
            args.csv_b,
            noise_threshold_pct=args.noise_pct,
            require_bit_exact=bool(getattr(args, "require_bit_exact", False)),
        )
        if args.output:
            write_json(args.output, res)
        if args.json:
            print(json.dumps(res, indent=2))
            return 0
        print(f"=== Shape Benchmark A/B Comparison ({res['matched_points']} matched points) ===")
        print(f"  Baseline (A):  {res['file_a']}")
        print(f"  Candidate (B): {res['file_b']}")
        print(
            f"  Geomean speedup: {res['geomean_speedup_x']:.4f}x   "
            f"Arith mean: {res['arith_mean_speedup_x']:.4f}x   "
            f"Total latency sum: {res['sum_base_us']:.2f} us -> {res['sum_cand_us']:.2f} us ({res['sum_delta_pct']:+.2f}%)"
        )
        print(
            f"  Wins / Ties / Regressions (+/-{res['noise_threshold_pct']:.1f}%): "
            f"{res['wins']}W / {res['ties']}T / {res['losses']}L   "
            f"Bit-exact points: {res['bit_exact_points']}/{res['matched_points']}   "
            f"Verdict: {res['verdict']}"
        )
        print(
            f"  {'qt':<6}{'pol':<5}{'N':>7}{'K':>7}{'T':>5}{'base_us':>10}{'cand_us':>10}"
            f"{'delta_us':>10}{'delta_%':>9}{'speedup':>9}{'status':>12}  {'bit_exact':<20}"
        )
        for p in res["points"]:
            be_str = "BIT_EXACT" if p.get("bit_exact") is True else ("REDUCTION_CHANGED" if p.get("bit_exact") is False else "-")
            print(
                f"  {p['qtype']:<6}{p['policy']:<5}{p['N']:>7}{p['K']:>7}{p['T']:>5}"
                f"{p['base_us']:>10.2f}{p['cand_us']:>10.2f}{p['delta_us']:>+10.2f}"
                f"{p['delta_pct']:>+8.2f}%{p['speedup_x']:>8.3f}x{p['status']:>12}  {be_str:<20}"
            )
        return 1 if res["verdict"] == "REJECT_BIT_EXACTNESS_VIOLATION" else 0

    cmd_a = args.cmd_a
    cmd_b = args.cmd_b
    if args.exe and not (cmd_a and cmd_b):
        cmd_a = args.exe
        cmd_b = args.exe
    if not (cmd_a and cmd_b):
        raise SystemExit("ckt ab requires either (--csv-a and --csv-b) or (--cmd-a and --cmd-b) or (--exe with --env-a/--env-b)")

    from .ab_runner import _parse_env_kv

    env_a = _parse_env_kv(args.env_a)
    env_b = _parse_env_kv(args.env_b)
    res = run_interleaved_ab(
        cmd_a=cmd_a,
        cmd_b=cmd_b,
        env_a=env_a,
        env_b=env_b,
        label_a=args.label_a,
        label_b=args.label_b,
        rounds=args.rounds,
        warmup_rounds=args.warmup_rounds,
        metric_regex=args.metric_regex,
        kernel_share_of_total=args.kernel_share,
    )
    if args.output:
        write_json(args.output, res)
    if args.json:
        print(json.dumps(res, indent=2))
        return 0

    st = res["stats"]
    b = st["baseline"]
    c = st["candidate"]
    print(f"=== Interleaved GPU A/B Benchmark ({res['measured_rounds']} measured rounds, {res['warmup_rounds']} warmup) ===")
    print(f"  A ({res['label_a']}): {res['cmd_a']}" + (f"  env={res['env_a']}" if res["env_a"] else ""))
    print(f"  B ({res['label_b']}): {res['cmd_b']}" + (f"  env={res['env_b']}" if res["env_b"] else ""))
    for rl in res["round_log"]:
        print(f"    round {rl['round']} ({rl['order']}): A={rl['a_us']:.3f}  B={rl['b_us']:.3f}  ({rl['delta_pct']:+.2f}%)")
    print(f"  Baseline A : median={b['median_us']:.3f}  mean={b['mean_us']:.3f}  stdev={b['stdev_us']:.3f}  CV={b['cv_pct']:.2f}%")
    print(f"  Candidate B: median={c['median_us']:.3f}  mean={c['mean_us']:.3f}  stdev={c['stdev_us']:.3f}  CV={c['cv_pct']:.2f}%")
    print(
        f"  Delta      : {st['delta_us']:+.3f} ({st['pct_change']:+.2f}%)  "
        f"speedup={st['speedup']:.4f}x  SNR={st['signal_to_noise_ratio']}  verdict={st['verdict']}"
    )
    if "amdahl_e2e_speedup" in st:
        print(f"  Amdahl E2E : {st['amdahl_e2e_speedup']:.4f}x ({st['amdahl_e2e_pct_gain']:+.2f}%)")
    return 0


def cmd_nsys(args: argparse.Namespace) -> int:
    """Generate or execute the optimal nsys profile command based on kernel permissions."""
    from .environment import detect_profiling_capabilities

    prof = detect_profiling_capabilities()
    paranoid = prof.get("perf_event_paranoid")
    kptr = prof.get("kptr_restrict")

    unlocked_pmu = paranoid is not None and paranoid <= 0
    unlocked_sampling = paranoid is not None and paranoid <= 1
    unlocked_symbols = kptr == 0

    trace_flags = ["-t", "cuda,nvtx,osrt", "-c", "cudaProfilerApi", "--cuda-graph-trace=node", "--stats=true"]
    if getattr(args, "full", False) or unlocked_sampling:
        trace_flags.extend(["--sample=process-tree", "--backtrace=dwarf", "--cudabacktrace=all"])
    if getattr(args, "full", False) or unlocked_pmu:
        trace_flags.extend(["--event-sample=system-wide"])

    out_file = args.output or "profile"
    trace_flags.extend(["-o", out_file])

    target_cmd = list(args.target or [])
    full_cmd = ["nsys", "profile", *trace_flags, *target_cmd]

    print("# NSYS Profiling Command Configurator")
    print(f"# Linux paranoid: {paranoid} (PMU hardware events {'unlocked' if unlocked_pmu else 'locked'})")
    print(f"# kptr_restrict:  {kptr} (Kernel symbol resolution {'unlocked' if unlocked_symbols else 'locked'})")
    if not unlocked_pmu or not unlocked_symbols:
        print("#")
        print("# [UNLOCK INSTRUCTION]")
        print("# To unlock CPU PMU hardware counters and kernel symbol resolution in call stacks, run:")
        print("#   sudo sysctl -w kernel.perf_event_paranoid=-1 kernel.kptr_restrict=0")
        print("#")

    cmd_str = " ".join(full_cmd)
    print(f"Recommended command:\n{cmd_str}")

    if args.run:
        if not target_cmd:
            print("error: --run requires a target command to profile", file=sys.stderr)
            return 2
        import subprocess
        print(f"\nExecuting: {cmd_str}\n")
        return subprocess.run(full_cmd).returncode
    return 0


def cmd_cliffs(args: argparse.Namespace) -> int:
    """Scan all requested Infernix kernel families for remaining latency cliffs."""
    if getattr(args, "list_families", False):
        for fam in ALL_FAMILIES:
            print(fam)
        return 0

    families = parse_family_list(getattr(args, "families", "all"))
    env = _open_env(args)
    dev = env.device
    progress_cb = None if getattr(args, "quiet", False) or getattr(args, "json", False) else (lambda msg: print(msg, flush=True))

    report = run_cliffs_analysis(
        dev=dev,
        repo_root=getattr(args, "repo_root", "."),
        families=families,
        max_tokens=args.max_tokens,
        warmup=args.warmup,
        repeat=args.repeat,
        flush_mib=args.flush_mib,
        upward_threshold_pct=args.upward_pct,
        inverted_threshold_pct=args.inverted_pct,
        min_delta_us=args.min_delta_us,
        progress=progress_cb,
    )

    if getattr(args, "merge_report", None):
        merge_path = Path(args.merge_report)
        if not merge_path.exists():
            raise SystemExit(f"--merge-report file not found: {merge_path}")
        report = merge_reports(read_json(merge_path), report)

    if getattr(args, "output", None):
        write_json(args.output, report.to_dict())
    if getattr(args, "markdown_out", None):
        md_path = Path(args.markdown_out)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(render_markdown_report(report))

    if getattr(args, "json", False):
        print(json.dumps(report.to_dict(), indent=2))
        return 0

    format_cli_summary(report)
    if getattr(args, "output", None):
        print(f"\nSaved JSON report: {args.output}")
    if getattr(args, "markdown_out", None):
        print(f"Saved Markdown report: {args.markdown_out}")
    return 0


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--cache-dir", default=DEFAULT_CACHE, help="artifact cache directory")
    p.add_argument("--config", help="YAML/TOML/JSON configuration file")
    p.add_argument("--device", type=int, default=0, help="CUDA device index")
    p.add_argument("-v", "--verbose", action="count", default=0, help="increase logging verbosity")
    p.add_argument("-q", "--quiet", action="store_true", help="suppress progress logging")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ckt",
        description="CUDA kernel analysis and optimisation toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Typical use:\n"
            "  ckt env                      # discover hardware and measure ceilings\n"
            "  ckt doctor                   # verify the environment supports each stage\n"
            "  ckt static --sources kernels --exe ./bench\n"
            "  ckt analyze --exe ./bench --workload saxpy --problem-size 4194304\n"
            "  ckt search  --exe ./bench --workload reduce_block --problem-size 16777216\n"
        ),
    )
    p.add_argument("--version", action="version", version=f"ckt {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    e = sub.add_parser("env", help="discover hardware and toolchain")
    _add_common(e)
    e.add_argument("--refresh", action="store_true", help="ignore a stored environment snapshot")
    e.add_argument("--rebuild", action="store_true", help="rebuild the native probes")
    e.add_argument("--tools", action="store_true", help="also print the toolchain table")
    e.add_argument("--json", action="store_true")
    e.add_argument("--no-measure-bandwidth", action="store_true")
    e.add_argument("--no-measure-compute", action="store_true")
    e.set_defaults(func=cmd_env)

    s = sub.add_parser("static", help="static source and binary analysis")
    _add_common(s)
    s.add_argument("exe_pos", nargs="?", help="built executable to inspect (optional positional form)")
    s.add_argument("--sources", nargs="*", help="source files or directories")
    s.add_argument("--exe", help="built executable to inspect")
    s.add_argument("--kernel", help="filter kernels by name or demangled template substring")
    s.add_argument("--spills-only", action="store_true", help="only display kernels with STACK > 0 or LOCAL > 0")
    s.add_argument("--threads", default="auto", help="threads per block (32..1024) or 'auto' to infer per-symbol from C++ schedule template (default: auto)")
    s.add_argument("--sass", action="store_true", help="also extract and summarise SASS instruction mix")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_static)

    pr = sub.add_parser("profile", help="collect NCU metrics for one kernel")
    _add_common(pr)
    pr.add_argument("--exe")
    pr.add_argument("--workload", required=True)
    pr.add_argument("--problem-size", type=int)
    pr.add_argument("--kernel-filter", help="kernel name substring for NCU")
    pr.add_argument("--stage", default="B", choices=["B", "C"])
    pr.add_argument("--focus", choices=["core", "memory", "compute", "shared", "launch", "control"])
    pr.add_argument("--launch-count", type=int, default=1)
    pr.add_argument("--launch-skip", type=int, default=2)
    pr.add_argument("--no-cache", action="store_true")
    pr.add_argument("--json", action="store_true")
    pr.set_defaults(func=cmd_profile)

    se = sub.add_parser("search", help="launch-parameter search with correctness gating")
    _add_common(se)
    se.add_argument("--exe")
    se.add_argument("--workload", required=True)
    se.add_argument("--problem-size", type=int)
    se.add_argument("--iterations", type=int)
    se.add_argument("--warmup", type=int)
    se.add_argument("--repeats", type=int)
    se.add_argument("--strategy", choices=["exhaustive", "coarse-to-fine", "stratified", "model-guided"])
    se.add_argument("--output")
    se.add_argument("--json", action="store_true")
    se.add_argument("--no-verify", action="store_true", help="skip correctness gating (not recommended)")
    se.add_argument("--no-cache", action="store_true")
    se.set_defaults(func=cmd_search)

    an = sub.add_parser("analyze", help="run the full pipeline and write a report")
    _add_common(an)
    an.add_argument("--exe")
    an.add_argument("--workload")
    an.add_argument("--workloads", nargs="*")
    an.add_argument("--problem-size", type=int)
    an.add_argument("--iterations", type=int)
    an.add_argument("--warmup", type=int)
    an.add_argument("--repeats", type=int)
    an.add_argument("--stage", default="B", choices=["B", "C"])
    an.add_argument("--focus", choices=["core", "memory", "compute", "shared", "launch", "control"])
    an.add_argument("--output", default=DEFAULT_RESULTS)
    an.add_argument("--stem")
    an.add_argument("--title")
    an.add_argument("--no-profile", action="store_true")
    an.add_argument("--no-html", action="store_true")
    an.add_argument("--no-cache", action="store_true")
    an.add_argument("--refresh", action="store_true")
    an.set_defaults(func=cmd_analyze)

    ex = sub.add_parser("experiments", help="inspect experiment history")
    ex.add_argument("--dir", default="experiments")
    ex.add_argument("--json", action="store_true")
    ex.add_argument("--prune", nargs="*", help="keep only these experiment names")
    ex.add_argument("--yes", action="store_true", help="actually delete when pruning")
    ex.set_defaults(func=cmd_experiments)

    d = sub.add_parser("doctor", help="check environment readiness")
    _add_common(d)
    d.add_argument("--refresh", action="store_true")
    d.set_defaults(func=cmd_doctor)

    # ---- ckt math: zero-hand-math calculators for agents ------------------ #
    m = sub.add_parser("math", help="exact analytical calculators (occupancy, waves, roofline, workload, smem, mlp, ab)")
    _add_common(m)
    msub = m.add_subparsers(dest="topic", required=True)

    mo = msub.add_parser("occupancy", help="occupancy, register step table, smem step table, and 32..1024 block-size sweep")
    _add_common(mo)
    mo.add_argument("--threads", default="auto", help="threads per block (32..1024) or 'auto'/'all' for full 32..1024 sweep (default: auto)")
    mo.add_argument("--regs", type=int, required=True, help="registers per thread (1..255)")
    mo.add_argument("--smem", type=int, default=0, help="shared memory bytes per block (0..101376)")
    mo.add_argument("--json", action="store_true")
    mo.set_defaults(func=cmd_math)

    mw = msub.add_parser("waves", help="grid wave quantisation and wave-aligned CTA counts")
    _add_common(mw)
    mw.add_argument("--ctas", type=int, required=True, help="total CTAs in grid")
    mw.add_argument("--threads", type=int, default=256)
    mw.add_argument("--regs", type=int, default=64)
    mw.add_argument("--smem", type=int, default=0)
    mw.add_argument("--items", type=int, help="optional total work items to divide across wave-aligned grids")
    mw.add_argument("--json", action="store_true")
    mw.set_defaults(func=cmd_math)

    mr = msub.add_parser("roofline", help="roofline bounds, ridge points, and L2 working-set residency")
    _add_common(mr)
    mr.add_argument("--bytes", type=float, required=True, help="bytes moved")
    mr.add_argument("--flops", type=float, required=True, help="FLOP count")
    mr.add_argument("--latency-us", type=float, help="optional measured kernel latency in us")
    mr.add_argument("--dtype", default="fp32", help="compute dtype (fp32, fp16, bf16, int8, fp64)")
    mr.add_argument("--json", action="store_true")
    mr.set_defaults(func=cmd_math)

    mwl = msub.add_parser("workload", help="exact GEMV/GEMM & GGML-quantised byte/FLOP/roofline calculator")
    _add_common(mwl)
    mwl.add_argument("--m", type=int, required=True, help="batch / token / column count M")
    mwl.add_argument("--n", type=int, required=True, help="output row dimension N")
    mwl.add_argument("--k", type=int, required=True, help="reduction dimension K")
    mwl.add_argument("--weight-dtype", required=True, help="weight format (e.g. iq3_s, q4_k, q6_k, q8_0, bf16, fp16)")
    mwl.add_argument("--act-dtype", default="fp32", help="activation format (default fp32, or q8_1, bf16)")
    mwl.add_argument("--out-dtype", default="fp32", help="output format (default fp32, or bf16)")
    mwl.add_argument("--compute-dtype", default="fp32", help="compute pipe dtype (fp32, fp16, bf16, int8)")
    mwl.add_argument("--latency-us", type=float, help="optional measured kernel latency in us")
    mwl.add_argument("--json", action="store_true")
    mwl.set_defaults(func=cmd_math)

    ms = msub.add_parser("smem", help="shared-memory tile footprint, bank conflict degree, and optimal padding")
    _add_common(ms)
    ms.add_argument("--rows", type=int, required=True)
    ms.add_argument("--cols", type=int, required=True)
    ms.add_argument("--elem-bytes", type=float, required=True)
    ms.add_argument("--stages", type=int, default=1)
    ms.add_argument("--other-bytes", type=int, default=0)
    ms.add_argument("--threads", type=int, default=256)
    ms.add_argument("--regs", type=int, default=64)
    ms.add_argument("--json", action="store_true")
    ms.set_defaults(func=cmd_math)

    mm = msub.add_parser("mlp", help="Little's Law memory-level parallelism & unroll factor calculator")
    _add_common(mm)
    mm.add_argument("--warps-per-sm", type=int, required=True)
    mm.add_argument("--load-width-bytes", type=int, default=4)
    mm.add_argument("--mem-latency-ns", type=float, default=250.0)
    mm.add_argument("--json", action="store_true")
    mm.set_defaults(func=cmd_math)

    mab = msub.add_parser("ab", help="A/B timing statistics, noise floor, SNR, verdict, and Amdahl token projection")
    _add_common(mab)
    mab.add_argument("--base", required=True, help="comma-separated baseline samples in us")
    mab.add_argument("--cand", required=True, help="comma-separated candidate samples in us")
    mab.add_argument("--launches-per-token", type=int, help="optional launches of this kernel per token")
    mab.add_argument("--token-ms", type=float, help="optional baseline ms/token for end-to-end projection")
    mab.add_argument("--json", action="store_true")
    mab.set_defaults(func=cmd_math)

    # ---- ckt compare: SASS/resource diff or analysis report diff --------- #
    cmp_p = sub.add_parser("compare", help="compare SASS/resources between two binaries/symbols or two analysis JSONs")
    _add_common(cmp_p)
    cmp_p.add_argument("exe_a_pos", nargs="?", help="first binary executable (optional positional form)")
    cmp_p.add_argument("exe_b_pos", nargs="?", help="second binary executable (optional positional form)")
    cmp_p.add_argument("--exe-a", help="first binary executable")
    cmp_p.add_argument("--exe-b", help="second binary executable (defaults to --exe-a)")
    cmp_p.add_argument("--symbol", help="kernel name, mangled substring, or demangled template signature to compare")
    cmp_p.add_argument("--kernel", help="alias for --symbol")
    cmp_p.add_argument("--symbol-a", help="kernel name in exe-a")
    cmp_p.add_argument("--symbol-b", help="kernel name in exe-b")
    cmp_p.add_argument("--threads", default="auto", help="block size (32..1024) or 'auto' to infer from template signature (default: auto)")
    cmp_p.add_argument("--sass", action="store_true", help="include SASS instruction mix delta (enabled by default for binary comparison)")
    cmp_p.add_argument("--report-a", help="first analysis.json report")
    cmp_p.add_argument("--report-b", help="second analysis.json report")
    cmp_p.add_argument("--json", action="store_true")
    cmp_p.set_defaults(func=cmd_compare)

    # ---- ckt hotspots: NSYS SQLite analyzer with occupancy & wave math --- #
    hs = sub.add_parser("hotspots", help="rank NSYS SQLite kernels & functional blocks with GB10 occupancy/wave math")
    _add_common(hs)
    hs.add_argument("sqlite", help="path to nsys .sqlite export")
    hs.add_argument("--top", type=int, default=30)
    hs.add_argument("--reps", type=int, default=1, help="divide totals by number of repetitions in window")
    hs.add_argument("--phase", choices=["all", "prefill", "decode"], default="all", help="filter analysis to inference phase (all, prefill, decode)")
    hs.add_argument("--exe", help="executable binary to cross-reference compiler register spills (default: auto-detect)")
    hs.add_argument("--by-name", action="store_true", help="show top kernels aggregated by symbol name")
    hs.add_argument("--json", action="store_true")
    hs.set_defaults(func=cmd_hotspots)

    # ---- ckt sweep: env/parameter sweep with correctness gating ---------- #
    sw = sub.add_parser("sweep", help="sweep environment variables or parameters with correctness gating")
    _add_common(sw)
    sw.add_argument("--exe", required=True, help="harness executable")
    sw.add_argument("--workload", required=True, help="workload name")
    sw.add_argument("--problem-size", type=int, default=None, help="optional --n problem size")
    sw.add_argument("--env", help="environment variable sweep, e.g. INFERNIX_CKT_COLUMNS=1,2,4,8")
    sw.add_argument("--ckt", action="store_true", help="pass --ckt to an Infernix test binary")
    sw.add_argument("--extra-arg", action="append", help="extra argument passed to executable before --workload")
    sw.add_argument("--iterations", type=int, default=20)
    sw.add_argument("--warmup", type=int, default=5)
    sw.add_argument("--repeats", type=int, default=3)
    sw.add_argument("--no-verify", action="store_true")
    sw.add_argument("--output")
    sw.add_argument("--json", action="store_true")
    sw.set_defaults(func=cmd_sweep)

    # ---- ckt shapes: discover, inspect, and benchmark /shapes/ ------------ #
    sh = sub.add_parser("shapes", help="discover, inspect, and benchmark Infernix /shapes/ dispatch tables")
    _add_common(sh)
    shsub = sh.add_subparsers(dest="shapes_action", required=True)

    sh_list = shsub.add_parser("list", help="list all /shapes/ files with weight footprint, L2 residency, and tuning origin")
    _add_common(sh_list)
    sh_list.add_argument("--qtype", choices=["q4", "q5", "q6", "q8", "fp8", "nvfp4", "bf16"], help="filter by quantization type")
    sh_list.add_argument("--n", type=int, help="filter by output rows N")
    sh_list.add_argument("--k", type=int, help="filter by input cols K")
    sh_list.add_argument("--repo-root", default=".", help="Infernix repository root")
    sh_list.add_argument("--json", action="store_true")
    sh_list.set_defaults(func=cmd_shapes)

    sh_insp = shsub.add_parser("inspect", help="inspect token dispatch bands, tile dimensions, GB10 wave efficiency, and roofline floor")
    _add_common(sh_insp)
    sh_insp.add_argument("--qtype", choices=["q4", "q5", "q6", "q8", "fp8", "nvfp4", "bf16"])
    sh_insp.add_argument("--n", type=int)
    sh_insp.add_argument("--k", type=int)
    sh_insp.add_argument("--file", help="explicit path to a src/ops/linear/*/shapes/n*_k*.{cu,cpp} file")
    sh_insp.add_argument("--repo-root", default=".")
    sh_insp.add_argument("--json", action="store_true")
    sh_insp.set_defaults(func=cmd_shapes)

    sh_bench = shsub.add_parser("bench", help="benchmark a shape on the current GPU using the fastest cold-L2 method")
    _add_common(sh_bench)
    sh_bench.add_argument("--qtype", required=True, choices=["q4", "q5", "q6", "q8", "fp8", "nvfp4", "bf16"])
    sh_bench.add_argument("--n", type=int, required=True, help="output dimension N")
    sh_bench.add_argument("--k", type=int, required=True, help="reduction dimension K")
    sh_bench.add_argument("--policy", default="a16", choices=["a16", "a8", "a4", "invariant"])
    sh_bench.add_argument("--mode", default="boundaries", choices=["boundaries", "sweep", "single"], help="boundaries (fastest: probes route cutoffs), sweep, or single")
    sh_bench.add_argument("--t", type=int, default=1, help="token count T when --mode single")
    sh_bench.add_argument("--tokens", help="comma-separated list of token counts T to probe, e.g. 1,8,16,64,96,128,129,320,321")
    sh_bench.add_argument("--max-tokens", type=int, help="maximum token boundary to probe in boundaries mode")
    sh_bench.add_argument("--sweep", help="explicit START:END[:STEP] token sweep when --mode sweep")
    sh_bench.add_argument("--execution", default="eager", choices=["eager", "graph"])
    sh_bench.add_argument("--warmup", type=int, default=2)
    sh_bench.add_argument("--repeat", type=int, default=10)
    sh_bench.add_argument("--flush-mib", type=int, help="override L2 flush buffer size in MiB (defaults to 32-48 MiB on GB10)")
    sh_bench.add_argument("--full-flush", action="store_true", help="use 2x L2 flush (48 MiB) instead of fast 1.35x L2 flush (32 MiB)")
    sh_bench.add_argument("--repo-root", default=".")
    sh_bench.add_argument("--bench-exe", help="explicit path to infernix_linear_bench")
    sh_bench.add_argument("--csv-out", help="save raw infernix_linear_bench CSV rows to this path (for `ckt ab --csv-a/--csv-b`)")
    sh_bench.add_argument("--output", help="save enriched JSON benchmark report to this path")
    sh_bench.add_argument("--json", action="store_true")
    sh_bench.set_defaults(func=cmd_shapes)

    # ---- ckt ab: interleaved command A/B and shape CSV A/B ---------------- #
    ab_p = sub.add_parser("ab", help="run interleaved GPU A/B benchmarks or compare two shape benchmark CSV/JSON files")
    _add_common(ab_p)
    ab_p.add_argument("--csv-a", help="baseline shape benchmark CSV or JSON file")
    ab_p.add_argument("--csv-b", help="candidate shape benchmark CSV or JSON file")
    ab_p.add_argument("--noise-pct", type=float, default=2.0, help="noise threshold percentage for CSV point classification (default 2.0%%)")
    ab_p.add_argument("--require-bit-exact", dest="require_bit_exact", action="store_true", default=True, help="reject A/B comparison if any point changes its K-slice reduction signature (default: True)")
    ab_p.add_argument("--allow-reduction-change", "--no-require-bit-exact", dest="require_bit_exact", action="store_false", help="allow K-slice reduction order changes (for fast-path evaluation)")
    ab_p.add_argument("--cmd-a", help="baseline command line for interleaved GPU execution")
    ab_p.add_argument("--cmd-b", help="candidate command line for interleaved GPU execution")
    ab_p.add_argument("--exe", help="shared executable when comparing --env-a vs --env-b")
    ab_p.add_argument("--env-a", action="append", help="env var(s) for A, e.g. VAR=0")
    ab_p.add_argument("--env-b", action="append", help="env var(s) for B, e.g. VAR=1")
    ab_p.add_argument("--label-a", default="baseline_A")
    ab_p.add_argument("--label-b", default="candidate_B")
    ab_p.add_argument("--rounds", type=int, default=6, help="number of measured interleaved A/B rounds (default 6)")
    ab_p.add_argument("--warmup-rounds", type=int, default=1, help="number of warmup rounds before measurement (default 1)")
    ab_p.add_argument("--metric-regex", help="custom regex with group(1) capturing latency/metric")
    ab_p.add_argument("--kernel-share", type=float, help="optional kernel share of end-to-end runtime (0..1) for Amdahl projection")
    ab_p.add_argument("--output", help="write full JSON A/B report to path")
    ab_p.add_argument("--json", action="store_true")
    ab_p.set_defaults(func=cmd_ab)

    # ---- ckt nsys: generate or run optimal nsys profile command ---------- #
    ns = sub.add_parser("nsys", help="generate or run optimal nsys command with unlocked PMU/CPU profiling")
    _add_common(ns)
    ns.add_argument("-o", "--output", default="profile", help="output file prefix (default: profile)")
    ns.add_argument("--full", action="store_true", help="force full PMU event sampling and DWARF backtraces")
    ns.add_argument("--run", action="store_true", help="execute the nsys profile command directly")
    ns.add_argument("target", nargs=argparse.REMAINDER, help="target executable and arguments to profile")
    ns.set_defaults(func=cmd_nsys)

    # ---- ckt cliffs: unified cross-family cliff analysis ----------------- #
    cl_p = sub.add_parser("cliffs", help="scan all kernel families for remaining latency cliffs and generate a unified report")
    _add_common(cl_p)
    cl_p.add_argument("--families", default="all", help="comma-separated kernel families or 'all' (default: all)")
    cl_p.add_argument("--list-families", action="store_true", help="list registered kernel families and exit")
    cl_p.add_argument("--max-tokens", type=int, default=512, help="maximum token width T to probe (default: 512)")
    cl_p.add_argument("--warmup", type=int, default=2, help="warmup iterations per point (default: 2)")
    cl_p.add_argument("--repeat", type=int, default=8, help="measured iterations per point (default: 8)")
    cl_p.add_argument("--flush-mib", type=int, help="override L2 flush buffer size in MiB (default: GB10 32 MiB)")
    cl_p.add_argument("--upward-pct", type=float, default=15.0, help="upward step threshold pct (default: 15.0)")
    cl_p.add_argument("--inverted-pct", type=float, default=-8.0, help="inverted cliff threshold pct (default: -8.0)")
    cl_p.add_argument("--min-delta-us", type=float, default=0.5, help="minimum absolute step change in us to flag (default: 0.5)")
    cl_p.add_argument("--repo-root", default=".", help="repository root directory")
    cl_p.add_argument("--merge-report", help="merge newly scanned families into an existing JSON cliff report")
    cl_p.add_argument("--output", help="write consolidated JSON cliff report to path")
    cl_p.add_argument("--markdown-out", help="write consolidated Markdown cliff report to path")
    cl_p.add_argument("--json", action="store_true")
    cl_p.set_defaults(func=cmd_cliffs)

    # ---- ckt autotune: offline benchmark-driven CUDA autotuning ---------- #
    register_autotune_subparsers(sub, _add_common)

    # ---- ckt kt: in-process Kernel Tuner qualification of product kernels - #
    register_kt_subparsers(sub, _add_common)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(0 if getattr(args, "quiet", False) else getattr(args, "verbose", 0) + 1)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except ToolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 4
    except SystemExit as exc:
        if isinstance(exc.code, str):
            print(exc.code, file=sys.stderr)
            return 2
        raise


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
