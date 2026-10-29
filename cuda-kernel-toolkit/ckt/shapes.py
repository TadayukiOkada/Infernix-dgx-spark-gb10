"""shapes — discover, inspect, and benchmark Infernix `/shapes/` dispatch tables.

Infernix stores per-shape token dispatch tables under:
    src/ops/linear/<qtype>/shapes/n<N>_k<K>.{cu,cpp}

Many shape tables were originally fitted on RTX 5090 (170 SMs, 96 MiB L2,
1792 GB/s GDDR7) and must be audited or retuned for GB10 (48 SMs, 24 MiB L2,
242.7 GB/s LPDDR5X unified memory).

This module provides:
1. ``discover_shapes()``: scans all ``/shapes/`` directories, parses every
   ``n<N>_k<K>.{cu,cpp}`` file, extracts its token dispatch thresholds and
   schedule names, parses tile dimensions ``(TileRows, TileTokens)`` from the
   schedule identifiers, and computes GB10 wave quantization and roofline bounds
   without hand math.
2. ``inspect_shape()``: produces a zero-hand-math analytical table for every
   token band in a shape file (CTA count, waves on 48 SMs, tail efficiency,
   weight/activation/output bytes, arithmetic intensity, and GB10 roofline
   floor latency).
3. ``benchmark_shape()``: runs ``infernix_linear_bench`` using the fastest
   cold-cache method on GB10:
   - Single in-process weight allocation per ``(qtype, N, K)`` shape.
   - GB10-sized L2 eviction buffer (``--flush-mib 48`` = 2x GB10's 24 MiB L2
     instead of the 256 MiB RTX 5090 default, or ``32 MiB`` in ``--fast`` mode,
     reducing L2 flush overhead by 5x-8x).
   - Boundary-aware token probing (``--mode boundaries``) or continuous sweep
     (``--mode sweep``), mapping every measured token count ``T`` back to the
     exact C++ schedule in the ``/shapes/`` file and automatically flagging
     route-transition cliffs and low-utilization bands.
"""

from __future__ import annotations

import csv
import math
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .environment import DeviceInfo
from .kernel_math import calculate_roofline_math, calculate_wave_tables, calculate_workload_math
from .util import ToolError, run, to_plain


# Map Infernix linear directory names to workload_math dtype names and default policies.
QTYPE_TO_WORKLOAD_DTYPE: dict[str, str] = {
    "q4": "q4_g64",
    "q5": "q5_g64",
    "q6": "q6_g64",
    "q8": "q8_g32",
    "fp8": "fp8_e4m3",
    "nvfp4": "nvfp4",
    "bf16": "bf16",
}

QTYPE_TO_COMPUTE_DTYPE: dict[str, dict[str, str]] = {
    "q4": {"a16": "bf16"},
    "q5": {"a16": "bf16"},
    "q6": {"a16": "bf16"},
    "q8": {"a16": "bf16"},
    "fp8": {"a16": "bf16", "a8": "fp8"},
    "nvfp4": {"a16": "bf16", "a4": "nvfp4"},
    "bf16": {"a16": "bf16"},
}


@dataclass
class DispatchBand:
    """One token-count dispatch branch in a ``/shapes/n<N>_k<K>.{cu,cpp}`` file."""

    policy: str  # "a16", "a8", "a4"
    condition: str  # e.g. "tokens <= 8", "tokens == 1", "default"
    min_tokens: int
    max_tokens: int | None  # None means unbounded (default/fallback branch)
    route_expr: str
    line_number: int = 0
    tile_rows: int | None = None
    tile_tokens: int | None = None
    tile_k: int | None = None
    threads_per_block: int | None = None
    k_slices: int | None = None
    reduction_signature: str | None = None
    ctas_at_min_t: int | None = None
    ctas_at_max_t: int | None = None
    wave_eff_at_min_t_pct: float | None = None
    wave_eff_at_max_t_pct: float | None = None
    roofline_floor_at_min_t_us: float | None = None
    roofline_floor_at_max_t_us: float | None = None

    @property
    def route(self) -> str:
        return self.route_expr

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass
class ShapeSpec:
    """Parsed metadata and dispatch ladder for one ``/shapes/n<N>_k<K>`` file."""

    qtype: str
    n: int
    k: int
    file_path: str
    tuning_origin: str = "unknown"  # "GB10", "RTX 5090", or "unknown"
    policies: list[str] = field(default_factory=list)
    weight_bytes: int = 0
    weight_mb: float = 0.0
    l2_residency_regime: str = "unknown"
    memory_floor_t1_us: float | None = None
    token_step: int = 1
    bands: list[DispatchBand] = field(default_factory=list)
    aliases: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


_FILENAME_RE = re.compile(r"^n(\d+)_k(\d+)\.(cu|cpp)$")
_USING_RE = re.compile(r"using\s+([A-Za-z0-9_]+)\s*=\s*([^;]+);")
_IF_LE_RE = re.compile(
    r"if\s*\(\s*(?:tokens|x\.ne\[1\])\s*(<=|==|<|>=|>)\s*(\d+)\s*\)\s*return\s+([^;]+);",
    re.S,
)
_RETURN_RE = re.compile(r"^\s*(?:return\s+)?([A-Za-z0-9_<>:,\s]+?)(?:\([^;]*\))?\s*;\s*$")


def _extract_tile_dims(route_text: str, aliases: dict[str, str]) -> tuple[int | None, int | None, int | None]:
    """Extract ``(tile_rows, tile_tokens, tile_k)`` from a schedule/route name or alias."""
    expanded = route_text
    for alias_name, alias_target in aliases.items():
        if re.search(rf"\b{re.escape(alias_name)}\b", expanded):
            expanded = f"{expanded} {alias_target}"

    # Pattern 1: r64_t128_k64 or r32_t16_w4
    m = re.search(r"(?:_|^|\b)r(\d+)(?:x\d+)?_t(\d+)(?:_k(\d+))?", expanded, re.I)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3)) if m.group(3) else None

    # Pattern 2: R64T128K64 or R32T16W4
    m = re.search(r"R(\d+)T(\d+)(?:K(\d+))?", expanded)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3)) if m.group(3) else None

    # Pattern 3: T64R128K128 or Tma64x128 (tokens x rows in Fp8/Nvfp4 schedules)
    m = re.search(r"T(?:ma)?(\d+)(?:R|x)(\d+)(?:K(\d+))?", expanded)
    if m:
        return int(m.group(2)), int(m.group(1)), int(m.group(3)) if m.group(3) else None

    # Pattern 4: Fp8A16TmaMmaSchedule<Tokens, Rows, ...> or Fp8A8TmaMmaSchedule<Tokens, Rows, BlockK, ...>
    m = re.search(r"(?:Fp8A16TmaMmaSchedule|Fp8A8TmaMmaSchedule|Nvfp4A4MmaSchedule)<\s*(\d+)\s*,\s*(\d+)(?:\s*,\s*(\d+))?", expanded)
    if m:
        t_tok = int(m.group(1))
        t_row = int(m.group(2))
        t_k = int(m.group(3)) if m.group(3) and int(m.group(3)) >= 16 else None
        return t_row, t_tok, t_k

    # Pattern 5: Fp8A16MmaSchedule<Rows, Tokens, BlockK, ...> or Nvfp4A16MmaSchedule<Rows, Tokens, BlockK, ...>
    m = re.search(r"(?:Fp8A16MmaSchedule|Nvfp4A16MmaSchedule)<\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", expanded)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))

    # Pattern 6: SlicedInstance<Rows, Warps, ...> or Q8A16SlicedKMmaSchedule<Tokens, ...>
    m = re.search(r"Q8A16SlicedKMmaSchedule<\s*(\d+)", expanded)
    if m:
        return 16, int(m.group(1)), None

    m = re.search(r"(?:Fp8SlicedInstance|Nvfp4SlicedInstance)<\s*(\d+)", expanded)
    if m:
        return int(m.group(1)), None, None

    # Pattern 7: sliced_k5120_t16 or sliced_k2048_t4 (16 rows, t tokens)
    m = re.search(r"sliced_k\d+_t(\d+)", expanded, re.I)
    if m:
        return 16, int(m.group(1)), None

    return None, None, None


def extract_route_signature(
    route_text: str,
    aliases: dict[str, str] | None = None,
    policy: str = "a16",
) -> dict[str, Any]:
    """Extract ``threads_per_block``, ``k_slices``, and ``reduction_signature`` from a route name.

    Two schedules with the same ``reduction_signature`` accumulate each output element ``(n, t)``
    across ``K`` in the exact same floating-point order and therefore produce **100% bit-exact**
    (``0 ULP``) outputs. Changing ``k_slices`` (e.g. ``w4 -> w2 -> w1``) or the compute pipe
    changes the FP32 reduction tree and requires a ``LinearRoutes`` fast-path flag under
    ``AGENTS.md``.
    """
    expanded = route_text
    if aliases:
        for alias_name, alias_target in aliases.items():
            if re.search(rf"\b{re.escape(alias_name)}\b", expanded):
                expanded = f"{expanded} {alias_target}"

    low = expanded.lower()
    pol = "a8" if "_a8_" in low else ("a4" if "_a4_" in low else policy.lower())

    # Determine compute family
    if "gemv" in low:
        family = "gemv"
    elif "stream" in low:
        family = "stream"
    elif "simt" in low:
        family = "simt"
    elif "sliced" in low:
        family = "sliced_mma"
    else:
        family = "mma"

    k_slices = 1
    threads = 256

    # Check explicit _w<N>_ in route name
    mw = re.search(r"(?:_|^|\b)w(\d+)(?:_|$|\b)", low)
    if mw:
        w = int(mw.group(1))
        if family == "gemv":
            mr = re.search(r"(?:_|^|\b)r(\d+)_w(\d+)", low)
            rows_per_blk = int(mr.group(1)) if mr else 1
            k_slices = w
            threads = rows_per_blk * w * 32
        elif family == "simt":
            mr = re.search(r"(?:_|^|\b)r(\d+)_t\d+_w(\d+)", low)
            rows_per_blk = int(mr.group(1)) if mr else 4
            k_slices = w
            threads = rows_per_blk * w * 32
        else:
            k_slices = w
            threads = w * 32
    elif "capacity" in low and family == "sliced_mma":
        # e.g. launch_q4_a16_sliced_r16_t8_capacity4 uses KWarps=4 (128 threads)
        k_slices = 4
        threads = 128
    elif re.search(r"q8a16slicedkmmaschedule<\s*\d+\s*,\s*(\d+)", low):
        m_q8s = re.search(r"q8a16slicedkmmaschedule<\s*\d+\s*,\s*(\d+)", low)
        if m_q8s:
            k_slices = int(m_q8s.group(1))
            threads = k_slices * 32
    elif re.search(r"sliced_k\d+_t\d+", low):
        # SlicedK5120T16 / SlicedK5120T24 / SlicedK5120T4 / SlicedK2048T4 all use KWarps=8 (256 threads)
        k_slices = 8
        threads = 256
    elif family == "stream":
        m_stream = re.search(r"q8a16streamschedule<\s*\d+\s*,\s*(\d+)\s*,\s*(\d+)", low)
        if m_stream:
            k_slices = int(m_stream.group(2))
        else:
            k_slices = 8
        threads = 256
    elif family == "mma":
        k_slices = 1
        # Check r<BR>_t<BT>...wr<WR>_wt<WT>
        m_wr = re.search(r"r(\d+)_t(\d+).*?wr(\d+)_wt(\d+)", low)
        if m_wr:
            br, bt, wr, wt = int(m_wr.group(1)), int(m_wr.group(2)), int(m_wr.group(3)), int(m_wr.group(4))
            if wr > 0 and wt > 0:
                warps = max(1, (br // wr) * (bt // wt))
                threads = warps * 32
        else:
            # Known Q4/Q5/Q6/Q8 MMA shorthand schedules
            if "r64_t48" in low or "r64_t96" in low:
                threads = 384
            elif "r64_t72" in low:
                threads = 192
            elif "r64_t80" in low:
                threads = 256
            elif "r64_t112" in low:
                threads = 448
            elif "r64_t120" in low:
                threads = 320
            elif "r64_t64_k128" in low:
                threads = 256
            else:
                threads = 128

    red_order = "pairwise" if "pairwise" in low else ("serial" if k_slices == 1 else "linear")
    sig = f"{pol}:{family}:k_slices={k_slices}:{red_order}"
    return {
        "threads_per_block": threads,
        "k_slices": k_slices,
        "reduction_signature": sig,
    }


def parse_shape_file(path: Path, dev: DeviceInfo | None = None) -> ShapeSpec | None:
    """Parse one ``n<N>_k<K>.{cu,cpp}`` file into a :class:`ShapeSpec`."""
    m = _FILENAME_RE.match(path.name)
    if not m:
        return None
    n = int(m.group(1))
    k = int(m.group(2))
    qtype = path.parent.parent.name.lower()
    text = path.read_text(errors="replace")

    origin = "unknown"
    if "GB10" in text or "sm_121" in text:
        origin = "GB10"
    elif "RTX 5090" in text or "5090" in text:
        origin = "RTX 5090"

    m_mod = re.search(r"tokens\s*%\s*(\d+)\s*!=\s*0", text)
    token_step = int(m_mod.group(1)) if m_mod else 1

    # Extract `using Alias = ...;` definitions so schedule parameters are visible.
    aliases: dict[str, str] = {}
    for um in _USING_RE.finditer(text):
        aliases[um.group(1).strip()] = " ".join(um.group(2).split())

    # Split into policy functions if multiple policies exist (e.g. launch_a16, launch_a8, select_a4).
    bands: list[DispatchBand] = []
    policies_seen: list[str] = []

    lines = text.splitlines()
    current_policy = "a16"
    current_min_t = token_step
    in_function = False
    brace_depth = 0

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        lineno = i + 1

        # Detect function header
        if re.search(r"\b(select_[a-z0-9_]+|launch_a16|launch_a8|select_a16|select_a4)\s*\(", stripped):
            if "a8" in stripped:
                current_policy = "a8"
            elif "a4" in stripped:
                current_policy = "a4"
            else:
                current_policy = "a16"
            if current_policy not in policies_seen:
                policies_seen.append(current_policy)
            current_min_t = token_step
            in_function = True
            brace_depth = line.count("{") - line.count("}")
            i += 1
            continue

        if in_function:
            # Join multi-line `if (...) return ...;` statements
            stmt = stripped
            end_i = i
            while (
                stmt.startswith("if ")
                and ";" not in stmt
                and end_i + 1 < len(lines)
                and (end_i - i) < 3
            ):
                end_i += 1
                stmt += " " + lines[end_i].strip()

            if_m = _IF_LE_RE.search(stmt)
            if if_m:
                op, val_s, route = if_m.group(1), if_m.group(2), " ".join(if_m.group(3).split())
                val = int(val_s)
                if op in ("<=", "=="):
                    min_t = current_min_t if op == "<=" else val
                    max_t = val
                    current_min_t = val + token_step
                elif op == "<":
                    min_t = current_min_t
                    max_t = max(current_min_t, val - token_step)
                    current_min_t = val
                else:
                    # >= or > (e.g. nvfp4 select_a4 large-token check at top)
                    min_t = val if op == ">=" else val + token_step
                    max_t = None
                t_rows, t_toks, t_k = _extract_tile_dims(route, aliases)
                sig_info = extract_route_signature(route, aliases, policy=current_policy)
                bands.append(
                    DispatchBand(
                        policy=current_policy,
                        condition=f"tokens {op} {val}",
                        min_tokens=min_t,
                        max_tokens=max_t,
                        route_expr=route,
                        line_number=lineno,
                        tile_rows=t_rows,
                        tile_tokens=t_toks,
                        tile_k=t_k,
                        threads_per_block=sig_info["threads_per_block"],
                        k_slices=sig_info["k_slices"],
                        reduction_signature=sig_info["reduction_signature"],
                    )
                )
                brace_depth += sum(lines[j].count("{") - lines[j].count("}") for j in range(i, end_i + 1))
                i = end_i + 1
                if brace_depth <= 0:
                    in_function = False
                continue

            # Final fallback `return ...;` or tail call in `launch_a16` / `launch_a8`
            if (
                not stripped.startswith("//")
                and not stripped.startswith("using ")
                and not stripped.startswith("const ")
                and not stripped.startswith("if ")
                and not stripped.startswith("select_a16(")
                and (
                    stripped.startswith("return ")
                    or re.match(r"^(fp8_linear_|launch_fp8_|nvfp4_|launch_)", stripped)
                )
                and ";" in stripped
            ):
                route = re.sub(r"^return\s+", "", stripped).rstrip(";").strip()
                route = re.sub(r"\(x,\s*weight.*$", "", route).strip()
                t_rows, t_toks, t_k = _extract_tile_dims(route, aliases)
                sig_info = extract_route_signature(route, aliases, policy=current_policy)
                bands.append(
                    DispatchBand(
                        policy=current_policy,
                        condition=f"tokens >= {current_min_t}",
                        min_tokens=current_min_t,
                        max_tokens=None,
                        route_expr=route,
                        line_number=lineno,
                        tile_rows=t_rows,
                        tile_tokens=t_toks,
                        tile_k=t_k,
                        threads_per_block=sig_info["threads_per_block"],
                        k_slices=sig_info["k_slices"],
                        reduction_signature=sig_info["reduction_signature"],
                    )
                )

            brace_depth += line.count("{") - line.count("}")
            if brace_depth <= 0:
                in_function = False

        i += 1

    if not policies_seen:
        policies_seen = ["a16"]

    # Compute workload bytes and GB10 roofline numbers if `dev` is available.
    wl_dtype = QTYPE_TO_WORKLOAD_DTYPE.get(qtype, "bf16")
    from .kernel_math import tensor_bytes

    w_info = tensor_bytes(n * k, wl_dtype)
    weight_bytes = int(w_info["bytes"])
    weight_mb = round(weight_bytes / (1024.0 * 1024.0), 3)

    l2_regime = "unknown"
    mem_floor_t1_us: float | None = None
    if dev is not None:
        wl_t1 = calculate_workload_math(
            dev, m=1, n=n, k=k, weight_dtype=wl_dtype, act_dtype="bf16", out_dtype="bf16"
        )
        rf_t1 = wl_t1["roofline_unique_weights"]
        l2_regime = str(rf_t1.get("l2_residency_regime", "unknown"))
        mem_floor_t1_us = rf_t1.get("memory_bound_latency_us")

        # Populate per-band wave and roofline metrics on `dev`
        for b in bands:
            comp_dtype = QTYPE_TO_COMPUTE_DTYPE.get(qtype, {}).get(b.policy, "bf16")
            for t_val, attr_prefix in ((b.min_tokens, "min_t"), (b.max_tokens, "max_t")):
                if t_val is None or t_val <= 0:
                    continue
                wl = calculate_workload_math(
                    dev,
                    m=t_val,
                    n=n,
                    k=k,
                    weight_dtype=wl_dtype,
                    act_dtype="bf16",
                    out_dtype="bf16",
                    compute_dtype=comp_dtype,
                )
                rf = wl["roofline_unique_weights"]
                setattr(b, f"roofline_floor_at_{attr_prefix}_us", rf.get("limit_us"))
                if b.tile_rows and b.tile_rows > 0:
                    ctas_n = math.ceil(n / b.tile_rows)
                    ctas_t = math.ceil(t_val / b.tile_tokens) if (b.tile_tokens and b.tile_tokens > 0) else 1
                    total_ctas = ctas_n * ctas_t
                    wv = calculate_wave_tables(
                        dev,
                        total_ctas=total_ctas,
                        threads_per_block=b.threads_per_block or 256,
                        registers_per_thread=168,
                    )
                    setattr(b, f"ctas_at_{attr_prefix}", total_ctas)
                    setattr(b, f"wave_eff_at_{attr_prefix}_pct", wv["wave_analysis"].get("tail_efficiency_pct"))

    return ShapeSpec(
        qtype=qtype,
        n=n,
        k=k,
        file_path=str(path),
        tuning_origin=origin,
        policies=policies_seen,
        weight_bytes=weight_bytes,
        weight_mb=weight_mb,
        l2_residency_regime=l2_regime,
        memory_floor_t1_us=mem_floor_t1_us,
        token_step=token_step,
        bands=bands,
        aliases=aliases,
    )


def discover_shapes(
    repo_root: str | Path = ".",
    *,
    qtype: str | None = None,
    n: int | None = None,
    k: int | None = None,
    dev: DeviceInfo | None = None,
) -> list[ShapeSpec]:
    """Scan ``src/ops/linear/*/shapes/`` and return parsed :class:`ShapeSpec` objects."""
    root = Path(repo_root).resolve()
    linear_dir = root / "src" / "ops" / "linear"
    if not linear_dir.exists():
        # Try parent if invoked from `cuda-kernel-toolkit/` subdirectory
        if (root.parent / "src" / "ops" / "linear").exists():
            linear_dir = root.parent / "src" / "ops" / "linear"
        else:
            return []

    specs: list[ShapeSpec] = []
    for shapes_dir in sorted(linear_dir.glob("*/shapes")):
        qt = shapes_dir.parent.name.lower()
        if qtype and qt != qtype.lower():
            continue
        for fpath in sorted(shapes_dir.iterdir()):
            if fpath.suffix not in (".cu", ".cpp"):
                continue
            spec = parse_shape_file(fpath, dev=dev)
            if spec is None:
                continue
            if n is not None and spec.n != n:
                continue
            if k is not None and spec.k != k:
                continue
            specs.append(spec)

    specs.sort(key=lambda s: (s.qtype, s.n, s.k))
    return specs


def boundary_tokens_for_shape(spec: ShapeSpec, policy: str = "a16", max_t: int | None = None) -> list[int]:
    """Return the minimal set of token counts ``T`` that test every dispatch boundary in ``spec``.

    For each ``tokens <= B`` branch, we test both ``B`` (the top of that branch)
    and ``B + token_step`` (the first valid token count of the next branch).
    """
    t_step = max(1, spec.token_step)
    if max_t is None:
        highest_band = max((b.max_tokens or 0 for b in spec.bands if b.policy == policy), default=256)
        max_t = max(512, highest_band + t_step)
    pts: set[int] = {t_step}
    for b in spec.bands:
        if b.policy != policy:
            continue
        if b.min_tokens and t_step <= b.min_tokens <= max_t and b.min_tokens % t_step == 0:
            pts.add(b.min_tokens)
        if b.max_tokens and t_step <= b.max_tokens <= max_t and b.max_tokens % t_step == 0:
            pts.add(b.max_tokens)
            if b.max_tokens + t_step <= max_t:
                pts.add(b.max_tokens + t_step)
    if not pts:
        pts = {t_step * x for x in (1, 2, 4, 8, 16, 32, 64)}
    return sorted(pts)


def active_band_for_tokens(spec: ShapeSpec, tokens: int, policy: str = "a16") -> DispatchBand | None:
    """Return the :class:`DispatchBand` selected at ``tokens`` for ``policy``."""
    policy_bands = [b for b in spec.bands if b.policy == policy]
    if not policy_bands:
        policy_bands = spec.bands
    # Check explicit upper-bounded bands in order, or >= bands if matched first.
    for b in policy_bands:
        if b.condition.startswith("tokens >=") and b.max_tokens is None and b.line_number < policy_bands[-1].line_number:
            if tokens >= b.min_tokens:
                return b
        elif b.max_tokens is not None:
            if b.min_tokens <= tokens <= b.max_tokens:
                return b
        else:
            if tokens >= b.min_tokens:
                return b
    return policy_bands[-1] if policy_bands else None


def optimal_flush_mib(dev: DeviceInfo, *, fast: bool = False) -> int:
    """Compute the minimum L2 flush buffer size in MiB that evicts the device's L2 cache.

    ``infernix_linear_bench`` defaults to ``256 MiB`` (sized for RTX 5090's 96 MiB L2).
    On GB10 (24 MiB L2, 242.7 GB/s LPDDR5X), flushing 256 MiB before every repeat
    wastes ~1.05 ms per repeat (~21 ms per T point). Sizing the flush buffer to
    2x L2 capacity (48 MiB on GB10, or 32 MiB in ``--fast`` mode) evicts L2
    completely in 0.13-0.19 ms — 5x-8x faster!
    """
    l2_bytes = dev.l2_cache_size or (24 * 1024 * 1024)
    l2_mib = max(8, math.ceil(l2_bytes / (1024 * 1024)))
    return max(16, int(round(l2_mib * (1.35 if fast else 2.0))))


def find_linear_bench(repo_root: str | Path = ".", bench_exe: str | None = None) -> Path:
    """Locate ``infernix_linear_bench`` in ``build-native/bench/`` or ``build/bench/``."""
    if bench_exe:
        p = Path(bench_exe).resolve()
        if p.exists():
            return p
        raise ToolError(f"specified benchmark executable not found: {p}")

    root = Path(repo_root).resolve()
    candidates = [
        root / "build-native" / "bench" / "infernix_linear_bench",
        root / "build" / "bench" / "infernix_linear_bench",
        root.parent / "build-native" / "bench" / "infernix_linear_bench",
        root.parent / "build" / "bench" / "infernix_linear_bench",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise ToolError(
        "infernix_linear_bench not found in build-native/bench/ or build/bench/. "
        "Build it with: cmake --build build-native --parallel \"$(nproc)\" --target infernix_linear_bench"
    )


def benchmark_shape(
    *,
    dev: DeviceInfo,
    qtype: str,
    n: int,
    k: int,
    policy: str = "a16",
    mode: str = "boundaries",  # "boundaries", "sweep", "single", "list"
    tokens: int = 1,
    token_list: list[int] | None = None,
    max_boundary_tokens: int | None = None,
    sweep: str | None = None,
    execution: str = "eager",
    warmup: int = 2,
    repeat: int = 10,
    fast: bool = True,
    flush_mib: int | None = None,
    repo_root: str | Path = ".",
    bench_exe: str | None = None,
    csv_out: str | Path | None = None,
    timeout: float = 300.0,
) -> dict[str, Any]:
    """Benchmark a shape using ``infernix_linear_bench`` and enrich with ``/shapes/`` route diagnostics.

    Why this is the fastest method on GB10:
    1. In-process execution via ``infernix_linear_bench`` allocates and quantizes the
       synthetic weight matrix once per ``(qtype, N, K)`` shape.
    2. ``flush_mib`` is automatically sized to GB10's 24 MiB L2 cache (32-48 MiB
       instead of 256 MiB), cutting L2-eviction overhead by 5x-8x.
    3. In ``boundaries`` mode, it parses ``src/ops/linear/<qtype>/shapes/n<N>_k<K>.{cu,cpp}``
       and probes the exact token cutoffs where the kernel route changes.
    """
    exe = find_linear_bench(repo_root, bench_exe)
    specs = discover_shapes(repo_root, qtype=qtype, n=n, k=k, dev=dev)
    spec = specs[0] if specs else None

    eff_flush_mib = flush_mib if flush_mib is not None else optimal_flush_mib(dev, fast=fast)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_csv = Path(csv_out) if csv_out else (Path(tmpdir) / "shape_bench.csv")
        tmp_csv.parent.mkdir(parents=True, exist_ok=True)

        base_cmd = [
            str(exe),
            "--qtype",
            qtype,
            "--policy",
            policy,
            "--n",
            str(n),
            "--k",
            str(k),
            "--execution",
            execution,
            "--warmup",
            str(warmup),
            "--repeat",
            str(repeat),
            "--flush-mib",
            str(eff_flush_mib),
        ]

        rows_raw: list[dict[str, str]] = []
        commands_run: list[str] = []

        if mode == "single":
            cmd = [*base_cmd, "--t", str(tokens), "--csv-out", str(tmp_csv)]
            commands_run.append(" ".join(cmd))
            res = run(cmd, timeout=timeout, check=False, log=False)
            if res.returncode != 0:
                raise ToolError(f"infernix_linear_bench failed ({res.returncode}): {res.stderr.strip()}")
            with tmp_csv.open(newline="") as f:
                rows_raw = list(csv.DictReader(f))

        elif mode == "sweep" or sweep is not None:
            sweep_arg = sweep or "1:64:1"
            cmd = [*base_cmd, "--sweep", sweep_arg, "--csv-out", str(tmp_csv)]
            commands_run.append(" ".join(cmd))
            res = run(cmd, timeout=timeout, check=False, log=False)
            if res.returncode != 0:
                raise ToolError(f"infernix_linear_bench failed ({res.returncode}): {res.stderr.strip()}")
            with tmp_csv.open(newline="") as f:
                rows_raw = list(csv.DictReader(f))

        else:
            # "boundaries" or explicit `token_list` mode: probe boundary T points from the parsed shape file.
            # If the max boundary is small (<= 32), a single in-process `--sweep 1:max_b:1`
            # reuses the single weight allocation in one process and finishes in < 0.3s!
            if token_list:
                b_tokens = sorted({int(x) for x in token_list if int(x) >= 1})
            else:
                b_tokens = (
                    boundary_tokens_for_shape(spec, policy=policy, max_t=max_boundary_tokens)
                    if spec is not None
                    else [1, 2, 4, 8, 16, 32, 64, 128]
                )
            t_step = max(1, spec.token_step if spec is not None else 1)
            max_b = max(b_tokens) if b_tokens else 32
            if max_b <= 32 and t_step == 1:
                cmd = [*base_cmd, "--sweep", f"1:{max_b}:1", "--csv-out", str(tmp_csv)]
                commands_run.append(" ".join(cmd))
                res = run(cmd, timeout=timeout, check=False, log=False)
                if res.returncode != 0:
                    raise ToolError(f"infernix_linear_bench failed ({res.returncode}): {res.stderr.strip()}")
                with tmp_csv.open(newline="") as f:
                    all_rows = list(csv.DictReader(f))
                b_set = set(b_tokens)
                rows_raw = [r for r in all_rows if int(r.get("T", 0) or 0) in b_set]
            else:
                # Group contiguous pairs (e.g. [8, 9], [16, 17], or [8, 12] when t_step=4) into tiny
                # `--sweep a:b:step` ranges so pairs share weight allocation and compute `delta_pct` directly!
                groups: list[tuple[int, int]] = []
                for t_pt in b_tokens:
                    if groups and t_pt == groups[-1][1] + t_step:
                        groups[-1] = (groups[-1][0], t_pt)
                    else:
                        groups.append((t_pt, t_pt))
                for idx, (t_start, t_end) in enumerate(groups):
                    part_csv = Path(tmpdir) / f"part_{idx}.csv"
                    if t_start == t_end:
                        cmd = [*base_cmd, "--t", str(t_start), "--csv-out", str(part_csv)]
                    else:
                        cmd = [*base_cmd, "--sweep", f"{t_start}:{t_end}:{t_step}", "--csv-out", str(part_csv)]
                    commands_run.append(" ".join(cmd))
                    res = run(cmd, timeout=timeout, check=False, log=False)
                    if res.returncode != 0:
                        raise ToolError(f"infernix_linear_bench failed ({res.returncode}): {res.stderr.strip()}")
                    with part_csv.open(newline="") as f:
                        rows_raw.extend(list(csv.DictReader(f)))
                if csv_out and rows_raw:
                    with Path(csv_out).open("w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=list(rows_raw[0].keys()))
                        writer.writeheader()
                        writer.writerows(rows_raw)

    # Enrich every measured row with active route from `/shapes/`, GB10 wave efficiency, and cliff diagnostics.
    points: list[dict[str, Any]] = []
    cliffs: list[dict[str, Any]] = []
    low_efficiency_points: list[dict[str, Any]] = []

    prev_point: dict[str, Any] | None = None
    for r in rows_raw:
        t_val = int(r["T"])
        median_us = float(r["median_us"])
        min_us = float(r["min_us"])
        p95_us = float(r["p95_us"])
        eff_gbs = float(r["effective_gbs"])
        read_pct = float(r["sustained_read_pct"]) if r.get("sustained_read_pct") else None
        tflops = float(r["useful_tflops"])
        tc_pct = float(r["tensor_peak_pct"]) if r.get("tensor_peak_pct") else None
        mem_floor_us = float(r["memory_floor_us"]) if r.get("memory_floor_us") else None

        band = active_band_for_tokens(spec, t_val, policy=policy) if spec else None
        route_expr = band.route_expr if band else "unknown"
        line_no = band.line_number if band else 0
        thr_blk = (band.threads_per_block if band and band.threads_per_block else 256)
        k_sl = (band.k_slices if band and band.k_slices else 1)
        red_sig = (band.reduction_signature if band and band.reduction_signature else f"{policy}:mma:k_slices=1:serial")

        r["active_route"] = route_expr
        r["threads_per_block"] = str(thr_blk)
        r["k_slices"] = str(k_sl)
        r["reduction_signature"] = red_sig

        ctas: int | None = None
        wave_eff_pct: float | None = None
        if band and band.tile_rows and band.tile_rows > 0:
            ctas_n = math.ceil(n / band.tile_rows)
            ctas_t = math.ceil(t_val / band.tile_tokens) if (band.tile_tokens and band.tile_tokens > 0) else 1
            ctas = ctas_n * ctas_t
            wv = calculate_wave_tables(dev, total_ctas=ctas, threads_per_block=thr_blk, registers_per_thread=168)
            wave_eff_pct = wv["wave_analysis"].get("tail_efficiency_pct")

        step_delta_pct: float | None = None
        route_changed = False
        if prev_point is not None:
            prev_us = float(prev_point["median_us"])
            if prev_us > 0:
                step_delta_pct = round((median_us / prev_us - 1.0) * 100.0, 2)
            route_changed = prev_point.get("active_route") != route_expr

        pt_dict: dict[str, Any] = {
            "T": t_val,
            "median_us": round(median_us, 3),
            "min_us": round(min_us, 3),
            "p95_us": round(p95_us, 3),
            "effective_gbs": round(eff_gbs, 2),
            "sustained_read_pct": round(read_pct, 2) if read_pct is not None else None,
            "useful_tflops": round(tflops, 3),
            "tensor_peak_pct": round(tc_pct, 2) if tc_pct is not None else None,
            "memory_floor_us": round(mem_floor_us, 3) if mem_floor_us is not None else None,
            "active_route": route_expr,
            "threads_per_block": thr_blk,
            "k_slices": k_sl,
            "reduction_signature": red_sig,
            "shape_line": line_no,
            "ctas": ctas,
            "wave_tail_eff_pct": wave_eff_pct,
            "step_delta_pct": step_delta_pct,
            "route_changed": route_changed,
        }
        points.append(pt_dict)

        if (
            prev_point is not None
            and step_delta_pct is not None
            and (t_val - int(prev_point["T"]) <= 4 or route_changed)
        ):
            if step_delta_pct >= 15.0:
                cliffs.append(
                    {
                        "kind": "upward_cliff",
                        "from_T": prev_point["T"],
                        "to_T": t_val,
                        "from_us": prev_point["median_us"],
                        "to_us": pt_dict["median_us"],
                        "step_delta_pct": step_delta_pct,
                        "from_route": prev_point["active_route"],
                        "to_route": route_expr,
                        "shape_line": line_no,
                        "to_ctas": ctas,
                        "to_wave_tail_eff_pct": wave_eff_pct,
                        "insight": (
                            f"[UPWARD CLIFF] Latency jumped +{step_delta_pct:.1f}% ({prev_point['median_us']:.2f} us -> "
                            f"{pt_dict['median_us']:.2f} us) from T={prev_point['T']} to T={t_val}"
                            + (
                                f" when switching from `{prev_point['active_route']}` to `{route_expr}` "
                                f"(line {line_no}). Extend the `{prev_point['active_route']}` threshold or "
                                "replace the destination schedule."
                                if route_changed
                                else f" within `{route_expr}` (wave tail efficiency: {wave_eff_pct}%)."
                            )
                        ),
                    }
                )
            elif step_delta_pct <= -8.0 and route_changed:
                cliffs.append(
                    {
                        "kind": "inverted_cliff_dominated_band",
                        "from_T": prev_point["T"],
                        "to_T": t_val,
                        "from_us": prev_point["median_us"],
                        "to_us": pt_dict["median_us"],
                        "step_delta_pct": step_delta_pct,
                        "from_route": prev_point["active_route"],
                        "to_route": route_expr,
                        "shape_line": line_no,
                        "to_ctas": ctas,
                        "to_wave_tail_eff_pct": wave_eff_pct,
                        "insight": (
                            f"[INVERTED CLIFF / DOMINATED BAND] Latency dropped {step_delta_pct:.1f}% "
                            f"({prev_point['median_us']:.2f} us at T={prev_point['T']} -> "
                            f"{pt_dict['median_us']:.2f} us at T={t_val}) when switching from "
                            f"`{prev_point['active_route']}` to `{route_expr}` (line {line_no}). "
                            f"The lower-token band `{prev_point['active_route']}` is slower even with fewer tokens; "
                            f"switch to `{route_expr}` earlier!"
                        ),
                    }
                )

        best_util = max(read_pct or 0.0, tc_pct or 0.0)
        if best_util > 0.0 and best_util < 55.0:
            low_efficiency_points.append(
                {
                    "T": t_val,
                    "median_us": pt_dict["median_us"],
                    "sustained_read_pct": pt_dict["sustained_read_pct"],
                    "tensor_peak_pct": pt_dict["tensor_peak_pct"],
                    "active_route": route_expr,
                    "ctas": ctas,
                    "wave_tail_eff_pct": wave_eff_pct,
                }
            )

        prev_point = pt_dict

    if csv_out and rows_raw:
        Path(csv_out).parent.mkdir(parents=True, exist_ok=True)
        with Path(csv_out).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows_raw[0].keys()))
            writer.writeheader()
            writer.writerows(rows_raw)

    return {
        "qtype": qtype,
        "policy": policy,
        "N": n,
        "K": k,
        "execution": execution,
        "mode": mode,
        "flush_mib": eff_flush_mib,
        "warmup": warmup,
        "repeat": repeat,
        "shape_file": spec.file_path if spec else None,
        "tuning_origin": spec.tuning_origin if spec else "unknown",
        "weight_mb": spec.weight_mb if spec else None,
        "l2_residency_regime": spec.l2_residency_regime if spec else None,
        "commands_run": commands_run,
        "points": points,
        "cliffs": cliffs,
        "low_efficiency_points": low_efficiency_points,
    }
