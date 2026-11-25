"""cliffs — cross-family kernel dispatch and latency cliff analyzer for Infernix on GB10.

Runs boundary-aware cold-L2 microbenchmarks across all Infernix kernel families
(``linear:*`` `/shapes/` dispatch tables, ``linear_add``, ``linear_swiglu``,
``attn_input_proj``, ``gdn_input_proj``, ``linear_pair``, ``ggml_linear``,
``causal_softmax_attention``, ``gated_delta_net``, and ``sparse_moe``), detects
upward latency cliffs and inverted cliffs (dominated bands where larger token
batches run faster than smaller ones), and renders a single consolidated JSON +
Markdown report.
"""

from __future__ import annotations

import csv
import re
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from .environment import DeviceInfo
from .shapes import benchmark_shape, discover_shapes, find_linear_bench, optimal_flush_mib
from .util import ToolError, run, to_plain


DEFAULT_PROBE_TOKENS: tuple[int, ...] = (
    1,
    2,
    3,
    4,
    5,
    8,
    9,
    12,
    13,
    16,
    17,
    24,
    25,
    32,
    33,
    48,
    49,
    64,
    65,
    96,
    97,
    128,
    129,
    192,
    193,
    256,
    257,
    384,
    385,
    512,
)

DEFAULT_ATTENTION_TOKENS: tuple[int, ...] = (
    1,
    2,
    3,
    4,
    5,
    8,
    9,
    16,
    17,
    32,
    33,
    64,
    65,
    128,
    129,
    256,
)

DEFAULT_MOE_TOKENS: tuple[int, ...] = (
    1,
    2,
    3,
    4,
    5,
    8,
    9,
    16,
    17,
    32,
    33,
    64,
    65,
)

ALL_FAMILIES: tuple[str, ...] = (
    "linear:bf16",
    "linear:fp8",
    "linear:mxfp8",
    "linear:nvfp4",
    "linear:q4",
    "linear:q5",
    "linear:q6",
    "linear:q8",
    "linear_add",
    "linear_swiglu",
    "attn_input_proj",
    "gdn_input_proj",
    "linear_pair",
    "ggml_linear",
    "causal_softmax_attention",
    "gated_delta_net",
    "sparse_moe",
    "hyper_connection",
    "qsa_attention",
    "rmsnorm_rope",
    "offloaded_sparse_moe",
    "candidate_selector",
    "dynamic_grouped_conv",
)

LINEAR_QTYPE_SUPPORTED_POLICIES: dict[str, tuple[str, ...]] = {
    "bf16": ("a16",),
    "fp8": ("a16", "a8"),
    "nvfp4": ("a16", "a4"),
    "q4": ("a16",),
    "q5": ("a16",),
    "q6": ("a16",),
    "q8": ("a16",),
}

_SEVERITY_RANK: dict[str, int] = {"critical": 0, "high": 1, "medium": 2}


@dataclass
class CliffEvent:
    """One detected upward latency cliff or inverted cliff (dominated band)."""

    family: str
    case_id: str
    kind: str  # "upward_cliff" | "inverted_cliff"
    severity: str  # "critical" | "high" | "medium"
    from_T: int
    to_T: int
    from_us: float
    to_us: float
    step_delta_pct: float
    work_growth_pct: float
    excess_delta_pct: float
    wasted_us: float
    from_route: str = ""
    to_route: str = ""
    route_changed: bool = False
    source_file: str | None = None
    shape_line: int = 0
    insight: str = ""

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass
class CaseResult:
    """Measured token-latency points and cliffs for one benchmark case."""

    family: str
    case_id: str
    status: str  # "ok" | "skipped" | "error"
    source_file: str | None = None
    command: str | None = None
    reason: str | None = None
    points: list[dict[str, Any]] = field(default_factory=list)
    cliffs: list[CliffEvent] = field(default_factory=list)
    elapsed_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass
class FamilySummary:
    """Aggregated cliff and case counts for one kernel family."""

    family: str
    cases_total: int = 0
    cases_ok: int = 0
    cases_skipped: int = 0
    cases_error: int = 0
    points_measured: int = 0
    cliffs_total: int = 0
    upward_cliffs: int = 0
    inverted_cliffs: int = 0
    critical_cliffs: int = 0
    high_cliffs: int = 0
    medium_cliffs: int = 0
    worst_upward_pct: float | None = None
    worst_inverted_pct: float | None = None
    elapsed_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass
class CliffsReport:
    """Complete cross-family cliff report across all inspected Infernix families."""

    device_name: str
    cuda_arch: str
    l2_mib: int
    flush_mib: int
    max_tokens: int
    warmup: int
    repeat: int
    upward_threshold_pct: float
    inverted_threshold_pct: float
    min_delta_us: float
    families_requested: list[str] = field(default_factory=list)
    family_summaries: list[FamilySummary] = field(default_factory=list)
    cases: list[CaseResult] = field(default_factory=list)
    top_cliffs: list[CliffEvent] = field(default_factory=list)
    total_elapsed_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


def parse_family_list(families_arg: str | None) -> list[str]:
    """Expand ``all`` or comma-separated family names into a validated ordered list."""
    if not families_arg or families_arg.strip().lower() == "all":
        return list(ALL_FAMILIES)
    selected: list[str] = []
    known = set(ALL_FAMILIES)
    for raw in families_arg.split(","):
        item = raw.strip()
        if not item:
            continue
        if item == "linear":
            for fam in ALL_FAMILIES:
                if fam.startswith("linear:") and fam not in selected:
                    selected.append(fam)
            continue
        if item not in known:
            raise ToolError(
                f"unknown kernel family '{item}'; valid families: linear, {', '.join(ALL_FAMILIES)}"
            )
        if item not in selected:
            selected.append(item)
    if not selected:
        raise ToolError("no kernel families selected")
    return selected


def _probe_tokens(tokens: Sequence[int], max_tokens: int) -> list[int]:
    pts = [int(t) for t in tokens if 1 <= int(t) <= max_tokens]
    return pts or [1]


def _classify_severity(kind: str, step_delta_pct: float) -> str:
    if kind == "upward_cliff":
        if step_delta_pct >= 35.0:
            return "critical"
        if step_delta_pct >= 25.0:
            return "high"
        return "medium"
    if step_delta_pct <= -20.0:
        return "critical"
    if step_delta_pct <= -12.0:
        return "high"
    return "medium"


def detect_cliffs(
    points: Sequence[dict[str, Any]],
    *,
    family: str,
    case_id: str,
    source_file: str | None = None,
    upward_threshold_pct: float = 15.0,
    inverted_threshold_pct: float = -8.0,
    min_delta_us: float = 0.5,
    max_adjacent_gap: int = 4,
) -> list[CliffEvent]:
    """Detect upward latency cliffs and inverted cliffs in sorted token-latency points."""
    ordered = sorted(points, key=lambda p: int(p["T"]))
    cliffs: list[CliffEvent] = []
    for prev_pt, cur_pt in zip(ordered, ordered[1:]):
        from_t = int(prev_pt["T"])
        to_t = int(cur_pt["T"])
        if to_t <= from_t:
            continue
        from_us = float(prev_pt["median_us"])
        to_us = float(cur_pt["median_us"])
        if from_us <= 0.0 or to_us <= 0.0:
            continue

        step_delta_pct = round((to_us / from_us - 1.0) * 100.0, 2)
        work_growth_pct = round((to_t / from_t - 1.0) * 100.0, 2)
        excess_delta_pct = round(step_delta_pct - max(0.0, work_growth_pct), 2)
        from_route = str(prev_pt.get("active_route") or "")
        to_route = str(cur_pt.get("active_route") or "")
        route_changed = bool(from_route and to_route and from_route != to_route) or bool(
            cur_pt.get("route_changed")
        )
        shape_line = int(cur_pt.get("shape_line") or 0)

        if step_delta_pct <= inverted_threshold_pct and (from_us - to_us) >= min_delta_us:
            wasted_us = round(from_us - to_us, 3)
            sev = _classify_severity("inverted_cliff", step_delta_pct)
            cliffs.append(
                CliffEvent(
                    family=family,
                    case_id=case_id,
                    kind="inverted_cliff",
                    severity=sev,
                    from_T=from_t,
                    to_T=to_t,
                    from_us=round(from_us, 3),
                    to_us=round(to_us, 3),
                    step_delta_pct=step_delta_pct,
                    work_growth_pct=work_growth_pct,
                    excess_delta_pct=excess_delta_pct,
                    wasted_us=wasted_us,
                    from_route=from_route,
                    to_route=to_route,
                    route_changed=route_changed,
                    source_file=source_file,
                    shape_line=shape_line,
                    insight=(
                        f"T={from_t} ({from_us:.2f} us) is slower than T={to_t} ({to_us:.2f} us), "
                        f"{step_delta_pct:+.1f}%"
                        + (f" [{from_route} -> {to_route}]" if route_changed else "")
                    ),
                )
            )
            continue

        is_adjacent = (to_t - from_t) <= max_adjacent_gap or (
            route_changed and (to_t / max(1, from_t)) <= 1.25
        )
        if is_adjacent and step_delta_pct >= upward_threshold_pct and (to_us - from_us) >= min_delta_us:
            wasted_us = round(to_us - from_us, 3)
            sev = _classify_severity("upward_cliff", step_delta_pct)
            cliffs.append(
                CliffEvent(
                    family=family,
                    case_id=case_id,
                    kind="upward_cliff",
                    severity=sev,
                    from_T=from_t,
                    to_T=to_t,
                    from_us=round(from_us, 3),
                    to_us=round(to_us, 3),
                    step_delta_pct=step_delta_pct,
                    work_growth_pct=work_growth_pct,
                    excess_delta_pct=excess_delta_pct,
                    wasted_us=wasted_us,
                    from_route=from_route,
                    to_route=to_route,
                    route_changed=route_changed,
                    source_file=source_file,
                    shape_line=shape_line,
                    insight=(
                        f"T={from_t} -> T={to_t} jumped {step_delta_pct:+.1f}% "
                        f"({from_us:.2f} -> {to_us:.2f} us)"
                        + (f" [{from_route} -> {to_route}]" if route_changed else "")
                    ),
                )
            )

    return cliffs


def _resolve_bench_exe(root: Path, name: str) -> Path | None:
    for build_dir in ("build-native", "build"):
        cand = root / build_dir / "bench" / name
        if cand.exists():
            return cand
    return None


def _pick_col(row: dict[str, str], names: Sequence[str]) -> str:
    lower_map = {k.strip().lower(): v for k, v in row.items() if k}
    for name in names:
        val = lower_map.get(name.lower())
        if val is not None and str(val).strip() != "":
            return str(val).strip()
    return ""


def parse_bench_csv_text(
    csv_text: str,
    *,
    token_cols: Sequence[str] = ("T", "t", "tokens", "W"),
    route_cols: Sequence[str] = ("route", "path", "active_route", "implementation", "schedule"),
    row_filter: Callable[[dict[str, str]], bool] | None = None,
) -> list[dict[str, Any]]:
    """Parse generic Infernix bench CSV output into normalized point dictionaries."""
    reader = csv.DictReader(line for line in csv_text.splitlines() if line.strip() and not line.lstrip().startswith("#"))
    points: list[dict[str, Any]] = []
    for row in reader:
        if row_filter is not None and not row_filter(row):
            continue
        t_raw = _pick_col(row, token_cols)
        med_raw = _pick_col(row, ("median_us",))
        if not t_raw or not med_raw:
            continue
        t_val = int(float(t_raw))
        med_us = float(med_raw)
        min_raw = _pick_col(row, ("min_us",))
        p95_raw = _pick_col(row, ("p95_us",))
        route = _pick_col(row, route_cols)
        extra_experts = _pick_col(row, ("unique_experts",))
        if extra_experts:
            route = f"{route or 'moe'}(experts={extra_experts})"
        points.append(
            {
                "T": t_val,
                "median_us": round(med_us, 3),
                "min_us": round(float(min_raw), 3) if min_raw else None,
                "p95_us": round(float(p95_raw), 3) if p95_raw else None,
                "active_route": route,
            }
        )
    return sorted(points, key=lambda p: int(p["T"]))


_FP8_SWIGLU_LINE_RE = re.compile(
    r"^(A16|A8|A8R|INVARIANT)\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)",
    re.M,
)

_GGML_LINE_RE = re.compile(
    r"^(\S+)\s+N=\s*(\d+)\s+K=\s*(\d+)\s+T=\s*(\d+)\s+route=(\S+)\s+median\s+([0-9.]+)\s+us\s+\(min\s+([0-9.]+),\s+p95\s+([0-9.]+)\)",
    re.M,
)


def parse_fp8_linear_swiglu_stdout(stdout: str) -> list[dict[str, Any]]:
    """Parse stdout table emitted by ``infernix_fp8_linear_swiglu_bench``."""
    points: list[dict[str, Any]] = []
    for m in _FP8_SWIGLU_LINE_RE.finditer(stdout):
        pol, t_s, med_s, min_s, p95_s, _gbs, _tf = m.groups()
        points.append(
            {
                "T": int(t_s),
                "median_us": round(float(med_s), 3),
                "min_us": round(float(min_s), 3),
                "p95_us": round(float(p95_s), 3),
                "active_route": pol.lower(),
            }
        )
    return sorted(points, key=lambda p: int(p["T"]))


def parse_ggml_linear_stdout(stdout: str) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Parse stdout emitted by ``infernix_ggml_linear_bench``, grouped by ``(shape, route)``."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for m in _GGML_LINE_RE.finditer(stdout):
        shape_name, _n, _k, t_s, route_name, med_s, min_s, p95_s = m.groups()
        key = (shape_name, route_name)
        grouped.setdefault(key, []).append(
            {
                "T": int(t_s),
                "median_us": round(float(med_s), 3),
                "min_us": round(float(min_s), 3),
                "p95_us": round(float(p95_s), 3),
                "active_route": route_name,
            }
        )
    for key in grouped:
        grouped[key].sort(key=lambda p: int(p["T"]))
    return grouped


_HYPER_CONN_RE = re.compile(
    r"^T=\s*(\d+)\s*\|\s*per call median\s+([0-9.]+)\s+us\s+\(min\s+([0-9.]+),\s*p95\s+([0-9.]+)\)",
    re.M,
)
_HC_FUSED_RE = re.compile(
    r"^(\S+)\t(\d+)\t(\S+)\t([0-9.]+)\t([0-9.]+)\t([0-9.]+)\t",
    re.M,
)
_QSA_RE = re.compile(
    r"^(bf16|int8|fp8|nvfp4|k8v4|vq2|k4v2)\s+(\d+)\s+(\d+)\s+(-?\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$",
    re.M,
)
_RMSNORM_ROPE_TEXT_RE = re.compile(
    r"^form=text\s+route=(\S+)\s+Q=(\d+)\s+K=(\d+)\s+T=(\d+)\s+execution=\S+\s+median=([0-9.]+)\s+us\s+min=([0-9.]+)\s+us\s+p95=([0-9.]+)\s+us",
    re.M,
)
_RMSNORM_ROPE_PAIR_RE = re.compile(
    r"^form=pair\s+W=(\d+)\s+B=(\d+)\s+T=(\d+)\s+heads_per_cta=\d+\s+execution=\S+\s+median=([0-9.]+)\s+us\s+min=([0-9.]+)\s+us\s+p95=([0-9.]+)\s+us",
    re.M,
)
_OFFLOADED_MOE_WIDE_RE = re.compile(
    r"^T=\s*(\d+)\s+placement=(\S+)\s+experts used\s+(\d+).*?moe_experts median\s+([0-9.]+)\s+us\s+\(min\s+([0-9.]+),\s*p95\s+([0-9.]+)\)",
    re.M,
)
_RESIDENT_MOE_RE = re.compile(
    r"^T=(\d+)\s+entries=\s*(\d+)\s*\|\s*median\s+([0-9.]+)\s+us\s+\(min\s+([0-9.]+),\s*p95\s+([0-9.]+)\)",
    re.M,
)
_CAUSAL_CONV1D_RE = re.compile(
    r"^causal_conv1d\s+(\S+)\s+\[(\d+),(\d+),(\d+)\]\s+median=\s*([0-9.]+)\s+us\s+min=\s*([0-9.]+)\s+us\s+p95=\s*([0-9.]+)\s+us",
    re.M,
)


def parse_hyper_connection_stdout(stdout: str, *, codec: str = "q8") -> list[dict[str, Any]]:
    """Parse stdout emitted by ``infernix_hyper_connection_bench``."""
    pts: list[dict[str, Any]] = []
    for m in _HYPER_CONN_RE.finditer(stdout):
        t_s, med_s, min_s, p95_s = m.groups()
        pts.append(
            {
                "T": int(t_s),
                "median_us": round(float(med_s), 3),
                "min_us": round(float(min_s), 3),
                "p95_us": round(float(p95_s), 3),
                "active_route": f"hyper_conn:{codec}",
            }
        )
    return sorted(pts, key=lambda p: int(p["T"]))


def parse_hc_mix_fused_stdout(stdout: str) -> dict[str, list[dict[str, Any]]]:
    """Parse tab-separated stdout emitted by ``infernix_hc_mix_fused_bench``, grouped by route."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for m in _HC_FUSED_RE.finditer(stdout):
        _codec, t_s, route_name, med_s, min_s, p95_s = m.groups()
        grouped.setdefault(route_name, []).append(
            {
                "T": int(t_s),
                "median_us": round(float(med_s), 3),
                "min_us": round(float(min_s), 3),
                "p95_us": round(float(p95_s), 3),
                "active_route": route_name,
            }
        )
    for k in grouped:
        grouped[k].sort(key=lambda p: int(p["T"]))
    return grouped


def parse_qsa_attention_stdout(stdout: str) -> dict[tuple[str, int], list[dict[str, Any]]]:
    """Parse table rows from ``infernix_qsa_attention_bench``, grouped by ``(storage, context)``."""
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for m in _QSA_RE.finditer(stdout):
        storage, ctx_s, w_s, v2_s, tot_s, _sel_s, _attn_s = m.groups()
        grouped.setdefault((storage, int(ctx_s)), []).append(
            {
                "T": int(w_s),
                "median_us": round(float(tot_s), 3),
                "min_us": None,
                "p95_us": None,
                "active_route": f"v2={v2_s}",
            }
        )
    for k in grouped:
        grouped[k].sort(key=lambda p: int(p["T"]))
    return grouped


def parse_rmsnorm_rope_stdout(stdout: str) -> dict[str, list[dict[str, Any]]]:
    """Parse ``text`` and ``pair`` output lines emitted by ``infernix_rmsnorm_rope_bench``."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for m in _RMSNORM_ROPE_TEXT_RE.finditer(stdout):
        route_name, q_s, k_s, t_s, med_s, min_s, p95_s = m.groups()
        cid = f"text:{route_name}:Q{q_s}_K{k_s}"
        grouped.setdefault(cid, []).append(
            {
                "T": int(t_s),
                "median_us": round(float(med_s), 3),
                "min_us": round(float(min_s), 3),
                "p95_us": round(float(p95_s), 3),
                "active_route": f"text:{route_name}",
            }
        )
    for m in _RMSNORM_ROPE_PAIR_RE.finditer(stdout):
        _w_s, b_s, t_s, med_s, min_s, p95_s = m.groups()
        cid = f"pair:B{b_s}"
        grouped.setdefault(cid, []).append(
            {
                "T": int(t_s),
                "median_us": round(float(med_s), 3),
                "min_us": round(float(min_s), 3),
                "p95_us": round(float(p95_s), 3),
                "active_route": "pair:fused",
            }
        )
    for k in grouped:
        grouped[k].sort(key=lambda p: int(p["T"]))
    return grouped


def parse_offloaded_moe_wide_stdout(stdout: str) -> dict[str, list[dict[str, Any]]]:
    """Parse stdout emitted by ``infernix_offloaded_moe_wide_bench``, grouped by placement."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for m in _OFFLOADED_MOE_WIDE_RE.finditer(stdout):
        t_s, placement, experts_s, med_s, min_s, p95_s = m.groups()
        grouped.setdefault(placement, []).append(
            {
                "T": int(t_s),
                "median_us": round(float(med_s), 3),
                "min_us": round(float(min_s), 3),
                "p95_us": round(float(p95_s), 3),
                "active_route": f"{placement}(experts={experts_s})",
            }
        )
    for k in grouped:
        grouped[k].sort(key=lambda p: int(p["T"]))
    return grouped


def parse_resident_moe_stdout(stdout: str) -> list[dict[str, Any]]:
    """Parse stdout emitted by ``infernix_resident_moe_bench``."""
    pts: list[dict[str, Any]] = []
    for m in _RESIDENT_MOE_RE.finditer(stdout):
        t_s, entries_s, med_s, min_s, p95_s = m.groups()
        pts.append(
            {
                "T": int(t_s),
                "median_us": round(float(med_s), 3),
                "min_us": round(float(min_s), 3),
                "p95_us": round(float(p95_s), 3),
                "active_route": f"resident(entries={entries_s})",
            }
        )
    return sorted(pts, key=lambda p: int(p["T"]))


def parse_causal_conv1d_stdout(stdout: str) -> dict[str, list[dict[str, Any]]]:
    """Parse stdout emitted by ``infernix_causal_conv1d_silu_bench``."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for m in _CAUSAL_CONV1D_RE.finditer(stdout):
        mode_tag, _c_s, t_s, _b_s, med_s, min_s, p95_s = m.groups()
        grouped.setdefault(mode_tag, []).append(
            {
                "T": int(t_s),
                "median_us": round(float(med_s), 3),
                "min_us": round(float(min_s), 3),
                "p95_us": round(float(p95_s), 3),
                "active_route": mode_tag,
            }
        )
    for k in grouped:
        grouped[k].sort(key=lambda p: int(p["T"]))
    return grouped


def _finalize_grouped_stdout_cases(
    *,
    family: str,
    source_file: str,
    cmd_str: str,
    grouped: dict[str, list[dict[str, Any]]],
    elapsed_s: float,
    upward_pct: float,
    inverted_pct: float,
    min_delta_us: float,
    progress: Callable[[str], None] | None = None,
) -> list[CaseResult]:
    """Build ``CaseResult`` objects with cliff detection for each grouped stdout series."""
    if not grouped:
        return [
            CaseResult(
                family=family,
                case_id=f"{family}:parse",
                status="error",
                source_file=source_file,
                command=cmd_str,
                reason="parsed 0 measurement rows from stdout",
                elapsed_s=elapsed_s,
            )
        ]
    per_s = round(elapsed_s / len(grouped), 3)
    results: list[CaseResult] = []
    for cid, pts in sorted(grouped.items()):
        cliffs = detect_cliffs(
            pts,
            family=family,
            case_id=cid,
            source_file=source_file,
            upward_threshold_pct=upward_pct,
            inverted_threshold_pct=inverted_pct,
            min_delta_us=min_delta_us,
        )
        cr = CaseResult(
            family=family,
            case_id=cid,
            status="ok",
            source_file=source_file,
            command=cmd_str,
            points=pts,
            cliffs=cliffs,
            elapsed_s=per_s,
        )
        results.append(cr)
        if progress:
            progress(f"  [{family}] {cid}: ok ({len(pts)} pts, {len(cliffs)} cliffs)")
    return results


def _run_csv_command(
    *,
    family: str,
    case_id: str,
    cmd_builder: Callable[[Path], list[str]],
    source_file: str | None = None,
    token_cols: Sequence[str] = ("T", "t", "tokens", "W"),
    route_cols: Sequence[str] = ("route", "path", "active_route", "implementation", "schedule"),
    row_filter: Callable[[dict[str, str]], bool] | None = None,
    upward_pct: float,
    inverted_pct: float,
    min_delta_us: float,
    timeout: float = 300.0,
) -> CaseResult:
    t0 = time.monotonic()
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "bench.csv"
        cmd = cmd_builder(csv_path)
        cmd_str = " ".join(cmd)
        try:
            res = run(cmd, timeout=timeout, check=False, log=False)
        except Exception as exc:
            return CaseResult(
                family=family,
                case_id=case_id,
                status="error",
                source_file=source_file,
                command=cmd_str,
                reason=str(exc),
                elapsed_s=round(time.monotonic() - t0, 3),
            )
        if res.returncode != 0:
            err = (res.stderr or res.stdout or "").strip().splitlines()
            msg = err[-1] if err else f"exit code {res.returncode}"
            return CaseResult(
                family=family,
                case_id=case_id,
                status="error",
                source_file=source_file,
                command=cmd_str,
                reason=msg,
                elapsed_s=round(time.monotonic() - t0, 3),
            )
        if not csv_path.exists():
            return CaseResult(
                family=family,
                case_id=case_id,
                status="error",
                source_file=source_file,
                command=cmd_str,
                reason="benchmark did not produce CSV output",
                elapsed_s=round(time.monotonic() - t0, 3),
            )
        pts = parse_bench_csv_text(
            csv_path.read_text(errors="replace"),
            token_cols=token_cols,
            route_cols=route_cols,
            row_filter=row_filter,
        )
    if not pts:
        return CaseResult(
            family=family,
            case_id=case_id,
            status="error",
            source_file=source_file,
            command=cmd_str,
            reason="parsed 0 measurement rows from CSV",
            elapsed_s=round(time.monotonic() - t0, 3),
        )
    cliffs = detect_cliffs(
        pts,
        family=family,
        case_id=case_id,
        source_file=source_file,
        upward_threshold_pct=upward_pct,
        inverted_threshold_pct=inverted_pct,
        min_delta_us=min_delta_us,
    )
    return CaseResult(
        family=family,
        case_id=case_id,
        status="ok",
        source_file=source_file,
        command=cmd_str,
        points=pts,
        cliffs=cliffs,
        elapsed_s=round(time.monotonic() - t0, 3),
    )


def _run_linear_family(
    *,
    dev: DeviceInfo,
    root: Path,
    qtype: str,
    max_tokens: int,
    warmup: int,
    repeat: int,
    flush_mib: int,
    upward_pct: float,
    inverted_pct: float,
    min_delta_us: float,
    progress: Callable[[str], None] | None = None,
) -> list[CaseResult]:
    family = f"linear:{qtype}"
    specs = discover_shapes(root, qtype=qtype, dev=dev)
    if qtype == "mxfp8":
        results: list[CaseResult] = []
        for s in specs:
            results.append(
                CaseResult(
                    family=family,
                    case_id=f"{qtype}:n{s.n}_k{s.k}",
                    status="skipped",
                    source_file=s.file_path,
                    reason="infernix_linear_bench does not support --qtype mxfp8",
                )
            )
        if progress:
            progress(f"[{family}] skipped {len(results)} shape files (no infernix_linear_bench support)")
        return results

    try:
        bench_exe = find_linear_bench(root)
    except ToolError as exc:
        return [CaseResult(family=family, case_id=f"{family}:binary", status="error", reason=str(exc))]

    supported_policies = LINEAR_QTYPE_SUPPORTED_POLICIES.get(qtype, ("a16",))
    results = []
    for spec in specs:
        pols = [p for p in spec.policies if p in supported_policies] or [supported_policies[0]]
        for pol in pols:
            case_id = f"{qtype}:{pol}:n{spec.n}_k{spec.k}"
            t0 = time.monotonic()
            try:
                res = benchmark_shape(
                    dev=dev,
                    qtype=qtype,
                    n=spec.n,
                    k=spec.k,
                    policy=pol,
                    mode="boundaries",
                    max_boundary_tokens=max_tokens,
                    execution="eager",
                    warmup=warmup,
                    repeat=repeat,
                    fast=True,
                    flush_mib=flush_mib,
                    repo_root=root,
                    bench_exe=str(bench_exe),
                )
                pts = list(res.get("points") or [])
                cliffs = detect_cliffs(
                    pts,
                    family=family,
                    case_id=case_id,
                    source_file=spec.file_path,
                    upward_threshold_pct=upward_pct,
                    inverted_threshold_pct=inverted_pct,
                    min_delta_us=min_delta_us,
                )
                cr = CaseResult(
                    family=family,
                    case_id=case_id,
                    status="ok",
                    source_file=spec.file_path,
                    command="; ".join(res.get("commands_run") or []),
                    points=pts,
                    cliffs=cliffs,
                    elapsed_s=round(time.monotonic() - t0, 3),
                )
            except Exception as exc:
                cr = CaseResult(
                    family=family,
                    case_id=case_id,
                    status="error",
                    source_file=spec.file_path,
                    reason=str(exc),
                    elapsed_s=round(time.monotonic() - t0, 3),
                )
            results.append(cr)
            if progress:
                progress(f"  [{family}] {case_id}: {cr.status} ({len(cr.points)} pts, {len(cr.cliffs)} cliffs, {cr.elapsed_s:.1f}s)")
    return results


def _run_linear_add_family(
    *,
    root: Path,
    max_tokens: int,
    warmup: int,
    repeat: int,
    flush_mib: int,
    upward_pct: float,
    inverted_pct: float,
    min_delta_us: float,
    progress: Callable[[str], None] | None = None,
) -> list[CaseResult]:
    t_str = ",".join(str(t) for t in _probe_tokens(DEFAULT_PROBE_TOKENS, max_tokens))
    specs: list[tuple[str, str, list[str], str | None]] = [
        (
            "bf16:5120x6144",
            "infernix_bf16_linear_add_bench",
            ["--route", "production", "--t-sweep", t_str, "--warmup", str(warmup), "--repeat", str(repeat)],
            "src/ops/linear_add/bf16",
        ),
        (
            "fp8:a16:5120x6144",
            "infernix_fp8_linear_add_bench",
            ["--n", "5120", "--k", "6144", "--policy", "a16", "--t-sweep", t_str, "--warmup", str(warmup), "--repeat", str(repeat)],
            "src/ops/linear_add/fp8/fp8_linear_add_a16.cu",
        ),
        (
            "fp8:a8:5120x6144",
            "infernix_fp8_linear_add_bench",
            ["--n", "5120", "--k", "6144", "--policy", "a8", "--t-sweep", t_str, "--warmup", str(warmup), "--repeat", str(repeat)],
            "src/ops/linear_add/fp8/fp8_linear_add_a8.cu",
        ),
        (
            "fp8:a16:5120x17408",
            "infernix_fp8_linear_add_bench",
            ["--n", "5120", "--k", "17408", "--policy", "a16", "--t-sweep", t_str, "--warmup", str(warmup), "--repeat", str(repeat)],
            "src/ops/linear_add/fp8/fp8_linear_add_a16.cu",
        ),
        (
            "fp8:a8:5120x17408",
            "infernix_fp8_linear_add_bench",
            ["--n", "5120", "--k", "17408", "--policy", "a8", "--t-sweep", t_str, "--warmup", str(warmup), "--repeat", str(repeat)],
            "src/ops/linear_add/fp8/fp8_linear_add_a8.cu",
        ),
        (
            "fp8:a16:3072x8192",
            "infernix_fp8_linear_add_bench",
            ["--n", "3072", "--k", "8192", "--policy", "a16", "--t-sweep", t_str, "--warmup", str(warmup), "--repeat", str(repeat)],
            "src/ops/linear_add/fp8/fp8_linear_add_n3072_k8192.cu",
        ),
        (
            "fp8:a8:3072x8192",
            "infernix_fp8_linear_add_bench",
            ["--n", "3072", "--k", "8192", "--policy", "a8", "--t-sweep", t_str, "--warmup", str(warmup), "--repeat", str(repeat)],
            "src/ops/linear_add/fp8/fp8_linear_add_n3072_k8192.cu",
        ),
        (
            "nvfp4:a16:5120x6144",
            "infernix_nvfp4_linear_add_bench",
            ["--n", "5120", "--k", "6144", "--policy", "a16", "--t-sweep", t_str, "--warmup", str(warmup), "--repeat", str(repeat)],
            "src/ops/linear_add/nvfp4/nvfp4_linear_add_a16.cu",
        ),
        (
            "nvfp4:a4:5120x6144",
            "infernix_nvfp4_linear_add_bench",
            ["--n", "5120", "--k", "6144", "--policy", "a4", "--t-sweep", t_str, "--warmup", str(warmup), "--repeat", str(repeat)],
            "src/ops/linear_add/nvfp4/nvfp4_linear_add_a4.cu",
        ),
        (
            "nvfp4:a16:5120x17408",
            "infernix_nvfp4_linear_add_bench",
            ["--n", "5120", "--k", "17408", "--policy", "a16", "--t-sweep", t_str, "--warmup", str(warmup), "--repeat", str(repeat)],
            "src/ops/linear_add/nvfp4/nvfp4_linear_add_a16.cu",
        ),
        (
            "nvfp4:a4:5120x17408",
            "infernix_nvfp4_linear_add_bench",
            ["--n", "5120", "--k", "17408", "--policy", "a4", "--t-sweep", t_str, "--warmup", str(warmup), "--repeat", str(repeat)],
            "src/ops/linear_add/nvfp4/nvfp4_linear_add_a4.cu",
        ),
        (
            "q5:k6144",
            "infernix_q5_linear_add_bench",
            ["--k", "6144", "--t-sweep", t_str, "--warmup", str(warmup), "--repeat", str(repeat), "--flush-mib", str(flush_mib)],
            "src/ops/linear_add/q5",
        ),
        (
            "q5:k17408",
            "infernix_q5_linear_add_bench",
            ["--k", "17408", "--t-sweep", t_str, "--warmup", str(warmup), "--repeat", str(repeat), "--flush-mib", str(flush_mib)],
            "src/ops/linear_add/q5",
        ),
        (
            "q8:k4096",
            "infernix_q8_linear_add_bench",
            ["--k", "4096", "--t-sweep", t_str, "--warmup", str(warmup), "--repeat", str(repeat), "--production-only"],
            "src/ops/linear_add/q8",
        ),
        (
            "q8:k6144",
            "infernix_q8_linear_add_bench",
            ["--k", "6144", "--t-sweep", t_str, "--warmup", str(warmup), "--repeat", str(repeat), "--production-only"],
            "src/ops/linear_add/q8",
        ),
    ]
    results: list[CaseResult] = []
    for cid, bin_name, args, src_hint in specs:
        exe = _resolve_bench_exe(root, bin_name)
        if exe is None:
            results.append(CaseResult(family="linear_add", case_id=cid, status="skipped", source_file=src_hint, reason=f"{bin_name} not found"))
            continue
        cr = _run_csv_command(
            family="linear_add",
            case_id=cid,
            cmd_builder=lambda cp, e=exe, a=args: [str(e), *a, "--csv-out", str(cp)],
            source_file=src_hint,
            upward_pct=upward_pct,
            inverted_pct=inverted_pct,
            min_delta_us=min_delta_us,
        )
        results.append(cr)
        if progress:
            progress(f"  [linear_add] {cid}: {cr.status} ({len(cr.points)} pts, {len(cr.cliffs)} cliffs, {cr.elapsed_s:.1f}s)")
    return results


def _run_linear_swiglu_family(
    *,
    root: Path,
    max_tokens: int,
    warmup: int,
    repeat: int,
    upward_pct: float,
    inverted_pct: float,
    min_delta_us: float,
    progress: Callable[[str], None] | None = None,
) -> list[CaseResult]:
    t_str = ",".join(str(t) for t in _probe_tokens(DEFAULT_PROBE_TOKENS, max_tokens))
    csv_specs: list[tuple[str, str, list[str], str | None]] = [
        (
            "nvfp4:a16",
            "infernix_nvfp4_linear_swiglu_bench",
            ["--policy", "a16", "--t-sweep", t_str, "--warmup", str(warmup), "--repeat", str(repeat)],
            "src/ops/linear_swiglu/nvfp4",
        ),
        (
            "nvfp4:a4",
            "infernix_nvfp4_linear_swiglu_bench",
            ["--policy", "a4", "--t-sweep", t_str, "--warmup", str(warmup), "--repeat", str(repeat)],
            "src/ops/linear_swiglu/nvfp4",
        ),
        (
            "q4:exact",
            "infernix_q4_linear_swiglu_bench",
            ["--t-sweep", t_str, "--warmup", str(warmup), "--repeat", str(repeat), "--production-only"],
            "src/ops/linear_swiglu/q4",
        ),
        (
            "q8:companion",
            "infernix_q8_linear_swiglu_bench",
            ["--problem", "companion", "--t-sweep", t_str, "--warmup", str(warmup), "--repeat", str(repeat)],
            "src/ops/linear_swiglu/q8",
        ),
        (
            "q8:dflash2",
            "infernix_q8_linear_swiglu_bench",
            ["--problem", "dflash2", "--t-sweep", t_str, "--warmup", str(warmup), "--repeat", str(repeat)],
            "src/ops/linear_swiglu/q8",
        ),
        (
            "q8:flash_next_shared",
            "infernix_q8_linear_swiglu_bench",
            ["--problem", "flash_next_shared", "--t-sweep", t_str, "--warmup", str(warmup), "--repeat", str(repeat)],
            "src/ops/linear_swiglu/q8",
        ),
    ]
    results: list[CaseResult] = []
    for cid, bin_name, args, src_hint in csv_specs:
        exe = _resolve_bench_exe(root, bin_name)
        if exe is None:
            results.append(CaseResult(family="linear_swiglu", case_id=cid, status="skipped", source_file=src_hint, reason=f"{bin_name} not found"))
            continue
        cr = _run_csv_command(
            family="linear_swiglu",
            case_id=cid,
            cmd_builder=lambda cp, e=exe, a=args: [str(e), *a, "--csv-out", str(cp)],
            source_file=src_hint,
            upward_pct=upward_pct,
            inverted_pct=inverted_pct,
            min_delta_us=min_delta_us,
        )
        results.append(cr)
        if progress:
            progress(f"  [linear_swiglu] {cid}: {cr.status} ({len(cr.points)} pts, {len(cr.cliffs)} cliffs, {cr.elapsed_s:.1f}s)")

    fp8_exe = _resolve_bench_exe(root, "infernix_fp8_linear_swiglu_bench")
    if fp8_exe is None:
        results.append(CaseResult(family="linear_swiglu", case_id="fp8", status="skipped", source_file="src/ops/linear_swiglu/fp8", reason="infernix_fp8_linear_swiglu_bench not found"))
        return results

    for pol in ("a16", "a8"):
        cid = f"fp8:{pol}"
        t0 = time.monotonic()
        cmd = [str(fp8_exe), "--policy", pol, "--t-sweep", t_str, "--warmup", str(warmup), "--repeat", str(repeat)]
        res = run(cmd, timeout=300.0, check=False, log=False)
        if res.returncode != 0:
            cr = CaseResult(
                family="linear_swiglu",
                case_id=cid,
                status="error",
                source_file="src/ops/linear_swiglu/fp8",
                command=" ".join(cmd),
                reason=(res.stderr or res.stdout or f"exit {res.returncode}").strip(),
                elapsed_s=round(time.monotonic() - t0, 3),
            )
        else:
            pts = parse_fp8_linear_swiglu_stdout(res.stdout)
            cliffs = detect_cliffs(
                pts,
                family="linear_swiglu",
                case_id=cid,
                source_file="src/ops/linear_swiglu/fp8",
                upward_threshold_pct=upward_pct,
                inverted_threshold_pct=inverted_pct,
                min_delta_us=min_delta_us,
            )
            cr = CaseResult(
                family="linear_swiglu",
                case_id=cid,
                status="ok" if pts else "error",
                source_file="src/ops/linear_swiglu/fp8",
                command=" ".join(cmd),
                reason=None if pts else "parsed 0 rows from stdout",
                points=pts,
                cliffs=cliffs,
                elapsed_s=round(time.monotonic() - t0, 3),
            )
        results.append(cr)
        if progress:
            progress(f"  [linear_swiglu] {cid}: {cr.status} ({len(cr.points)} pts, {len(cr.cliffs)} cliffs, {cr.elapsed_s:.1f}s)")
    return results


def _run_input_proj_family(
    *,
    family: str,
    bin_name: str,
    cases: Sequence[tuple[str, list[str], str]],
    root: Path,
    max_tokens: int,
    warmup: int,
    repeat: int,
    upward_pct: float,
    inverted_pct: float,
    min_delta_us: float,
    progress: Callable[[str], None] | None = None,
) -> list[CaseResult]:
    exe = _resolve_bench_exe(root, bin_name)
    if exe is None:
        return [CaseResult(family=family, case_id=f"{family}:binary", status="skipped", reason=f"{bin_name} not found")]
    t_str = ",".join(str(t) for t in _probe_tokens(DEFAULT_PROBE_TOKENS, max_tokens))
    results: list[CaseResult] = []
    for cid, extra_args, src_hint in cases:
        cr = _run_csv_command(
            family=family,
            case_id=cid,
            cmd_builder=lambda cp, e=exe, ea=extra_args: [
                str(e),
                *ea,
                "--tokens",
                t_str,
                "--cache",
                "cold",
                "--execution",
                "eager",
                "--warmup",
                str(warmup),
                "--repeat",
                str(repeat),
                "--csv-out",
                str(cp),
            ],
            source_file=src_hint,
            upward_pct=upward_pct,
            inverted_pct=inverted_pct,
            min_delta_us=min_delta_us,
        )
        results.append(cr)
        if progress:
            progress(f"  [{family}] {cid}: {cr.status} ({len(cr.points)} pts, {len(cr.cliffs)} cliffs, {cr.elapsed_s:.1f}s)")
    return results


def _run_linear_pair_family(
    *,
    root: Path,
    max_tokens: int,
    warmup: int,
    repeat: int,
    upward_pct: float,
    inverted_pct: float,
    min_delta_us: float,
    progress: Callable[[str], None] | None = None,
) -> list[CaseResult]:
    exe = _resolve_bench_exe(root, "infernix_linear_pair_bench")
    if exe is None:
        return [CaseResult(family="linear_pair", case_id="linear_pair:binary", status="skipped", reason="infernix_linear_pair_bench not found")]
    t_str = ",".join(str(t) for t in _probe_tokens(DEFAULT_PROBE_TOKENS, max_tokens))
    results: list[CaseResult] = []
    for k_val in (2048, 5120):
        cid = f"q8:k{k_val}"
        cr = _run_csv_command(
            family="linear_pair",
            case_id=cid,
            cmd_builder=lambda cp, e=exe, kv=k_val: [
                str(e),
                "--k",
                str(kv),
                "--t-sweep",
                t_str,
                "--warmup",
                str(warmup),
                "--repeat",
                str(repeat),
                "--production-only",
                "--csv-out",
                str(cp),
            ],
            source_file="src/ops/linear_pair/q8",
            upward_pct=upward_pct,
            inverted_pct=inverted_pct,
            min_delta_us=min_delta_us,
        )
        results.append(cr)
        if progress:
            progress(f"  [linear_pair] {cid}: {cr.status} ({len(cr.points)} pts, {len(cr.cliffs)} cliffs, {cr.elapsed_s:.1f}s)")
    return results


def _run_ggml_linear_family(
    *,
    root: Path,
    max_tokens: int,
    warmup: int,
    repeat: int,
    upward_pct: float,
    inverted_pct: float,
    min_delta_us: float,
    progress: Callable[[str], None] | None = None,
) -> list[CaseResult]:
    exe = _resolve_bench_exe(root, "infernix_ggml_linear_bench")
    if exe is None:
        return [CaseResult(family="ggml_linear", case_id="ggml_linear:binary", status="skipped", reason="infernix_ggml_linear_bench not found")]
    t_str = ",".join(str(t) for t in _probe_tokens(DEFAULT_PROBE_TOKENS, max_tokens))
    invocations: list[tuple[str, list[str]]] = [
        ("builtin", [str(exe), "--tokens", t_str, "--cold", "--warmup", str(warmup), "--repeat", str(repeat)]),
        (
            "iq3xxs_8192",
            [str(exe), "--shape", "iq3xxs_8192=iq3_xxs:8192:2560", "--tokens", t_str, "--cold", "--warmup", str(warmup), "--repeat", str(repeat)],
        ),
    ]
    results: list[CaseResult] = []
    for label, cmd in invocations:
        t0 = time.monotonic()
        res = run(cmd, timeout=600.0, check=False, log=False)
        dt = round(time.monotonic() - t0, 3)
        if res.returncode != 0:
            results.append(
                CaseResult(
                    family="ggml_linear",
                    case_id=label,
                    status="error",
                    source_file="src/ops/ggml",
                    command=" ".join(cmd),
                    reason=(res.stderr or res.stdout or f"exit {res.returncode}").strip(),
                    elapsed_s=dt,
                )
            )
            continue
        grouped = parse_ggml_linear_stdout(res.stdout)
        if not grouped:
            results.append(
                CaseResult(
                    family="ggml_linear",
                    case_id=label,
                    status="error",
                    source_file="src/ops/ggml",
                    command=" ".join(cmd),
                    reason="parsed 0 GGML rows from stdout",
                    elapsed_s=dt,
                )
            )
            continue
        per_case_s = round(dt / max(1, len(grouped)), 3)
        for (shape_name, route_name), pts in sorted(grouped.items()):
            cid = f"{shape_name}:{route_name}"
            cliffs = detect_cliffs(
                pts,
                family="ggml_linear",
                case_id=cid,
                source_file="src/ops/ggml",
                upward_threshold_pct=upward_pct,
                inverted_threshold_pct=inverted_pct,
                min_delta_us=min_delta_us,
            )
            cr = CaseResult(
                family="ggml_linear",
                case_id=cid,
                status="ok",
                source_file="src/ops/ggml",
                command=" ".join(cmd),
                points=pts,
                cliffs=cliffs,
                elapsed_s=per_case_s,
            )
            results.append(cr)
            if progress:
                progress(f"  [ggml_linear] {cid}: ok ({len(pts)} pts, {len(cliffs)} cliffs)")
    return results


def _run_causal_softmax_attention_family(
    *,
    root: Path,
    max_tokens: int,
    warmup: int,
    repeat: int,
    upward_pct: float,
    inverted_pct: float,
    min_delta_us: float,
    progress: Callable[[str], None] | None = None,
) -> list[CaseResult]:
    exe = _resolve_bench_exe(root, "infernix_causal_softmax_attention_bench")
    if exe is None:
        return [CaseResult(family="causal_softmax_attention", case_id="csa:binary", status="skipped", reason="infernix_causal_softmax_attention_bench not found")]
    w_str = ",".join(str(t) for t in _probe_tokens(DEFAULT_ATTENTION_TOKENS, min(max_tokens, 512)))
    results: list[CaseResult] = []
    for geom in ("d256-h24-kv4", "d256-h16-kv2"):
        for kv_dtype in ("bf16", "int8", "fp8", "nvfp4", "vq2"):
            cid = f"cached:{geom}:{kv_dtype}"
            cr = _run_csv_command(
                family="causal_softmax_attention",
                case_id=cid,
                cmd_builder=lambda cp, e=exe, g=geom, kv=kv_dtype: [
                    str(e),
                    "--entry",
                    "cached",
                    "--geometry",
                    g,
                    "--kv-dtype",
                    kv,
                    "--batch",
                    "1",
                    "--tokens",
                    w_str,
                    "--context",
                    "4096",
                    "--execution",
                    "graph",
                    "--cache",
                    "cold",
                    "--warmup",
                    str(warmup),
                    "--repeat",
                    str(repeat),
                    "--csv-out",
                    str(cp),
                ],
                source_file=f"src/ops/softmax_attention/dense/causal_cache/{kv_dtype}",
                token_cols=("W", "T", "tokens"),
                upward_pct=upward_pct,
                inverted_pct=inverted_pct,
                min_delta_us=min_delta_us,
            )
            results.append(cr)
            if progress:
                progress(f"  [causal_softmax_attention] {cid}: {cr.status} ({len(cr.points)} pts, {len(cr.cliffs)} cliffs, {cr.elapsed_s:.1f}s)")
    return results


def _run_gated_delta_net_family(
    *,
    root: Path,
    warmup: int,
    repeat: int,
    flush_mib: int,
    upward_pct: float,
    inverted_pct: float,
    min_delta_us: float,
    progress: Callable[[str], None] | None = None,
) -> list[CaseResult]:
    exe = _resolve_bench_exe(root, "infernix_gated_delta_net_bench")
    if exe is None:
        return [CaseResult(family="gated_delta_net", case_id="gdn:binary", status="skipped", reason="infernix_gated_delta_net_bench not found")]
    t0 = time.monotonic()
    cmd = [
        str(exe),
        "--running",
        "--sweep",
        "--warmup",
        str(warmup),
        "--repeat",
        str(repeat),
        "--flush-mib",
        str(flush_mib),
        "--csv",
    ]
    res = run(cmd, timeout=300.0, check=False, log=False)
    dt = round(time.monotonic() - t0, 3)
    if res.returncode != 0:
        return [
            CaseResult(
                family="gated_delta_net",
                case_id="running:sweep",
                status="error",
                source_file="src/ops/linear_attention",
                command=" ".join(cmd),
                reason=(res.stderr or res.stdout or f"exit {res.returncode}").strip(),
                elapsed_s=dt,
            )
        ]
    # Header starts with state_form,...
    csv_lines = [ln for ln in res.stdout.splitlines() if "," in ln]
    pts = parse_bench_csv_text(
        "\n".join(csv_lines),
        token_cols=("tokens", "T"),
        route_cols=("implementation", "state_form"),
        row_filter=lambda r: (r.get("implementation") or "").strip() == "public.total",
    )
    cliffs = detect_cliffs(
        pts,
        family="gated_delta_net",
        case_id="running:sweep",
        source_file="src/ops/linear_attention",
        upward_threshold_pct=upward_pct,
        inverted_threshold_pct=inverted_pct,
        min_delta_us=min_delta_us,
    )
    cr = CaseResult(
        family="gated_delta_net",
        case_id="running:sweep",
        status="ok" if pts else "error",
        source_file="src/ops/linear_attention",
        command=" ".join(cmd),
        reason=None if pts else "parsed 0 CSV rows from stdout",
        points=pts,
        cliffs=cliffs,
        elapsed_s=dt,
    )
    if progress:
        progress(f"  [gated_delta_net] running:sweep: {cr.status} ({len(pts)} pts, {len(cliffs)} cliffs, {dt:.1f}s)")
    return [cr]


def _run_sparse_moe_family(
    *,
    root: Path,
    max_tokens: int,
    warmup: int,
    repeat: int,
    flush_mib: int,
    upward_pct: float,
    inverted_pct: float,
    min_delta_us: float,
    progress: Callable[[str], None] | None = None,
) -> list[CaseResult]:
    exe = _resolve_bench_exe(root, "infernix_sparse_moe_bench")
    if exe is None:
        return [CaseResult(family="sparse_moe", case_id="sparse_moe:binary", status="skipped", reason="infernix_sparse_moe_bench not found")]
    t_list = _probe_tokens(DEFAULT_MOE_TOKENS, min(max_tokens, 65))
    results: list[CaseResult] = []
    for codec in ("q4-q5", "q4-q6", "q8-q8"):
        cid = f"trace-like:{codec}"
        t0 = time.monotonic()
        all_pts: list[dict[str, Any]] = []
        err_msg: str | None = None
        last_cmd = ""
        with tempfile.TemporaryDirectory() as tmpdir:
            for idx, tok in enumerate(t_list):
                csv_p = Path(tmpdir) / f"moe_{idx}.csv"
                cmd = [
                    str(exe),
                    "--codec",
                    codec,
                    "--tokens",
                    str(tok),
                    "--distribution",
                    "trace-like",
                    "--cache",
                    "cold",
                    "--warmup",
                    str(warmup),
                    "--repeat",
                    str(repeat),
                    "--flush-mib",
                    str(flush_mib),
                    "--csv-out",
                    str(csv_p),
                ]
                last_cmd = " ".join(cmd)
                res = run(cmd, timeout=300.0, check=False, log=False)
                if res.returncode != 0 or not csv_p.exists():
                    err_msg = (res.stderr or res.stdout or f"exit {res.returncode} at T={tok}").strip()
                    break
                all_pts.extend(
                    parse_bench_csv_text(
                        csv_p.read_text(errors="replace"),
                        token_cols=("tokens", "T"),
                        route_cols=("schedule", "timed_scope"),
                    )
                )
        dt = round(time.monotonic() - t0, 3)
        if err_msg:
            cr = CaseResult(
                family="sparse_moe",
                case_id=cid,
                status="error",
                source_file="src/ops/sparse_moe",
                command=last_cmd,
                reason=err_msg,
                elapsed_s=dt,
            )
        else:
            all_pts.sort(key=lambda p: int(p["T"]))
            cliffs = detect_cliffs(
                all_pts,
                family="sparse_moe",
                case_id=cid,
                source_file="src/ops/sparse_moe",
                upward_threshold_pct=upward_pct,
                inverted_threshold_pct=inverted_pct,
                min_delta_us=min_delta_us,
            )
            cr = CaseResult(
                family="sparse_moe",
                case_id=cid,
                status="ok",
                source_file="src/ops/sparse_moe",
                command=last_cmd,
                points=all_pts,
                cliffs=cliffs,
                elapsed_s=dt,
            )
        results.append(cr)
        if progress:
            progress(f"  [sparse_moe] {cid}: {cr.status} ({len(cr.points)} pts, {len(cr.cliffs)} cliffs, {dt:.1f}s)")
    return results


def _run_hyper_connection_family(
    *,
    root: Path,
    max_tokens: int,
    warmup: int,
    repeat: int,
    upward_pct: float,
    inverted_pct: float,
    min_delta_us: float,
    progress: Callable[[str], None] | None = None,
) -> list[CaseResult]:
    results: list[CaseResult] = []
    t_str = ",".join(str(t) for t in _probe_tokens(DEFAULT_PROBE_TOKENS, max_tokens))
    hc_exe = _resolve_bench_exe(root, "infernix_hyper_connection_bench")
    if hc_exe is None:
        results.append(CaseResult(family="hyper_connection", case_id="hyper_connection:binary", status="skipped", source_file="src/ops/hyper_connection", reason="infernix_hyper_connection_bench not found"))
    else:
        for codec in ("q8", "q4", "bf16"):
            cid = f"mix:{codec}"
            t0 = time.monotonic()
            cmd = [str(hc_exe), "--columns", t_str, "--codec", codec, "--warmup", str(warmup), "--repeat", str(repeat)]
            res = run(cmd, timeout=300.0, check=False, log=False)
            dt = round(time.monotonic() - t0, 3)
            if res.returncode != 0:
                cr = CaseResult(family="hyper_connection", case_id=cid, status="error", source_file="src/ops/hyper_connection", command=" ".join(cmd), reason=(res.stderr or res.stdout or f"exit {res.returncode}").strip(), elapsed_s=dt)
            else:
                pts = parse_hyper_connection_stdout(res.stdout, codec=codec)
                cliffs = detect_cliffs(pts, family="hyper_connection", case_id=cid, source_file="src/ops/hyper_connection", upward_threshold_pct=upward_pct, inverted_threshold_pct=inverted_pct, min_delta_us=min_delta_us)
                cr = CaseResult(family="hyper_connection", case_id=cid, status="ok" if pts else "error", source_file="src/ops/hyper_connection", command=" ".join(cmd), reason=None if pts else "parsed 0 rows from stdout", points=pts, cliffs=cliffs, elapsed_s=dt)
            results.append(cr)
            if progress:
                progress(f"  [hyper_connection] {cid}: {cr.status} ({len(cr.points)} pts, {len(cr.cliffs)} cliffs, {dt:.1f}s)")

    fused_exe = _resolve_bench_exe(root, "infernix_hc_mix_fused_bench")
    if fused_exe is None:
        results.append(CaseResult(family="hyper_connection", case_id="hc_mix_fused:binary", status="skipped", source_file="src/ops/hyper_connection", reason="infernix_hc_mix_fused_bench not found"))
    else:
        hc_tok_str = ",".join(str(t) for t in _probe_tokens(DEFAULT_PROBE_TOKENS, max_tokens))
        for codec in ("q8", "bf16"):
            t0 = time.monotonic()
            cmd = [str(fused_exe), "--codec", codec, "--tokens", hc_tok_str, "--warmup", str(warmup), "--repeat", str(repeat)]
            res = run(cmd, timeout=300.0, check=False, log=False)
            dt = round(time.monotonic() - t0, 3)
            if res.returncode != 0:
                results.append(CaseResult(family="hyper_connection", case_id=f"hc_mix_fused:{codec}", status="error", source_file="src/ops/hyper_connection", command=" ".join(cmd), reason=(res.stderr or res.stdout or f"exit {res.returncode}").strip(), elapsed_s=dt))
            else:
                raw_grouped = parse_hc_mix_fused_stdout(res.stdout)
                grouped = {f"hc_mix_fused:{codec}:{route}": pts for route, pts in raw_grouped.items() if route != "fused_warm"}
                results.extend(_finalize_grouped_stdout_cases(family="hyper_connection", source_file="src/ops/hyper_connection", cmd_str=" ".join(cmd), grouped=grouped, elapsed_s=dt, upward_pct=upward_pct, inverted_pct=inverted_pct, min_delta_us=min_delta_us, progress=progress))
    return results


def _run_qsa_attention_family(
    *,
    root: Path,
    max_tokens: int,
    warmup: int,
    repeat: int,
    upward_pct: float,
    inverted_pct: float,
    min_delta_us: float,
    progress: Callable[[str], None] | None = None,
) -> list[CaseResult]:
    exe = _resolve_bench_exe(root, "infernix_qsa_attention_bench")
    if exe is None:
        return [CaseResult(family="qsa_attention", case_id="qsa:binary", status="skipped", source_file="src/ops/qsa", reason="infernix_qsa_attention_bench not found")]
    w_str = ",".join(str(t) for t in _probe_tokens(DEFAULT_MOE_TOKENS, min(max_tokens, 65)))
    t0 = time.monotonic()
    cmd = [str(exe), "--storage", "bf16,int8,fp8,nvfp4,vq2", "--contexts", "8192", "--widths", w_str, "--warmup", str(warmup), "--repeat", str(repeat)]
    res = run(cmd, timeout=600.0, check=False, log=False)
    dt = round(time.monotonic() - t0, 3)
    if res.returncode != 0:
        return [CaseResult(family="qsa_attention", case_id="qsa:ctx8192", status="error", source_file="src/ops/qsa", command=" ".join(cmd), reason=(res.stderr or res.stdout or f"exit {res.returncode}").strip(), elapsed_s=dt)]
    raw_grouped = parse_qsa_attention_stdout(res.stdout)
    grouped = {f"{storage}:ctx{ctx}": pts for (storage, ctx), pts in raw_grouped.items()}
    return _finalize_grouped_stdout_cases(family="qsa_attention", source_file="src/ops/qsa", cmd_str=" ".join(cmd), grouped=grouped, elapsed_s=dt, upward_pct=upward_pct, inverted_pct=inverted_pct, min_delta_us=min_delta_us, progress=progress)


def _run_rmsnorm_rope_family(
    *,
    root: Path,
    max_tokens: int,
    warmup: int,
    repeat: int,
    upward_pct: float,
    inverted_pct: float,
    min_delta_us: float,
    progress: Callable[[str], None] | None = None,
) -> list[CaseResult]:
    results: list[CaseResult] = []
    t_str = ",".join(str(t) for t in _probe_tokens(DEFAULT_PROBE_TOKENS, max_tokens))
    rr_exe = _resolve_bench_exe(root, "infernix_rmsnorm_rope_bench")
    if rr_exe is None:
        results.append(CaseResult(family="rmsnorm_rope", case_id="rmsnorm_rope:binary", status="skipped", source_file="src/ops/rmsnorm_rope", reason="infernix_rmsnorm_rope_bench not found"))
    else:
        w_str = ",".join(str(t) for t in _probe_tokens((2, 3, 4, 5, 6, 8, 9, 12, 16), min(max_tokens, 16)))
        for label, extra in (
            ("text:fused", ["--form", "text", "--route", "fused", "--tokens", t_str]),
            ("pair:fused", ["--form", "pair", "--route", "fused", "--widths", w_str, "--batches", "1"]),
        ):
            t0 = time.monotonic()
            cmd = [str(rr_exe), *extra, "--warmup", str(warmup), "--repeat", str(repeat)]
            res = run(cmd, timeout=300.0, check=False, log=False)
            dt = round(time.monotonic() - t0, 3)
            if res.returncode != 0:
                results.append(CaseResult(family="rmsnorm_rope", case_id=label, status="error", source_file="src/ops/rmsnorm_rope", command=" ".join(cmd), reason=(res.stderr or res.stdout or f"exit {res.returncode}").strip(), elapsed_s=dt))
            else:
                grouped = parse_rmsnorm_rope_stdout(res.stdout)
                results.extend(_finalize_grouped_stdout_cases(family="rmsnorm_rope", source_file="src/ops/rmsnorm_rope", cmd_str=" ".join(cmd), grouped=grouped, elapsed_s=dt, upward_pct=upward_pct, inverted_pct=inverted_pct, min_delta_us=min_delta_us, progress=progress))

    rm_exe = _resolve_bench_exe(root, "infernix_rmsnorm_bench")
    if rm_exe is None:
        results.append(CaseResult(family="rmsnorm_rope", case_id="rmsnorm:binary", status="skipped", source_file="src/ops/rows", reason="infernix_rmsnorm_bench not found"))
    else:
        for kind in ("hidden27", "target_hidden35", "gated35"):
            cid = f"rmsnorm:{kind}"
            cr = _run_csv_command(
                family="rmsnorm_rope",
                case_id=cid,
                cmd_builder=lambda cp, e=rm_exe, k=kind: [str(e), "--kind", k, "--tokens", t_str, "--cache", "cold", "--execution", "graph", "--warmup", str(warmup), "--repeat", str(repeat), "--csv-out", str(cp)],
                source_file="src/ops/rows",
                route_cols=("kind", "execution"),
                upward_pct=upward_pct,
                inverted_pct=inverted_pct,
                min_delta_us=min_delta_us,
            )
            results.append(cr)
            if progress:
                progress(f"  [rmsnorm_rope] {cid}: {cr.status} ({len(cr.points)} pts, {len(cr.cliffs)} cliffs, {cr.elapsed_s:.1f}s)")
    return results


def _run_offloaded_sparse_moe_family(
    *,
    root: Path,
    max_tokens: int,
    warmup: int,
    repeat: int,
    upward_pct: float,
    inverted_pct: float,
    min_delta_us: float,
    progress: Callable[[str], None] | None = None,
) -> list[CaseResult]:
    results: list[CaseResult] = []
    t_str = ",".join(str(t) for t in _probe_tokens(DEFAULT_MOE_TOKENS, min(max_tokens, 65)))

    wide_exe = _resolve_bench_exe(root, "infernix_offloaded_moe_wide_bench")
    if wide_exe is None:
        results.append(CaseResult(family="offloaded_sparse_moe", case_id="offloaded_wide:binary", status="skipped", source_file="src/ops/offloaded_sparse_moe", reason="infernix_offloaded_moe_wide_bench not found"))
    else:
        t0 = time.monotonic()
        cmd = [str(wide_exe), "--tokens", t_str, "--warmup", str(warmup), "--repeat", str(repeat)]
        res = run(cmd, timeout=600.0, check=False, log=False)
        dt = round(time.monotonic() - t0, 3)
        if res.returncode != 0:
            results.append(CaseResult(family="offloaded_sparse_moe", case_id="offloaded_wide", status="error", source_file="src/ops/offloaded_sparse_moe", command=" ".join(cmd), reason=(res.stderr or res.stdout or f"exit {res.returncode}").strip(), elapsed_s=dt))
        else:
            raw_grouped = parse_offloaded_moe_wide_stdout(res.stdout)
            grouped = {f"wide:{placement}": pts for placement, pts in raw_grouped.items()}
            results.extend(_finalize_grouped_stdout_cases(family="offloaded_sparse_moe", source_file="src/ops/offloaded_sparse_moe", cmd_str=" ".join(cmd), grouped=grouped, elapsed_s=dt, upward_pct=upward_pct, inverted_pct=inverted_pct, min_delta_us=min_delta_us, progress=progress))

    res_exe = _resolve_bench_exe(root, "infernix_resident_moe_bench")
    if res_exe is None:
        results.append(CaseResult(family="offloaded_sparse_moe", case_id="resident_moe:binary", status="skipped", source_file="src/ops/resident_moe", reason="infernix_resident_moe_bench not found"))
    else:
        for codec in ("q4", "q8"):
            cid = f"resident:{codec}"
            t0 = time.monotonic()
            cmd = [str(res_exe), "--columns", t_str, "--codec", codec, "--warmup", str(warmup), "--repeat", str(repeat)]
            res = run(cmd, timeout=300.0, check=False, log=False)
            dt = round(time.monotonic() - t0, 3)
            if res.returncode != 0:
                cr = CaseResult(family="offloaded_sparse_moe", case_id=cid, status="error", source_file="src/ops/resident_moe", command=" ".join(cmd), reason=(res.stderr or res.stdout or f"exit {res.returncode}").strip(), elapsed_s=dt)
            else:
                pts = parse_resident_moe_stdout(res.stdout)
                cliffs = detect_cliffs(pts, family="offloaded_sparse_moe", case_id=cid, source_file="src/ops/resident_moe", upward_threshold_pct=upward_pct, inverted_threshold_pct=inverted_pct, min_delta_us=min_delta_us)
                cr = CaseResult(family="offloaded_sparse_moe", case_id=cid, status="ok" if pts else "error", source_file="src/ops/resident_moe", command=" ".join(cmd), reason=None if pts else "parsed 0 rows from stdout", points=pts, cliffs=cliffs, elapsed_s=dt)
            results.append(cr)
            if progress:
                progress(f"  [offloaded_sparse_moe] {cid}: {cr.status} ({len(cr.points)} pts, {len(cr.cliffs)} cliffs, {dt:.1f}s)")
    return results


def _run_candidate_selector_family(
    *,
    root: Path,
    warmup: int,
    repeat: int,
    flush_mib: int,
    upward_pct: float,
    inverted_pct: float,
    min_delta_us: float,
    progress: Callable[[str], None] | None = None,
) -> list[CaseResult]:
    exe = _resolve_bench_exe(root, "infernix_candidate_selector_bench")
    if exe is None:
        return [CaseResult(family="candidate_selector", case_id="candidate_selector:binary", status="skipped", source_file="src/ops/candidate_selector", reason="infernix_candidate_selector_bench not found")]
    steps = (1, 2, 3, 4, 5, 6, 8, 9, 12, 15)
    t0 = time.monotonic()
    by_mode: dict[str, list[dict[str, Any]]] = {}
    last_cmd = ""
    for k_step in steps:
        cmd = [str(exe), "--steps", str(k_step), "--batch", "1", "--warmup", str(warmup), "--repeat", str(repeat), "--flush-mib", str(flush_mib)]
        last_cmd = " ".join(cmd)
        res = run(cmd, timeout=120.0, check=False, log=False)
        if res.returncode != 0:
            return [CaseResult(family="candidate_selector", case_id="candidate_selector", status="error", source_file="src/ops/candidate_selector", command=last_cmd, reason=(res.stderr or res.stdout or f"exit {res.returncode}").strip(), elapsed_s=round(time.monotonic() - t0, 3))]
        reader = csv.DictReader(ln for ln in res.stdout.splitlines() if ln.strip() and not ln.lstrip().startswith("#"))
        for row in reader:
            mode = (row.get("mode") or "unknown").strip()
            by_mode.setdefault(f"B1:{mode}", []).append(
                {
                    "T": int(row["K"]),
                    "median_us": round(float(row["median_us"]), 3),
                    "min_us": round(float(row["min_us"]), 3),
                    "p95_us": round(float(row["p95_us"]), 3),
                    "active_route": (row.get("route") or "").strip(),
                }
            )
    for k in by_mode:
        by_mode[k].sort(key=lambda p: int(p["T"]))
    dt = round(time.monotonic() - t0, 3)
    return _finalize_grouped_stdout_cases(family="candidate_selector", source_file="src/ops/candidate_selector", cmd_str=last_cmd, grouped=by_mode, elapsed_s=dt, upward_pct=upward_pct, inverted_pct=inverted_pct, min_delta_us=min_delta_us, progress=progress)


def _run_dynamic_grouped_conv_family(
    *,
    root: Path,
    max_tokens: int,
    warmup: int,
    repeat: int,
    flush_mib: int,
    upward_pct: float,
    inverted_pct: float,
    min_delta_us: float,
    progress: Callable[[str], None] | None = None,
) -> list[CaseResult]:
    results: list[CaseResult] = []
    prep_exe = _resolve_bench_exe(root, "infernix_dynamic_grouped_conv_prepare_bench")
    if prep_exe is None:
        results.append(CaseResult(family="dynamic_grouped_conv", case_id="prepare:binary", status="skipped", source_file="src/ops/dynamic_grouped_conv", reason="infernix_dynamic_grouped_conv_prepare_bench not found"))
    else:
        widths = _probe_tokens((2, 3, 4, 5, 6, 8, 9, 12, 16), min(max_tokens, 16))
        t0 = time.monotonic()
        pts: list[dict[str, Any]] = []
        last_cmd = ""
        err_msg: str | None = None
        for w in widths:
            cmd = [str(prep_exe), "--width", str(w), "--batch", "1", "--warmup", str(warmup), "--repeat", str(repeat), "--flush-mib", str(flush_mib)]
            last_cmd = " ".join(cmd)
            res = run(cmd, timeout=120.0, check=False, log=False)
            if res.returncode != 0:
                err_msg = (res.stderr or res.stdout or f"exit {res.returncode} at W={w}").strip()
                break
            pts.extend(parse_bench_csv_text(res.stdout, token_cols=("W", "T")))
        dt = round(time.monotonic() - t0, 3)
        if err_msg:
            cr = CaseResult(family="dynamic_grouped_conv", case_id="prepare:B1", status="error", source_file="src/ops/dynamic_grouped_conv", command=last_cmd, reason=err_msg, elapsed_s=dt)
        else:
            pts.sort(key=lambda p: int(p["T"]))
            cliffs = detect_cliffs(pts, family="dynamic_grouped_conv", case_id="prepare:B1", source_file="src/ops/dynamic_grouped_conv", upward_threshold_pct=upward_pct, inverted_threshold_pct=inverted_pct, min_delta_us=min_delta_us)
            cr = CaseResult(family="dynamic_grouped_conv", case_id="prepare:B1", status="ok" if pts else "error", source_file="src/ops/dynamic_grouped_conv", command=last_cmd, points=pts, cliffs=cliffs, elapsed_s=dt)
        results.append(cr)
        if progress:
            progress(f"  [dynamic_grouped_conv] prepare:B1: {cr.status} ({len(cr.points)} pts, {len(cr.cliffs)} cliffs, {dt:.1f}s)")

    conv_exe = _resolve_bench_exe(root, "infernix_causal_conv1d_silu_bench")
    if conv_exe is None:
        results.append(CaseResult(family="dynamic_grouped_conv", case_id="causal_conv1d:binary", status="skipped", source_file="src/ops/dynamic_grouped_conv", reason="infernix_causal_conv1d_silu_bench not found"))
    else:
        t_str = ",".join(str(t) for t in _probe_tokens(DEFAULT_PROBE_TOKENS, max_tokens))
        t0 = time.monotonic()
        cmd = [str(conv_exe), "--prefill", "--cache", "cold", "--tokens", t_str]
        res = run(cmd, timeout=300.0, check=False, log=False)
        dt = round(time.monotonic() - t0, 3)
        if res.returncode != 0:
            results.append(CaseResult(family="dynamic_grouped_conv", case_id="causal_conv1d:prefill", status="error", source_file="src/ops/dynamic_grouped_conv", command=" ".join(cmd), reason=(res.stderr or res.stdout or f"exit {res.returncode}").strip(), elapsed_s=dt))
        else:
            grouped = parse_causal_conv1d_stdout(res.stdout)
            results.extend(_finalize_grouped_stdout_cases(family="dynamic_grouped_conv", source_file="src/ops/dynamic_grouped_conv", cmd_str=" ".join(cmd), grouped=grouped, elapsed_s=dt, upward_pct=upward_pct, inverted_pct=inverted_pct, min_delta_us=min_delta_us, progress=progress))
    return results


def _summarize_family(family: str, cases: Sequence[CaseResult]) -> FamilySummary:
    summary = FamilySummary(family=family, cases_total=len(cases))
    up_vals: list[float] = []
    inv_vals: list[float] = []
    for c in cases:
        summary.elapsed_s = round(summary.elapsed_s + c.elapsed_s, 3)
        if c.status == "ok":
            summary.cases_ok += 1
        elif c.status == "skipped":
            summary.cases_skipped += 1
        else:
            summary.cases_error += 1
        summary.points_measured += len(c.points)
        summary.cliffs_total += len(c.cliffs)
        for ev in c.cliffs:
            if ev.kind == "upward_cliff":
                summary.upward_cliffs += 1
                up_vals.append(ev.step_delta_pct)
            else:
                summary.inverted_cliffs += 1
                inv_vals.append(ev.step_delta_pct)
            if ev.severity == "critical":
                summary.critical_cliffs += 1
            elif ev.severity == "high":
                summary.high_cliffs += 1
            else:
                summary.medium_cliffs += 1
    summary.worst_upward_pct = max(up_vals) if up_vals else None
    summary.worst_inverted_pct = min(inv_vals) if inv_vals else None
    return summary


def run_cliffs_analysis(
    *,
    dev: DeviceInfo,
    repo_root: str | Path = ".",
    families: Sequence[str] | None = None,
    max_tokens: int = 512,
    warmup: int = 2,
    repeat: int = 8,
    flush_mib: int | None = None,
    upward_threshold_pct: float = 15.0,
    inverted_threshold_pct: float = -8.0,
    min_delta_us: float = 0.5,
    progress: Callable[[str], None] | None = None,
) -> CliffsReport:
    """Execute boundary cliff analysis across the requested Infernix kernel families."""
    root = Path(repo_root).resolve()
    if not (root / "src" / "ops").exists() and (root.parent / "src" / "ops").exists():
        root = root.parent
    eff_flush = flush_mib if flush_mib is not None else optimal_flush_mib(dev, fast=True)
    fams = list(families) if families else list(ALL_FAMILIES)
    t_all = time.monotonic()

    all_cases: list[CaseResult] = []
    summaries: list[FamilySummary] = []

    for fam in fams:
        if progress:
            progress(f"==> Scanning family: {fam}")
        if fam.startswith("linear:"):
            qtype = fam.split(":", 1)[1]
            fam_cases = _run_linear_family(
                dev=dev,
                root=root,
                qtype=qtype,
                max_tokens=max_tokens,
                warmup=warmup,
                repeat=repeat,
                flush_mib=eff_flush,
                upward_pct=upward_threshold_pct,
                inverted_pct=inverted_threshold_pct,
                min_delta_us=min_delta_us,
                progress=progress,
            )
        elif fam == "linear_add":
            fam_cases = _run_linear_add_family(
                root=root,
                max_tokens=max_tokens,
                warmup=warmup,
                repeat=repeat,
                flush_mib=eff_flush,
                upward_pct=upward_threshold_pct,
                inverted_pct=inverted_threshold_pct,
                min_delta_us=min_delta_us,
                progress=progress,
            )
        elif fam == "linear_swiglu":
            fam_cases = _run_linear_swiglu_family(
                root=root,
                max_tokens=max_tokens,
                warmup=warmup,
                repeat=repeat,
                upward_pct=upward_threshold_pct,
                inverted_pct=inverted_threshold_pct,
                min_delta_us=min_delta_us,
                progress=progress,
            )
        elif fam == "attn_input_proj":
            fam_cases = _run_input_proj_family(
                family="attn_input_proj",
                bin_name="infernix_attn_input_proj_bench",
                cases=[
                    ("q4q5", ["--format", "q4q5", "--q4q5-policy", "a16"], "src/ops/attn_input_proj/q4_q5"),
                    ("q8-qgkv", ["--format", "q8-qgkv"], "src/ops/attn_input_proj/q8"),
                    ("q8-qkv", ["--format", "q8-qkv"], "src/ops/attn_input_proj/q8"),
                    ("bf16", ["--format", "bf16"], "src/ops/attn_input_proj/bf16"),
                    ("nvfp4:a16", ["--format", "nvfp4", "--nvfp4-policy", "a16"], "src/ops/attn_input_proj/nvfp4"),
                    ("nvfp4:a4", ["--format", "nvfp4", "--nvfp4-policy", "a4"], "src/ops/attn_input_proj/nvfp4"),
                    ("fp8:a16", ["--format", "fp8", "--fp8-policy", "a16"], "src/ops/attn_input_proj/fp8"),
                    ("fp8:a8", ["--format", "fp8", "--fp8-policy", "a8"], "src/ops/attn_input_proj/fp8"),
                ],
                root=root,
                max_tokens=max_tokens,
                warmup=warmup,
                repeat=repeat,
                upward_pct=upward_threshold_pct,
                inverted_pct=inverted_threshold_pct,
                min_delta_us=min_delta_us,
                progress=progress,
            )
        elif fam == "gdn_input_proj":
            fam_cases = _run_input_proj_family(
                family="gdn_input_proj",
                bin_name="infernix_gdn_input_proj_bench",
                cases=[
                    ("q4q5", ["--format", "q4q5"], "src/ops/gdn_input_proj/q4_q5"),
                    ("q8", ["--format", "q8"], "src/ops/gdn_input_proj/q8"),
                    ("nvfp4:a16", ["--format", "nvfp4", "--nvfp4-policy", "a16"], "src/ops/gdn_input_proj/nvfp4"),
                    ("nvfp4:a4", ["--format", "nvfp4", "--nvfp4-policy", "a4"], "src/ops/gdn_input_proj/nvfp4"),
                    ("fp8:a16", ["--format", "fp8", "--fp8-policy", "a16"], "src/ops/gdn_input_proj/fp8"),
                    ("fp8:a8", ["--format", "fp8", "--fp8-policy", "a8"], "src/ops/gdn_input_proj/fp8"),
                ],
                root=root,
                max_tokens=max_tokens,
                warmup=warmup,
                repeat=repeat,
                upward_pct=upward_threshold_pct,
                inverted_pct=inverted_threshold_pct,
                min_delta_us=min_delta_us,
                progress=progress,
            )
        elif fam == "linear_pair":
            fam_cases = _run_linear_pair_family(
                root=root,
                max_tokens=max_tokens,
                warmup=warmup,
                repeat=repeat,
                upward_pct=upward_threshold_pct,
                inverted_pct=inverted_threshold_pct,
                min_delta_us=min_delta_us,
                progress=progress,
            )
        elif fam == "ggml_linear":
            fam_cases = _run_ggml_linear_family(
                root=root,
                max_tokens=max_tokens,
                warmup=warmup,
                repeat=repeat,
                upward_pct=upward_threshold_pct,
                inverted_pct=inverted_threshold_pct,
                min_delta_us=min_delta_us,
                progress=progress,
            )
        elif fam == "causal_softmax_attention":
            fam_cases = _run_causal_softmax_attention_family(
                root=root,
                max_tokens=max_tokens,
                warmup=warmup,
                repeat=repeat,
                upward_pct=upward_threshold_pct,
                inverted_pct=inverted_threshold_pct,
                min_delta_us=min_delta_us,
                progress=progress,
            )
        elif fam == "gated_delta_net":
            fam_cases = _run_gated_delta_net_family(
                root=root,
                warmup=warmup,
                repeat=repeat,
                flush_mib=eff_flush,
                upward_pct=upward_threshold_pct,
                inverted_pct=inverted_threshold_pct,
                min_delta_us=min_delta_us,
                progress=progress,
            )
        elif fam == "sparse_moe":
            fam_cases = _run_sparse_moe_family(
                root=root,
                max_tokens=max_tokens,
                warmup=warmup,
                repeat=repeat,
                flush_mib=eff_flush,
                upward_pct=upward_threshold_pct,
                inverted_pct=inverted_threshold_pct,
                min_delta_us=min_delta_us,
                progress=progress,
            )
        elif fam == "hyper_connection":
            fam_cases = _run_hyper_connection_family(
                root=root,
                max_tokens=max_tokens,
                warmup=warmup,
                repeat=repeat,
                upward_pct=upward_threshold_pct,
                inverted_pct=inverted_threshold_pct,
                min_delta_us=min_delta_us,
                progress=progress,
            )
        elif fam == "qsa_attention":
            fam_cases = _run_qsa_attention_family(
                root=root,
                max_tokens=max_tokens,
                warmup=warmup,
                repeat=repeat,
                upward_pct=upward_threshold_pct,
                inverted_pct=inverted_threshold_pct,
                min_delta_us=min_delta_us,
                progress=progress,
            )
        elif fam == "rmsnorm_rope":
            fam_cases = _run_rmsnorm_rope_family(
                root=root,
                max_tokens=max_tokens,
                warmup=warmup,
                repeat=repeat,
                upward_pct=upward_threshold_pct,
                inverted_pct=inverted_threshold_pct,
                min_delta_us=min_delta_us,
                progress=progress,
            )
        elif fam == "offloaded_sparse_moe":
            fam_cases = _run_offloaded_sparse_moe_family(
                root=root,
                max_tokens=max_tokens,
                warmup=warmup,
                repeat=repeat,
                upward_pct=upward_threshold_pct,
                inverted_pct=inverted_threshold_pct,
                min_delta_us=min_delta_us,
                progress=progress,
            )
        elif fam == "candidate_selector":
            fam_cases = _run_candidate_selector_family(
                root=root,
                warmup=warmup,
                repeat=repeat,
                flush_mib=eff_flush,
                upward_pct=upward_threshold_pct,
                inverted_pct=inverted_threshold_pct,
                min_delta_us=min_delta_us,
                progress=progress,
            )
        elif fam == "dynamic_grouped_conv":
            fam_cases = _run_dynamic_grouped_conv_family(
                root=root,
                max_tokens=max_tokens,
                warmup=warmup,
                repeat=repeat,
                flush_mib=eff_flush,
                upward_pct=upward_threshold_pct,
                inverted_pct=inverted_threshold_pct,
                min_delta_us=min_delta_us,
                progress=progress,
            )
        else:
            fam_cases = [CaseResult(family=fam, case_id=fam, status="error", reason=f"unhandled family {fam}")]

        all_cases.extend(fam_cases)
        summaries.append(_summarize_family(fam, fam_cases))

    all_cliffs = [ev for c in all_cases for ev in c.cliffs]
    all_cliffs.sort(
        key=lambda e: (
            _SEVERITY_RANK.get(e.severity, 9),
            -abs(e.step_delta_pct),
            -e.wasted_us,
        )
    )
    l2_mib = (dev.l2_cache_size or (24 * 1024 * 1024)) // (1024 * 1024)
    return CliffsReport(
        device_name=dev.name,
        cuda_arch=dev.compute_capability,
        l2_mib=l2_mib,
        flush_mib=eff_flush,
        max_tokens=max_tokens,
        warmup=warmup,
        repeat=repeat,
        upward_threshold_pct=upward_threshold_pct,
        inverted_threshold_pct=inverted_threshold_pct,
        min_delta_us=min_delta_us,
        families_requested=fams,
        family_summaries=summaries,
        cases=all_cases,
        top_cliffs=all_cliffs,
        total_elapsed_s=round(time.monotonic() - t_all, 2),
    )


def _cliff_from_dict(d: dict[str, Any]) -> CliffEvent:
    return CliffEvent(**d)


def _case_from_dict(d: dict[str, Any]) -> CaseResult:
    cliffs = [_cliff_from_dict(ev) if isinstance(ev, dict) else ev for ev in (d.get("cliffs") or [])]
    return CaseResult(
        family=d["family"],
        case_id=d["case_id"],
        status=d["status"],
        source_file=d.get("source_file"),
        command=d.get("command"),
        reason=d.get("reason"),
        points=list(d.get("points") or []),
        cliffs=cliffs,
        elapsed_s=float(d.get("elapsed_s") or 0.0),
    )


def merge_reports(base_report_dict: dict[str, Any], new_report: CliffsReport) -> CliffsReport:
    """Merge ``new_report`` families into an existing serialized ``CliffsReport`` dict."""
    replaced_families = {s.family for s in new_report.family_summaries}
    kept_cases = [
        _case_from_dict(c)
        for c in (base_report_dict.get("cases") or [])
        if c.get("family") not in replaced_families
    ]
    merged_cases = kept_cases + list(new_report.cases)

    by_family: dict[str, list[CaseResult]] = {}
    for c in merged_cases:
        by_family.setdefault(c.family, []).append(c)

    ordered_families = [f for f in ALL_FAMILIES if f in by_family]
    for f in by_family:
        if f not in ordered_families:
            ordered_families.append(f)

    ordered_cases: list[CaseResult] = []
    summaries: list[FamilySummary] = []
    for fam in ordered_families:
        fc = by_family[fam]
        ordered_cases.extend(fc)
        summaries.append(_summarize_family(fam, fc))

    all_cliffs = [ev for c in ordered_cases for ev in c.cliffs]
    all_cliffs.sort(
        key=lambda e: (
            _SEVERITY_RANK.get(e.severity, 9),
            -abs(e.step_delta_pct),
            -e.wasted_us,
        )
    )
    total_elapsed = round(sum(s.elapsed_s for s in summaries), 2)
    return CliffsReport(
        device_name=new_report.device_name or str(base_report_dict.get("device_name", "")),
        cuda_arch=new_report.cuda_arch or str(base_report_dict.get("cuda_arch", "")),
        l2_mib=new_report.l2_mib or int(base_report_dict.get("l2_mib", 24)),
        flush_mib=new_report.flush_mib or int(base_report_dict.get("flush_mib", 32)),
        max_tokens=new_report.max_tokens,
        warmup=new_report.warmup,
        repeat=new_report.repeat,
        upward_threshold_pct=new_report.upward_threshold_pct,
        inverted_threshold_pct=new_report.inverted_threshold_pct,
        min_delta_us=new_report.min_delta_us,
        families_requested=ordered_families,
        family_summaries=summaries,
        cases=ordered_cases,
        top_cliffs=all_cliffs,
        total_elapsed_s=total_elapsed,
    )


def render_markdown_report(report: CliffsReport, *, top_n: int = 40) -> str:
    """Render a unified Markdown cliff report across all benchmarked kernel families."""
    lines: list[str] = []
    total_cases = sum(s.cases_total for s in report.family_summaries)
    ok_cases = sum(s.cases_ok for s in report.family_summaries)
    skipped_cases = sum(s.cases_skipped for s in report.family_summaries)
    error_cases = sum(s.cases_error for s in report.family_summaries)
    total_pts = sum(s.points_measured for s in report.family_summaries)
    total_cliffs = sum(s.cliffs_total for s in report.family_summaries)
    total_up = sum(s.upward_cliffs for s in report.family_summaries)
    total_inv = sum(s.inverted_cliffs for s in report.family_summaries)

    lines.append("# Infernix Cross-Family Kernel Cliff Report")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append(
        f"- **Device**: `{report.device_name}` (`sm_{report.cuda_arch.replace('.', '')}`, L2={report.l2_mib} MiB, flush={report.flush_mib} MiB)"
    )
    lines.append(
        f"- **Protocol**: `max_tokens={report.max_tokens}`, `warmup={report.warmup}`, `repeat={report.repeat}`, "
        f"`upward >= +{report.upward_threshold_pct:.1f}%`, `inverted <= {report.inverted_threshold_pct:.1f}%`, `min_delta >= {report.min_delta_us:.2f} us`"
    )
    lines.append(
        f"- **Coverage**: `{len(report.family_summaries)}` families, `{total_cases}` cases "
        f"(`{ok_cases}` ok, `{skipped_cases}` skipped, `{error_cases}` error), `{total_pts}` measured `(T, latency)` points in `{report.total_elapsed_s:.1f}s`"
    )
    lines.append(
        f"- **Cliffs Found**: **{total_cliffs}** total (**{total_up}** upward cliffs, **{total_inv}** inverted cliffs / dominated bands)"
    )
    lines.append("")

    lines.append("## Cross-Family Summary")
    lines.append("")
    lines.append(
        "| Family | Cases (ok/skip/err) | Points | Cliffs | Upward | Inverted | Critical/High/Med | Worst Upward | Worst Inverted | Time (s) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in report.family_summaries:
        wu = f"+{s.worst_upward_pct:.1f}%" if s.worst_upward_pct is not None else "-"
        wi = f"{s.worst_inverted_pct:.1f}%" if s.worst_inverted_pct is not None else "-"
        lines.append(
            f"| `{s.family}` | {s.cases_ok}/{s.cases_skipped}/{s.cases_error} | {s.points_measured} | "
            f"{s.cliffs_total} | {s.upward_cliffs} | {s.inverted_cliffs} | "
            f"{s.critical_cliffs}/{s.high_cliffs}/{s.medium_cliffs} | {wu} | {wi} | {s.elapsed_s:.1f} |"
        )
    lines.append("")

    lines.append(f"## Top Ranked Cliffs Across All Families (top {min(top_n, len(report.top_cliffs))} of {len(report.top_cliffs)})")
    lines.append("")
    if not report.top_cliffs:
        lines.append("_No cliffs met the threshold across the benchmarked cases._")
        lines.append("")
    else:
        lines.append(
            "| Sev | Kind | Family | Case | T -> T' | us -> us' | Step Δ% | Route Transition |"
        )
        lines.append("|---|---|---|---|---|---:|---:|---|")
        for ev in report.top_cliffs[:top_n]:
            rt = (
                f"`{ev.from_route}` -> `{ev.to_route}`"
                if ev.route_changed
                else f"`{ev.to_route or ev.from_route or '-'}`"
            )
            lines.append(
                f"| `{ev.severity}` | `{ev.kind}` | `{ev.family}` | `{ev.case_id}` | "
                f"`{ev.from_T}->{ev.to_T}` | `{ev.from_us:.2f} -> {ev.to_us:.2f}` | "
                f"**{ev.step_delta_pct:+.1f}%** | {rt} |"
            )
        lines.append("")

    lines.append("## Per-Family Cliff Details")
    lines.append("")
    for s in report.family_summaries:
        fam_cases = [c for c in report.cases if c.family == s.family]
        fam_cliffs = [ev for c in fam_cases for ev in c.cliffs]
        fam_cliffs.sort(key=lambda e: (_SEVERITY_RANK.get(e.severity, 9), -abs(e.step_delta_pct)))
        lines.append(f"### `{s.family}` ({s.cliffs_total} cliffs across {s.cases_ok} ok / {s.cases_skipped} skipped / {s.cases_error} error)")
        lines.append("")
        if fam_cliffs:
            lines.append("| Sev | Kind | Case | T -> T' | us -> us' | Step Δ% | Route / Line |")
            lines.append("|---|---|---|---|---:|---:|---|")
            for ev in fam_cliffs:
                line_suffix = f" (L{ev.shape_line})" if ev.shape_line else ""
                rt = (
                    f"`{ev.from_route}` -> `{ev.to_route}`{line_suffix}"
                    if ev.route_changed
                    else f"`{ev.to_route or ev.from_route or '-'}`{line_suffix}"
                )
                lines.append(
                    f"| `{ev.severity}` | `{ev.kind}` | `{ev.case_id}` | `{ev.from_T}->{ev.to_T}` | "
                    f"`{ev.from_us:.2f} -> {ev.to_us:.2f}` | **{ev.step_delta_pct:+.1f}%** | {rt} |"
                )
            lines.append("")
        else:
            lines.append("_No cliffs detected in completed cases._")
            lines.append("")

        non_ok = [c for c in fam_cases if c.status != "ok"]
        if non_ok:
            lines.append("Skipped / Error cases:")
            for c in non_ok:
                lines.append(f"- `{c.case_id}` (`{c.status}`): {c.reason or 'unknown'}")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def format_cli_summary(report: CliffsReport, *, stream: Any = None) -> None:
    """Print a concise console table of cross-family cliff counts and top cliffs."""
    out = stream or sys.stdout
    print(
        f"=== Cross-Family Cliff Analysis ({report.device_name}, flush={report.flush_mib} MiB, "
        f"elapsed={report.total_elapsed_s:.1f}s) ===",
        file=out,
    )
    print(
        f"  {'family':<26}{'cases':>10}{'pts':>7}{'cliffs':>8}{'up':>6}{'inv':>6}{'worst_up':>10}{'worst_inv':>11}",
        file=out,
    )
    for s in report.family_summaries:
        cases_s = f"{s.cases_ok}/{s.cases_total}"
        wu = f"+{s.worst_upward_pct:.1f}%" if s.worst_upward_pct is not None else "-"
        wi = f"{s.worst_inverted_pct:.1f}%" if s.worst_inverted_pct is not None else "-"
        print(
            f"  {s.family:<26}{cases_s:>10}{s.points_measured:>7}{s.cliffs_total:>8}"
            f"{s.upward_cliffs:>6}{s.inverted_cliffs:>6}{wu:>10}{wi:>11}",
            file=out,
        )
    if report.top_cliffs:
        print("\nTop 15 cliffs:", file=out)
        for ev in report.top_cliffs[:15]:
            print(
                f"  [{ev.severity:<8}] {ev.family:<24} {ev.case_id:<24} "
                f"T={ev.from_T}->{ev.to_T:<4} {ev.from_us:>7.2f}->{ev.to_us:<7.2f} us ({ev.step_delta_pct:+.1f}%)",
                file=out,
            )
