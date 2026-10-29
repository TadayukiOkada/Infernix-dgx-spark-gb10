"""reporting — human- and machine-readable output that never blurs provenance.

The single design rule here is that **every number carries a label**:

* ``measured``  — produced by running something on the target (timings, NCU
                  counters, probe results);
* ``compiler``  — reported by the toolchain about the code (registers, spills,
                  static shared-memory size);
* ``modelled``  — arithmetic on the above using documented formulas (occupancy,
                  roofline bounds, wave counts);
* ``static``    — read from the source or the SASS without execution;
* ``unavailable`` — could not be determined, and is printed as such.

A report never prints a modelled figure in a column of measured figures.  The
Markdown form puts the provenance in the column header or in an explicit label
column so a reader cannot mistake the two.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from .util import ensure_dir, write_json, write_text


PROVENANCE_LABELS = {
    "measured": "measured on this device",
    "compiler": "reported by the compiler",
    "modelled": "computed from measured/compiler inputs",
    "static": "read from source or SASS",
    "unavailable": "not determined",
}


def provenance_tag(kind: str) -> str:
    return PROVENANCE_LABELS.get(kind, kind)


# --------------------------------------------------------------------------- #
# Report assembly
# --------------------------------------------------------------------------- #


@dataclass
class ReportSection:
    title: str
    body: list[str] = field(default_factory=list)
    table_headers: list[str] = field(default_factory=list)
    table_rows: list[list[Any]] = field(default_factory=list)
    provenance: str = "measured"
    notes: list[str] = field(default_factory=list)


@dataclass
class Report:
    title: str = "CUDA Kernel Analysis Report"
    generated_at: str = ""
    environment: dict[str, Any] = field(default_factory=dict)
    sections: list[ReportSection] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)

    def add(self, section: ReportSection) -> None:
        self.sections.append(section)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "generated_at": self.generated_at,
            "environment": self.environment,
            "warnings": self.warnings,
            "sections": [
                {
                    "title": s.title,
                    "body": s.body,
                    "table_headers": s.table_headers,
                    "table_rows": s.table_rows,
                    "provenance": s.provenance,
                    "provenance_label": provenance_tag(s.provenance),
                    "notes": s.notes,
                }
                for s in self.sections
            ],
            "data": self.data,
        }

    # ------------------------------------------------------------------ #
    def to_markdown(self) -> str:
        lines: list[str] = []
        lines.append(f"# {self.title}")
        lines.append("")
        if self.generated_at:
            lines.append(f"*Generated {self.generated_at}*")
            lines.append("")
        if self.environment:
            lines.append("## Environment")
            lines.append("")
            lines.append("| Field | Value |")
            lines.append("| --- | --- |")
            for k, v in self.environment.items():
                lines.append(f"| {k} | {_cell(v)} |")
            lines.append("")
        for s in self.sections:
            lines.append(f"## {s.title}")
            lines.append("")
            if s.provenance != "measured":
                lines.append(f"> Value provenance: **{provenance_tag(s.provenance)}**")
                lines.append("")
            for b in s.body:
                lines.append(b)
                # A run of contiguous table rows must stay together: a blank
                # line inside a Markdown table terminates it, splitting the
                # header from its body.  Non-table body text keeps its blank
                # separator.
                if not str(b).lstrip().startswith("|"):
                    lines.append("")
            if s.table_headers:
                lines.append("| " + " | ".join(str(h) for h in s.table_headers) + " |")
                lines.append("| " + " | ".join("---" for _ in s.table_headers) + " |")
                for row in s.table_rows:
                    lines.append("| " + " | ".join(_cell(c) for c in row) + " |")
                lines.append("")
            for n in s.notes:
                lines.append(f"- {n}")
            if s.notes:
                lines.append("")
        if self.warnings:
            lines.append("## Warnings and limitations")
            lines.append("")
            for w in self.warnings:
                lines.append(f"- {w}")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def to_html(self) -> str:
        """Minimal standalone HTML rendering (no templates, no dependencies)."""
        def esc(t: str) -> str:
            return (
                str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            )

        parts = [
            "<!doctype html><html><head><meta charset='utf-8'>",
            f"<title>{esc(self.title)}</title>",
            "<style>body{font-family:system-ui,sans-serif;max-width:70rem;margin:2rem auto;padding:0 1rem;line-height:1.45}"
            "table{border-collapse:collapse;margin:1rem 0;font-size:.92em}"
            "th,td{border:1px solid #ccc;padding:.35rem .6rem;text-align:left}"
            "th{background:#f2f2f2}code{background:#f6f6f6;padding:.1rem .25rem}"
            ".prov{color:#7a4b00;background:#fff5e0;padding:.35rem .6rem;border-left:3px solid #e0a800;margin:.6rem 0}"
            "</style></head><body>",
            f"<h1>{esc(self.title)}</h1>",
        ]
        if self.generated_at:
            parts.append(f"<p><em>Generated {esc(self.generated_at)}</em></p>")
        if self.environment:
            parts.append("<h2>Environment</h2><table><tr><th>Field</th><th>Value</th></tr>")
            for k, v in self.environment.items():
                parts.append(f"<tr><td>{esc(k)}</td><td>{esc(_cell(v))}</td></tr>")
            parts.append("</table>")
        for s in self.sections:
            parts.append(f"<h2>{esc(s.title)}</h2>")
            if s.provenance != "measured":
                parts.append(
                    f"<div class='prov'>Value provenance: <strong>{esc(provenance_tag(s.provenance))}</strong></div>"
                )
            for b in s.body:
                parts.append(f"<p>{esc(b)}</p>")
            if s.table_headers:
                parts.append("<table><tr>" + "".join(f"<th>{esc(h)}</th>" for h in s.table_headers) + "</tr>")
                for row in s.table_rows:
                    parts.append("<tr>" + "".join(f"<td>{esc(_cell(c))}</td>" for c in row) + "</tr>")
                parts.append("</table>")
            if s.notes:
                parts.append("<ul>" + "".join(f"<li>{esc(n)}</li>" for n in s.notes) + "</ul>")
        if self.warnings:
            parts.append("<h2>Warnings and limitations</h2><ul>")
            for w in self.warnings:
                parts.append(f"<li>{esc(w)}</li>")
            parts.append("</ul>")
        parts.append("</body></html>")
        return "\n".join(parts)

    def write(self, out_dir: str | Path, stem: str = "report", *, html: bool = True) -> dict[str, str]:
        d = ensure_dir(out_dir)
        written: dict[str, str] = {}
        jp = d / f"{stem}.json"
        write_json(jp, self.to_dict())
        written["json"] = str(jp)
        mp = d / f"{stem}.md"
        write_text(mp, self.to_markdown())
        written["markdown"] = str(mp)
        if html:
            hp = d / f"{stem}.html"
            write_text(hp, self.to_html())
            written["html"] = str(hp)
        return written


def _cell(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        if v != v:  # NaN
            return "n/a"
        return f"{v:.4g}"
    return str(v)


def _md_table(rows: Iterable[Sequence[Any]], headers: Sequence[str]) -> tuple[list[str], list[list[Any]]]:
    return list(headers), [list(r) for r in rows]


# --------------------------------------------------------------------------- #
# Builders for the standard report
# --------------------------------------------------------------------------- #


def environment_section(env: Any) -> dict[str, Any]:
    dev = getattr(env, "device", None)
    bw, bw_src = dev.peak_bandwidth_gbps() if dev else (None, "unknown")
    ct, ct_src = dev.compute_ceiling_tflops if dev else (None, "unknown")
    peaks = getattr(dev, "dtype_peaks", {}) or {}
    fp16_dp = peaks.get("fp16")
    int8_dp = peaks.get("int8")
    fp16_tflops = (fp16_dp.get("tflops") if isinstance(fp16_dp, dict) else getattr(fp16_dp, "tflops", None)) if fp16_dp else None
    int8_tops = (int8_dp.get("tflops") if isinstance(int8_dp, dict) else getattr(int8_dp, "tflops", None)) if int8_dp else None
    smem_bw = getattr(dev, "measured_smem_bandwidth_gbps", None)
    tools = getattr(env, "tools", {}) or {}
    out = {
        "GPU": getattr(dev, "name", "unknown"),
        "Architecture": getattr(dev, "arch", "unknown"),
        "SM count": getattr(dev, "multi_processor_count", None),
        "Max threads/SM": getattr(dev, "max_threads_per_multiprocessor", None),
        "Registers/SM": getattr(dev, "regs_per_multiprocessor", None),
        "Shared memory/SM": getattr(dev, "shared_mem_per_multiprocessor", None),
        "L2 cache": getattr(dev, "l2_cache_size", None),
        "Unified memory": getattr(dev, "integrated_uma", None),
        "Measured bandwidth (GB/s)": f"{bw:.1f} ({bw_src})" if bw else f"unknown ({bw_src})",
        "Measured FP32 ceiling (TFLOP/s)": f"{ct:.2f} ({ct_src})" if ct else f"unknown ({ct_src})",
    }
    if fp16_tflops:
        out["Measured FP16/BF16 Tensor ceiling (TFLOP/s)"] = f"{float(fp16_tflops):.2f} (measured)"
    if int8_tops:
        out["Measured INT8 Tensor ceiling (TOPS)"] = f"{float(int8_tops):.2f} (measured)"
    if smem_bw:
        out["Measured shared-memory bandwidth (GB/s)"] = f"{float(smem_bw):.1f} (measured)"
    out["nvcc"] = tools.get("nvcc").version if tools.get("nvcc") else None
    out["ncu"] = tools.get("ncu").version if tools.get("ncu") else None
    out["python"] = getattr(env, "python_version", None)
    return out


def report_from_analysis(
    *,
    env: Any,
    results: Sequence[dict[str, Any]],
    title: str = "CUDA Kernel Analysis Report",
    baseline: dict[str, Any] | None = None,
) -> Report:
    """Assemble the standard report from per-kernel analysis records.

    Each entry in ``results`` is expected to contain the keys emitted by the
    pipeline: ``kernel``, ``baseline_timing``, ``roofline``, ``occupancy``,
    ``waves``, ``classification``, ``recommendations``, and ``correctness``.
    Missing keys are rendered as ``n/a`` rather than omitted, so a gap is visible.
    """
    rep = Report(title=title, generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    rep.environment = environment_section(env)
    rep.data["kernels"] = list(results)

    dev = getattr(env, "device", None)
    rep.environment["Device"] = getattr(dev, "name", "unknown")

    # ---- measured timings --------------------------------------------- #
    sec = ReportSection(title="Measured kernel timings", provenance="measured")
    headers = ["Kernel", "Median (us)", "p10 (us)", "p90 (us)", "CV", "Samples", "Stable", "Effective BW (GB/s)", "TFLOP/s"]
    rows = []
    for r in results:
        bt = r.get("baseline_timing") or {}
        st = bt.get("stats") or {}
        dv = bt.get("derived") or {}
        rows.append([
            _short(r.get("kernel", "?")),
            st.get("median"), st.get("p10"), st.get("p90"), st.get("cv"), st.get("n"),
            bt.get("stable"),
            dv.get("derived.effective_bandwidth_gbps"),
            dv.get("derived.tflops"),
        ])
    sec.table_headers, sec.table_rows = _md_table(rows, headers)
    if not any(r[1] is not None for r in rows):
        sec.body.append("No timing measurements were available for any kernel.")
    rep.add(sec)

    # ---- bottleneck classification ------------------------------------- #
    sec = ReportSection(title="Bottleneck classification", provenance="measured")
    headers = ["Kernel", "Category", "Confidence", "Supporting counters", "Confirming experiment"]
    rows = []
    for r in results:
        cl = r.get("classification") or {}
        cs = cl.get("counter_summary") or {}
        cs_txt = ", ".join(f"{k}={v:g}" for k, v in list(cs.items())[:5]) if cs else "n/a"
        rows.append([
            _short(r.get("kernel", "?")),
            cl.get("category", "n/a"),
            cl.get("confidence", "n/a"),
            cs_txt,
            (cl.get("confirming_experiment") or "n/a")[:160],
        ])
    sec.table_headers, sec.table_rows = _md_table(rows, headers)
    for r in results:
        cl = r.get("classification") or {}
        for e in (cl.get("evidence") or [])[:4]:
            sec.notes.append(f"{_short(r.get('kernel', '?'))}: {e}")
    rep.add(sec)

    # ---- roofline ------------------------------------------------------- #
    sec = ReportSection(title="Roofline and memory analysis", provenance="modelled")
    headers = ["Kernel", "Arith. intensity", "Regime", "Eff. BW (GB/s)",
               "BW vs measured DRAM ceiling (>100% = L2-assisted)",
               "Sectors/req", "L2 hit %", "Roofline bound (us)"]
    rows = []
    for r in results:
        rf = r.get("roofline") or {}
        rows.append([
            _short(r.get("kernel", "?")),
            rf.get("arithmetic_intensity"),
            rf.get("regime"),
            rf.get("effective_bandwidth_gbps"),
            rf.get("memory_utilization_pct"),
            rf.get("achieved_sectors_per_request"),
            rf.get("achieved_l2_hit_rate_pct"),
            rf.get("limit_us"),
        ])
    sec.table_headers, sec.table_rows = _md_table(rows, headers)
    for r in results:
        rf = r.get("roofline") or {}
        for n in (rf.get("notes") or [])[:3]:
            sec.notes.append(f"{_short(r.get('kernel', '?'))}: {n}")
    rep.add(sec)

    # ---- resources and occupancy ---------------------------------------- #
    sec = ReportSection(title="Resources and predicted occupancy", provenance="modelled")
    headers = ["Kernel", "Regs/thread", "Spills (st/ld)", "SMEM/block", "Block size",
               "Blocks/SM", "Occupancy %", "Limiting factor", "Regs for +1 block", "Free reg headroom"]
    rows = []
    for r in results:
        occ = r.get("occupancy") or {}
        rr = r.get("resource_report") or {}
        rows.append([
            _short(r.get("kernel", "?")),
            occ.get("registers_per_thread"),
            f"{rr.get('spill_stores', 0)}/{rr.get('spill_loads', 0)}" if rr else "n/a",
            occ.get("smem_allocated_per_block"),
            occ.get("threads_per_block"),
            occ.get("active_blocks_per_sm"),
            occ.get("occupancy_pct"),
            occ.get("limiting_factor"),
            occ.get("max_regs_per_thread_to_gain_one_block"),
            occ.get("free_reg_headroom_per_thread"),
        ])
    sec.table_headers, sec.table_rows = _md_table(rows, headers)
    sec.body.append(
        "Occupancy is predicted from compiler-reported registers and shared memory using the "
        "target's allocation granularity; column headers therefore carry no measured provenance."
    )
    for r in results:
        occ = r.get("occupancy") or {}
        steps = occ.get("register_step_table") or []
        if steps:
            step_txt = ", ".join(
                f"{s['blocks_per_sm']} blk ({s['occupancy_pct']:.0f}% occ) <= {s['max_regs_per_thread']} regs"
                for s in steps[:6]
            )
            sec.notes.append(f"{_short(r.get('kernel', '?'))} register step table: {step_txt}")
    rep.add(sec)

    # ---- wave quantisation ---------------------------------------------- #
    sec = ReportSection(title="Grid and wave analysis", provenance="modelled")
    headers = ["Kernel", "CTAs", "Blocks/wave", "Waves", "Tail CTAs", "Tail efficiency %", "Structure"]
    rows = []
    for r in results:
        w = r.get("waves") or {}
        rows.append([
            _short(r.get("kernel", "?")),
            w.get("total_ctas"), w.get("blocks_per_wave"), w.get("waves"),
            w.get("tail_ctas"), w.get("tail_efficiency_pct"), w.get("problem"),
        ])
    sec.table_headers, sec.table_rows = _md_table(rows, headers)
    for r in results:
        w = r.get("waves") or {}
        for n in (w.get("notes") or [])[:2]:
            sec.notes.append(f"{_short(r.get('kernel', '?'))}: {n}")
    rep.add(sec)

    # ---- correctness ---------------------------------------------------- #
    sec = ReportSection(title="Correctness status", provenance="measured")
    headers = ["Kernel", "Status", "Method", "Detail"]
    rows = []
    for r in results:
        cr = r.get("correctness") or {}
        rows.append([
            _short(r.get("kernel", "?")),
            cr.get("status", "unvalidated"),
            cr.get("method", "n/a"),
            (cr.get("detail") or "")[:120],
        ])
    sec.table_headers, sec.table_rows = _md_table(rows, headers)
    unvalidated = [r for r in results if (r.get("correctness") or {}).get("status") != "pass"]
    if unvalidated:
        sec.notes.append(
            f"{len(unvalidated)} kernel(s) are not in a passed state. No performance conclusion "
            "may be drawn for them."
        )
    rep.add(sec)

    # ---- recommendations ------------------------------------------------- #
    sec = ReportSection(title="Recommendations", provenance="modelled")
    for r in results:
        recs = r.get("recommendations") or {}
        items = recs.get("recommendations") or []
        sec.body.append(f"### {_short(r.get('kernel', '?'))} — {recs.get('classification', 'n/a')}")
        if not items:
            sec.body.append("No recommendation was produced.")
            for u in recs.get("unavailable_checks") or []:
                sec.notes.append(f"{_short(r.get('kernel', '?'))}: {u}")
            continue
        headers = ["#", "Title", "Kind", "Confidence", "Effort", "Risk", "Impact"]
        rows = [
            [i + 1, it.get("title", ""), it.get("kind", ""), it.get("confidence", ""),
             it.get("effort", ""), it.get("risk", ""), it.get("impact", "")]
            for i, it in enumerate(items)
        ]
        sec.body.append("| " + " | ".join(headers) + " |")
        sec.body.append("| " + " | ".join("---" for _ in headers) + " |")
        for row in rows:
            sec.body.append("| " + " | ".join(_cell(c) for c in row) + " |")
        for it in items:
            ev = "; ".join(it.get("evidence") or [])
            sec.notes.append(
                f"{it.get('title')}: {it.get('suspected_cause', '')} Validation: {it.get('validation', '')}"
            )
            if ev:
                sec.notes.append(f"    evidence: {ev[:300]}")
    rep.add(sec)

    # ---- search results -------------------------------------------------- #
    if baseline:
        sec = ReportSection(title="Launch-parameter search", provenance="measured")
        sec.body.append(
            f"Strategy: {baseline.get('strategy', 'n/a')}; "
            f"wall time {baseline.get('wall_time_s', 0):.1f}s; "
            f"{len(baseline.get('candidates') or [])} configuration(s) evaluated."
        )
        headers = ["Configuration", "Latency (us)", "Speedup", "Passed verification", "Occupancy %"]
        rows = []
        for c in baseline.get("candidates") or []:
            occ = c.get("occupancy") or {}
            rows.append([
                c.get("label"), c.get("latency_us"),
                c.get("speedup"), c.get("passed_correctness"),
                occ.get("occupancy_pct"),
            ])
        sec.table_headers, sec.table_rows = _md_table(rows, headers)
        for n in baseline.get("notes") or []:
            sec.notes.append(n)
        rep.add(sec)

    rep.warnings.extend(getattr(env, "warnings", []) or [])
    return rep


def _short(name: str, limit: int = 46) -> str:
    n = str(name)
    return n if len(n) <= limit else n[: limit - 3] + "..."


# --------------------------------------------------------------------------- #
# CSV export
# --------------------------------------------------------------------------- #


def write_csv(path: str | Path, rows: Sequence[dict[str, Any]]) -> Path:
    """Write a list of flat dicts as CSV, using the union of keys as the header."""
    p = Path(path)
    ensure_dir(p.parent)
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=keys, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    write_text(p, buf.getvalue())
    return p
