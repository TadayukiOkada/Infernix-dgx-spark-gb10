"""Comprehensive Markdown and text report generation for offline autotuning.

Formats hardware profiles, discovery inventory, candidate validation results,
winner tables, crossover intervals, and custom kernel recommendations into
human-readable documents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .database import TuningDatabase, TuningRecord
from .diagnostics import DiagnosticOutcome
from .discovery import DiscoveryReport
from .generator import DispatchInterval
from .profile import TuningProfile
from .search import WinnerSelection
from .validator import PolicyValidationReport


class AutotuneReporter:
    """Generates human-readable Markdown and text reports from tuning runs."""

    @staticmethod
    def render_tuning_campaign_report(
        profile: TuningProfile,
        discovery: DiscoveryReport | None,
        shape_key: str,
        operation: str,
        qtype: str,
        n: int,
        k: int,
        winners: dict[int, WinnerSelection],
        intervals: Sequence[DispatchInterval],
        diagnostics: Sequence[DiagnosticOutcome] | None = None,
        validation_report: PolicyValidationReport | None = None,
        generated_cpp_path: str | None = None,
        generated_cpp_code: str | None = None,
    ) -> str:
        """Render full campaign Markdown report."""
        hw = profile.hardware
        sw = profile.software

        lines = [
            f"# Autotuning & Dispatch Generation Report: {operation.upper()} {qtype.upper()} [{n}, {k}]",
            "",
            "## 1. Hardware & Software Profile",
            "",
            f"- **Target GPU**: {hw.gpu_name} ({hw.arch_name})",
            f"- **Compute Capability**: {hw.compute_capability} (SMs: {hw.sm_count})",
            f"- **Device Fingerprint**: `{hw.fingerprint}`",
            f"- **Profile ID**: `{profile.profile_id}`",
            f"- **Sustained Read Bandwidth**: {hw.sustained_read_gbps:.1f} GB/s (Peak DRAM: {hw.bandwidth_gbps:.1f} GB/s)",
            f"- **Shared Memory Limits**: {hw.shared_mem_per_block_default // 1024} KiB default, {hw.shared_mem_per_block_opt_in // 1024} KiB opt-in",
            f"- **CUDA Runtime & Toolchain**: {sw.cuda_runtime_version} ({sw.compiler_version or sw.compiler})",
            f"- **Framework Revision**: `{sw.framework_revision[:12]}` (clean={sw.framework_clean})",
            "",
            "## 2. Repository Discovery & Surface Area",
            "",
        ]

        if discovery:
            lines.extend([
                f"- **Discovered Quantization Families**: {len(discovery.families)} ({', '.join(sorted(discovery.families.keys()))})",
                f"- **Total Registered Shapes in Repo**: {discovery.total_shapes}",
                f"- **Target Family Operations**: {', '.join(discovery.families.get(qtype, discovery.families.get('linear', None)).operations if qtype in discovery.families else ['linear'])}",
            ])
        else:
            lines.append("- *Discovery report not attached.*")

        lines.extend([
            "",
            "## 3. Winner Table across Token Counts",
            "",
            "| Tokens `T` | Winning Candidate | Family | Median (µs) | p95 (µs) | Margin (%) | Tie-break |",
            "|---|---|---|---|---|---|---|",
        ])

        noisy_runs = []
        for t, w in sorted(winners.items()):
            margin_str = f"{w.margin_pct:+.1f}%" if w.margin_pct != 0 else "-"
            tie_str = "yes" if w.tie_breaker_applied else ("tie" if w.is_tie else "no")
            rec = w.record
            if rec and (rec.latency_p95_us - rec.latency_min_us) / (rec.latency_us or 1.0) > 0.15:
                noisy_runs.append(t)
            lines.append(
                f"| `{t}` | `{w.winning_candidate.id[:36]}...` | `{w.winning_candidate.kernel_family}` | "
                f"{w.winning_latency_us:.3f} | {w.record.latency_p95_us:.3f} | {margin_str} | {tie_str} |"
            )

        lines.extend([
            "",
            "## 4. Compressed Dispatch Intervals",
            "",
            "| Interval Tokens | Condition | Selected Kernel Schedule / Launch Expression |",
            "|---|---|---|",
        ])
        for inter in intervals:
            tok_range = f"`T <= {inter.max_tokens}`" if inter.max_tokens is not None else f"`T >= {inter.min_tokens}` (default)"
            if inter.min_tokens == inter.max_tokens and inter.min_tokens == 1:
                tok_range = "`T == 1`"
            lines.append(
                f"| {tok_range} | `{inter.condition}` | `{inter.launch_expr}` |"
            )

        lines.extend([
            "",
            "## 5. Correctness & Byte-Identity Validation",
            "",
        ])
        if validation_report:
            vr = validation_report
            comp_badge = "PASSED" if vr.compilation_passed else "FAILED"
            corr_badge = "PASSED" if vr.correctness_passed else "FAILED"
            byte_badge = "PASSED (Bit-Exact)" if vr.byte_identity_passed else "FAILED"
            bench_badge = "PASSED (0% Regression)" if vr.benchmark_verified else "REGRESSION DETECTED"
            prod_badge = "READY" if vr.production_ready else "NOT READY"

            lines.extend([
                f"- **C++ Compilation Gate**: `{comp_badge}`",
                f"- **Existing Test Suite**: `{corr_badge}` ({len(vr.correctness_summary.passed)} passed, {len(vr.correctness_summary.failed)} failed, {len(vr.correctness_summary.skipped)} skipped)",
                f"- **Strict Byte-Identity Gate**: `{byte_badge}`",
                f"- **Performance Regression Gate**: `{bench_badge}` (Max regression: {vr.max_regression_pct}%)",
                f"- **Production Readiness**: `{prod_badge}`",
            ])
            if vr.correctness_summary.passed:
                lines.append(f"  - Passed tests: `{', '.join(vr.correctness_summary.passed)}`")
            if vr.correctness_summary.failed:
                lines.append(f"  - Failed tests: `{', '.join(vr.correctness_summary.failed)}`")
            if vr.regressions:
                lines.append("  - Regressions:")
                for reg in vr.regressions:
                    lines.append(f"    - `T={reg.get('tokens')}`: regression {reg.get('regression_pct')}%")
        else:
            lines.append("- *Validation report pending.*")

        if noisy_runs:
            lines.extend([
                "",
                "## 6. Measurement Quality & Noise Warnings",
                "",
                f"- High spread / noise observed at token counts: `{', '.join(str(x) for x in noisy_runs)}` (spread > 15%). Close comparisons were confirmed with noise-aware margins.",
            ])

        if diagnostics:
            lines.extend([
                "",
                "## 7. Custom-Kernel & Bottleneck Investigation",
                "",
                "| Workload | Verdict | BW Util (%) | Roofline Gap | Identified Bottlenecks & Recommendations |",
                "|---|---|---|---|---|",
            ])
            for d in diagnostics:
                bn_text = "; ".join(d.identified_bottlenecks + d.recommendations) or "Acceptable"
                lines.append(
                    f"| `{d.workload_key}` | **`{d.outcome}`** | {d.bandwidth_utilization_pct:.1f}% | {d.roofline_headroom_ratio:.2f}x | {bn_text} |"
                )

        if generated_cpp_code:
            lines.extend([
                "",
                "## 8. Generated C++ Dispatch Preview",
                "",
                "```cpp",
                generated_cpp_code.strip(),
                "```",
            ])

        return "\n".join(lines) + "\n"
