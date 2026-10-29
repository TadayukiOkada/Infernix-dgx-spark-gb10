"""Deterministic C++ dispatch code generation and policy compression.

Compresses measured winner points into minimal token interval ladders and emits
deterministic C++ source with reproducible provenance headers and out-of-range fallback.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .database import TuningRecord
from .profile import TuningProfile
from .registry import CandidateSpec
from .search import WinnerSelection
from ..util import to_plain

GENERATOR_VERSION = "1.0.0"


@dataclass
class DispatchInterval:
    """A compressed interval where one candidate is active."""

    min_tokens: int
    max_tokens: int | None  # None indicates final unbounded fallback
    candidate: CandidateSpec
    launch_expr: str
    condition: str  # e.g. "tokens <= 8", "tokens == 1", "default"

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_tokens": self.min_tokens,
            "max_tokens": self.max_tokens,
            "candidate_id": self.candidate.id,
            "launch_expr": self.launch_expr,
            "condition": self.condition,
        }


class PolicyCompressor:
    """Compresses point-by-point winner tables into contiguous token intervals."""

    @staticmethod
    def compress(
        winners_per_token: dict[int, WinnerSelection],
        token_step: int = 1,
        latencies_per_token: dict[int, dict[str, float]] | None = None,
        tolerance_pct: float = 2.0,
    ) -> list[DispatchInterval]:
        """Merge consecutive winner points into intervals within tolerance."""
        if not winners_per_token:
            return []

        sorted_tokens = sorted(winners_per_token.keys())
        if not latencies_per_token:
            # Fallback to exact candidate identity merging
            intervals: list[DispatchInterval] = []
            cur_cand = winners_per_token[sorted_tokens[0]].winning_candidate
            cur_min = sorted_tokens[0]
            cur_max = sorted_tokens[0]

            for t in sorted_tokens[1:]:
                cand = winners_per_token[t].winning_candidate
                if cand.launch_expr == cur_cand.launch_expr:
                    cur_max = t
                else:
                    cond = f"tokens == {cur_min}" if cur_min == cur_max and cur_min == 1 else f"tokens <= {cur_max}"
                    intervals.append(
                        DispatchInterval(
                            min_tokens=cur_min,
                            max_tokens=cur_max,
                            candidate=cur_cand,
                            launch_expr=cur_cand.launch_expr,
                            condition=cond,
                        )
                    )
                    cur_cand = cand
                    cur_min = t
                    cur_max = t

            intervals.append(
                DispatchInterval(
                    min_tokens=cur_min,
                    max_tokens=None,
                    candidate=cur_cand,
                    launch_expr=cur_cand.launch_expr,
                    condition="default",
                )
            )
            return intervals

        # Tolerance-based greedy span creation and merging (same principle as bf16/fp8 sweep C++)
        def within(cand: CandidateSpec, t: int) -> bool:
            if cand.max_tokens is not None and t > cand.max_tokens:
                return False
            if t < cand.min_tokens:
                return False
            if cand.kernel_family == "gemv" and t > 1:
                return False
            t_lats = latencies_per_token.get(t, {})
            cand_lat = t_lats.get(cand.launch_expr)
            if cand_lat is None or math.isinf(cand_lat) or cand_lat <= 0:
                return cand.launch_expr == winners_per_token[t].winning_candidate.launch_expr
            best_lat = winners_per_token[t].winning_latency_us
            return cand_lat <= best_lat * (1.0 + tolerance_pct / 100.0)

        # 1. Greedy forward pass
        class Span:
            def __init__(self, first_t: int, last_t: int, cand: CandidateSpec):
                self.first_t = first_t
                self.last_t = last_t
                self.cand = cand

        spans: list[Span] = []
        for t in sorted_tokens:
            if spans and within(spans[-1].cand, t):
                spans[-1].last_t = t
            else:
                spans.append(Span(t, t, winners_per_token[t].winning_candidate))

        # 2. Merge adjacent spans if either candidate covers both within tolerance
        merged = True
        while merged:
            merged = False
            for s in range(len(spans) - 1):
                span_a = spans[s]
                span_b = spans[s + 1]
                # Check if span_a's candidate covers span_b
                tokens_in_ab = [t for t in sorted_tokens if span_a.first_t <= t <= span_b.last_t]
                if all(within(span_a.cand, t) for t in tokens_in_ab):
                    span_a.last_t = span_b.last_t
                    spans.pop(s + 1)
                    merged = True
                    break
                # Or check if span_b's candidate covers span_a
                if all(within(span_b.cand, t) for t in tokens_in_ab):
                    span_b.first_t = span_a.first_t
                    spans.pop(s)
                    merged = True
                    break

        intervals: list[DispatchInterval] = []
        for idx, span in enumerate(spans):
            is_last = idx == len(spans) - 1
            max_t = None if is_last else span.last_t
            cond = f"tokens == 1" if span.first_t == span.last_t and span.first_t == 1 else (f"tokens <= {span.last_t}" if not is_last else "default")
            intervals.append(
                DispatchInterval(
                    min_tokens=span.first_t,
                    max_tokens=max_t,
                    candidate=span.cand,
                    launch_expr=span.cand.launch_expr,
                    condition=cond,
                )
            )

        return intervals


class DispatchGenerator:
    """Emits deterministic C++ dispatch code."""

    def __init__(self, profile: TuningProfile):
        self.profile = profile

    def generate_cpp(
        self,
        operation: str,
        qtype: str,
        n: int,
        k: int,
        intervals: Sequence[DispatchInterval],
        *,
        token_step: int = 1,
        disabled_routes: Sequence[str] | None = None,
        coupled_routes: Sequence[str] | None = None,
        results_hash: str = "",
        existing_code: str | None = None,
    ) -> str:
        """Generate deterministic C++ shape dispatch file."""
        import re

        # 1. Header with provenance and reproducible metadata
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        header_lines = [
            "// =============================================================================",
            f"// Autotuned Dispatch Policy generated by cuda-kernel-toolkit (ckt autotune v{GENERATOR_VERSION})",
            f"// Target GPU: {self.profile.hardware.gpu_name} ({self.profile.hardware.arch_name})",
            f"// Hardware Fingerprint: {self.profile.hardware.fingerprint}",
            f"// Profile ID: {self.profile.profile_id}",
            f"// Software Build: {self.profile.software.framework_revision[:12]} (clean={self.profile.software.framework_clean})",
            f"// Results Hash: {results_hash or 'reproducible-offline'}",
            f"// Generated at: {now_iso}",
            "// =============================================================================",
        ]

        q = qtype.lower()
        fn_name = ""
        ret_type = ""
        if q == "nvfp4":
            fn_name = "select_a16"
            ret_type = "Nvfp4Launch"
        elif q == "fp8":
            fn_name = "launch_a16"
            ret_type = "void"
        elif q == "q8":
            fn_name = f"select_q8_n{n}_k{k}"
            ret_type = "Q8Launch"
        elif q in ("q4", "q5", "q6"):
            fn_name = f"select_{q}_n{n}_k{k}"
            ret_type = f"{q.upper()}Launch"
        elif q == "bf16":
            fn_name = f"select_bf16_n{n}_k{k}"
            ret_type = "Bf16Launch"

        # Build dispatch ladder body
        body_lines = []
        if q == "fp8":
            body_lines.append("    const int tokens = x.ne[1];")

        for idx, inter in enumerate(intervals):
            is_last = idx == len(intervals) - 1
            launch = inter.launch_expr.strip()
            if not launch.startswith("return") and q != "fp8":
                launch_stmt = f"return {launch};"
            elif q == "fp8":
                launch_stmt = f"{launch}; return;" if not launch.endswith(";") else launch
            else:
                launch_stmt = f"{launch};"

            if is_last or inter.max_tokens is None:
                body_lines.append(f"    {launch_stmt}")
            elif inter.min_tokens == inter.max_tokens and inter.min_tokens == 1:
                body_lines.append(f"    if (tokens == 1) {launch_stmt}")
            else:
                body_lines.append(f"    if (tokens <= {inter.max_tokens}) {launch_stmt}")

        new_body = "\n" + "\n".join(body_lines) + "\n"

        # If existing_code is present and contains fn_name, update in place
        if existing_code and fn_name in existing_code:
            sig_pattern = re.compile(
                rf'(\b(?:' + re.escape(ret_type) + r'|void|Q8Launch|Bf16Launch|Nvfp4Launch|[A-Za-z0-9_]+Launch)\s+' +
                re.escape(fn_name) + r'\s*\([^)]*\)\s*\{)([\s\S]*?)(\n\})'
            )
            m = sig_pattern.search(existing_code)
            if m:
                old_inner = m.group(2)
                # Preserve leading unaligned-token guard blocks such as `if (tokens % 4 != 0) { ... }`
                guard_match = re.search(r'(\s*if\s*\(\s*tokens\s*%\s*\d+\s*!=\s*0\s*\)\s*\{[\s\S]*?\})', old_inner)
                if guard_match:
                    guard_block = guard_match.group(1).strip("\n")
                    # Strip `if (tokens == 1)` from body_lines if already inside the guard block
                    if "tokens == 1" in guard_block:
                        filtered_lines = [ln for ln in body_lines if "if (tokens == 1)" not in ln]
                    else:
                        filtered_lines = body_lines
                    new_body = "\n" + guard_block + "\n" + "\n".join(filtered_lines) + "\n"
                updated = sig_pattern.sub(rf'\g<1>{new_body}\g<3>', existing_code)
                prov_block = "\n".join(header_lines) + "\n\n"
                old_prov = re.compile(r'// =+\n// Autotuned Dispatch Policy[\s\S]*?// =+\n\n?')
                if old_prov.search(updated):
                    updated = old_prov.sub(prov_block, updated)
                else:
                    updated = prov_block + updated
                return updated

        # Otherwise generate complete file from scratch
        lines = list(header_lines)
        lines.append("")

        # 2. Includes matching quantization family conventions
        q = qtype.lower()
        if q == "nvfp4":
            lines.extend([
                '#include "ops/linear/nvfp4/nvfp4_shapes.h"',
                '#include "ops/linear/nvfp4/nvfp4_launch.cuh"',
                '#include "ops/linear/nvfp4/nvfp4_instances.cuh"',
                "",
                "#include <stdexcept>",
                "",
                "namespace infernix::ops::detail {",
                "namespace {",
                f"using Geometry = Nvfp4Geometry<{n}, {k}>;",
            ])
        elif q == "fp8":
            lines.extend([
                '#include "ops/linear/fp8/fp8_shapes.h"',
                '#include "ops/linear/fp8/fp8_launch.cuh"',
                '#include "ops/linear/fp8/fp8_a8_pair_tma_mma.cuh"',
                "",
                "#include <stdexcept>",
                "",
                "namespace infernix::ops::detail {",
                "namespace {",
                f"using Geometry = Fp8Geometry<{n}, {k}>;",
            ])
        elif q == "q8":
            has_templates = any(inter.candidate.using_decl for inter in intervals)
            if has_templates:
                lines.extend([
                    '#include "ops/linear/q8/q8_shapes.h"',
                    '#include "ops/linear/q8/q8_gemv_launch.cuh"',
                    '#include "ops/linear/q8/q8_mma_launch.cuh"',
                    '#include "ops/linear/q8/q8_sliced_k_launch.cuh"',
                    "",
                    "namespace infernix::ops::detail {",
                ])
            else:
                lines.extend([
                    '#include "ops/linear/q8/q8_shapes.h"',
                    "",
                    "namespace infernix::ops::detail {",
                ])
        elif q in ("q4", "q5", "q6"):
            lines.extend([
                f'#include "ops/linear/{q}/{q}_shapes.h"',
                f'#include "ops/linear/{q}/{q}_launch.h"',
                "",
                "namespace infernix::ops::detail {",
            ])
        elif q == "bf16":
            lines.extend([
                '#include "ops/linear/bf16/bf16_shapes.h"',
                '#include "ops/linear/bf16/bf16_launch.h"',
                "",
                "namespace infernix::ops::detail {",
            ])
        else:
            lines.extend([
                f'#include "ops/linear/{q}/{q}_shapes.h"',
                "",
                "namespace infernix::ops::detail {",
            ])

        # 3. Emit using aliases if any
        using_decls: set[str] = set()
        for inter in intervals:
            if inter.candidate.using_decl:
                ud = inter.candidate.using_decl.strip()
                if not ud.startswith("using Geometry"):
                    using_decls.add(ud)
        for ud in sorted(using_decls):
            lines.append(ud)
        if using_decls:
            lines.append("")

        # 4. Dispatch function body
        fn_name = ""
        ret_type = ""
        if q == "nvfp4":
            fn_name = "select_a16"
            ret_type = "Nvfp4Launch"
            lines.append(f"{ret_type} {fn_name}(int tokens) {{")
        elif q == "fp8":
            fn_name = "launch_a16"
            lines.append("void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {")
            lines.append("    const int tokens = x.ne[1];")
        elif q == "q8":
            fn_name = f"select_q8_n{n}_k{k}"
            ret_type = "Q8Launch"
            lines.append(f"{ret_type} {fn_name}(std::int32_t tokens) {{")
        elif q in ("q4", "q5", "q6"):
            fn_name = f"select_{q}_n{n}_k{k}"
            ret_type = f"{q.upper()}Launch"
            lines.append(f"{ret_type} {fn_name}(std::int32_t tokens) {{")
        elif q == "bf16":
            fn_name = f"select_bf16_n{n}_k{k}"
            ret_type = "Bf16Launch"
            lines.append(f"{ret_type} {fn_name}(std::int32_t tokens) {{")

        # Emit intervals
        for idx, inter in enumerate(intervals):
            is_last = idx == len(intervals) - 1
            launch = inter.launch_expr.strip()
            if not launch.startswith("return") and q != "fp8":
                launch_stmt = f"return {launch};"
            elif q == "fp8":
                launch_stmt = f"{launch}; return;" if not launch.endswith(";") else launch
            else:
                launch_stmt = f"{launch};"

            if is_last or inter.max_tokens is None:
                lines.append(f"    {launch_stmt}")
            elif inter.min_tokens == inter.max_tokens and inter.min_tokens == 1:
                lines.append(f"    if (tokens == 1) {launch_stmt}")
            else:
                lines.append(f"    if (tokens <= {inter.max_tokens}) {launch_stmt}")

        lines.append("}")
        lines.append("")

        # 5. Emit policy wrapper structs/helpers for nvfp4/fp8
        if q == "nvfp4":
            has_disabled_a4 = disabled_routes and any("a4" in r for r in disabled_routes)
            lines.extend([
                "void launch_a16(const Tensor& x, const Weight& w, Tensor& y, cudaStream_t stream) {",
                "    select_a16(x.ne[1])(x, w, y, stream);",
                "}",
                "",
            ])
            if has_disabled_a4:
                lines.extend([
                    "[[noreturn]] void launch_a4(const Tensor&, const Weight&, Tensor&, Nvfp4A4Workspace, cudaStream_t) {",
                    f'    throw std::logic_error("nvfp4 A4 route is not enabled for the [{n}, {k}] shape");',
                    "}",
                    "",
                    "bool uses_a4(std::int32_t, std::int32_t) { return false; }",
                    "} // namespace",
                    "",
                    f"const Nvfp4LinearShape kNvfp4N{n}K{k}{{{n}, {k}, launch_a16, launch_a4, uses_a4}};",
                    "} // namespace infernix::ops::detail",
                ])
            else:
                lines.extend([
                    "bool uses_a4(std::int32_t, std::int32_t) { return false; }",
                    "} // namespace",
                    "",
                    f"const Nvfp4LinearShape kNvfp4N{n}K{k}{{{n}, {k}, launch_a16, nullptr, uses_a4}};",
                    "} // namespace infernix::ops::detail",
                ])
        elif q == "fp8":
            lines.extend([
                "bool uses_a8(std::int32_t, std::int32_t) { return false; }",
                "} // namespace",
                "",
                f"const Fp8LinearShape kFp8N{n}K{k}{{{n}, {k}, launch_a16, nullptr, uses_a8}};",
                "} // namespace infernix::ops::detail",
            ])
        else:
            lines.append("} // namespace infernix::ops::detail")

        return "\n".join(lines) + "\n"
