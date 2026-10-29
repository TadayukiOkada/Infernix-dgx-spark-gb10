"""Quantization-family agnostic repository discovery and inspection.

Discovers all supported operations, quantization families, shape/geometry registrations,
kernel schedules, launch functions, restrictions, and existing test contracts from
the codebase without hard-coding assumptions to any specific family.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..shapes import ShapeSpec, discover_shapes, parse_shape_file
from ..util import to_plain


@dataclass
class DiscoveredShape:
    """A shape/geometry discovered in the repository."""

    operation: str
    quantization_family: str
    n: int
    k: int
    file_path: str
    policies: list[str] = field(default_factory=list)
    token_step: int = 1
    disabled_routes: list[str] = field(default_factory=list)
    coupled_routes: list[str] = field(default_factory=list)
    existing_bands: list[dict[str, Any]] = field(default_factory=list)
    aliases: dict[str, str] = field(default_factory=dict)
    associated_tests: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass
class DiscoveredFamily:
    """A quantization family and its supported operations."""

    name: str  # e.g. "nvfp4", "fp8", "q4", "q5", "q6", "q8", "bf16"
    operations: list[str] = field(default_factory=list)
    shapes: list[DiscoveredShape] = field(default_factory=list)
    kernel_families: list[str] = field(default_factory=list)
    available_tests: list[str] = field(default_factory=list)
    byte_identity_tests: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass
class DiscoveryReport:
    """Full repository discovery inventory and status."""

    repo_root: str
    families: dict[str, DiscoveredFamily] = field(default_factory=dict)
    total_shapes: int = 0
    total_candidates: int = 0
    unsupported_families: list[dict[str, str]] = field(default_factory=list)
    skipped_paths: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_root": self.repo_root,
            "total_shapes": self.total_shapes,
            "total_candidates": self.total_candidates,
            "families": {k: v.to_dict() for k, v in self.families.items()},
            "unsupported_families": self.unsupported_families,
            "skipped_paths": self.skipped_paths,
        }

    def render_markdown(self) -> str:
        """Render a clean, human-readable discovery report."""
        lines = [
            "# Infernix CUDA Repository Discovery Report",
            "",
            f"**Repository Root**: `{self.repo_root}`  ",
            f"**Quantization Families Discovered**: {len(self.families)}  ",
            f"**Total Registered Shapes**: {self.total_shapes}  ",
            "",
            "## Discovered Quantization Families & Operations",
            "",
            "| Family | Operations | Shapes | Kernel Families | Correctness & Byte-Identity Tests |",
            "|---|---|---|---|---|",
        ]
        for name, fam in sorted(self.families.items()):
            ops_str = ", ".join(f"`{o}`" for o in sorted(fam.operations))
            k_fams = ", ".join(fam.kernel_families) or "standard"
            test_count = len(fam.available_tests)
            exact_count = len(fam.byte_identity_tests)
            tests_desc = f"{test_count} tests ({exact_count} byte-exact)"
            lines.append(
                f"| **{name.upper()}** | {ops_str} | {len(fam.shapes)} | {k_fams} | {tests_desc} |"
            )

        lines.extend([
            "",
            "## Registered Geometries per Family",
            "",
        ])
        for name, fam in sorted(self.families.items()):
            lines.append(f"### {name.upper()} ({len(fam.shapes)} registered shapes)")
            if not fam.shapes:
                lines.append("*No shapes found.*")
                continue
            lines.append("| Geometry `[N, K]` | File | Policies | Restrictions / Couplings | Tests |")
            lines.append("|---|---|---|---|---|")
            for s in fam.shapes:
                restr = []
                if s.disabled_routes:
                    restr.append(f"Disabled: {', '.join(s.disabled_routes)}")
                if s.coupled_routes:
                    restr.append(f"Coupled: {', '.join(s.coupled_routes)}")
                restr_str = "; ".join(restr) if restr else "None"
                t_str = ", ".join(s.associated_tests) if s.associated_tests else "infernix_linear_bench"
                lines.append(
                    f"| `[{s.n}, {s.k}]` | `{Path(s.file_path).name}` | {', '.join(s.policies)} | {restr_str} | `{t_str}` |"
                )
            lines.append("")

        if self.unsupported_families or self.skipped_paths:
            lines.extend([
                "## Explicitly Skipped or Restricted Paths",
                "",
                "| Path / Family | Reason |",
                "|---|---|",
            ])
            for u in self.unsupported_families:
                lines.append(f"| `{u.get('path', 'unknown')}` | {u.get('reason', '')} |")
            for sk in self.skipped_paths:
                lines.append(f"| `{sk.get('path', 'unknown')}` | {sk.get('reason', '')} |")
            lines.append("")

        return "\n".join(lines)


class RepositoryDiscovery:
    """Discovers all supported families, shapes, schedules, and test suites."""

    def __init__(self, repo_root: str | Path = "."):
        self.repo_root = Path(repo_root).resolve()
        # Handle running from `cuda-kernel-toolkit/` subfolder
        if not (self.repo_root / "src" / "ops").exists() and (self.repo_root.parent / "src" / "ops").exists():
            self.repo_root = self.repo_root.parent

    def discover(self) -> DiscoveryReport:
        report = DiscoveryReport(repo_root=str(self.repo_root))

        ops_dir = self.repo_root / "src" / "ops"
        linear_dir = ops_dir / "linear"

        # 1. Discover all quantization families in src/ops/linear/
        known_families: set[str] = set()
        if linear_dir.exists():
            for child in sorted(linear_dir.iterdir()):
                if child.is_dir() and (child / "shapes").exists():
                    known_families.add(child.name.lower())

        # Also inspect other operations for quantization families (linear_add, linear_swiglu, etc.)
        op_candidates = ["linear", "linear_add", "linear_swiglu", "attn_input_proj", "gdn_input_proj", "dynamic_grouped_conv"]
        family_to_ops: dict[str, list[str]] = {}

        for op_name in op_candidates:
            op_path = ops_dir / op_name
            if not op_path.exists():
                continue
            for sub in sorted(op_path.iterdir()):
                if sub.is_dir() and sub.name.lower() in ("nvfp4", "fp8", "q4", "q5", "q6", "q8", "bf16", "q4_q5"):
                    f_name = sub.name.lower()
                    family_to_ops.setdefault(f_name, []).append(op_name)
                    known_families.add(f_name)

        # 2. For each family, discover shapes and parse restrictions
        for fam_name in sorted(known_families):
            fam = DiscoveredFamily(
                name=fam_name,
                operations=family_to_ops.get(fam_name, ["linear"]),
            )

            # Discover tests
            fam.available_tests = self._discover_tests_for_family(fam_name)
            fam.byte_identity_tests = [
                t for t in fam.available_tests
                if "column_invariant" in t or "exact" in t or "a16" in t
            ]

            # Discover shapes from src/ops/linear/<family>/shapes/
            fam_shapes_dir = linear_dir / fam_name / "shapes"
            if fam_shapes_dir.exists():
                specs = discover_shapes(self.repo_root, qtype=fam_name)
                for s in specs:
                    dis = self._inspect_shape_spec(fam_name, s)
                    fam.shapes.append(dis)
                    report.total_shapes += 1

            # Discover kernel families present in headers
            fam.kernel_families = self._discover_kernel_families(fam_name)

            report.families[fam_name] = fam

        return report

    def _inspect_shape_spec(self, family: str, spec: ShapeSpec) -> DiscoveredShape:
        fpath = Path(spec.file_path)
        content = fpath.read_text(errors="replace") if fpath.exists() else ""

        disabled_routes = []
        coupled_routes = []

        # Check for explicitly disabled routes
        if "throw std::logic_error" in content:
            for m in re.finditer(r'throw std::logic_error\("([^"]+)"\)', content):
                disabled_routes.append(m.group(1))
        if "uses_a4" in content and "return false;" in content:
            disabled_routes.append("a4_route_disabled")
        if "uses_a8" in content and "return false;" in content:
            disabled_routes.append("a8_route_disabled")

        # Check coupled routes (e.g. TMA scale layouts, residual profiles)
        if "Nvfp4ScaleLayout" in content:
            coupled_routes.append("scale_layout_quantizer_coupled")
        if "Tma" in content or "tma" in content:
            coupled_routes.append("tma_barrier_workspace_coupled")

        # Map to matching test binary
        tests = []
        test_bin = f"infernix_linear_{family}_a16_test"
        if (self.repo_root / "build-native" / "tests" / test_bin).exists():
            tests.append(test_bin)
        if "a4" in spec.policies:
            a4_test = f"infernix_linear_{family}_a4_test"
            if (self.repo_root / "build-native" / "tests" / a4_test).exists():
                tests.append(a4_test)
        if "a8" in spec.policies:
            a8_test = f"infernix_linear_{family}_a8_test"
            if (self.repo_root / "build-native" / "tests" / a8_test).exists():
                tests.append(a8_test)

        # Byte-identity test for column invariance
        col_inv_test = "infernix_column_invariant_projections_test"
        if (self.repo_root / "build-native" / "tests" / col_inv_test).exists():
            tests.append(col_inv_test)

        bands_dicts = [b.to_dict() for b in spec.bands]

        return DiscoveredShape(
            operation="linear",
            quantization_family=family,
            n=spec.n,
            k=spec.k,
            file_path=spec.file_path,
            policies=spec.policies,
            token_step=spec.token_step,
            disabled_routes=disabled_routes,
            coupled_routes=coupled_routes,
            existing_bands=bands_dicts,
            aliases=spec.aliases,
            associated_tests=tests,
        )

    def _discover_tests_for_family(self, family: str) -> list[str]:
        tests = []
        test_dir = self.repo_root / "build-native" / "tests"
        if not test_dir.exists():
            return []

        for p in test_dir.iterdir():
            if not p.is_file() or not os.access(p, os.X_OK):
                continue
            name = p.name
            if family in name:
                tests.append(name)
            elif name == "infernix_column_invariant_projections_test":
                tests.append(name)

        return sorted(tests)

    def _discover_kernel_families(self, family: str) -> list[str]:
        linear_fam_dir = self.repo_root / "src" / "ops" / "linear" / family
        if not linear_fam_dir.exists():
            return []

        k_fams = set()
        for header in linear_fam_dir.glob("*.cuh"):
            name = header.stem.lower()
            if "gemv" in name:
                k_fams.add("gemv")
            if "simt" in name:
                k_fams.add("simt")
            if "sliced" in name:
                k_fams.add("sliced_k")
            if "tma" in name:
                k_fams.add("tma")
            if "mma" in name:
                k_fams.add("mma")
            if "stream" in name:
                k_fams.add("stream")
            if "pair" in name:
                k_fams.add("pair")

        return sorted(k_fams)
