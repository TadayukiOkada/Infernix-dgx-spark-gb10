"""Workload specifications and token/shape sweep generation.

Represents benchmark workloads explicitly across operations, quantization families,
and geometries, avoiding assumptions that every operation is identical.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..util import to_plain


# Default representative sweep covering decode, verification, and early prefill
DEFAULT_TOKEN_SWEEP: list[int] = [
    1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512
]

# Dense decode sweep focusing on single-token to small speculative batch widths
DENSE_DECODE_SWEEP: list[int] = [
    1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32
]


@dataclass(frozen=True)
class WorkloadSpec:
    """Explicit description of a single tuning or benchmark workload."""

    operation: str
    quantization_family: str
    n: int
    k: int
    tokens: int = 1
    policy: str = "a16"
    execution_mode: str = "eager"
    routes: str = "exact"
    test_target: str | None = None
    requires_byte_identity: bool = True
    extra_dims: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return (
            f"{self.operation}_{self.quantization_family}_"
            f"n{self.n}_k{self.k}_t{self.tokens}_{self.policy}_{self.routes}"
        )

    @property
    def geometry_key(self) -> str:
        return f"{self.operation}_{self.quantization_family}_n{self.n}_k{self.k}"

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkloadSpec:
        return cls(
            operation=str(data.get("operation", "linear")),
            quantization_family=str(data.get("quantization_family", "unknown")),
            n=int(data.get("n", 0)),
            k=int(data.get("k", 0)),
            tokens=int(data.get("tokens", 1)),
            policy=str(data.get("policy", "a16")),
            execution_mode=str(data.get("execution_mode", "eager")),
            routes=str(data.get("routes", "exact")),
            test_target=data.get("test_target"),
            requires_byte_identity=bool(data.get("requires_byte_identity", True)),
            extra_dims=dict(data.get("extra_dims", {})),
        )


class WorkloadSweep:
    """Helper to generate token counts and workload parameter sweeps."""

    @staticmethod
    def standard_sweep() -> list[int]:
        return list(DEFAULT_TOKEN_SWEEP)

    @staticmethod
    def dense_decode_sweep() -> list[int]:
        return list(DENSE_DECODE_SWEEP)

    @staticmethod
    def from_string(spec: str) -> list[int]:
        """Parse token sweep string.

        Supports:
        - Comma-separated: "1,2,4,8,16,32"
        - Range with step: "START:END:STEP" (e.g. "1:32:1" or "8:64:8")
        - Range with default step 1: "START:END"
        """
        spec = spec.strip()
        if ":" in spec:
            parts = [int(p.strip()) for p in spec.split(":") if p.strip()]
            if len(parts) == 2:
                start, end = parts
                step = 1
            elif len(parts) == 3:
                start, end, step = parts
            else:
                raise ValueError(f"Invalid range spec '{spec}'; expected START:END[:STEP]")
            if step <= 0:
                raise ValueError(f"Step must be positive in '{spec}'")
            return list(range(start, end + 1, step))
        elif "," in spec:
            return sorted({int(x.strip()) for x in spec.split(",") if x.strip()})
        else:
            return [int(spec)]

    @staticmethod
    def boundary_probe_tokens(
        existing_boundaries: Sequence[int],
        step: int = 1,
        max_tokens: int = 512,
    ) -> list[int]:
        """Generate token probing points centered on known route transitions."""
        pts: set[int] = {1}
        for b in existing_boundaries:
            if b <= 0:
                continue
            if b <= max_tokens:
                pts.add(b)
            if b + step <= max_tokens:
                pts.add(b + step)
            if b - step > 0 and b - step <= max_tokens:
                pts.add(b - step)
        return sorted(pts)
