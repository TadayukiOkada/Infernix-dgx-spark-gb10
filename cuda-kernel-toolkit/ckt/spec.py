"""Architecture-specific facts that the CUDA runtime does *not* report.

The device probe supplies every measurable quantity (SM count, register file,
cache sizes, feature gates).  This module supplies only the constants that are
properties of the ISA rather than the individual board:

* register allocation granularity (warps are allocated in fixed blocks of
  registers, which is why trimming one register per thread can change occupancy
  by a whole resident block or not at all),
* shared-memory allocation granularity,
* peak dense throughput per data type,
* whether the naive ``bus_width * mem_clock`` bandwidth formula is meaningful on
  this class of device.

Anything not listed for a compute capability is reported as unknown rather than
guessed: ``None`` propagates and the reporting layer prints ``n/a``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ArchFacts:
    """ISA-level constants for one compute-capability generation."""

    family: str
    cc: tuple[int, int]
    # Register allocation: hardware grants registers to a warp in units of
    # `reg_alloc_unit_warp` registers (256 = 8 regs/thread x 32 lanes on
    # CC 7.0+).  Per-thread counts are also rounded up to `reg_alloc_step`.
    reg_alloc_step: int
    reg_alloc_unit_warp: int
    smem_alloc_unit_bytes: int
    # Peak dense throughput (FP32 accumulation) in TFLOP/s is a *board*
    # property, not an ISA one; it is left None here and filled from a measured
    # or explicitly configured value.  The ISA table only records whether a
    # Tensor Core path exists for a dtype.
    tensor_dtypes: tuple[str, ...] = ()
    # True when memory_bus_width/memory_clock_rate describe a real dedicated
    # bus.  False on unified-memory SoCs where the numbers are not meaningful.
    dedicated_memory_bus: bool = True
    notes: str = ""


_ARCH_TABLE: dict[tuple[int, int], ArchFacts] = {
    (7, 0): ArchFacts("Volta", (7, 0), 8, 256, 256, ("fp16",), True),
    (7, 5): ArchFacts("Turing", (7, 5), 8, 256, 256, ("fp16", "int8"), True),
    (8, 0): ArchFacts("Ampere-A100", (8, 0), 8, 256, 128, ("fp16", "bf16", "tf32", "int8"), True),
    (8, 6): ArchFacts("Ampere", (8, 6), 8, 256, 128, ("fp16", "bf16", "tf32", "int8"), True),
    (8, 9): ArchFacts("Ada", (8, 9), 8, 256, 128, ("fp16", "bf16", "tf32", "int8", "fp8"), True),
    (9, 0): ArchFacts("Hopper", (9, 0), 8, 256, 128, ("fp16", "bf16", "tf32", "int8", "fp8"), True),
    (10, 0): ArchFacts("Blackwell", (10, 0), 8, 256, 128, ("fp16", "bf16", "tf32", "int8", "fp8", "fp4"), True),
    (10, 3): ArchFacts("Blackwell", (10, 3), 8, 256, 128, ("fp16", "bf16", "tf32", "int8", "fp8", "fp4"), True),
    (11, 0): ArchFacts("Blackwell", (11, 0), 8, 256, 128, ("fp16", "bf16", "tf32", "int8", "fp8", "fp4"), True),
    (12, 0): ArchFacts(
        "Blackwell-GeForce",
        (12, 0),
        8,
        256,
        128,
        ("fp16", "bf16", "tf32", "int8", "fp8", "fp4"),
        True,
        "Consumer Blackwell; dedicated GDDR7 bus.",
    ),
    (12, 1): ArchFacts(
        "Blackwell-DGXSpark",
        (12, 1),
        8,
        256,
        128,
        ("fp16", "bf16", "tf32", "int8", "fp8", "fp4"),
        False,
        "GB10 Grace Blackwell / DGX Spark: unified LPDDR5X. "
        "memory_bus_width and memory_clock_rate do not describe the real bus; "
        "bandwidth MUST be measured.",
    ),
}


def arch_facts(major: int, minor: int) -> ArchFacts | None:
    """Look up ISA facts, falling back to the nearest lower minor revision."""
    key = (int(major), int(minor))
    if key in _ARCH_TABLE:
        return _ARCH_TABLE[key]
    candidates = [k for k in _ARCH_TABLE if k[0] == key[0] and k[1] <= key[1]]
    if candidates:
        return _ARCH_TABLE[max(candidates)]
    # Unknown generation: fall back to the newest family's allocation rules,
    # which have been stable since CC 7.x, and flag the uncertainty.
    return ArchFacts(
        f"Unknown(CC {major}.{minor})",
        key,
        8,
        256,
        128,
        (),
        True,
        "Unrecognised compute capability; register/shared allocation assumed "
        "CC 7.0+ defaults. Verify occupancy predictions against a measurement.",
    )


@dataclass
class DTypePeak:
    """Peak sustained throughput for one data type, with provenance."""

    dtype: str
    tflops: float | None
    source: str  # "measured" | "vendor-spec" | "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {"dtype": self.dtype, "tflops": self.tflops, "source": self.source}
