"""static_analyzer — source, resource and instruction level analysis.

This module answers questions that are knowable *before* any profiling, and it
labels every answer with its confidence:

* ``measured``  — reported by a compiler or disassembler (facts).
* ``inferred``  — derived from source structure (a reading, not a measurement).
* ``estimated`` — produced by a model with stated assumptions.

Nothing here claims exact memory traffic, arithmetic intensity or occupancy from
source inspection alone, because compiler transformations and cache effects can
change all three.  Those quantities come from the profiler and the performance
model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .binary_inspector import Disassembly, InstructionProfile, profile_instructions
from .source_discovery import KernelDecl, LaunchSite
from .util import to_plain


@dataclass
class Finding:
    """One observation with its provenance."""

    kind: str
    detail: str
    confidence: str = "inferred"  # measured | inferred | estimated
    evidence: str = ""
    location: str = ""

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass
class SourceAnalysis:
    """Source-derived (inferred) properties of one kernel."""

    name: str = ""
    file: str = ""
    line: int = 0
    loops: int = 0
    nested_loop_depth: int = 0
    has_grid_stride: bool = False
    global_loads: int = 0
    global_stores: int = 0
    shared_decls: int = 0
    shared_bytes_declared: int | None = None
    syncthreads: int = 0
    warp_primitives: int = 0
    atomics: int = 0
    tensor_core_intrinsics: int = 0
    fp_ops_approx: int = 0
    branches: int = 0
    vector_types: list[str] = field(default_factory=list)
    uses_restrict: bool = False
    launch_bounds: dict[str, int] | None = None
    template_params: str | None = None
    findings: list[Finding] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


_VEC_TYPES = ("float4", "float2", "int4", "int2", "uint4", "uint2", "double2", "half2", "__half2")
_TC_INTRINSICS = ("wmma", "mma", "wgmma", "__hmma", "cute::", "cublas", "tensor", "tcgen")
_ATOMIC_RE = re.compile(r"\batomic[A-Za-z]+|__threadfence|red\.")
_SYNC_RE = re.compile(r"__syncthreads|__syncwarp|cooperative_groups::|cg::this_thread_block")
_WARP_RE = re.compile(r"__shfl|__ballot|__any|__all|__match|__activemask|__reduce_")
_SHARED_RE = re.compile(r"__shared__\s+([A-Za-z_][A-Za-z0-9_:<>]*)\s+([A-Za-z_][A-Za-z0-9_]*)\s*(\[[^\]]*\])?")
_GLOBAL_PTR_RE = re.compile(r"__global__|__restrict__")
_LOOP_RE = re.compile(r"\bfor\s*\(|\bwhile\s*\(|\bdo\s*\{")
_FP_RE = re.compile(r"\b(fmaf?|fadd|fmul|fsub|fdiv|expf?|logf?|sqrtf?|tanhf?|__expf|__logf|rsqrtf?)\s*\(")


def analyze_source(decl: KernelDecl) -> SourceAnalysis:
    """Infer source-level properties of one kernel.  All results are `inferred`."""
    body = decl.body
    a = SourceAnalysis(
        name=decl.name,
        file=decl.file,
        line=decl.line,
        launch_bounds=decl.launch_bounds,
        template_params=decl.template_params,
    )
    a.loops = len(_LOOP_RE.findall(body))
    a.syncthreads = len(re.findall(r"__syncthreads", body))
    a.warp_primitives = len(_WARP_RE.findall(body))
    a.atomics = len(_ATOMIC_RE.findall(body))
    a.branches = len(re.findall(r"\bif\s*\(|\belse\b|\bswitch\s*\(|\?[^:]", body))
    a.fp_ops_approx = len(_FP_RE.findall(body))
    a.uses_restrict = "__restrict__" in body or "restrict" in decl.signature
    for vt in _VEC_TYPES:
        if re.search(rf"\b{re.escape(vt)}\b", body):
            a.vector_types.append(vt)
    lower = body.lower()
    for t in _TC_INTRINSICS:
        if t in lower:
            a.tensor_core_intrinsics += 1

    # Global memory accesses (approximate: count array subscripts on
    # pointer-typed parameters).
    pointer_params = [n for t, n in zip(decl.param_types, decl.param_names) if "*" in t]
    for p in pointer_params:
        a.global_loads += len(re.findall(rf"\b{re.escape(p)}\s*\[", body))
        a.global_stores += len(re.findall(rf"\b{re.escape(p)}\s*\[[^\]]*\]\s*=", body))

    # Shared declarations and their static size when the extent is literal.
    smem_total = 0
    for m in _SHARED_RE.finditer(body):
        a.shared_decls += 1
        dim = m.group(3)
        if dim:
            nums = re.findall(r"\[(\d+)\]", dim)
            size = 1
            for n in nums:
                size *= int(n)
            # Element size unknown without type resolution; record element count.
            smem_total += size
    a.shared_bytes_declared = smem_total if a.shared_decls else None

    # Grid-stride detection.
    if re.search(r"gridDim\s*\.\s*x\s*\*\s*blockDim\s*\.\s*x", body) and re.search(r"<[^>]*\+=\s*stride", body):
        a.has_grid_stride = True
    elif re.search(r"gid\s*\+=\s*(blockDim|gridDim)", body):
        a.has_grid_stride = True

    # Nested-loop depth (naive brace scan around loops).
    depth = 0
    max_depth = 0
    for token in re.finditer(r"\bfor\s*\(|\bwhile\s*\(|\{|\}", body):
        t = token.group(0)
        if t in ("for (", "while ("):
            depth += 1
            max_depth = max(max_depth, depth)
        elif t == "{":
            pass
        elif t == "}":
            depth = max(0, depth - 1)
    a.nested_loop_depth = max_depth

    # ---- findings ----
    if a.has_grid_stride:
        a.findings.append(
            Finding("structure", "Kernel uses a grid-stride loop: grid size is a free parameter.", "inferred")
        )
    if not a.launch_bounds:
        a.findings.append(
            Finding(
                "resource",
                "No __launch_bounds__: ptxas may raise register usage and lower occupancy "
                "without the kernel author's intent.",
                "inferred",
            )
        )
    if a.atomics:
        a.findings.append(
            Finding("synchronization", f"{a.atomics} atomic/ordered-memory operations found.", "inferred")
        )
    if a.vector_types:
        a.findings.append(
            Finding("memory", f"Vector types used ({', '.join(a.vector_types)}): wide accesses likely.", "inferred")
        )
    else:
        a.findings.append(
            Finding(
                "memory",
                "No vector types in the source. If global traffic is significant, check the SASS for "
                "narrow LDG.E.32/STG.E.32 and consider 128-bit accesses.",
                "inferred",
            )
        )
    if a.shared_decls and a.shared_bytes_declared:
        a.findings.append(
            Finding(
                "memory",
                f"{a.shared_decls} static __shared__ declaration(s), "
                f"~{a.shared_bytes_declared} elements total (element size not resolved).",
                "inferred",
            )
        )
    return a


@dataclass
class ResourceAnalysis:
    """Compiler-reported resource usage (measured facts)."""

    symbol: str = ""
    registers: int | None = None
    stack_bytes: int | None = None
    spill_stores: int | None = None
    spill_loads: int | None = None
    shared_bytes: int | None = None
    local_bytes: int | None = None
    instruction_profile: InstructionProfile | None = None
    findings: list[Finding] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = to_plain(self)
        if self.instruction_profile:
            d["instruction_profile"] = self.instruction_profile.to_dict()
        return d


def analyze_resources(
    symbol: str,
    *,
    registers: int | None = None,
    stack_bytes: int | None = None,
    spill_stores: int | None = None,
    spill_loads: int | None = None,
    shared_bytes: int | None = None,
    local_bytes: int | None = None,
    disassembly: Disassembly | None = None,
) -> ResourceAnalysis:
    """Combine compiler resource data with a SASS instruction profile."""
    r = ResourceAnalysis(
        symbol=symbol,
        registers=registers,
        stack_bytes=stack_bytes,
        spill_stores=spill_stores,
        spill_loads=spill_loads,
        shared_bytes=shared_bytes,
        local_bytes=local_bytes,
    )
    if spill_stores or spill_loads:
        total = (spill_stores or 0) + (spill_loads or 0)
        r.findings.append(
            Finding(
                "resource",
                f"Register spilling detected: {total} bytes. Spills hit local memory (L1/L2/DRAM) "
                f"inside the hot path and usually cost more than the occupancy they buy.",
                "measured",
                evidence=f"spill_stores={spill_stores} spill_loads={spill_loads}",
            )
        )
    if disassembly and disassembly.instructions:
        prof = profile_instructions(disassembly.instructions)
        r.instruction_profile = prof
        r.findings.extend(_findings_from_profile(prof, registers))
    return r


def _findings_from_profile(p: InstructionProfile, registers: int | None) -> list[Finding]:
    out: list[Finding] = []
    if p.has_local_memory:
        out.append(
            Finding(
                "resource",
                f"SASS contains local-memory instructions (LDL={p.local_loads}, STL={p.local_stores}), "
                f"confirming spills reached the instruction stream.",
                "measured",
            )
        )
    if p.global_loads == 0 and p.global_stores == 0:
        out.append(Finding("memory", "No global memory instructions: the kernel is not memory bound.", "measured"))
    else:
        if p.load_widths and max(p.load_widths) < 16 and p.global_loads > 4:
            out.append(
                Finding(
                    "memory",
                    f"Global loads are narrow (max {max(p.load_widths)} B/thread). "
                    f"Widening to 16 B/thread (LDG.E.128) cuts the load instruction count ~4x.",
                    "measured",
                    evidence=f"load_width_histogram={p.load_widths}",
                )
            )
        elif p.load_widths and max(p.load_widths) >= 16:
            out.append(
                Finding(
                    "memory",
                    f"Global loads are already vectorized (up to {max(p.load_widths)} B/thread).",
                    "measured",
                    evidence=f"load_width_histogram={p.load_widths}",
                )
            )
    if p.fp64_ops:
        out.append(
            Finding(
                "compute",
                f"SASS contains {p.fp64_ops} FP64 instruction(s) (DFMA/DMUL/DADD/.F64). On GB10, "
                f"FP64 throughput is ~67x lower than FP32 (~0.42 vs ~27.8 TFLOP/s); verify whether "
                f"double literals or promotions were unintentional.",
                "measured",
                evidence=f"fp64_ops={p.fp64_ops}",
            )
        )
    if p.async_copies:
        out.append(Finding("memory", f"{p.async_copies} cp.async (LDGSTS) operations: asynchronous staging in use.", "measured"))
    if p.tma_ops:
        out.append(Finding("memory", f"{p.tma_ops} TMA operations detected.", "measured"))
    if p.tensor_core_ops:
        tc_detail = (
            f"{p.tensor_core_ops} tensor-core instructions detected "
            f"(int_mma={p.int_tensor_core_ops}, float_mma={p.float_tensor_core_ops})."
        )
        out.append(Finding("compute", tc_detail, "measured"))
    else:
        out.append(
            Finding(
                "compute",
                "No tensor-core instructions. If the workload is FP16/BF16/FP8/INT8 GEMM-like, "
                "a Tensor Core path may be several times faster.",
                "measured",
            )
        )
    if p.shared_loads > 0 and p.shared_load_widths and max(p.shared_load_widths) < 16:
        out.append(
            Finding(
                "memory",
                f"Shared memory loads are narrow (max {max(p.shared_load_widths)} B/thread). "
                f"Widening to 16 B (LDS.128 / LDSM) reduces MIO/LSU wavefront pressure.",
                "measured",
                evidence=f"shared_load_width_histogram={p.shared_load_widths}",
            )
        )
    if p.shuffles:
        out.append(
            Finding(
                "synchronization",
                f"{p.shuffles} warp-shuffle (SHFL) instructions emitted for intra-warp communication.",
                "measured",
            )
        )
    if p.uniform_ops:
        out.append(
            Finding(
                "compute",
                f"{p.uniform_ops} uniform-datapath instructions (UIADD/ULDC/ULOP) emitted, reducing vector register and ALU pressure.",
                "measured",
            )
        )
    # Integer address arithmetic share.
    if p.total:
        addr_ops = sum(v for k, v in p.counts.items() if k.startswith(("IMAD", "IADD3", "IADD", "SHF", "LOP3", "LEA")))
        share = addr_ops / p.total
        if share > 0.2:
            out.append(
                Finding(
                    "compute",
                    f"Address/integer arithmetic is {share*100:.0f}% of all instructions. "
                    f"Hoisting index math out of the inner loop (pointer induction) can remove it.",
                    "measured",
                    evidence=f"integer_ops={addr_ops}/{p.total}",
                )
            )
    if registers is not None and registers > 128:
        out.append(
            Finding(
                "resource",
                f"High register count ({registers}/thread) limits resident warps; verify that the "
                f"extra registers buy ILP or a larger thread tile before trimming.",
                "measured",
            )
        )
    return out


@dataclass
class LaunchAnalysis:
    """Launch geometry from source, plus the inferred grid shape."""

    site: LaunchSite | None = None
    grid_expr: str = ""
    block_expr: str = ""
    shared_mem_expr: str | None = None
    stream_expr: str | None = None
    findings: list[Finding] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


def analyze_launch(site: LaunchSite) -> LaunchAnalysis:
    a = LaunchAnalysis(
        site=site,
        grid_expr=site.grid,
        block_expr=site.block,
        shared_mem_expr=site.shared_mem,
        stream_expr=site.stream,
    )
    if not site.stream:
        a.findings.append(
            Finding("launch", "Launch does not specify a stream (uses the default stream).", "inferred", location=f"{site.file}:{site.line}")
        )
    # A literal block size is worth checking against the warp/block limits.
    m = re.fullmatch(r"\s*(\d+)\s*", site.block or "")
    if m:
        bt = int(m.group(1))
        if bt % 32 != 0:
            a.findings.append(
                Finding(
                    "launch",
                    f"Block size {bt} is not a multiple of the warp size (32); the last warp is partial.",
                    "inferred",
                    location=f"{site.file}:{site.line}",
                )
            )
        if bt > 1024:
            a.findings.append(
                Finding("launch", f"Block size {bt} exceeds the 1024-thread hardware limit.", "inferred")
            )
    return a
