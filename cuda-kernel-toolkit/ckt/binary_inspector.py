"""binary_inspector — cuobjdump / nvdisasm integration with per-function filtering.

Design constraints that shape this module (all verified against the installed
CUDA 13.0 toolchain on this host, and documented in ``docs/limitations.md``):

* ``cuobjdump --function`` matches the *stored symbol exactly*.  A short or
  mangled-only name returns exit code 0 with **no instructions**, so every
  extraction verifies a nonzero instruction count before it is trusted.
* This build's ``cuobjdump`` has **no ``regex:`` function filter**.
* An unfiltered ``cuobjdump -sass`` on a large linked executable can emit
  gigabytes; the inspector therefore always filters, and it extracts a tiny
  cubin first when the binary format does not support filtering directly.
* Every extraction is cached, keyed by binary digest + tool version + symbol.

The module separates *extraction* from *analysis*: the same cached disassembly
serves the static analyzer, the SASS-pattern checks, and the report writer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .cache import Cache, CacheKey
from .source_discovery import ResourceUsage, parse_res_usage
from .util import LOG, ToolError, file_sha256, read_json, run, to_plain, which, write_json


@dataclass
class SASSInstruction:
    address: str
    opcode: str
    predicate: str | None
    operands: str
    raw: str

    @property
    def is_global_load(self) -> bool:
        return self.opcode.startswith("LDG")

    @property
    def is_global_store(self) -> bool:
        return self.opcode.startswith("STG")

    @property
    def is_shared_load(self) -> bool:
        return self.opcode.startswith("LDS")

    @property
    def is_shared_store(self) -> bool:
        return self.opcode.startswith("STS")

    @property
    def is_local(self) -> bool:
        return self.opcode.startswith(("LDL", "STL"))

    @property
    def is_async_copy(self) -> bool:
        return self.opcode.startswith("LDGSTS") or self.opcode.startswith("CP.ASYNC")

    @property
    def is_tma(self) -> bool:
        return self.opcode.startswith(("UTMALDG", "UTMASTG", "UBLKCP", "TMA"))

    @property
    def is_tensor_core(self) -> bool:
        return self.opcode.startswith(("HMMA", "IMMA", "OMMA", "QMMA", "BMMA", "WGMMA", "MMA", "UTMMA"))

    @property
    def is_int_tensor_core(self) -> bool:
        return self.opcode.startswith(("IMMA", "BMMA"))

    @property
    def is_fp8_tensor_core(self) -> bool:
        return self.opcode.startswith("QMMA") or (".E4M3" in self.opcode or ".E5M2" in self.opcode or ".E3M2" in self.opcode)

    @property
    def is_fp4_tensor_core(self) -> bool:
        return self.opcode.startswith("OMMA") or ".E2M1" in self.opcode

    @property
    def is_float_tensor_core(self) -> bool:
        return self.opcode.startswith(("HMMA", "WGMMA", "MMA", "UTMMA"))

    @property
    def is_fp64(self) -> bool:
        return self.opcode.startswith(("DFMA", "DMUL", "DADD", "DSETP", "DMNMX")) or ".F64" in self.opcode

    @property
    def is_shuffle(self) -> bool:
        return self.opcode.startswith("SHFL")

    @property
    def is_uniform(self) -> bool:
        return (
            self.opcode.startswith(("UIADD", "UIMAD", "ULDC", "UMOV", "ULOP", "UFLO", "UISETP", "USHF", "UBREV", "VOTEU"))
        )

    @property
    def is_mufu(self) -> bool:
        return self.opcode.startswith("MUFU")

    @property
    def is_depbar(self) -> bool:
        return self.opcode.startswith("DEPBAR")

    @property
    def is_sync_barrier(self) -> bool:
        return self.opcode.startswith(("BAR", "BSSY", "BSYNC", "WARPSYNC"))

    @property
    def is_barrier(self) -> bool:
        return self.opcode.startswith("BAR") or self.opcode.startswith("DEPBAR")

    @property
    def is_atomic(self) -> bool:
        return self.opcode.startswith(("ATOM", "ATOMS", "RED"))

    @property
    def is_alu(self) -> bool:
        return self.opcode.startswith(("FADD", "FMUL", "FFMA", "FMA", "IADD", "IMAD", "IADD3", "LOP3", "SHF", "ISETP", "FSETP", "MOV", "SEL"))

    @property
    def width_bytes(self) -> int | None:
        """Access width in bytes for a memory instruction.

        SASS omits the width suffix when the access is the default 32-bit form,
        so ``LDG.E`` and ``STG.E`` mean 4 bytes.  Returning ``None`` for those
        (as an "unknown width") makes a scalar kernel look as though it had no
        memory traffic at all, which inverts the very comparison the inspector
        exists to make: the vectorised kernel must show 16-byte accesses and the
        scalar one 4-byte accesses.
        """
        if not self.is_global_load and not self.is_global_store and not self.is_global:
            if not (self.is_shared_load or self.is_shared_store or self.is_local):
                return None
        # An explicit width suffix always wins.  Match on the suffix chain so
        # that `.E.128` and `.E.U8` are read the same way.
        m = re.search(r"\.(\d+)\b", self.opcode)
        if m:
            bits = int(m.group(1))
            if bits in (8, 16, 32, 64, 128, 256):
                return bits // 8
        # `LDG.E.U8` style: the width appears as a type suffix.
        m2 = re.search(r"\.U?(8|16|32|64)\b", self.opcode)
        if m2:
            return int(m2.group(1)) // 8
        # No suffix at all: the ISA default for a memory access is 32 bits.
        return 4

    @property
    def is_global(self) -> bool:
        return self.opcode.startswith(("LDG", "STG", "LD", "ST"))


@dataclass
class Disassembly:
    symbol: str
    arch: str
    instructions: list[SASSInstruction] = field(default_factory=list)
    registers_used: int | None = None
    header: str = ""
    raw: str = ""
    source: str = "cuobjdump"

    def to_dict(self, *, include_instructions: bool = True) -> dict[str, Any]:
        d = {
            "symbol": self.symbol,
            "arch": self.arch,
            "instruction_count": len(self.instructions),
            "registers_used": self.registers_used,
            "source": self.source,
        }
        if include_instructions:
            d["instructions"] = [i.raw for i in self.instructions]
        return d

    @property
    def profile(self) -> "InstructionProfile":
        """Instruction mix for this disassembly (computed on first access)."""
        cached = getattr(self, "_profile_cache", None)
        if cached is None:
            cached = profile_instructions(self.instructions)
            object.__setattr__(self, "_profile_cache", cached)
        return cached


@dataclass
class InstructionProfile:
    """Aggregated instruction mix for one kernel."""

    total: int = 0
    counts: dict[str, int] = field(default_factory=dict)
    global_loads: int = 0
    global_stores: int = 0
    shared_loads: int = 0
    shared_stores: int = 0
    local_loads: int = 0
    local_stores: int = 0
    async_copies: int = 0
    tma_ops: int = 0
    tensor_core_ops: int = 0
    int_tensor_core_ops: int = 0
    float_tensor_core_ops: int = 0
    fp8_tensor_core_ops: int = 0
    fp4_tensor_core_ops: int = 0
    fp64_ops: int = 0
    shuffles: int = 0
    uniform_ops: int = 0
    mufu_ops: int = 0
    barriers: int = 0
    sync_barriers: int = 0
    depbars: int = 0
    atomics: int = 0
    alu_ops: int = 0
    load_widths: dict[int, int] = field(default_factory=dict)
    store_widths: dict[int, int] = field(default_factory=dict)
    shared_load_widths: dict[int, int] = field(default_factory=dict)
    shared_store_widths: dict[int, int] = field(default_factory=dict)
    branches: int = 0

    @property
    def has_local_memory(self) -> bool:
        return self.local_loads > 0 or self.local_stores > 0

    @property
    def max_load_width(self) -> int:
        return max(self.load_widths) if self.load_widths else 0

    @property
    def max_shared_load_width(self) -> int:
        return max(self.shared_load_widths) if self.shared_load_widths else 0

    def to_dict(self) -> dict[str, Any]:
        d = to_plain(self)
        d["has_local_memory"] = self.has_local_memory
        d["max_load_width"] = self.max_load_width
        d["max_shared_load_width"] = self.max_shared_load_width
        return d


def profile_instructions(instructions: list[SASSInstruction]) -> InstructionProfile:
    p = InstructionProfile()
    for ins in instructions:
        p.total += 1
        p.counts[ins.opcode] = p.counts.get(ins.opcode, 0) + 1
        if ins.is_fp64:
            p.fp64_ops += 1
        if ins.is_shuffle:
            p.shuffles += 1
        if ins.is_uniform:
            p.uniform_ops += 1
        if ins.is_mufu:
            p.mufu_ops += 1
        if ins.is_depbar:
            p.depbars += 1
        if ins.is_sync_barrier:
            p.sync_barriers += 1
        if ins.is_global_load:
            p.global_loads += 1
            if ins.width_bytes:
                p.load_widths[ins.width_bytes] = p.load_widths.get(ins.width_bytes, 0) + 1
        elif ins.is_global_store:
            p.global_stores += 1
            if ins.width_bytes:
                p.store_widths[ins.width_bytes] = p.store_widths.get(ins.width_bytes, 0) + 1
        elif ins.is_shared_load:
            p.shared_loads += 1
            if ins.width_bytes:
                p.shared_load_widths[ins.width_bytes] = p.shared_load_widths.get(ins.width_bytes, 0) + 1
        elif ins.is_shared_store:
            p.shared_stores += 1
            if ins.width_bytes:
                p.shared_store_widths[ins.width_bytes] = p.shared_store_widths.get(ins.width_bytes, 0) + 1
        elif ins.opcode.startswith("LDL"):
            p.local_loads += 1
        elif ins.opcode.startswith("STL"):
            p.local_stores += 1
        elif ins.is_async_copy:
            p.async_copies += 1
        elif ins.is_tma:
            p.tma_ops += 1
        elif ins.is_tensor_core:
            p.tensor_core_ops += 1
            if ins.is_fp4_tensor_core:
                p.fp4_tensor_core_ops += 1
            elif ins.is_fp8_tensor_core:
                p.fp8_tensor_core_ops += 1
            elif ins.is_int_tensor_core:
                p.int_tensor_core_ops += 1
            elif ins.is_float_tensor_core:
                p.float_tensor_core_ops += 1
        elif ins.is_barrier:
            p.barriers += 1
        elif ins.is_atomic:
            p.atomics += 1
        elif ins.is_alu:
            p.alu_ops += 1
        if ins.opcode.startswith(("BRA", "BSSY", "BSYNC", "JMP", "RET", "EXIT", "@")):
            if ins.opcode.startswith(("BRA", "JMP")):
                p.branches += 1
    return p


def compare_profiles(
    prof_a: InstructionProfile,
    prof_b: InstructionProfile,
    *,
    regs_a: int | None = None,
    regs_b: int | None = None,
    shared_a: int | None = None,
    shared_b: int | None = None,
    stack_a: int | None = None,
    stack_b: int | None = None,
) -> dict[str, Any]:
    """Compute exact instruction-mix and compiler-resource deltas (B minus A)."""
    fields = (
        "total",
        "global_loads",
        "global_stores",
        "shared_loads",
        "shared_stores",
        "local_loads",
        "local_stores",
        "async_copies",
        "tma_ops",
        "tensor_core_ops",
        "int_tensor_core_ops",
        "float_tensor_core_ops",
        "fp8_tensor_core_ops",
        "fp4_tensor_core_ops",
        "fp64_ops",
        "shuffles",
        "uniform_ops",
        "mufu_ops",
        "barriers",
        "sync_barriers",
        "depbars",
        "atomics",
        "alu_ops",
        "branches",
        "max_load_width",
        "max_shared_load_width",
    )
    deltas: dict[str, dict[str, Any]] = {}
    for f in fields:
        va = int(getattr(prof_a, f, 0) or 0)
        vb = int(getattr(prof_b, f, 0) or 0)
        diff = vb - va
        pct = (100.0 * diff / va) if va != 0 else (100.0 if diff > 0 else 0.0)
        deltas[f] = {"before": va, "after": vb, "delta": diff, "pct_change": round(pct, 2)}

    res_deltas: dict[str, dict[str, Any]] = {}
    for name, va, vb in (
        ("registers", regs_a, regs_b),
        ("shared_bytes", shared_a, shared_b),
        ("stack_bytes", stack_a, stack_b),
    ):
        if va is not None or vb is not None:
            diff = (vb - va) if (va is not None and vb is not None) else None
            pct = (100.0 * diff / va) if (diff is not None and va) else None
            res_deltas[name] = {
                "before": va,
                "after": vb,
                "delta": diff,
                "pct_change": round(pct, 2) if pct is not None else None,
            }
    return {
        "resources": res_deltas,
        "instructions": deltas,
        "load_widths_before": dict(prof_a.load_widths),
        "load_widths_after": dict(prof_b.load_widths),
        "shared_load_widths_before": dict(prof_a.shared_load_widths),
        "shared_load_widths_after": dict(prof_b.shared_load_widths),
    }


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #

# nvdisasm / cuobjdump SASS lines:
#        /*0000*/                   IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;
#        /*0010*/              @P0 LDG.E.128 R4, [R2.64] ;
_SASS_RE = re.compile(
    r"^\s*/\*(?P<addr>[0-9a-fA-F]+)\*/\s*(?:@!?(?P<pred>\w+)\s+)?(?P<op>[A-Z][A-Z0-9._]*)\s*(?P<ops>.*?)\s*;\s*$"
)
_SASS_RE_NOSEMI = re.compile(
    r"^\s*/\*(?P<addr>[0-9a-fA-F]+)\*/\s*(?:@!?(?P<pred>\w+)\s+)?(?P<op>[A-Z][A-Z0-9._]*)\s*(?P<ops>.*?)\s*$"
)
_HEADER_RE = re.compile(r"^\s*//-+\s*(?P<name>.+?)\s*-+\s*$")
_REG_HEADER_RE = re.compile(r"REG:(\d+)")


def parse_sass(text: str) -> list[SASSInstruction]:
    """Parse SASS text into instructions, tolerating both semicolon styles."""
    out: list[SASSInstruction] = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("//") or s.startswith("/*") and "*/" not in s:
            continue
        if s.startswith("//") or s.startswith("."):
            continue
        m = _SASS_RE.match(line) or _SASS_RE_NOSEMI.match(line)
        if not m:
            continue
        out.append(
            SASSInstruction(
                address=m.group("addr"),
                opcode=m.group("op"),
                predicate=m.group("pred"),
                operands=m.group("ops"),
                raw=s,
            )
        )
    return out


def arch_from_text(text: str) -> str:
    m = re.search(r"(sm_\d+\w*)", text)
    return m.group(1) if m else "unknown"


# --------------------------------------------------------------------------- #
# Inspector
# --------------------------------------------------------------------------- #


@dataclass
class InspectorInfo:
    cuobjdump: str | None = None
    nvdisasm: str | None = None
    versions: dict[str, str] = field(default_factory=dict)
    features: dict[str, bool] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


class BinaryInspector:
    """Cached, function-filtered access to compiled CUDA binaries."""

    def __init__(self, cache: Cache, *, binary_digest: str | None = None):
        self.cache = cache
        self._binary_digest = binary_digest
        self.info = InspectorInfo(
            cuobjdump=which("cuobjdump"),
            nvdisasm=which("nvdisasm"),
        )
        self._probe_features()

    # ------------------------------------------------------------------ #
    def _probe_features(self) -> None:
        """Verify option support rather than assuming flags across releases."""
        if self.info.cuobjdump:
            res = run([self.info.cuobjdump, "--help"], timeout=30, check=False, log=False)
            help_text = res.stdout + res.stderr
            self.info.features["function_filter"] = "-function" in help_text or "-fun " in help_text
            self.info.features["regex_filter"] = "regex:" in help_text
            self.info.features["res_usage"] = "-res-usage" in help_text
            self.info.features["extract_elf"] = "-xelf" in help_text
            m = re.search(r"CUDA \(?([0-9.]+)", help_text)
            self.info.versions["cuobjdump"] = m.group(1) if m else "unknown"
            if self.info.features["function_filter"] and not self.info.features["regex_filter"]:
                self.info.warnings.append(
                    "cuobjdump has no regex: function filter; the toolkit resolves "
                    "exact stored symbols before extracting SASS."
                )
        else:
            self.info.warnings.append("cuobjdump unavailable; SASS inspection disabled.")
        if self.info.nvdisasm:
            res = run([self.info.nvdisasm, "--help"], timeout=30, check=False, log=False)
            t = res.stdout + res.stderr
            self.info.features["nvdisasm_function"] = "--function" in t
            self.info.features["nvdisasm_print_code"] = "-c" in t

    # ------------------------------------------------------------------ #
    def _digest(self, binary: Path) -> str:
        if self._binary_digest:
            return self._binary_digest
        return file_sha256(binary)

    def _key(self, binary: Path, kind: str, parts: dict[str, Any]) -> CacheKey:
        return CacheKey(
            kind="disasm",
            parts={"kind": kind, "binary": file_sha256(binary), **parts},
        )

    # ------------------------------------------------------------------ #
    def resource_usage(self, binary: str | Path, *, force: bool = False) -> dict[str, ResourceUsage]:
        """Return per-symbol REG/STACK/SHARED/LOCAL for the whole binary.

        One ``cuobjdump -res-usage`` pass (~1 s on this host) is far cheaper than
        per-symbol SASS extraction and gives the exact stored symbol names, which
        the rest of the inspector needs.
        """
        binary = Path(binary)
        if not self.info.cuobjdump:
            raise ToolError("cuobjdump unavailable")
        key = self._key(binary, "res-usage", {"tool": self.info.versions.get("cuobjdump")})
        cached = self.cache.get_json(key, "res-usage.json")
        if cached and not force:
            return {k: _ru_from_dict(v) for k, v in cached.items()}
        res = run([self.info.cuobjdump, "-res-usage", str(binary)], timeout=600, check=True, log=False)
        parsed = parse_res_usage(res.stdout)
        self.cache.put_json(key, "res-usage.json", {k: v.to_dict() for k, v in parsed.items()})
        return parsed

    def list_symbols(self, binary: str | Path, *, force: bool = False) -> list[str]:
        """Exact stored text-symbol names, in a stable order."""
        return sorted(self.resource_usage(binary, force=force).keys())

    # ------------------------------------------------------------------ #
    def disassemble(
        self,
        binary: str | Path,
        symbol: str,
        *,
        force: bool = False,
        with_line_info: bool = False,
    ) -> Disassembly:
        """Disassemble exactly one symbol, verifying a nonzero instruction count."""
        binary = Path(binary)
        if not self.info.cuobjdump:
            raise ToolError("cuobjdump unavailable")
        key = self._key(
            binary,
            "sass",
            {"symbol": symbol, "line_info": with_line_info, "tool": self.info.versions.get("cuobjdump")},
        )
        cached = self.cache.get_json(key, "sass.json")
        if cached and not force:
            dis = Disassembly(
                symbol=cached["symbol"],
                arch=cached.get("arch", "unknown"),
                instructions=parse_sass(cached.get("raw", "")),
                registers_used=cached.get("registers_used"),
                raw=cached.get("raw", ""),
            )
            if dis.instructions:
                return dis

        flags = ["-sass"]
        if with_line_info:
            flags.append("-lineinfo")
        cmd = [self.info.cuobjdump, "--function", symbol, *flags, str(binary)]
        res = run(cmd, timeout=900, check=False, log=False)
        text = res.stdout
        instructions = parse_sass(text)
        arch = arch_from_text(text)
        if not instructions:
            # Distinguish "tool failed" from "symbol matched nothing": both look
            # the same (exit 0, no instructions) and must never be treated as a
            # valid extraction.
            raise ToolError(
                f"cuobjdump produced no instructions for symbol '{symbol}'. "
                f"This usually means the symbol name is not the exact stored "
                f"symbol (exit {res.returncode}). Verify with list_symbols().",
                result=res,
            )
        regs = None
        m = _REG_HEADER_RE.search(text)
        if m:
            regs = int(m.group(1))
        dis = Disassembly(symbol=symbol, arch=arch, instructions=instructions, registers_used=regs, raw=text)
        self.cache.put_json(
            key,
            "sass.json",
            {
                "symbol": symbol,
                "arch": arch,
                "registers_used": regs,
                "instruction_count": len(instructions),
                "raw": text,
            },
        )
        return dis

    # ------------------------------------------------------------------ #
    def disassemble_via_cubin(self, binary: str | Path, symbol: str, arch_pattern: str = "sm_") -> Disassembly:
        """Extract a cubin then disassemble with nvdisasm (fallback path).

        Used when ``cuobjdump --function`` is unavailable or when nvdisasm's
        control-flow / register-life annotations are needed.
        """
        binary = Path(binary)
        if not self.info.cuobjdump or not self.info.nvdisasm:
            raise ToolError("cuobjdump/nvdisasm unavailable")
        key = self._key(binary, "nvdisasm", {"symbol": symbol, "arch": arch_pattern})
        cached = self.cache.get_json(key, "nvdisasm.json")
        if cached:
            return Disassembly(
                symbol=symbol,
                arch=cached.get("arch", "unknown"),
                instructions=parse_sass(cached.get("raw", "")),
                raw=cached.get("raw", ""),
                source="nvdisasm",
            )
        work = self.cache.dir_for(key)
        res = run([self.info.cuobjdump, "-xelf", "all", str(binary)], timeout=900, check=False, log=False, cwd=work)
        cubins = sorted(work.glob(f"*.{arch_pattern}*")) or sorted(work.glob("*.cubin"))
        if not cubins:
            raise ToolError(f"cuobjdump -xelf produced no cubin in {work}")
        cubin = cubins[0]
        res = run([self.info.nvdisasm, "--function", symbol, "-c", str(cubin)], timeout=900, check=False, log=False)
        text = res.stdout
        instructions = parse_sass(text)
        if not instructions:
            raise ToolError(f"nvdisasm produced no instructions for '{symbol}' in {cubin.name}")
        dis = Disassembly(symbol=symbol, arch=arch_from_text(text) or cubin.suffix, instructions=instructions, raw=text, source="nvdisasm")
        self.cache.put_json(key, "nvdisasm.json", {"symbol": symbol, "arch": dis.arch, "raw": text})
        return dis

    # ------------------------------------------------------------------ #
    def resolve_symbol(self, binary: str | Path, kernel_name: str) -> list[str]:
        """Map a source-level kernel name to exact stored symbols.

        Matching is deliberately loose at the name level and records every
        candidate: templates and overloads legitimately share a bare name.
        """
        symbols = self.list_symbols(binary)
        out: list[str] = []
        for s in symbols:
            if _symbol_matches(s, kernel_name):
                out.append(s)
        return out


def _symbol_matches(symbol: str, kernel_name: str) -> bool:
    """Heuristic C++ symbol -> bare kernel name match for one name."""
    if symbol == kernel_name:
        return True
    # File-static kernels look like:
    #   _ZN51_GLOBAL__N__83ff31e1_10_k_saxpy_cu_e2ec47e8...
    # The bare name often appears verbatim inside the mangled form.
    if kernel_name in symbol:
        # Guard against "k_read" matching "k_read_stream" in a substring sense
        # by requiring a non-identifier character (or string edge) after it.
        idx = symbol.find(kernel_name)
        while idx != -1:
            end = idx + len(kernel_name)
            nxt = symbol[end] if end < len(symbol) else ""
            if not (nxt.isalnum() or nxt == "_") or symbol[end:].startswith(("IL", "EE")):
                return True
            idx = symbol.find(kernel_name, idx + 1)
        return True
    return False


def _ru_from_dict(d: dict[str, Any]) -> ResourceUsage:
    return ResourceUsage(
        symbol=d.get("symbol", ""),
        reg=d.get("reg"),
        stack=d.get("stack"),
        shared=d.get("shared"),
        local=d.get("local"),
    )
