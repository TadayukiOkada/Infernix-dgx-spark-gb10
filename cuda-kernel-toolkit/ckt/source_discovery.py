"""source_discovery — kernel discovery and source <-> symbol mapping.

Two independent questions are answered separately, because the answers differ:

* **What kernels exist in the sources?**  Parsed from ``.cu``/``.cuh`` with a
  brace-matching scanner (not a regex over the whole file) so template
  declarations, attributes and multi-line signatures are handled.
* **What kernels exist in the executable, and what actually runs?**  Read from
  ``cuobjdump --list-text`` and from the profiler's launch counts.

The toolkit never assumes a source kernel is present in the binary, nor that a
compiled kernel is executed by a given workload.  The inventory records presence
and execution separately.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .util import LOG, ToolError, read_json, run, which, write_json

# --------------------------------------------------------------------------- #
# Source parsing
# --------------------------------------------------------------------------- #

_GLOBAL_RE = re.compile(r"__global__\b")
_DEVICE_RE = re.compile(r"__device__\b")
_HOST_RE = re.compile(r"__host__\b")


@dataclass
class KernelDecl:
    """A kernel declaration found in a source file."""

    name: str
    qualified_name: str
    file: str
    line: int
    signature: str
    return_type: str
    template_params: str | None = None
    is_extern_c: bool = False
    namespace: str | None = None
    attributes: list[str] = field(default_factory=list)
    launch_bounds: dict[str, int] | None = None
    param_types: list[str] = field(default_factory=list)
    param_names: list[str] = field(default_factory=list)
    body: str = ""
    body_start_line: int = 0
    # Populated later.
    mangled_symbols: list[str] = field(default_factory=list)
    instantiated: bool = False

    @property
    def symbol_hint(self) -> str:
        return self.name

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d.pop("body", None)  # keep reports small
        return d

    def to_json(self, *, include_body: bool = False) -> dict[str, Any]:
        d = dict(self.__dict__)
        if not include_body:
            d.pop("body", None)
        return d


@dataclass
class LaunchSite:
    """A ``kernel<<<grid, block, smem, stream>>>(...)`` occurrence."""

    kernel: str
    file: str
    line: int
    grid: str
    block: str
    shared_mem: str | None = None
    stream: str | None = None
    raw: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class KernelInventoryEntry:
    """One kernel, with static analysis + binary presence + runtime counts."""

    name: str
    source: KernelDecl | None = None
    mangled_symbols: list[str] = field(default_factory=list)
    in_binary: bool = False
    launch_count: int = 0
    launch_configs: list[LaunchSite] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source.to_json() if self.source else None,
            "mangled_symbols": self.mangled_symbols,
            "in_binary": self.in_binary,
            "launch_count": self.launch_count,
            "launch_configs": [c.to_dict() for c in self.launch_configs],
        }


def _strip_comments(text: str) -> str:
    """Remove // and /* */ comments while preserving line structure."""
    out = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if c == "/" and nxt == "/":
            while i < n and text[i] != "\n":
                i += 1
        elif c == "/" and nxt == "*":
            i += 2
            while i < n and not (text[i] == "*" and i + 1 < n and text[i + 1] == "/"):
                if text[i] == "\n":
                    out.append("\n")
                i += 1
            i += 2
        elif c in ('"', "'"):
            quote = c
            out.append(c)
            i += 1
            while i < n and text[i] != quote:
                if text[i] == "\\" and i + 1 < n:
                    out.append(text[i])
                    i += 1
                out.append(text[i])
                i += 1
            if i < n:
                out.append(text[i])
                i += 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _match_brace(text: str, start: int) -> int:
    """Index just past the ``}`` matching the ``{`` at ``start`` (-1 if none)."""
    depth = 0
    i = start
    n = len(text)
    while i < n:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return -1


def _parse_params(param_str: str) -> tuple[list[str], list[str]]:
    """Split a parameter list into types and names, honouring nested <> and ())."""
    params: list[str] = []
    depth = 0
    angle = 0
    cur = []
    for ch in param_str:
        if ch in "(<[":
            depth += 1 if ch in "([" else 0
            angle += 1 if ch == "<" else 0
        elif ch in ")>]":
            depth -= 1 if ch in ")]" else 0
            angle -= 1 if ch == ">" else 0
        if ch == "," and depth == 0 and angle == 0:
            params.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur:
        tail = "".join(cur).strip()
        if tail:
            params.append(tail)

    types, names = [], []
    for p in params:
        if not p or p == "void":
            continue
        p = re.sub(r"^\s*(const|volatile|restrict|__restrict__|__restrict)\s+", "", p).strip()
        m = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*(\[[^\]]*\])*\s*$", p)
        name = m.group(1) if m else ""
        idx = p.rfind(name) if name else len(p)
        types.append(p[:idx].strip() if name else p)
        names.append(name)
    return types, names


_LAUNCH_BOUNDS_RE = re.compile(r"__launch_bounds__\s*\(\s*([^)]*)\)")


def discover_kernels_in_source(path: str | Path) -> tuple[list[KernelDecl], list[LaunchSite]]:
    """Parse one CUDA translation unit for kernels and launch sites."""
    p = Path(path)
    raw = p.read_text(errors="replace")
    text = _strip_comments(raw)
    line_starts = [0]
    for m in re.finditer(r"\n", text):
        line_starts.append(m.end())

    def lineno(pos: int) -> int:
        import bisect

        return bisect.bisect_right(line_starts, pos)

    kernels: list[KernelDecl] = []
    namespace_at: dict[int, str] = {}

    # Track namespace nesting so qualified names are correct.
    ns_stack: list[tuple[str, int]] = []
    for m in re.finditer(r"\bnamespace\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{", text):
        ns_stack.append((m.group(1), _match_brace(text, text.index("{", m.start()))))
    # Simple approach below recomputes namespace per kernel position.

    for m in _GLOBAL_RE.finditer(text):
        # Find the end of the declaration: first '{' at brace depth 0 or a ';'.
        i = m.end()
        # Skip attributes and the return type / name / params.
        paren_depth = 0
        decl_end = -1
        body_start = -1
        n = len(text)
        while i < n:
            c = text[i]
            if c == "(":
                paren_depth += 1
            elif c == ")":
                paren_depth -= 1
            elif c == "{" and paren_depth == 0:
                body_start = i
                decl_end = i
                break
            elif c == ";" and paren_depth == 0:
                decl_end = i
                break
            i += 1
        if decl_end < 0:
            continue
        decl = text[m.end() : decl_end]
        # Locate the last identifier before '(' -> kernel name.
        lm = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", decl)
        if not lm:
            continue
        name = lm.group(1)
        # Skip C++ keywords that could match.
        if name in ("if", "for", "while", "switch", "return"):
            continue
        before_name = decl[: lm.start(1)]
        # Template parameters preceding the declaration.
        tparams = None
        tm = re.search(r"template\s*<", before_name)
        if tm:
            depth = 0
            j = before_name.index("<", tm.start())
            k = j
            while k < len(before_name):
                if before_name[k] == "<":
                    depth += 1
                elif before_name[k] == ">":
                    depth -= 1
                    if depth == 0:
                        break
                k += 1
            tparams = before_name[j + 1 : k].strip()
            before_name = before_name[: tm.start()]

        # Return type = tokens between __global__ and name, minus attributes.
        ret = before_name
        ret = _LAUNCH_BOUNDS_RE.sub("", ret)
        ret = re.sub(r"__device__|__host__|__global__|__forceinline__|inline|__inline__", "", ret)
        ret = re.sub(r"extern\s+\"C\"", "", ret)
        extern_c = 'extern "C"' in decl[: lm.start(1)]
        # Strip leading '::' and whitespace.
        ret = ret.strip()
        return_type = ret.split()[-1] if ret.split() else "void"

        # Parameters.
        ps = decl.index("(", lm.start(1))
        depth = 0
        pe = ps
        for k in range(ps, len(decl)):
            if decl[k] == "(":
                depth += 1
            elif decl[k] == ")":
                depth -= 1
                if depth == 0:
                    pe = k
                    break
        param_types, param_names = _parse_params(decl[ps + 1 : pe])

        # Body.
        body = ""
        body_line = 0
        if body_start >= 0:
            end = _match_brace(text, body_start)
            if end > 0:
                body = text[body_start:end]
                body_line = lineno(body_start)

        # Namespace: find the innermost namespace enclosing this position.
        ns = None
        for nm, close in reversed(ns_stack):
            if m.start() < close:
                ns = nm
                break
        qualified = f"{ns}::{name}" if ns else name

        lb = None
        lbm = _LAUNCH_BOUNDS_RE.search(text[max(0, m.start() - 400) : decl_end])
        if lbm:
            parts = [x.strip() for x in lbm.group(1).split(",")]
            lb = {}
            # The values may be named constants (e.g. `kThreads`), so store the
            # literal expression and only add an int when it truly is one.
            if parts and parts[0]:
                lb["max_threads_per_block_expr"] = parts[0]
                if parts[0].isdigit():
                    lb["max_threads_per_block"] = int(parts[0])
            if len(parts) > 1 and parts[1]:
                lb["min_blocks_per_multiprocessor_expr"] = parts[1]
                if parts[1].isdigit():
                    lb["min_blocks_per_multiprocessor"] = int(parts[1])

        kernels.append(
            KernelDecl(
                name=name,
                qualified_name=qualified,
                file=str(p),
                line=lineno(m.start()),
                signature=decl.strip(),
                return_type=return_type,
                template_params=tparams,
                is_extern_c=extern_c,
                namespace=ns,
                launch_bounds=lb,
                param_types=param_types,
                param_names=param_names,
                body=body,
                body_start_line=body_line,
            )
        )

    launches = _find_launch_sites(text, str(p), lineno)
    return kernels, launches


_LAUNCH_RE = re.compile(
    r"(?P<name>[A-Za-z_][A-Za-z0-9_:<>]*)\s*<<<\s*(?P<cfg>[^>]*(?:>(?!>)[^>]*)*?)>>>\s*\(",
    re.S,
)


def _find_launch_sites(text: str, path: str, lineno) -> list[LaunchSite]:
    sites: list[LaunchSite] = []
    for m in _LAUNCH_RE.finditer(text):
        cfg = m.group("cfg")
        parts = _split_launch_config(cfg)
        sites.append(
            LaunchSite(
                kernel=m.group("name").strip(),
                file=path,
                line=lineno(m.start()),
                grid=parts[0] if len(parts) > 0 else "",
                block=parts[1] if len(parts) > 1 else "",
                shared_mem=parts[2] if len(parts) > 2 else None,
                stream=parts[3] if len(parts) > 3 else None,
                raw=m.group(0),
            )
        )
    return sites


def _split_launch_config(cfg: str) -> list[str]:
    out: list[str] = []
    depth = 0
    cur: list[str] = []
    for ch in cfg:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur:
        out.append("".join(cur).strip())
    return out


def discover_kernels(paths: Iterable[str | Path]) -> tuple[list[KernelDecl], list[LaunchSite]]:
    """Discover kernels across a set of files or directories."""
    all_kernels: list[KernelDecl] = []
    all_launches: list[LaunchSite] = []
    for path in _expand(paths):
        try:
            ks, ls = discover_kernels_in_source(path)
        except (OSError, UnicodeDecodeError) as exc:
            LOG.warning("could not parse %s: %s", path, exc)
            continue
        all_kernels.extend(ks)
        all_launches.extend(ls)
    return all_kernels, all_launches


def _expand(paths: Iterable[str | Path]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            out.extend(sorted(path.rglob("*.cu")))
            out.extend(sorted(path.rglob("*.cuh")))
        elif path.exists():
            out.append(path)
        else:
            LOG.warning("source path does not exist: %s", path)
    return out


# --------------------------------------------------------------------------- #
# Binary symbol enumeration
# --------------------------------------------------------------------------- #


@dataclass
class BinarySymbol:
    symbol: str  # exact stored (possibly mangled) name
    demangled: str
    kernel_name: str
    source: str | None = None
    static: bool = False

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


_DEMANGLE_STRIP = re.compile(
    r"^\((?P<params>.*)\)\s*$|^\((?P<params2>.*)\)$"
)


def _kernel_name_from_demangled(demangled: str) -> str:
    """Extract the bare kernel name from a cuobjdump demangled signature."""
    # e.g. "k_read(const uint4 *, unsigned long long, float *)"
    name = demangled.split("(", 1)[0]
    name = name.split("::")[-1]
    return name.strip()


def list_binary_symbols(binary: str | Path, *, timeout: float = 300.0) -> list[BinarySymbol]:
    """Enumerate text symbols in a fat binary via ``cuobjdump --list-text``.

    ``--list-text <partial>`` requires a name filter, so the call is made with a
    broad-but-bounded pattern and the output parsed.  When that yields nothing
    the toolkit falls back to ``-res-usage``, which lists every stored symbol
    together with its register/shared-memory footprint in a single pass.
    """
    exe = which("cuobjdump")
    if not exe:
        raise ToolError("cuobjdump not available; cannot enumerate binary symbols")
    binary = Path(binary)
    if not binary.exists():
        raise ToolError(f"binary not found: {binary}")

    symbols: list[BinarySymbol] = []
    res = run([exe, "-res-usage", str(binary)], timeout=timeout, check=False, log=False)
    if res.ok:
        symbols.extend(_parse_res_usage(res.stdout))
    if not symbols:
        LOG.warning("cuobjdump -res-usage produced no symbols for %s", binary)
    return symbols


_RES_USAGE_RE = re.compile(
    r"Function\s+(?P<sym>\S+)\s*\n(?P<body>(?:.*\n)*?)(?=Function\s|\Z)",
)
_REG_LINE = re.compile(r"REG:\s*(\d+)")
_STACK_LINE = re.compile(r"STACK:\s*(\d+)")
_SHARED_LINE = re.compile(r"SHARED:\s*(\d+)")
_LOCAL_LINE = re.compile(r"LOCAL:\s*(\d+)")


@dataclass
class ResourceUsage:
    symbol: str
    reg: int | None = None
    stack: int | None = None
    shared: int | None = None
    local: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def parse_res_usage(text: str) -> dict[str, ResourceUsage]:
    """Parse ``cuobjdump -res-usage`` into per-symbol resource records."""
    out: dict[str, ResourceUsage] = {}
    # Records look like:
    #   Function _Z6k_readPK6uint4yPf:
    #   ...
    #           REG:64 STACK:0 SHARED:0 LOCAL:0
    cur: str | None = None
    for line in text.splitlines():
        # CUDA 13 prints ` Function <mangled>:`; older tools omit the colon.
        # `\S+?` (lazy) plus the optional colon keeps the colon OUT of the symbol,
        # which matters because `--function` matches the stored name exactly.
        m = re.match(r"\s*Function\s+(\S+?):?\s*$", line)
        if m:
            cur = m.group(1)
            out.setdefault(cur, ResourceUsage(symbol=cur))
            continue
        if cur and "REG:" in line:
            ru = out[cur]
            if (mm := _REG_LINE.search(line)):
                ru.reg = int(mm.group(1))
            if (mm := _STACK_LINE.search(line)):
                ru.stack = int(mm.group(1))
            if (mm := _SHARED_LINE.search(line)):
                ru.shared = int(mm.group(1))
            if (mm := _LOCAL_LINE.search(line)):
                ru.local = int(mm.group(1))
    return out


def _parse_res_usage(text: str) -> list[BinarySymbol]:
    out: list[BinarySymbol] = []
    for sym in parse_res_usage(text):
        out.append(
            BinarySymbol(
                symbol=sym,
                demangled=sym,
                kernel_name=_kernel_name_from_demangled(sym),
                static="__nv_static_" in sym,
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Inventory construction
# --------------------------------------------------------------------------- #


def build_inventory(
    kernels: list[KernelDecl],
    binary_symbols: list[BinarySymbol],
    launches: list[LaunchSite] | None = None,
    launch_counts: dict[str, int] | None = None,
) -> list[KernelInventoryEntry]:
    """Associate source kernels, binary symbols and observed launch counts.

    Matching is by bare kernel name, which is a heuristic: overloads and
    templates share a name.  Every match is therefore recorded with the full
    symbol list rather than pretending a 1:1 mapping.
    """
    launches = launches or []
    launch_counts = launch_counts or {}
    by_name: dict[str, list[BinarySymbol]] = {}
    for s in binary_symbols:
        by_name.setdefault(s.kernel_name, []).append(s)

    entries: list[KernelInventoryEntry] = []
    seen: set[str] = set()
    for k in kernels:
        syms = by_name.get(k.name, [])
        entry = KernelInventoryEntry(
            name=k.name,
            source=k,
            mangled_symbols=[s.symbol for s in syms],
            in_binary=bool(syms),
            launch_count=launch_counts.get(k.name, 0),
            launch_configs=[l for l in launches if l.kernel.split("<")[0].endswith(k.name)],
        )
        k.mangled_symbols = entry.mangled_symbols
        k.instantiated = entry.in_binary
        entries.append(entry)
        seen.add(k.name)

    # Kernels present in the binary but not found in the supplied sources.
    for name, syms in sorted(by_name.items()):
        if name in seen:
            continue
        if not name or name.startswith("__"):
            continue
        entries.append(
            KernelInventoryEntry(
                name=name,
                source=None,
                mangled_symbols=[s.symbol for s in syms],
                in_binary=True,
                launch_count=launch_counts.get(name, 0),
                launch_configs=[l for l in launches if l.kernel.split("<")[0].endswith(name)],
            )
        )
    return entries


@dataclass
class SourceSet:
    roots: list[str] = field(default_factory=list)
    include_dirs: list[str] = field(default_factory=list)
    kernels: list[KernelDecl] = field(default_factory=list)
    launches: list[LaunchSite] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "roots": self.roots,
            "include_dirs": self.include_dirs,
            "kernels": [k.to_json() for k in self.kernels],
            "launches": [l.to_dict() for l in self.launches],
        }
