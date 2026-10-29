"""build_manager — compilation, build caching and compiler resource reports.

Responsibilities:

* Compile CUDA translation units with an explicit, recorded command line.
* Parse ``ptxas -v`` resource reports into structured per-kernel records.
* Cache build products keyed by source digest + flags + arch + compiler version.
* Keep diagnostic builds (``-lineinfo``, ``--keep``, ``-Xptxas=-v``) in a
  separate namespace from production builds, and warn that diagnostic flags can
  change generated code.

The builder never silently reuses an artifact produced with different flags; the
cache key includes the full flag list.
"""

from __future__ import annotations

import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .cache import Cache, CacheKey, source_digest
from .util import LOG, ToolError, ensure_dir, read_json, run, to_plain, which, write_json


@dataclass
class CompilerResourceReport:
    """One ``ptxas -v`` record."""

    symbol: str = ""
    function: str = ""
    registers: int | None = None
    stack_bytes: int | None = None
    spill_stores: int | None = None
    spill_loads: int | None = None
    smem_bytes: int | None = None
    cmem_bytes: int | None = None
    barriers: int | None = None
    source_file: str | None = None

    @property
    def has_spills(self) -> bool:
        return bool(self.spill_stores or self.spill_loads)

    def to_dict(self) -> dict[str, Any]:
        d = to_plain(self)
        d["has_spills"] = self.has_spills
        return d


# ptxas -v output looks like:
#   ptxas info    : Compiling entry function '_Z7k_gemmPKfS0_Pfiii' for 'sm_121a'
#   ptxas info    : Function properties for _Z7k_gemmPKfS0_Pfiii
#       0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
#   ptxas info    : Used 40 registers, used 1 barriers, 2048 bytes smem
_RE_ENTRY = re.compile(r"Compiling entry function '([^']+)' for '([^']+)'")
_RE_FUNC = re.compile(r"Function properties for (\S+)")
_RE_FRAME = re.compile(
    r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads"
)
_RE_USED = re.compile(
    r"Used (\d+) registers(?:, used (\d+) barriers)?(?:, (\d+) bytes smem)?(?:, (\d+) bytes cmem\[0\])?"
)
_RE_FILE = re.compile(r"Function properties for .*\((\S+?)\)")


def parse_ptxas_verbose(text: str) -> list[CompilerResourceReport]:
    """Parse ``-Xptxas=-v`` output into per-function resource records."""
    reports: list[CompilerResourceReport] = []
    current: CompilerResourceReport | None = None
    pending_mangled: str | None = None
    for line in text.splitlines():
        m = _RE_ENTRY.search(line)
        if m:
            current = CompilerResourceReport(symbol=m.group(1))
            reports.append(current)
            continue
        m = _RE_FUNC.search(line)
        if m:
            pending_mangled = m.group(1)
            if current is None or current.symbol != pending_mangled:
                # A "Function properties" line for a function not preceded by a
                # "Compiling entry function" line (device function).
                current = CompilerResourceReport(symbol=pending_mangled)
                reports.append(current)
            continue
        if current is not None:
            m = _RE_FRAME.search(line)
            if m:
                current.stack_bytes = int(m.group(1))
                current.spill_stores = int(m.group(2))
                current.spill_loads = int(m.group(3))
                continue
            m = _RE_USED.search(line)
            if m:
                current.registers = int(m.group(1))
                if m.group(2):
                    current.barriers = int(m.group(2))
                if m.group(3):
                    current.smem_bytes = int(m.group(3))
                if m.group(4):
                    current.cmem_bytes = int(m.group(4))
                continue
            m = _RE_FILE.search(line)
            if m:
                current.source_file = m.group(1)
    return [r for r in reports if r.registers is not None or r.spill_stores is not None]


@dataclass
class BuildResult:
    ok: bool = False
    output: Path | None = None
    command: list[str] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    resources: list[CompilerResourceReport] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    cached: bool = False
    duration_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "output": str(self.output) if self.output else None,
            "command": " ".join(self.command),
            "resources": [r.to_dict() for r in self.resources],
            "errors": self.errors,
            "cached": self.cached,
            "duration_s": self.duration_s,
        }


@dataclass
class BuildConfig:
    """Compilation configuration; every field participates in the cache key."""

    sources: list[str] = field(default_factory=list)
    include_dirs: list[str] = field(default_factory=list)
    library_dirs: list[str] = field(default_factory=list)
    libraries: list[str] = field(default_factory=list)
    defines: list[str] = field(default_factory=list)
    undefines: list[str] = field(default_factory=list)
    arch: str = ""
    gen_codes: list[str] = field(default_factory=list)
    host_flags: list[str] = field(default_factory=list)
    device_flags: list[str] = field(default_factory=list)
    ld_flags: list[str] = field(default_factory=list)
    opt_level: str = "-O3"
    line_info: bool = False
    diagnostic: bool = False  # ptxas -v + keep intermediates
    maxrregcount: int | None = None
    entry_point: str | None = None
    language: str = "c++17"
    extra: list[str] = field(default_factory=list)
    output_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


class BuildManager:
    """Compile CUDA sources with caching and resource-report extraction."""

    def __init__(self, cache: Cache, env_fingerprint: str = ""):
        self.cache = cache
        self.env_fingerprint = env_fingerprint
        self._nvcc = which("nvcc")

    # ------------------------------------------------------------------ #
    @property
    def nvcc(self) -> str:
        if not self._nvcc:
            raise ToolError("nvcc not found; compilation is unavailable")
        return self._nvcc

    def nvcc_version(self) -> str:
        try:
            res = run([self.nvcc, "--version"], timeout=30, check=False, log=False)
            m = re.search(r"release\s+([0-9.]+)", res.stdout)
            return m.group(1) if m else "unknown"
        except ToolError:
            return "unknown"

    # ------------------------------------------------------------------ #
    def command(self, cfg: BuildConfig, output: Path) -> list[str]:
        cmd = [self.nvcc]
        for d in cfg.include_dirs:
            cmd += ["-I", d]
        for d in cfg.defines:
            cmd += ["-D", d]
        for d in cfg.undefines:
            cmd += ["-U", d]
        if cfg.arch:
            cmd.append(f"-arch={cfg.arch}")
        for gc in cfg.gen_codes:
            cmd.append(f"-gencode={gc}")
        cmd.append(cfg.opt_level)
        cmd += [f"-std={cfg.language}"]
        if cfg.line_info:
            cmd.append("-lineinfo")
        if cfg.diagnostic:
            cmd.append("-Xptxas=-v")
        if cfg.maxrregcount:
            cmd.append(f"--maxrregcount={cfg.maxrregcount}")
        if cfg.entry_point:
            cmd.append(f"--entry-point={cfg.entry_point}")
        cmd += cfg.host_flags
        cmd += cfg.device_flags
        cmd += cfg.extra
        cmd += cfg.library_dirs and sum((["-L", d] for d in cfg.library_dirs), []) or []
        cmd += [f"-l{l}" for l in cfg.libraries]
        cmd += ["-o", str(output)]
        cmd += cfg.sources
        cmd += cfg.ld_flags
        return cmd

    # ------------------------------------------------------------------ #
    def build(self, cfg: BuildConfig, *, force: bool = False) -> BuildResult:
        """Compile ``cfg`` (cached) and return the artifact path + resources."""
        if not cfg.sources:
            return BuildResult(ok=False, errors=["no sources supplied"])
        sources = [Path(s) for s in cfg.sources]
        for s in sources:
            if not s.exists():
                return BuildResult(ok=False, errors=[f"source not found: {s}"])

        digest = source_digest(sources)
        key = CacheKey(
            kind="build",
            parts={
                "config": cfg.to_dict(),
                "source_digest": digest,
                "nvcc": self.nvcc_version(),
            },
            env_fingerprint=self.env_fingerprint,
        )
        out_name = cfg.output_name or "artifact"
        result = BuildResult()
        cached_bin = self.cache.path_for(key, out_name)
        cached_json = self.cache.path_for(key, "build.json")

        if cached_bin.exists() and not force:
            data = self.cache.get_json(key, "build.json") or {}
            result.ok = True
            result.output = cached_bin
            result.command = data.get("command", [])
            result.cached = True
            result.resources = [_report_from_dict(d) for d in data.get("resources", [])]
            LOG.info("build cache hit (%s)", key.digest[:10])
            return result

        out_path = self.cache.path_for(key, out_name)
        cmd = self.command(cfg, out_path)
        try:
            res = run(cmd, timeout=1800, check=False, log=True, what="nvcc")
        except ToolError as exc:
            result.errors.append(str(exc))
            return result

        result.command = cmd
        result.stdout = res.stdout
        result.stderr = res.stderr
        result.duration_s = res.duration_s
        result.resources = parse_ptxas_verbose(res.stdout + "\n" + res.stderr)

        if res.returncode != 0 or not out_path.exists():
            result.ok = False
            # nvcc errors look like "file.cu(12): error: message"
            for line in (res.stderr + res.stdout).splitlines():
                if "error" in line.lower():
                    result.errors.append(line.strip())
            if not result.errors:
                result.errors.append(f"nvcc exited {res.returncode}")
            self.cache.record(key, status="failed", detail="; ".join(result.errors[:3]))
            return result

        result.ok = True
        result.output = out_path
        self.cache.put_json(
            key,
            "build.json",
            {"command": cmd, "resources": [r.to_dict() for r in result.resources], "source_digest": digest},
        )
        self.cache.record(
            key,
            status="built",
            detail=out_name,
            extra={"sources": [str(s) for s in sources], "arch": cfg.arch},
        )
        return result

    # ------------------------------------------------------------------ #
    def compile_to_ptx(self, source: Path, cfg: BuildConfig) -> str:
        """Produce PTX for a single source (used for instruction inspection)."""
        cfg = BuildConfig(**{**cfg.to_dict(), "sources": [str(source)], "diagnostic": False})
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out.ptx"
            cmd = self.command(cfg, out)
            cmd = [c for c in cmd if not c.startswith("-o")]
            cmd += ["-ptx", "-o", str(out), str(source)]
            res = run(cmd, timeout=900, check=False, log=False)
            return out.read_text() if out.exists() else res.stderr


def _report_from_dict(d: dict[str, Any]) -> CompilerResourceReport:
    r = CompilerResourceReport()
    for k, v in d.items():
        if hasattr(r, k):
            setattr(r, k, v)
    return r
