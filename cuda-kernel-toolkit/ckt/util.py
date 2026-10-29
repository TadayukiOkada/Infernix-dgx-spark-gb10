"""Shared utilities: process execution, hashing, logging, structured IO.

Everything here is deliberately dependency-free so the toolkit runs on a bare
Python 3.11+ interpreter.  All external process invocations funnel through
:func:`run` so that timeouts, error capture and logging stay consistent, and so
that the expensive-tool calls (``ncu``, ``cuobjdump``) are easy to audit.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

LOG = logging.getLogger("ckt")


class ToolError(RuntimeError):
    """Raised when an external tool fails in a way the caller cannot ignore."""

    def __init__(self, message: str, *, result: "ProcResult | None" = None):
        super().__init__(message)
        self.result = result


class ToolMissing(ToolError):
    """Raised when a required external tool is not on PATH."""


@dataclass
class ProcResult:
    """Outcome of one external process invocation."""

    argv: list[str]
    returncode: int
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    @property
    def command(self) -> str:
        return " ".join(shlex.quote(a) for a in self.argv)

    def raise_for_status(self, what: str) -> "ProcResult":
        if self.timed_out:
            raise ToolError(f"{what} timed out after {self.duration_s:.1f}s: {self.command}", result=self)
        if self.returncode != 0:
            tail = (self.stderr or self.stdout or "").strip().splitlines()[-8:]
            raise ToolError(
                f"{what} failed (exit {self.returncode}): {self.command}\n" + "\n".join(tail),
                result=self,
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "returncode": self.returncode,
            "duration_s": round(self.duration_s, 4),
            "timed_out": self.timed_out,
            "stdout_tail": self.stdout[-4000:],
            "stderr_tail": self.stderr[-4000:],
        }


def run(
    argv: Sequence[str],
    *,
    timeout: float | None = 600.0,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    stdin: str | None = None,
    check: bool = True,
    log: bool = True,
    what: str | None = None,
) -> ProcResult:
    """Run ``argv`` capturing output, with an explicit timeout.

    Never uses a shell, so callers must pass argument lists (safe against
    spaces and globs).  ``check=False`` returns the result even on failure.
    """
    argv = [str(a) for a in argv]
    label = what or Path(argv[0]).name
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    if log:
        LOG.debug("exec: %s", " ".join(shlex.quote(a) for a in argv))
    began = time.monotonic()
    timed_out = False
    try:
        proc = subprocess.run(
            argv,
            cwd=str(cwd) if cwd else None,
            env=full_env,
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        rc, out, err = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        rc = -9
        out = _decode(exc.stdout)
        err = _decode(exc.stderr)
    except FileNotFoundError as exc:
        raise ToolMissing(f"{label}: executable not found ({argv[0]})") from exc
    duration = time.monotonic() - began
    result = ProcResult(argv, rc, out or "", err or "", duration, timed_out)
    if check and not result.ok:
        result.raise_for_status(label)
    return result


def _decode(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def which(tool: str) -> str | None:
    """Resolve ``tool`` on PATH (or as an absolute path) without a shell."""
    import shutil

    return shutil.which(tool)


def require_tool(tool: str) -> str:
    path = which(tool)
    if not path:
        raise ToolMissing(f"required tool '{tool}' not found on PATH")
    return path


# --------------------------------------------------------------------------- #
# Hashing / identity
# --------------------------------------------------------------------------- #


def file_sha256(path: str | Path, *, limit: int | None = None) -> str:
    """Content hash of a file (or of its first ``limit`` bytes)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        remaining = limit
        while True:
            chunk = fh.read(1 << 20 if remaining is None else min(1 << 20, remaining))
            if not chunk:
                break
            h.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
                if remaining <= 0:
                    break
    return h.hexdigest()


def stable_hash(obj: Any, *, length: int = 16) -> str:
    """Deterministic hash of a JSON-serialisable object (key order normalised)."""
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:length]


def mtime_sha(path: str | Path) -> str:
    """Cheap fingerprint: size + mtime.  Used for cache invalidation."""
    st = os.stat(path)
    return f"{st.st_size}:{int(st.st_mtime)}"


# --------------------------------------------------------------------------- #
# Structured IO
# --------------------------------------------------------------------------- #


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, obj: Any) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=False, default=str) + "\n")
    tmp.replace(p)
    return p


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def write_text(path: str | Path, text: str) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(text)
    return p


def append_jsonl(path: str | Path, obj: Any) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, default=str) + "\n")


def read_jsonl(path: str | Path) -> list[Any]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #


def percentile(values: Sequence[float], pct: float) -> float:
    """Linear-interpolation percentile; matches numpy's default method."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (pct / 100.0) * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def stdev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return (sum((v - m) ** 2 for v in values) / (len(values) - 1)) ** 0.5


@dataclass
class SampleStats:
    """Summary of a timing sample set."""

    n: int
    median: float
    p10: float
    p90: float
    p99: float
    minimum: float
    maximum: float
    mean: float
    stdev: float

    @property
    def cv(self) -> float:
        """Coefficient of variation (relative dispersion)."""
        return (self.stdev / self.mean) if self.mean else float("nan")

    @property
    def iqr(self) -> float:
        return self.p90 - self.p10

    @classmethod
    def from_values(cls, values: Sequence[float]) -> "SampleStats":
        vals = list(values)
        return cls(
            n=len(vals),
            median=percentile(vals, 50),
            p10=percentile(vals, 10),
            p90=percentile(vals, 90),
            p99=percentile(vals, 99),
            minimum=min(vals) if vals else float("nan"),
            maximum=max(vals) if vals else float("nan"),
            mean=mean(vals),
            stdev=stdev(vals),
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "n": self.n,
            "median": self.median,
            "p10": self.p10,
            "p90": self.p90,
            "p99": self.p99,
            "min": self.minimum,
            "max": self.maximum,
            "mean": self.mean,
            "stdev": self.stdev,
            "cv": self.cv,
            "iqr": self.iqr,
        }


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #


_LOG_CONFIGURED = False


def setup_logging(verbosity: int = 1, *, logfile: str | Path | None = None) -> None:
    """Configure the root ``ckt`` logger.

    ``verbosity``: 0=warnings, 1=info, 2=debug, 3=trace (debug + subprocess cmds).
    """
    global _LOG_CONFIGURED
    level = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}.get(verbosity, logging.DEBUG)
    LOG.handlers.clear()
    fmt = logging.Formatter("[%(levelname)-7s] %(message)s" if verbosity < 2 else "[%(levelname)-7s] %(name)s: %(message)s")
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(fmt)
    LOG.addHandler(stream)
    if logfile:
        ensure_dir(Path(logfile).parent)
        fh = logging.FileHandler(str(logfile))
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)-7s] %(message)s"))
        LOG.addHandler(fh)
    LOG.setLevel(level)
    LOG.propagate = False
    _LOG_CONFIGURED = True


# --------------------------------------------------------------------------- #
# Misc
# --------------------------------------------------------------------------- #


def chunked(seq: Sequence[Any], size: int) -> Iterable[list[Any]]:
    for i in range(0, len(seq), size):
        yield list(seq[i : i + size])


def human_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0:
            return f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}PiB"


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into ``base`` (override wins)."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def to_plain(obj: Any) -> Any:
    """Recursively convert dataclasses/paths/tuples into JSON-safe primitives."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {str(k): to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_plain(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float):
        if obj != obj:
            return None
        return obj
    return obj
