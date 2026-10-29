"""cache — content-addressed artifact storage with toolchain-aware invalidation.

Every expensive artifact (compiled binary, disassembly, symbol table, profiler
report, measured timing) is keyed by a hash of *everything* that determines it:
source content, compiler flags, architecture, and tool versions.  A cache hit is
therefore never a coincidence of filename.

Layout::

    <cache_root>/
        env/                    environment snapshots
        build/<key>/            compiled artifacts + logs
        disasm/<key>/           cuobjdump / nvdisasm output
        profile/<key>/          raw ncu / nsys output + parsed metrics
        timing/<key>.json       measured launch timings
        index.jsonl             append-only provenance log

The index is append-only so an interrupted run never corrupts prior provenance.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .util import (
    LOG,
    append_jsonl,
    ensure_dir,
    file_sha256,
    mtime_sha,
    read_json,
    read_jsonl,
    stable_hash,
    to_plain,
    write_json,
)


@dataclass
class CacheKey:
    """Everything that must match for an artifact to be reusable."""

    kind: str
    parts: dict[str, Any]
    env_fingerprint: str = ""

    @property
    def digest(self) -> str:
        return stable_hash({"kind": self.kind, "env": self.env_fingerprint, "parts": self.parts}, length=20)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "env": self.env_fingerprint, "parts": to_plain(self.parts), "digest": self.digest}


def source_digest(paths: list[Path]) -> str:
    """Combined content digest of a source set (order-independent)."""
    entries = []
    for p in sorted(set(paths)):
        if p.is_file():
            entries.append((p.name, file_sha256(p)))
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and f.suffix in (".cu", ".cuh", ".h", ".hpp", ".cpp"):
                    entries.append((str(f.relative_to(p)), file_sha256(f)))
    return stable_hash(entries, length=24)


class Cache:
    """Filesystem cache rooted at a directory."""

    def __init__(self, root: str | Path):
        self.root = ensure_dir(root)
        self.index_path = self.root / "index.jsonl"

    # ------------------------------------------------------------------ #
    def dir_for(self, key: CacheKey) -> Path:
        return ensure_dir(self.root / key.kind / key.digest)

    def path_for(self, key: CacheKey, name: str) -> Path:
        return self.dir_for(key) / name

    def has(self, key: CacheKey, name: str) -> bool:
        return self.path_for(key, name).exists()

    def record(self, key: CacheKey, *, status: str, detail: str = "", extra: dict[str, Any] | None = None) -> None:
        append_jsonl(
            self.index_path,
            {
                "ts": time.time(),
                "kind": key.kind,
                "digest": key.digest,
                "status": status,
                "detail": detail,
                **(extra or {}),
            },
        )

    # ------------------------------------------------------------------ #
    def get_or_compute(
        self,
        key: CacheKey,
        name: str,
        producer: Callable[[Path], Any],
        *,
        force: bool = False,
    ) -> Path:
        """Return the cached artifact path, computing it via ``producer`` on miss.

        ``producer`` receives the destination path and must write it.  The
        artifact is written to a temporary name first so an interrupted producer
        cannot leave a truncated file that later looks like a cache hit.
        """
        target = self.path_for(key, name)
        if target.exists() and not force:
            LOG.debug("cache hit: %s", target)
            return target
        tmp = target.with_suffix(target.suffix + ".part")
        try:
            producer(tmp)
        except Exception:
            if tmp.exists():
                tmp.unlink()
            raise
        tmp.replace(target)
        self.record(key, status="computed", detail=name)
        return target

    def get_json(self, key: CacheKey, name: str) -> Any | None:
        p = self.path_for(key, name)
        if p.exists():
            try:
                return read_json(p)
            except (OSError, ValueError) as exc:  # pragma: no cover
                LOG.warning("corrupt cache entry %s: %s", p, exc)
        return None

    def put_json(self, key: CacheKey, name: str, obj: Any) -> Path:
        p = write_json(self.path_for(key, name), obj)
        self.record(key, status="stored", detail=name)
        return p

    # ------------------------------------------------------------------ #
    def entries(self) -> list[dict[str, Any]]:
        return read_jsonl(self.index_path)

    def stats(self) -> dict[str, Any]:
        entries = self.entries()
        by_status: dict[str, int] = {}
        for e in entries:
            by_status[e.get("status", "?")] = by_status.get(e.get("status", "?"), 0) + 1
        return {"entries": len(entries), "by_status": by_status, "root": str(self.root)}

    def clear(self, kind: str | None = None) -> None:
        """Remove cached artifacts (never the provenance index)."""
        import shutil

        target = self.root / kind if kind else self.root
        if kind:
            if target.exists():
                shutil.rmtree(target)
        else:
            for child in self.root.iterdir():
                if child.name == "index.jsonl":
                    continue
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()


@dataclass
class ProvenanceStamp:
    """Everything needed to reproduce and trust one measurement."""

    env_fingerprint: str = ""
    source_digest: str = ""
    binary_digest: str = ""
    compiler: str = ""
    flags: list[str] = field(default_factory=list)
    arch: str = ""
    workload: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    command: str = ""
    cache_digest: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)
