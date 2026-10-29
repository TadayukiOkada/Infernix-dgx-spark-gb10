"""experiment_manager — reproducible runs, provenance, and resumable iteration.

An experiment record must answer, without ambiguity: *what* was measured, on
*which* hardware and toolchain, at *which* workload size, under *which*
configuration, and whether the result is still valid.  Anything less and a
comparison across sessions is meaningless.

The central rule enforced here is **cache-invalidation by identity**: a stored
measurement is only reusable when the source digest, compiler flags, target
architecture, tool versions, hardware fingerprint, workload, and launch
configuration all match.  A result from a different binary or a different GPU is
never silently reused — it is reported as stale and re-measured.
"""

from __future__ import annotations

import json
import platform
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from .util import (
    append_jsonl,
    ensure_dir,
    read_json,
    read_jsonl,
    stable_hash,
    write_json,
    write_text,
)


EXPERIMENT_SCHEMA = "ckt.experiment/1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# Identity
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ExperimentIdentity:
    """The tuple that determines whether two measurements are comparable.

    Two results may be compared only when these fields are equal.  The identity
    deliberately excludes the *outcome* (timing, counters) and includes only the
    inputs that could change it.
    """

    source_digest: str = ""
    build_flags: str = ""
    arch: str = ""
    nvcc_version: str = ""
    ncu_version: str = ""
    device_fingerprint: str = ""
    workload: str = ""
    problem_size: str = ""
    launch_config: str = ""
    iterations: int = 0
    warmup: int = 0

    def digest(self) -> str:
        return stable_hash(self.__dict__, length=16)

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["identity_digest"] = self.digest()
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "ExperimentIdentity":
        fields = {k: v for k, v in d.items() if k in ExperimentIdentity.__dataclass_fields__}
        return ExperimentIdentity(**fields)  # type: ignore[arg-type]


@dataclass
class ComparabilityCheck:
    comparable: bool = False
    differences: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"comparable": self.comparable, "differences": self.differences, "notes": self.notes}


#: Fields whose difference makes two results non-comparable, with an explanation
#: of why.  Ordered so the most fundamental difference is reported first.
CRITICAL_FIELDS: list[tuple[str, str]] = [
    ("device_fingerprint", "measured on different hardware"),
    ("arch", "compiled for a different architecture"),
    ("source_digest", "the kernel source or its includes changed"),
    ("build_flags", "compiled with different flags"),
    ("nvcc_version", "compiled with a different toolchain"),
    ("workload", "different workload"),
    ("problem_size", "different problem size, so the work is not the same"),
    ("launch_config", "different launch configuration"),
    ("ncu_version", "profiled with a different NCU version"),
]

#: Fields that are informational but do not invalidate a comparison.
ADVISORY_FIELDS: list[tuple[str, str]] = [
    ("iterations", "measured over a different number of iterations"),
    ("warmup", "measured with a different warm-up length"),
]


def check_comparability(a: ExperimentIdentity, b: ExperimentIdentity) -> ComparabilityCheck:
    """Decide whether two experiment records may be compared directly."""
    c = ComparabilityCheck()
    for f, why in CRITICAL_FIELDS:
        va, vb = getattr(a, f), getattr(b, f)
        if va != vb:
            c.differences.append(f"{f}: {va!r} vs {vb!r} — {why}")
    for f, why in ADVISORY_FIELDS:
        va, vb = getattr(a, f), getattr(b, f)
        if va != vb:
            c.notes.append(f"{f}: {va!r} vs {vb!r} — {why}; dispersion may differ")
    c.comparable = not c.differences
    if not c.comparable:
        c.notes.append(
            "These results must not be compared. Differences at this level invalidate a "
            "speedup claim; re-measure both under one identity."
        )
    return c


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #


@dataclass
class ExperimentRecord:
    name: str = ""
    identity: ExperimentIdentity = field(default_factory=ExperimentIdentity)
    created_at: str = _now()
    status: str = "created"          # created | running | complete | failed
    outcome: str = "unknown"         # improved | regressed | neutral | failed | unknown
    hypothesis: str = ""
    change: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    schema: str = EXPERIMENT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "name": self.name,
            "created_at": self.created_at,
            "status": self.status,
            "outcome": self.outcome,
            "hypothesis": self.hypothesis,
            "change": self.change,
            "identity": self.identity.to_dict(),
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "notes": self.notes,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "ExperimentRecord":
        r = ExperimentRecord(name=d.get("name", ""))
        r.identity = ExperimentIdentity.from_dict(d.get("identity") or {})
        r.created_at = d.get("created_at", "")
        r.status = d.get("status", "unknown")
        r.outcome = d.get("outcome", "unknown")
        r.hypothesis = d.get("hypothesis", "")
        r.change = d.get("change", "")
        r.metrics = d.get("metrics") or {}
        r.artifacts = d.get("artifacts") or {}
        r.notes = d.get("notes") or []
        r.schema = d.get("schema", EXPERIMENT_SCHEMA)
        return r


# --------------------------------------------------------------------------- #
# Manager
# --------------------------------------------------------------------------- #


@dataclass
class ResumeState:
    """What a subsequent run needs in order to continue rather than restart."""

    experiment_dir: str = ""
    completed: list[str] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)
    stale: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_dir": self.experiment_dir,
            "completed": self.completed,
            "pending": self.pending,
            "stale": self.stale,
            "notes": self.notes,
        }


class ExperimentManager:
    """Owns the on-disk layout and provenance for a set of experiments."""

    def __init__(self, root: str | Path):
        self.root = ensure_dir(root)
        self._index_path = self.root / "experiments.jsonl"

    # ------------------------------------------------------------------ #
    @property
    def index_path(self) -> Path:
        return self._index_path

    def dir_for(self, name: str) -> Path:
        safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in name)[:64]
        return ensure_dir(self.root / safe)

    def record(self, rec: ExperimentRecord) -> Path:
        """Append a record to the index and write its own JSON file."""
        d = self.dir_for(rec.name)
        path = d / "experiment.json"
        write_json(path, rec.to_dict())
        # The append-only index is the source of truth for history; a record is
        # never edited in place, so a failed run stays in the history.
        append_jsonl(self._index_path, {"name": rec.name, "path": str(path), **rec.to_dict()})
        return path

    def load(self, name: str) -> ExperimentRecord | None:
        p = self.dir_for(name) / "experiment.json"
        if not p.exists():
            return None
        return ExperimentRecord.from_dict(read_json(p))

    def history(self) -> list[ExperimentRecord]:
        out: list[ExperimentRecord] = []
        for entry in read_jsonl(self._index_path):
            inner = entry.get("experiment") if "experiment" in entry else entry
            out.append(ExperimentRecord.from_dict(inner))
        return out

    def find_comparable(
        self, identity: ExperimentIdentity, *, exclude_name: str | None = None
    ) -> list[tuple[ExperimentRecord, ComparabilityCheck]]:
        """Every prior record that can be compared with ``identity``.

        Records that differ in any critical field are returned too, but marked
        non-comparable, so a caller can report *why* a comparison was refused
        rather than silently finding nothing.
        """
        out: list[tuple[ExperimentRecord, ComparabilityCheck]] = []
        seen: set[str] = set()
        for rec in self.history():
            if rec.name in seen:
                continue
            seen.add(rec.name)
            if exclude_name and rec.name == exclude_name:
                continue
            out.append((rec, check_comparability(identity, rec.identity)))
        return out

    def resume_state(
        self, planned: Sequence[str], identity: ExperimentIdentity
    ) -> ResumeState:
        """Describe which planned experiments are done, pending, or now stale."""
        st = ResumeState(experiment_dir=str(self.root))
        prior: dict[str, ExperimentRecord] = {}
        for rec in self.history():
            prior[rec.name] = rec  # later entries win
        for name in planned:
            rec = prior.get(name)
            if rec is None:
                st.pending.append(name)
                continue
            if rec.status != "complete":
                st.pending.append(name)
                st.notes.append(f"{name}: previously recorded as '{rec.status}', will re-run")
                continue
            cc = check_comparability(identity, rec.identity)
            if cc.comparable:
                st.completed.append(name)
            else:
                st.stale.append(name)
                st.notes.append(f"{name}: stored result is stale — {cc.differences[0]}")
        if st.stale:
            st.notes.append(
                f"{len(st.stale)} experiment(s) must be re-measured because the identity changed; "
                "their cached numbers are not comparable with the current configuration."
            )
        return st

    # ------------------------------------------------------------------ #
    def measurement_valid(
        self,
        name: str,
        identity: ExperimentIdentity,
        *,
        max_age_s: float | None = None,
    ) -> tuple[bool, str]:
        """Whether a stored measurement for ``name`` may be reused as-is."""
        rec = self.load(name)
        if rec is None:
            return False, "no stored record"
        if rec.status != "complete":
            return False, f"stored record status is '{rec.status}'"
        cc = check_comparability(identity, rec.identity)
        if not cc.comparable:
            return False, cc.differences[0]
        if max_age_s is not None:
            try:
                age = time.time() - datetime.fromisoformat(rec.created_at).timestamp()
            except (ValueError, TypeError):
                return False, "record has no usable timestamp"
            if age > max_age_s:
                return False, f"record is {age / 3600:.1f}h old, beyond the {max_age_s / 3600:.1f}h limit"
        return True, "reusable"

    def prune(self, *, keep: Sequence[str] = (), dry_run: bool = True) -> list[str]:
        """List (or remove) experiment directories not in ``keep``."""
        keep_set = set(keep)
        removed: list[str] = []
        for child in sorted(self.root.iterdir()):
            if not child.is_dir() or child.name in keep_set:
                continue
            removed.append(str(child))
            if not dry_run:
                shutil.rmtree(child, ignore_errors=True)
        return removed

    # ------------------------------------------------------------------ #
    def write_summary(self, records: Sequence[ExperimentRecord], path: str | Path | None = None) -> Path:
        """Write a compact progression log, retaining rejected experiments.

        A rejected experiment is retained deliberately: the reason it failed is
        the most valuable part of the record, and an optimisation loop that
        forgets it will retry the same failing idea.
        """
        p = Path(path) if path else self.root / "PROGRESSION.md"
        lines = ["# Experiment progression", ""]
        lines.append("| # | Experiment | Status | Outcome | Change | Key metric | Created |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for i, r in enumerate(records, 1):
            key = "; ".join(f"{k}={v}" for k, v in list(r.metrics.items())[:2]) or "n/a"
            lines.append(
                f"| {i} | {r.name} | {r.status} | {r.outcome} | "
                f"{(r.change or '').replace('|', '/')[:60]} | {key[:60]} | {r.created_at} |"
            )
        lines.append("")
        failed = [r for r in records if r.outcome in ("failed", "regressed")]
        if failed:
            lines.append("## Rejected or failed experiments (retained deliberately)")
            lines.append("")
            for r in failed:
                lines.append(f"### {r.name} — {r.outcome}")
                if r.hypothesis:
                    lines.append(f"- Hypothesis: {r.hypothesis}")
                if r.change:
                    lines.append(f"- Change: {r.change}")
                for n in r.notes:
                    lines.append(f"- {n}")
                lines.append("")
        write_text(p, "\n".join(lines).rstrip() + "\n")
        return p


# --------------------------------------------------------------------------- #
# Convenience: identity from an environment
# --------------------------------------------------------------------------- #


def identity_from(
    *,
    env: Any = None,
    source_digest: str = "",
    build_flags: str = "",
    workload: str = "",
    problem_size: Any = None,
    launch_config: Any = None,
    iterations: int = 0,
    warmup: int = 0,
) -> ExperimentIdentity:
    """Build an identity, filling hardware/toolchain fields from the environment."""
    dev = getattr(env, "device", None)
    tools = getattr(env, "tools", {}) or {}

    def tool_version(name: str) -> str:
        t = tools.get(name)
        return (t.version or "") if t else ""

    fp = ""
    if dev is not None:
        fp = stable_hash(
            {
                "name": getattr(dev, "name", ""),
                "cc": getattr(dev, "compute_capability", ""),
                "sms": getattr(dev, "multi_processor_count", 0),
                "l2": getattr(dev, "l2_cache_size", 0),
                "uma": getattr(dev, "integrated_uma", False),
                "mem": getattr(dev, "total_global_mem", 0),
            },
            length=16,
        )
    return ExperimentIdentity(
        source_digest=source_digest,
        build_flags=build_flags,
        arch=getattr(dev, "arch", ""),
        nvcc_version=tool_version("nvcc"),
        ncu_version=tool_version("ncu"),
        device_fingerprint=fp or platform.machine(),
        workload=workload,
        problem_size=str(problem_size) if problem_size is not None else "",
        launch_config=str(launch_config) if launch_config is not None else "",
        iterations=iterations,
        warmup=warmup,
    )
