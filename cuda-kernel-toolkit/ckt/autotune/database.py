"""Results database and persistence for offline autotuning.

Stores versioned, reproducible benchmark results, raw timing samples,
compilation logs, and strict correctness validation outcomes in JSON/JSONL.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .profile import HardwareProfile, SoftwareProfile, TuningProfile
from .registry import CandidateSpec
from .workload import WorkloadSpec
from ..util import to_plain

SCHEMA_VERSION = 1


@dataclass
class CorrectnessTestSummary:
    """Outcome of correctness tests run for a candidate."""

    required_tests: list[str] = field(default_factory=list)
    passed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    byte_identity_enforced: bool = True
    byte_identity_passed: bool = True
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CorrectnessTestSummary:
        return cls(
            required_tests=list(data.get("required_tests", [])),
            passed=list(data.get("passed", [])),
            failed=list(data.get("failed", [])),
            skipped=list(data.get("skipped", [])),
            byte_identity_enforced=bool(data.get("byte_identity_enforced", True)),
            byte_identity_passed=bool(data.get("byte_identity_passed", True)),
            details=dict(data.get("details", {})),
        )


@dataclass
class TuningRecord:
    """Complete, self-contained record of one candidate evaluation on one workload."""

    schema_version: int
    hardware_profile_id: str
    software_profile: dict[str, Any]
    workload: dict[str, Any]
    candidate: dict[str, Any]
    compile_status: str  # "success", "failed", "skipped"
    correctness_status: str  # "pass", "fail", "unvalidated"
    correctness_tests: CorrectnessTestSummary
    latency_us: float  # median
    latency_p95_us: float
    latency_min_us: float = 0.0
    latency_max_us: float = 0.0
    latency_mad_us: float = 0.0
    samples_us: list[float] = field(default_factory=list)
    effective_gbs: float = 0.0
    useful_tflops: float = 0.0
    benchmark_mode: str = "production"
    warmup: int = 3
    repeat: int = 20
    flush_mib: int = 32
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = to_plain(self)
        d["correctness_tests"] = self.correctness_tests.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TuningRecord:
        ct_data = data.get("correctness_tests", {})
        ct = (
            CorrectnessTestSummary.from_dict(ct_data)
            if isinstance(ct_data, dict)
            else CorrectnessTestSummary()
        )
        return cls(
            schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
            hardware_profile_id=str(data.get("hardware_profile_id", "")),
            software_profile=dict(data.get("software_profile", {})),
            workload=dict(data.get("workload", {})),
            candidate=dict(data.get("candidate", {})),
            compile_status=str(data.get("compile_status", "success")),
            correctness_status=str(data.get("correctness_status", "unvalidated")),
            correctness_tests=ct,
            latency_us=float(data.get("latency_us", 0.0)),
            latency_p95_us=float(data.get("latency_p95_us", 0.0)),
            latency_min_us=float(data.get("latency_min_us", 0.0)),
            latency_max_us=float(data.get("latency_max_us", 0.0)),
            latency_mad_us=float(data.get("latency_mad_us", 0.0)),
            samples_us=[float(x) for x in data.get("samples_us", [])],
            effective_gbs=float(data.get("effective_gbs", 0.0)),
            useful_tflops=float(data.get("useful_tflops", 0.0)),
            benchmark_mode=str(data.get("benchmark_mode", "production")),
            warmup=int(data.get("warmup", 3)),
            repeat=int(data.get("repeat", 20)),
            flush_mib=int(data.get("flush_mib", 32)),
            timestamp=str(data.get("timestamp", "")),
            notes=list(data.get("notes", [])),
        )


class TuningDatabase:
    """In-memory and file-backed database of tuning records."""

    def __init__(self) -> None:
        self.records: list[TuningRecord] = []

    def add(self, record: TuningRecord) -> None:
        self.records.append(record)

    def count(self) -> int:
        return len(self.records)

    def find_records(
        self,
        *,
        profile_id: str | None = None,
        operation: str | None = None,
        qtype: str | None = None,
        n: int | None = None,
        k: int | None = None,
        tokens: int | None = None,
    ) -> list[TuningRecord]:
        """Query matching records."""
        matched = []
        for r in self.records:
            if profile_id and r.hardware_profile_id != profile_id:
                continue
            wl = r.workload
            if operation and wl.get("operation") != operation:
                continue
            if qtype and wl.get("quantization_family") != qtype:
                continue
            if n is not None and wl.get("n") != n:
                continue
            if k is not None and wl.get("k") != k:
                continue
            if tokens is not None and wl.get("tokens") != tokens:
                continue
            matched.append(r)
        return matched

    def get_valid_winners_per_token(
        self,
        operation: str,
        qtype: str,
        n: int,
        k: int,
        profile_id: str | None = None,
        metric: str = "latency_us",
    ) -> dict[int, TuningRecord]:
        """Return the fastest valid candidate for each token count."""
        # Group by tokens
        by_token: dict[int, list[TuningRecord]] = {}
        for r in self.find_records(
            profile_id=profile_id, operation=operation, qtype=qtype, n=n, k=k
        ):
            # Strict correctness filter: must pass correctness and compilation
            if r.compile_status != "success":
                continue
            if r.correctness_status != "pass":
                continue
            if not r.correctness_tests.byte_identity_passed:
                continue
            t = int(r.workload.get("tokens", 1))
            by_token.setdefault(t, []).append(r)

        winners: dict[int, TuningRecord] = {}
        for t, recs in sorted(by_token.items()):
            best = min(recs, key=lambda x: getattr(x, metric, x.latency_us))
            winners[t] = best

        return winners

    def save_json(self, path: str | Path) -> None:
        """Save entire database to formatted JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "schema_version": SCHEMA_VERSION,
            "record_count": len(self.records),
            "records": [r.to_dict() for r in self.records],
        }
        p.write_text(json.dumps(data, indent=2))

    def save_jsonl(self, path: str | Path) -> None:
        """Save records as JSONL (one record per line)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for r in self.records:
                f.write(json.dumps(r.to_dict()) + "\n")

    def load(self, path: str | Path) -> None:
        """Load from JSON or JSONL file."""
        p = Path(path)
        if not p.exists():
            return
        text = p.read_text(encoding="utf-8")
        if text.strip().startswith("{") and '"records"' in text:
            # Full JSON
            data = json.loads(text)
            for item in data.get("records", []):
                self.records.append(TuningRecord.from_dict(item))
        else:
            # JSONL
            for line in text.splitlines():
                if line.strip():
                    self.records.append(TuningRecord.from_dict(json.loads(line)))

    def export_csv(self, path: str | Path) -> None:
        """Export tabular summary as CSV."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "operation",
            "qtype",
            "N",
            "K",
            "T",
            "candidate_id",
            "kernel_family",
            "compile_status",
            "correctness_status",
            "byte_identity_passed",
            "median_us",
            "p95_us",
            "min_us",
            "max_us",
            "effective_gbs",
            "useful_tflops",
        ]
        with p.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.records:
                wl = r.workload
                cand = r.candidate
                writer.writerow({
                    "operation": wl.get("operation"),
                    "qtype": wl.get("quantization_family"),
                    "N": wl.get("n"),
                    "K": wl.get("k"),
                    "T": wl.get("tokens"),
                    "candidate_id": cand.get("id"),
                    "kernel_family": cand.get("kernel_family"),
                    "compile_status": r.compile_status,
                    "correctness_status": r.correctness_status,
                    "byte_identity_passed": r.correctness_tests.byte_identity_passed,
                    "median_us": r.latency_us,
                    "p95_us": r.latency_p95_us,
                    "min_us": r.latency_min_us,
                    "max_us": r.latency_max_us,
                    "effective_gbs": r.effective_gbs,
                    "useful_tflops": r.useful_tflops,
                })
