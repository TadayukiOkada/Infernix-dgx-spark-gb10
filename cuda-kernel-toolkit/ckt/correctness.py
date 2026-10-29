"""correctness — mandatory correctness gating before any performance claim.

Performance work is only meaningful on a kernel that computes the right answer.
This module establishes correctness from *independent* evidence and records what
was actually checked, so that a kernel which was never validated is reported as
``unvalidated`` rather than implicitly treated as correct.

Verification layers, in increasing cost:

1. **Harness self-verification** — the workload declares a reference check
   (``verify.status``) computed against an independently-stored pre-image or a
   host-side oracle.  Cheap, and the first thing to consult.
2. **Reference comparison** — compare a candidate's outputs against a reference
   run of the baseline, with dtype-appropriate tolerances.
3. **Compute-sanitizer** — memory-safety and race instrumentation for kernels
   that were modified.  Slow, so it is opt-in.

The module never infers correctness from the absence of a crash, and never
treats a tolerance as satisfied when the compared quantities are not finite.
"""

from __future__ import annotations

import math
import re
import struct
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .util import ToolError, run, which


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #

# Tolerances by dtype.  These are comparison thresholds for *candidate vs
# reference* agreement, chosen so that reordering-driven differences pass while
# genuine arithmetic mistakes fail.  They are not a statement about the accuracy
# the kernel is required to achieve.
#
# Both the long and short spellings are accepted: the harness and CLI emit
# `fp32`/`bf16`, while some callers use `float32`/`bfloat16`.  Normalising once
# here means an unknown spelling cannot silently fall through to a wrong default.
DEFAULT_TOLERANCES: dict[str, tuple[float, float]] = {
    "float16": (1e-2, 1e-2),
    "half": (1e-2, 1e-2),
    "fp16": (1e-2, 1e-2),
    "bfloat16": (2e-2, 1e-2),
    "bf16": (2e-2, 1e-2),
    "float32": (1e-5, 1e-4),
    "fp32": (1e-5, 1e-4),
    "float64": (1e-9, 1e-9),
    "double": (1e-9, 1e-9),
    "fp64": (1e-9, 1e-9),
    "float8_e4m3": (5e-2, 5e-2),
    "fp8_e4m3": (5e-2, 5e-2),
    "float8_e5m2": (5e-2, 5e-2),
    "fp8_e5m2": (5e-2, 5e-2),
    "int32": (0.0, 0.0),
    "int64": (0.0, 0.0),
}

_FLOAT_DTYPES = {
    "float16", "half", "fp16",
    "bfloat16", "bf16",
    "float32", "fp32",
    "float64", "double", "fp64",
    "float8_e4m3", "fp8_e4m3",
    "float8_e5m2", "fp8_e5m2",
}


@dataclass
class ComparisonResult:
    n: int = 0
    max_abs_err: float = 0.0
    max_rel_err: float = 0.0
    mean_abs_err: float = 0.0
    mismatches: int = 0
    first_mismatch_index: int | None = None
    nan_candidate: int = 0
    nan_reference: int = 0
    inf_candidate: int = 0
    inf_reference: int = 0
    signed_zero_mismatches: int = 0
    nan_equal_pairs: int = 0
    atol: float = 0.0
    rtol: float = 0.0
    dtype: str = "float32"
    exact: bool = False
    passed: bool = False
    criteria: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "max_abs_err": self.max_abs_err,
            "max_rel_err": self.max_rel_err,
            "mean_abs_err": self.mean_abs_err,
            "mismatches": self.mismatches,
            "first_mismatch_index": self.first_mismatch_index,
            "nan_candidate": self.nan_candidate,
            "nan_reference": self.nan_reference,
            "inf_candidate": self.inf_candidate,
            "inf_reference": self.inf_reference,
            "signed_zero_mismatches": self.signed_zero_mismatches,
            "nan_equal_pairs": self.nan_equal_pairs,
            "atol": self.atol,
            "rtol": self.rtol,
            "dtype": self.dtype,
            "exact": self.exact,
            "passed": self.passed,
            "criteria": self.criteria,
            "notes": self.notes,
        }


def compare_arrays(
    candidate: Sequence[float],
    reference: Sequence[float],
    *,
    dtype: str = "float32",
    atol: float | None = None,
    rtol: float | None = None,
    exact: bool | None = None,
    treat_nan_equal: bool = True,
) -> ComparisonResult:
    """Compare candidate against reference with dtype-appropriate tolerances.

    Integer dtypes default to exact comparison.  Floats default to the
    per-dtype tolerance table.  Non-finite values are counted and reported
    separately: a NaN in the candidate is a failure even if the tolerance would
    otherwise be satisfied, because ``NaN > atol`` is false and would silently
    pass a naive check.
    """
    if len(candidate) != len(reference):
        r = ComparisonResult(n=min(len(candidate), len(reference)), dtype=dtype)
        r.notes.append(
            f"length mismatch: candidate has {len(candidate)} elements, reference has "
            f"{len(reference)}; comparison is not meaningful"
        )
        return r

    is_float = dtype in _FLOAT_DTYPES
    if exact is None:
        exact = not is_float
    if atol is None or rtol is None:
        d_atol, d_rtol = DEFAULT_TOLERANCES.get(dtype, (1e-5, 1e-4))
        atol = d_atol if atol is None else atol
        rtol = d_rtol if rtol is None else rtol

    r = ComparisonResult(
        n=len(candidate), atol=atol, rtol=rtol, dtype=dtype, exact=exact
    )
    r.criteria = (
        "bit-exact" if exact else f"|a-b| <= atol + rtol*|b| with atol={atol:g}, rtol={rtol:g}"
    )

    abs_sum = 0.0
    for i, (a, b) in enumerate(zip(candidate, reference)):
        # Non-finite bookkeeping first: these dominate any tolerance decision.
        a_nan = isinstance(a, float) and math.isnan(a)
        b_nan = isinstance(b, float) and math.isnan(b)
        if a_nan:
            r.nan_candidate += 1
        if b_nan:
            r.nan_reference += 1
        if isinstance(a, float) and math.isinf(a):
            r.inf_candidate += 1
        if isinstance(b, float) and math.isinf(b):
            r.inf_reference += 1

        if a_nan or b_nan:
            # When both sides are NaN they denote the same absent value; the
            # bookkeeping counters above already record that this happened, but
            # it must not be treated as a correctness failure under
            # ``treat_nan_equal`` — otherwise every pair of NaNs fails even
            # though nothing was compared differently.
            if a_nan and b_nan and treat_nan_equal:
                r.nan_equal_pairs += 1
                continue
            r.mismatches += 1
            if r.first_mismatch_index is None:
                r.first_mismatch_index = i
            continue

        if isinstance(a, float) and isinstance(b, float) and math.isinf(a) and math.isinf(b):
            if (a > 0) == (b > 0):
                continue
            r.mismatches += 1
            if r.first_mismatch_index is None:
                r.first_mismatch_index = i
            continue

        # Signed zero: `a != b` is false for +0.0 vs -0.0, so an exact check must
        # distinguish them explicitly when the sign is meaningful.
        if (
            isinstance(a, float) and isinstance(b, float)
            and a == 0.0 and b == 0.0
            and math.copysign(1.0, a) != math.copysign(1.0, b)
        ):
            r.signed_zero_mismatches += 1
            if exact:
                r.mismatches += 1
                if r.first_mismatch_index is None:
                    r.first_mismatch_index = i
            continue

        if exact:
            if a != b:
                r.mismatches += 1
                if r.first_mismatch_index is None:
                    r.first_mismatch_index = i
            continue

        abs_err = abs(a - b)
        rel_err = abs_err / abs(b) if b != 0 else (abs_err if abs_err == 0 else float("inf"))
        abs_sum += abs_err
        if abs_err > r.max_abs_err:
            r.max_abs_err = abs_err
        if rel_err > r.max_rel_err and math.isfinite(rel_err):
            r.max_rel_err = rel_err
        if abs_err > atol + rtol * abs(b):
            r.mismatches += 1
            if r.first_mismatch_index is None:
                r.first_mismatch_index = i

    r.mean_abs_err = abs_sum / r.n if r.n else 0.0
    if not is_float:
        r.max_abs_err = float(r.mismatches > 0)
        r.max_rel_err = 0.0

    # NaN values are a hard failure only when the candidate produced NaNs the
    # reference did not have. A pair where *both* sides are NaN is the same
    # absent value; under ``treat_nan_equal`` those pairs (counted above as
    # ``nan_equal_pairs``) must not fail the comparison.
    unexplained_nan = max(0, r.nan_candidate - r.nan_equal_pairs)
    missing_nan = max(0, r.nan_reference - r.nan_equal_pairs)
    r.passed = (
        r.mismatches == 0
        and unexplained_nan == 0
        and missing_nan == 0
        and r.inf_candidate == 0
    )
    if unexplained_nan:
        r.notes.append(
            f"Candidate produced {unexplained_nan} NaN value(s) where the reference had none; "
            "arithmetic is incorrect regardless of tolerance."
        )
    if r.inf_candidate and not r.inf_reference:
        r.notes.append(
            f"Candidate produced {r.inf_candidate} Inf value(s) where the reference had none."
        )
    if r.signed_zero_mismatches and not exact:
        r.notes.append(
            f"{r.signed_zero_mismatches} signed-zero difference(s) present; these are within "
            "tolerance for a float comparison but would fail a bit-exact requirement."
        )
    return r


def compare_scalars(
    candidate: float,
    reference: float,
    *,
    dtype: str = "float32",
    atol: float | None = None,
    rtol: float | None = None,
) -> ComparisonResult:
    return compare_arrays([candidate], [reference], dtype=dtype, atol=atol, rtol=rtol)


# --------------------------------------------------------------------------- #
# Harness verification
# --------------------------------------------------------------------------- #


@dataclass
class VerificationRecord:
    workload: str = ""
    status: str = "unvalidated"       # pass | fail | unvalidated
    detail: str = ""
    method: str = ""                  # what the harness actually compared against
    config: dict[str, str] = field(default_factory=dict)
    source: str = "harness-self-check"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "workload": self.workload,
            "status": self.status,
            "detail": self.detail,
            "method": self.method,
            "config": self.config,
            "source": self.source,
            "notes": self.notes,
        }


def verification_from_records(records: dict[str, str], *, workload: str = "") -> VerificationRecord:
    """Read the harness's self-declared verification outcome.

    The distinction that matters is between "the harness checked and passed" and
    "no check exists".  A missing ``verify.status`` yields ``unvalidated``, not
    ``pass``: silence is not evidence of correctness.
    """
    v = VerificationRecord(workload=workload or records.get("workload", ""))
    status = records.get("verify.status")
    if status is None:
        v.notes.append(
            "The harness reported no verification result. The workload is UNVALIDATED: "
            "no performance conclusion may be drawn from it."
        )
        return v
    v.status = status.strip().lower()
    v.detail = records.get("verify.detail", "")
    v.method = records.get("verify.method", "harness-internal reference check")
    for k, val in records.items():
        if k.startswith("config."):
            v.config[k] = val
    if v.status not in ("pass", "fail"):
        v.notes.append(f"Unrecognised verification status '{status}'; treating as unvalidated.")
        v.status = "unvalidated"
    if v.status == "pass":
        if "rel err" in v.detail or "err" in v.detail:
            v.method = "max relative error against an independent reference"
    return v


# --------------------------------------------------------------------------- #
# compute-sanitizer
# --------------------------------------------------------------------------- #

SANITIZER_TOOLS = ("memcheck", "racecheck", "initcheck", "synccheck")

_RE_SANITIZER_ERROR = re.compile(r"^=========\s*(?P<kind>[A-Za-z ]+?)\s*:", re.M)
_RE_SANITIZER_SUMMARY = re.compile(
    r"ERROR SUMMARY:\s*(?P<errors>\d+)\s+error", re.I
)


@dataclass
class SanitizerResult:
    tool: str = ""
    clean: bool = False
    error_count: int | None = None
    error_kinds: dict[str, int] = field(default_factory=dict)
    excerpt: str = ""
    command: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "clean": self.clean,
            "error_count": self.error_count,
            "error_kinds": self.error_kinds,
            "command": self.command,
            "excerpt": self.excerpt,
            "notes": self.notes,
        }


def run_sanitizer(
    exe: str | Path,
    args: Sequence[str],
    *,
    tool: str = "memcheck",
    timeout: float = 3600.0,
    sanitizer_path: str | None = None,
) -> SanitizerResult:
    """Run ``compute-sanitizer`` and summarise the outcome.

    Absence of the tool is reported as unknown, not as a pass: skipping a check
    is different from succeeding at it.
    """
    exe_path = sanitizer_path or which("compute-sanitizer")
    r = SanitizerResult(tool=tool)
    if tool not in SANITIZER_TOOLS:
        r.notes.append(f"unknown sanitizer tool '{tool}'; expected one of {SANITIZER_TOOLS}")
        return r
    if not exe_path:
        r.notes.append(
            "compute-sanitizer is not installed. Memory safety is UNKNOWN, not verified. "
            "Install the CUDA toolkit's sanitizer or run validation on a machine that has it."
        )
        return r
    cmd = [exe_path, "--tool", tool, "--error-exitcode", "1", str(exe), *args]
    r.command = " ".join(cmd)
    try:
        res = run(cmd, timeout=timeout, check=False, log=True)
    except ToolError as exc:
        r.notes.append(f"sanitizer run failed to complete: {exc}")
        return r

    text = res.stdout + "\n" + res.stderr
    kinds: dict[str, int] = {}
    for m in _RE_SANITIZER_ERROR.finditer(text):
        kind = m.group("kind").strip()
        if kind.lower() in ("launch", "sanitizer"):
            continue
        kinds[kind] = kinds.get(kind, 0) + 1
    r.error_kinds = kinds

    m = _RE_SANITIZER_SUMMARY.search(text)
    if m:
        r.error_count = int(m.group("errors"))
    elif kinds:
        r.error_count = sum(kinds.values())

    if r.error_count is None:
        # No summary and no recognisable errors: infer from exit status, and say
        # that this is an inference.
        r.clean = res.returncode == 0
        r.notes.append(
            "No ERROR SUMMARY line found; cleanliness was inferred from the process exit "
            "status, which is weaker evidence than a parsed summary."
        )
    else:
        r.clean = r.error_count == 0

    if not r.clean:
        lines = [ln for ln in text.splitlines() if "=========" in ln or " at " in ln]
        r.excerpt = "\n".join(lines[:40])
    return r


# --------------------------------------------------------------------------- #
# Aggregate gate
# --------------------------------------------------------------------------- #


@dataclass
class CorrectnessGate:
    """The single decision a caller should consult before timing anything."""

    workload: str = ""
    validated: bool = False
    checks: list[dict[str, Any]] = field(default_factory=list)
    blocking_issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add(self, kind: str, outcome: dict[str, Any], *, blocking: bool = True, detail: str = "") -> None:
        self.checks.append({"kind": kind, "detail": detail, **outcome})
        if blocking and not outcome.get("passed", outcome.get("clean", False)):
            self.blocking_issues.append(f"{kind}: {detail or outcome}")

    def finalise(self) -> "CorrectnessGate":
        self.validated = not self.blocking_issues and bool(self.checks)
        if not self.checks:
            self.blocking_issues.append(
                "no correctness check was performed; the kernel is UNVALIDATED"
            )
            self.validated = False
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "workload": self.workload,
            "validated": self.validated,
            "checks": self.checks,
            "blocking_issues": self.blocking_issues,
            "warnings": self.warnings,
        }


def gate_from_verification(v: VerificationRecord) -> CorrectnessGate:
    g = CorrectnessGate(workload=v.workload)
    g.checks.append({"kind": "harness-verification", **v.to_dict()})
    if v.status != "pass":
        g.blocking_issues.append(
            f"harness verification status is '{v.status}'"
            + (f": {v.detail}" if v.detail else "")
        )
    g.warnings.extend(v.notes)
    return g.finalise()
