"""profiler — baseline benchmarking and staged Nsight Compute metric collection.

Three stages, each entered only when its diagnostic value justifies the cost:

* **Stage A — baseline**: repeated execution of the target executable with
  warm-up, reporting median/p10/p90 latency and dispersion.  Cheap; no counters.
* **Stage B — targeted metrics**: one focused ``ncu`` pass with a metric set
  chosen for the *current bottleneck hypothesis*, not a fixed list.
* **Stage C — deep diagnosis**: heavier counters (full sections, warp-state
  statistics, cache breakdowns) for kernels whose bottleneck is still unclear.

Hard policy (mirrors the project's operating rules): **NCU is only ever run
against a synthetic-data harness**, never with real model weights.  This module
therefore profiles a harness executable whose inputs are generated in-process
from shapes and dtypes; it does not load checkpoints.

Metric availability is discovered from ``ncu --query-metrics`` rather than
assumed.  Metrics that do not exist on the target (for example ``dram__*`` on a
unified-memory GB10) are reported as unavailable, never substituted with a
fabricated value.
"""

from __future__ import annotations

import csv
import io
import json
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .cache import Cache, CacheKey
from .util import (
    LOG,
    SampleStats,
    ToolError,
    ensure_dir,
    percentile,
    run,
    to_plain,
    which,
)


# --------------------------------------------------------------------------- #
# Stage A: baseline timing
# --------------------------------------------------------------------------- #


@dataclass
class BaselineTiming:
    workload: str = ""
    command: str = ""
    samples: list[float] = field(default_factory=list)
    stats: SampleStats | None = None
    raw_records: dict[str, str] = field(default_factory=dict)
    derived: dict[str, float] = field(default_factory=dict)
    stable: bool = True
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "workload": self.workload,
            "command": self.command,
            "samples": self.samples,
            "stats": self.stats.to_dict() if self.stats else None,
            "derived": self.derived,
            "stable": self.stable,
            "notes": self.notes,
        }


def parse_kv_records(text: str) -> dict[str, str]:
    """Parse ``key<TAB>value`` records emitted by the harness."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        if "\t" in line:
            k, _, v = line.partition("\t")
            out[k.strip()] = v.strip()
    return out


def run_harness(
    exe: str | Path,
    args: Sequence[str],
    *,
    timeout: float = 900.0,
    repeats: int = 1,
    log: bool = True,
) -> tuple[dict[str, str], list[float]]:
    """Run the target executable ``repeats`` times, returning merged records."""
    records: dict[str, str] = {}
    medians: list[float] = []
    for i in range(repeats):
        res = run([str(exe), *args], timeout=timeout, check=False, log=log)
        rec = parse_kv_records(res.stdout)
        records.update(rec)
        if res.returncode != 0 and res.returncode not in (3,):
            raise ToolError(
                f"harness exited {res.returncode}: {(res.stderr or res.stdout)[-400:]}",
                result=res,
            )
        m = rec.get("timing.median_us")
        if m:
            try:
                medians.append(float(m))
            except ValueError:
                pass
    if len(medians) > 1:
        records["timing.median_us"] = f"{percentile(medians, 50):.4f}"
        records["timing.repeat_stdev_us"] = f"{statistics.pstdev(medians):.4f}" if len(medians) > 1 else "0"
    return records, medians


def baseline_timing(
    exe: str | Path,
    args: Sequence[str],
    *,
    workload: str = "",
    iters: int = 30,
    warmup: int = 5,
    repeats: int = 3,
    timeout: float = 900.0,
) -> BaselineTiming:
    """Stage A: measure latency with warm-up, repeated to expose instability."""
    all_args = list(args) + ["--iters", str(iters), "--warmup", str(warmup)]
    records: dict[str, str] = {}
    medians: list[float] = []
    samples: list[float] = []
    cmd_str = " ".join([str(exe), *all_args])
    for _ in range(max(1, repeats)):
        try:
            rec, meds = run_harness(exe, all_args, timeout=timeout, repeats=1, log=False)
        except ToolError as exc:
            return BaselineTiming(workload=workload, command=cmd_str, stable=False, notes=[str(exc)])
        records.update(rec)
        medians.extend(meds)
    if medians:
        samples = medians
    stats = SampleStats.from_values(samples) if samples else None
    bt = BaselineTiming(workload=workload, command=cmd_str, samples=samples, stats=stats, raw_records=records)
    for key in ("derived.effective_bandwidth_gbps", "derived.tflops", "timing.cv"):
        if key in records:
            try:
                bt.derived[key] = float(records[key])
            except ValueError:
                pass
    if stats:
        if stats.cv > 0.05:
            bt.stable = False
            bt.notes.append(
                f"High timing dispersion (CV={stats.cv:.3f}). Treat differences below "
                f"~{max(3.0, stats.iqr):.1f} us as noise, and check for competing GPU work."
            )
        if stats.n < 3:
            bt.notes.append("Fewer than 3 repeats; dispersion is not reliably estimated.")
    return bt


def verify_harness(
    exe: str | Path,
    args: Sequence[str],
    *,
    timeout: float = 900.0,
) -> dict[str, str]:
    """Run the harness's self-check for one configuration.

    Verification is always a separate, single launch.  The timed loop is run
    many times over the same buffers, so for an in-place kernel such as
    ``y = a*x + y`` the accumulated state after N iterations is not the state
    any reference comparison should be performed against.  A dedicated launch
    also keeps the correctness gate cheap, since it does not pay for the timing
    loop.
    """
    res = run([str(exe), *list(args), "--verify"], timeout=timeout, check=False, log=False)
    records = parse_kv_records(res.stdout)
    records.setdefault("_command", f"{exe} {' '.join(map(str, args))} --verify")
    if not records:
        detail = (res.stderr or res.stdout or "").strip()[-400:]
        records["verify.status"] = "fail"
        records["verify.detail"] = f"harness produced no records (exit {res.returncode}): {detail}"
    records["_exit_code"] = str(res.returncode)
    return records


# --------------------------------------------------------------------------- #
# Kernel-name resolution
# --------------------------------------------------------------------------- #

# NCU's `--kernel-name` matches the *undecorated function base name* exactly.
# It is not a substring match: with a kernel named `k_saxpy_vec4`, the filters
# `saxpy`, `k_saxpy` and `saxpy_vec4` all select nothing while `k_saxpy_vec4`
# selects it.  A workload label such as `saxpy` therefore never matches, and a
# silent no-op profile is indistinguishable from a failed one unless the
# resolved name is checked.  These helpers translate a source-level kernel name
# into the exact token NCU expects and verify the result.

_RE_AVAILABLE_KERNEL = re.compile(r"^\s*\d+\.\s+(\S+)\s*$", re.M)


def parse_available_kernels(text: str) -> list[str]:
    """Extract the kernel names NCU lists under ``Available Kernels``.

    NCU prints this list when a filter matches nothing, which turns a
    mis-specified filter into a usable hint instead of an opaque failure.
    """
    marker = text.find("Available Kernels")
    if marker == -1:
        return []
    return _RE_AVAILABLE_KERNEL.findall(text[marker:])


def demangle_symbol(symbol: str, *, demangler: str | None = None) -> str:
    """Demangle a C++ symbol, falling back to the mangled text."""
    tool = demangler or which("cu++filt") or which("c++filt")
    if not tool:
        return symbol
    res = run([tool, symbol], timeout=60, check=False, log=False)
    out = (res.stdout or "").strip()
    return out if out else symbol


# The callee in a demangled signature is the first identifier immediately
# followed by its template-argument list or its parameter list.  Splitting on
# "::" instead is wrong for the signatures CUDA actually produces: a
# file-static templated kernel demangles to
#   void infernix::ops::<unnamed>::rows_kernel<(QType)10, ...>(RowsGroup, int, ...)
# whose *last* "::" sits inside the parameter list, which made the old
# scope-splitting code return the token ``int)`` -- a filter that selects
# nothing and silently falls back to an unrelated kernel.
_RE_CALLEE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*(?=[<(])")


def ncu_base_name(symbol: str) -> str:
    """Reduce a mangled symbol or demangled signature to NCU's base name.

    ``_ZN...k_saxpy_vec4EPK6float4PS0_fi`` demangles to
    ``(anonymous namespace)::k_saxpy_vec4(float4 const*, float4*, float, int)``
    and NCU indexes it as ``k_saxpy_vec4``.

    Template instantiations collapse to the bare template name: the demangled
    ``k_gemm_tiled<(int)16>(const float *)`` is listed by NCU as
    ``k_gemm_tiled``.  Namespaces (``<unnamed>``) and parameter types (which
    themselves contain ``::``, templates and nested parentheses) must not be
    mistaken for the callee, so the name is taken from where a template- or
    parameter list begins rather than from the last scope separator.
    """
    demangled = demangle_symbol(symbol)
    m = _RE_CALLEE.search(demangled)
    if m:
        return m.group(1)
    tail = demangled.rsplit("::", 1)[-1].strip()
    cuts = [i for i in (tail.find("<"), tail.find("(")) if i > 0]
    if cuts:
        tail = tail[: min(cuts)]
    if " " in tail:
        tail = tail.split()[-1]
    return tail.strip() or demangled or symbol


def resolve_kernel_name(
    exe: str | Path,
    kernel_name: str,
    *,
    symbols: Sequence[str] = (),
) -> list[str]:
    """Return the NCU filter tokens that can select ``kernel_name``.

    Prefers the demangled base name of the symbols the binary actually stores,
    which is what NCU indexes.  Falls back to the supplied name so an unusual
    binary still produces a working filter.
    """
    out: list[str] = []
    for s in symbols:
        base = ncu_base_name(s)
        if base and base not in out:
            out.append(base)
    if not out and kernel_name:
        out.append(ncu_base_name(kernel_name))
    # A workload label may already be the exact base name.
    if kernel_name and kernel_name in out:
        out.remove(kernel_name)
        out.insert(0, kernel_name)
    return out


# --------------------------------------------------------------------------- #
# Metric selection
# --------------------------------------------------------------------------- #

# Metric names are grouped by question.  Each entry lists candidate spellings
# (different NCU versions and architectures name these slightly differently);
# the profiler resolves them against the target's actual metric list.
METRIC_GROUPS: dict[str, list[list[str]]] = {
    "duration": [["gpu__time_duration.sum", "gpu__time_duration.avg"]],
    "compute_throughput": [["sm__throughput.avg.pct_of_peak_sustained_elapsed"]],
    "memory_throughput": [
        ["gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"],
        ["gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"],
        ["dram__throughput.avg.pct_of_peak_sustained_elapsed"],
    ],
    "l2_throughput": [["lts__throughput.avg.pct_of_peak_sustained_elapsed"]],
    "l1_throughput": [["l1tex__throughput.avg.pct_of_peak_sustained_elapsed"]],
    "l2_hit_rate": [["lts__t_sector_hit_rate.pct"]],
    "l2_bytes": [["lts__t_bytes.sum", "lts__t_sectors.sum"]],
    "dram_bytes": [["dram__bytes.sum", "dram__bytes_read.sum"]],
    "warps_active": [
        ["sm__warps_active.avg.pct_of_peak_sustained_active"],
        ["smsp__warps_active.avg.pct_of_peak_sustained_active"],
    ],
    "issue_active": [["smsp__issue_active.avg.pct_of_peak_sustained_active"]],
    "tensor_active": [["sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed"]],
    "inst_executed": [["smsp__inst_executed.sum"]],
    # Instruction *utilisation* is what separates a genuinely saturated pipe
    # from a kernel whose aggregate looks busy because every thread sits on a
    # serial dependency chain; the raw count alone cannot say which case applies.
    "inst_utilization": [["sm__inst_executed.avg.pct_of_peak_sustained_elapsed"]],
    "occupancy": [["sm__maximum_warps_per_active_cycle_pct"]],
    # Stall reasons: the primary evidence for latency-bound classification.
    "stall_long_scoreboard": [["smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio"]],
    "stall_short_scoreboard": [["smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio"]],
    "stall_wait": [["smsp__average_warps_issue_stalled_wait_per_issue_active.ratio"]],
    "stall_barrier": [["smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio"]],
    "stall_math_pipe_throttle": [["smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio"]],
    "stall_lg_throttle": [["smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio"]],
    "stall_mio_throttle": [["smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio"]],
    "stall_not_selected": [["smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio"]],
    "stall_no_instruction": [["smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio"]],
    "stall_branch_resolving": [["smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio"]],
    "stall_drain": [["smsp__average_warps_issue_stalled_drain_per_issue_active.ratio"]],
    "stall_membar": [["smsp__average_warps_issue_stalled_membar_per_issue_active.ratio"]],
    "bank_conflicts": [["l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"]],
    "shared_load": [["l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum"]],
    "shared_store": [["l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum"]],
    # Per-pipe utilisation.  `sm__throughput` is the *maximum* over all
    # sub-units, so it cannot by itself say which pipe is saturated; on GB10 it
    # was observed to report exactly the L1TEX figure.  Resolving the individual
    # pipes lets the classifier distinguish a saturated on-chip memory path
    # (shared-memory traffic, L1 bandwidth) from saturated arithmetic.
    "l1_throughput": [["l1tex__throughput.avg.pct_of_peak_sustained_elapsed"]],
    "lsu_throughput": [["l1tex__lsu_writeback_active.avg.pct_of_peak_sustained_elapsed"]],
    "smem_throughput": [["l1tex__data_pipe_lsu_wavefronts_mem_shared.avg.pct_of_peak_sustained_elapsed"]],
    "fma_throughput": [["sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed"]],
    "alu_throughput": [["sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_elapsed"]],
    "fp64_throughput": [["sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_elapsed"]],
    "xu_throughput": [["sm__pipe_xu_cycles_active.avg.pct_of_peak_sustained_elapsed"]],
    "tensor_throughput": [["sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed"]],
    "global_load_sectors": [["l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"]],
    "global_store_sectors": [["l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum"]],
    "global_load_requests": [["l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum"]],
    "global_store_requests": [["l1tex__t_requests_pipe_lsu_mem_global_op_st.sum"]],
    "sectors_per_request": [
        ["l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio"],
        ["l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio"],
    ],
    "launch_registers": [["launch__registers_per_thread"]],
    "launch_occupancy": [
        ["launch__occupancy_limit_registers"],
        ["launch__occupancy_limit_shared_mem"],
        ["launch__occupancy_limit_warps"],
        ["launch__occupancy_limit_blocks"],
        ["launch__waves_per_multiprocessor"],
        ["sm__maximum_warps_per_active_cycle_pct"],
    ],
    "launch_blocks": [
        ["launch__grid_size"],
        ["launch__block_size"],
    ],
    "branch_efficiency": [["smsp__branch_efficiency.pct"]],
}

# Which groups belong to each stage / hypothesis.
STAGE_B_METRICS: dict[str, list[str]] = {
    "core": [
        "duration",
        "compute_throughput",
        "memory_throughput",
        "l2_throughput",
        "l2_hit_rate",
        "warps_active",
        "issue_active",
        "stall_long_scoreboard",
        "stall_short_scoreboard",
        "stall_wait",
        "stall_barrier",
        "stall_math_pipe_throttle",
        "stall_lg_throttle",
        "stall_not_selected",
        # On GB10 `sm__throughput` is the maximum over all sub-units, so it does
        # not say which resource saturates.  The per-pipe figures below are what
        # let the classifier separate an arithmetic limit (FMA/ALU/tensor) from
        # an on-chip memory-path limit (L1TEX, shared-memory wavefronts).
        "l1_throughput",
        "lsu_throughput",
        "smem_throughput",
        "fma_throughput",
        "alu_throughput",
        "fp64_throughput",
        "xu_throughput",
        "tensor_throughput",
        "inst_utilization",
        # Coalescing evidence belongs in the core set too: a kernel whose loads
        # waste cache-line space cannot be interpreted as clean traffic, and the
        # roofline and classifier both read this figure.
        "global_load_requests",
        "global_store_requests",
        "sectors_per_request",
        # Launch resources belong to the core set: the occupancy model needs the
        # register count, and without it every occupancy conclusion is guesswork.
        "launch_registers",
        "launch_blocks",
    ],
    "memory": [
        "global_load_sectors",
        "global_store_sectors",
        "global_load_requests",
        "global_store_requests",
        "sectors_per_request",
        "l2_bytes",
        "dram_bytes",
    ],
    "compute": ["tensor_active", "inst_executed", "stall_math_pipe_throttle"],
    "shared": ["bank_conflicts", "shared_load", "shared_store", "stall_mio_throttle"],
    "launch": ["launch_registers", "launch_occupancy", "launch_blocks"],
    "control": ["branch_efficiency", "stall_branch_resolving", "stall_no_instruction"],
}

STAGE_C_EXTRA: list[str] = [
    "stall_mio_throttle",
    "stall_drain",
    "stall_membar",
    "launch_registers",
    "launch_occupancy",
    "launch_blocks",
    "branch_efficiency",
    "tensor_active",
    "inst_executed",
]

# `launch__*` metrics are collectable but do NOT appear in
# `ncu --query-metrics-mode all` on this build.  They are therefore seeded into
# the catalog by name; if a target genuinely lacks one, NCU reports it back and
# the profiler records it under `unavailable` rather than fabricating a value.
KNOWN_UNLISTED_METRICS: list[str] = [
    "launch__registers_per_thread",
    "launch__block_size",
    "launch__grid_size",
    "launch__waves_per_multiprocessor",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
    "launch__occupancy_limit_warps",
    "launch__occupancy_limit_blocks",
    "launch__shared_mem_per_block_static",
    "launch__shared_mem_per_block_dynamic",
    "launch__shared_mem_config_size",
]


@dataclass
class MetricCatalog:
    """Which metric names actually exist on the target."""

    available: set[str] = field(default_factory=set)
    raw_count: int = 0
    ncu_version: str = ""

    def resolve(self, candidates: Sequence[str]) -> str | None:
        for c in candidates:
            if c in self.available:
                return c
            # Some metrics exist only with a suffix (e.g. .sum / .avg).
            base = c.split(".")[0]
            for a in self.available:
                if a == base or a.startswith(base + "."):
                    return a
        return None

    def resolve_group(self, group: str) -> list[str]:
        out: list[str] = []
        for cands in METRIC_GROUPS.get(group, []):
            r = self.resolve(cands)
            if r:
                out.append(r)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"raw_count": self.raw_count, "ncu_version": self.ncu_version, "available": len(self.available)}


@dataclass
class KernelMetricResult:
    kernel: str = ""
    grid: tuple[int, int, int] = (0, 0, 0)
    block: tuple[int, int, int] = (0, 0, 0)
    metrics: dict[str, float] = field(default_factory=dict)
    groups: dict[str, list[str]] = field(default_factory=dict)
    unavailable: list[str] = field(default_factory=list)
    raw_csv: str = ""
    stage: str = "B"
    notes: list[str] = field(default_factory=list)

    @property
    def duration_us(self) -> float | None:
        v = self.metrics.get("gpu__time_duration.sum") or self.metrics.get("gpu__time_duration.avg")
        return v / 1000.0 if v else None

    def get(self, *names: str) -> float | None:
        for n in names:
            if n in self.metrics:
                return self.metrics[n]
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kernel": self.kernel,
            "grid": list(self.grid),
            "block": list(self.block),
            "metrics": self.metrics,
            "groups": self.groups,
            "unavailable": self.unavailable,
            "stage": self.stage,
            "notes": self.notes,
        }


class Profiler:
    """Wraps ``ncu`` with availability-aware metric selection and caching."""

    def __init__(self, cache: Cache, *, ncu_path: str | None = None):
        self.cache = cache
        self.ncu = ncu_path or which("ncu")
        self.catalog = self._load_catalog()

    # ------------------------------------------------------------------ #
    def _load_catalog(self) -> MetricCatalog:
        cat = MetricCatalog()
        if not self.ncu:
            return cat
        res = run([self.ncu, "--version"], timeout=60, check=False, log=False)
        m = re.search(r"Version\s+([0-9][0-9.]*)", res.stdout + res.stderr)
        cat.ncu_version = m.group(1) if m else "unknown"
        cache_file = self.cache.root / "ncu_metrics.json"
        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text())
                if data.get("ncu_version") == cat.ncu_version:
                    cat.available = set(data.get("metrics", [])) | set(KNOWN_UNLISTED_METRICS)
                    cat.raw_count = len(cat.available)
                    return cat
            except (OSError, ValueError):
                pass
        q = run([self.ncu, "--query-metrics-mode", "all", "--csv"], timeout=600, check=False, log=False)
        names: set[str] = set()
        # Parse the WHOLE output as CSV: metric descriptions contain commas and
        # embedded newlines inside quoted fields, so line-by-line parsing
        # silently truncates the catalog.
        try:
            reader = csv.reader(io.StringIO(q.stdout))
            header: list[str] | None = None
            for row in reader:
                if not row:
                    continue
                if header is None:
                    # First field of the header row identifies the name column.
                    if row and row[0].strip().lower().startswith("metric name"):
                        header = row
                    continue
                if row and row[0] and not row[0].strip().startswith('"'):
                    names.add(row[0].strip())
        except csv.Error as exc:
            LOG.warning("metric catalog CSV parse failed: %s", exc)
        if not names:
            # Fall back to the line-oriented listing.
            q2 = run([self.ncu, "--query-metrics"], timeout=600, check=False, log=False)
            for mm in re.finditer(r"^([a-z][a-z0-9_]*__[a-z0-9_]+)", q2.stdout, re.M):
                names.add(mm.group(1))
        cat.available = names | set(KNOWN_UNLISTED_METRICS)
        cat.raw_count = len(cat.available)
        try:
            cache_file.write_text(json.dumps({"ncu_version": cat.ncu_version, "metrics": sorted(names)}))
        except OSError:
            pass
        return cat
    # ------------------------------------------------------------------ #
    def metrics_available(self, name: str) -> bool:
        return self.catalog.resolve([name]) is not None

    def build_metric_list(self, *, stage: str = "B", focus: str | None = None) -> list[str]:
        """Choose metrics for a stage and (optionally) a bottleneck focus."""
        groups = list(STAGE_B_METRICS["core"])
        if focus and focus in STAGE_B_METRICS:
            groups += STAGE_B_METRICS[focus]
        if stage == "C":
            groups += STAGE_C_EXTRA
        seen: set[str] = set()
        out: list[str] = []
        for g in groups:
            for m in self.catalog.resolve_group(g):
                if m not in seen:
                    seen.add(m)
                    out.append(m)
        return out

    # ------------------------------------------------------------------ #
    def profile(
        self,
        exe: str | Path,
        args: Sequence[str],
        *,
        kernel_filter: str | None = None,
        symbols: Sequence[str] = (),
        stage: str = "B",
        focus: str | None = None,
        launch_count: int = 1,
        launch_skip: int = 2,
        use_cache: bool = True,
        timeout: float = 1800.0,
    ) -> list[KernelMetricResult]:
        """Run one NCU pass and return per-kernel metric records.

        ``launch_skip``/``launch_count`` bound the replay cost: by default the
        profiler measures a single launch after a couple of warm-ups rather than
        replaying a launch loop.

        ``symbols`` should carry the exact stored symbols for the target kernel
        (from :meth:`BinaryInspector.resolve_symbol`).  NCU's ``--kernel-name``
        is an exact match on the base function name, so passing symbols lets the
        filter be derived from the binary rather than guessed from a workload
        label.
        """
        if not self.ncu:
            raise ToolError("ncu unavailable; profiling is disabled")
        metric_list = self.build_metric_list(stage=stage, focus=focus)
        if not metric_list:
            raise ToolError(
                "no usable NCU metrics were found on this target; "
                "run `ckt env --tools` to inspect the NCU installation"
            )
        key = CacheKey(
            kind="profile",
            parts={
                "exe": str(exe),
                "args": list(args),
                "metrics": metric_list,
                "kernel": kernel_filter,
                "symbols": list(symbols),
                "skip": launch_skip,
                "count": launch_count,
                "stage": stage,
                "focus": focus,
                "ncu": self.catalog.ncu_version,
            },
        )
        if use_cache:
            cached = self.cache.get_json(key, "metrics.json")
            if cached is not None:
                return [_result_from_dict(d) for d in cached]

        cmd = [
            self.ncu,
            "--csv",
            "--metrics",
            ",".join(metric_list),
            "--launch-skip",
            str(launch_skip),
            "--launch-count",
            str(launch_count),
            "--target-processes",
            "all",
        ]
        wanted: list[str] = []
        if kernel_filter:
            # This NCU build has no `regex:` filter; `--kernel-name` is an exact
            # match on the base function name, so the workload label is resolved
            # to the stored symbol first.
            base = resolve_kernel_name(exe, kernel_filter, symbols=symbols)
            wanted = [b for b in base if b and b != kernel_filter]
            cmd += ["--kernel-name", base[0] if base else kernel_filter]
        cmd += [str(exe), *args]
        res = run(cmd, timeout=timeout, check=False, log=True, what="ncu")
        results = parse_ncu_csv(res.stdout)
        if not results and kernel_filter:
            # The filter selected nothing.  NCU lists the kernels it did see;
            # retry each candidate before giving up, so a name mismatch
            # instead of silently producing no metrics.  A candidate is only
            # accepted when it is plausibly the requested kernel: substituting an
            # arbitrary other kernel from the same process would attribute that
            # kernel's counters to this workload, which is worse than no metrics.
            available = parse_available_kernels(res.stderr + res.stdout)
            for cand in available:
                if kernel_filter not in cand and cand not in wanted:
                    continue
                retry = [c for c in cmd]
                i = retry.index("--kernel-name")
                retry[i + 1] = cand
                res2 = run(retry, timeout=timeout, check=False, log=True, what="ncu")
                results = parse_ncu_csv(res2.stdout)
                if results:
                    res = res2
                    LOG.info("resolved kernel filter %r to %r", kernel_filter, cand)
                    break
        if not results:
            available = parse_available_kernels(res.stderr + res.stdout)
            hint = (
                f"\nKernels NCU observed: {', '.join(available)}"
                if available
                else "\nNCU observed no kernels at all, so the target never launched one."
            )
            err = (res.stderr or res.stdout).strip().splitlines()[-6:]
            raise ToolError(
                "ncu produced no kernel records. Common causes: the kernel was never "
                "launched with the given arguments, --launch-skip exceeded the number "
                "of launches, --kernel-name did not match a kernel exactly, or counter "
                "permission is unavailable.\n" + "\n".join(err) + hint,
                result=res,
            )
        # Record which requested metrics the target does not provide.
        have = set()
        for r in results:
            have |= set(r.metrics)
        unavailable = [m for m in metric_list if m not in have]
        for r in results:
            r.unavailable = unavailable
            r.stage = stage
            if unavailable:
                r.notes.append(
                    f"{len(unavailable)} requested metric(s) unavailable on this target; "
                    "they are reported as missing, not substituted."
                )
        if use_cache:
            self.cache.put_json(key, "metrics.json", [r.to_dict() for r in results])
        # Also keep the raw CSV for provenance.
        self.cache.path_for(key, "ncu_raw.csv").write_text(res.stdout)
        # The profiled process re-runs its warm-up/timing loop under NCU replay,
        # so any `timing.*` records it printed are NOT valid wall-clock
        # measurements.  Use Stage A (`baseline_timing`) for latency.
        for r in results:
            r.notes.append(
                "measured under NCU replay; use Stage A baseline timing for latency, "
                "not the harness's own timing records"
            )
        return results

    # ------------------------------------------------------------------ #
    def check_permissions(self) -> tuple[bool, str]:
        """Probe whether hardware counters are collectable in this environment."""
        if not self.ncu:
            return False, "ncu not found"
        return True, "ncu present (counter access is verified on the first real pass)"


def parse_ncu_csv(text: str) -> list[KernelMetricResult]:
    """Parse ``ncu --csv --metrics`` output into per-kernel results.

    The CSV has one row per (kernel, metric) with columns including
    ``Kernel Name``, ``Block Size``, ``Grid Size``, ``Metric Name``,
    ``Metric Value``.
    """
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        return []
    header: list[str] = []
    start = 0
    for i, row in enumerate(rows):
        if "Metric Name" in row and "Kernel Name" in row:
            header = row
            start = i + 1
            break
    if not header:
        return []

    def idx(name: str) -> int | None:
        for i, h in enumerate(header):
            if h.strip() == name:
                return i
        return None

    i_kernel = idx("Kernel Name")
    i_block = idx("Block Size")
    i_grid = idx("Grid Size")
    i_metric = idx("Metric Name")
    i_value = idx("Metric Value")
    if i_kernel is None or i_metric is None or i_value is None:
        return []

    by_kernel: dict[str, KernelMetricResult] = {}
    for row in rows[start:]:
        if len(row) <= max(i_kernel, i_metric, i_value):
            continue
        kname = row[i_kernel].strip()
        if not kname:
            continue
        r = by_kernel.get(kname)
        if r is None:
            r = KernelMetricResult(kernel=kname)
            if i_block is not None and i_block < len(row):
                r.block = _parse_dims(row[i_block])
            if i_grid is not None and i_grid < len(row):
                r.grid = _parse_dims(row[i_grid])
            by_kernel[kname] = r
        mname = row[i_metric].strip()
        mval = row[i_value].strip()
        v = _parse_metric_value(mval)
        if v is not None:
            r.metrics[mname] = v
        else:
            r.notes.append(f"metric {mname} reported non-numeric value '{mval}'")
    return list(by_kernel.values())


def _parse_dims(text: str) -> tuple[int, int, int]:
    nums = re.findall(r"\d+", text or "")
    vals = [int(x) for x in nums[:3]]
    while len(vals) < 3:
        vals.append(1)
    return tuple(vals)  # type: ignore[return-value]


def _parse_metric_value(text: str) -> float | None:
    if text is None:
        return None
    t = text.strip().replace(",", "")
    if not t or t.lower() in ("n/a", "na", "", "-"):
        return None
    try:
        return float(t)
    except ValueError:
        pass
    # Values may carry unit suffixes or use SI notation.
    m = re.match(r"([-+]?[0-9.]+(?:[eE][-+]?\d+)?)\s*([a-zA-Z%]*)", t)
    if m:
        try:
            val = float(m.group(1))
        except ValueError:
            return None
        unit = m.group(2).lower()
        mult = {"k": 1e3, "m": 1e6, "g": 1e9, "t": 1e12}.get(unit[0] if unit else "", 1.0) if unit else 1.0
        if unit and unit[0] in "kmgt" and not m.group(2).startswith("%"):
            val *= mult
        return val
    return None


def _result_from_dict(d: dict[str, Any]) -> KernelMetricResult:
    r = KernelMetricResult(
        kernel=d.get("kernel", ""),
        grid=tuple(d.get("grid", (1, 1, 1))),  # type: ignore[arg-type]
        block=tuple(d.get("block", (1, 1, 1))),  # type: ignore[arg-type]
        metrics={k: float(v) for k, v in (d.get("metrics") or {}).items()},
        groups=d.get("groups") or {},
        unavailable=list(d.get("unavailable") or []),
        stage=d.get("stage", "B"),
        notes=list(d.get("notes") or []),
    )
    return r


def stage_c_needed(result: KernelMetricResult, catalog: MetricCatalog) -> bool:
    """Decide whether a kernel warrants the more expensive Stage C pass.

    Escalation is justified only when the Stage B evidence is *ambiguous*: both
    compute and memory utilisation are mid-range so no ceiling is clearly the
    limit, or the stall breakdown is missing.
    """
    comp = _first(result.metrics, "sm__throughput.avg.pct_of_peak_sustained_elapsed")
    mem = _first(result.metrics, "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
                 "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed")
    have_stalls = any("issue_stalled" in k for k in result.metrics)
    if not have_stalls:
        return True
    if comp is None or mem is None:
        return True
    if 35.0 <= comp <= 65.0 and 35.0 <= mem <= 65.0:
        return True
    return False


def _first(metrics: dict[str, float], *names: str) -> float | None:
    for n in names:
        if n in metrics:
            return metrics[n]
    return None
