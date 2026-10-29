"""parameter_search — hardware-constrained launch-configuration search.

The search explores launch parameters (block size, grid size, shared memory) that
are *legal on the target* and *pruned by the occupancy model* before anything is
executed.  The point is to run a small number of informative experiments rather
than a large number of doomed ones: a configuration whose block cannot be
resident, or whose shared-memory request exceeds the per-block opt-in limit, is
rejected analytically instead of costing a launch.

Two rules are enforced throughout:

* **Correctness gates every result.**  A candidate whose output fails
  verification is recorded as a failure and never becomes the selected
  configuration, however fast it looks.
* **A candidate is only comparable if the work is identical.**  Every candidate
  for one search runs the same workload and the same problem size; only launch
  parameters vary.  Changing the problem size between candidates (or letting a
  tile-width parameter change the number of elements processed) is what produces
  the spurious speedups described in the project's reporting rules, so the
  problem size is held fixed and recorded with the results.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from .correctness import CorrectnessGate, gate_from_verification, verification_from_records
from .environment import DeviceInfo
from .performance_model import OccupancyModel, occupancy_model, wave_analysis
from .profiler import parse_kv_records, run_harness
from .util import LOG, SampleStats, percentile


# --------------------------------------------------------------------------- #
# Configuration candidates
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LaunchConfig:
    blocks: int
    threads: int
    shared_mem: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"blocks": self.blocks, "threads": self.threads, "shared_mem": self.shared_mem}

    def args(self) -> list[str]:
        return [
            "--blocks", str(self.blocks),
            "--threads", str(self.threads),
            "--smem", str(self.shared_mem),
        ]

    @property
    def label(self) -> str:
        return f"b{self.blocks}_t{self.threads}_s{self.shared_mem}"


@dataclass
class ConstraintViolation:
    config: LaunchConfig
    reason: str


@dataclass
class CandidateEvaluation:
    config: LaunchConfig
    occupancy: OccupancyModel | None = None
    waves: dict[str, Any] | None = None
    stats: SampleStats | None = None
    latency_us: float | None = None
    baseline_us: float | None = None
    speedup: float | None = None
    correctness: dict[str, Any] | None = None
    passed_correctness: bool = False
    records: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    pruned: bool = False
    prune_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.config.label,
            "config": self.config.to_dict(),
            "latency_us": self.latency_us,
            "baseline_us": self.baseline_us,
            "speedup": self.speedup,
            "passed_correctness": self.passed_correctness,
            "correctness": self.correctness,
            "occupancy": self.occupancy.to_dict() if self.occupancy else None,
            "waves": self.waves,
            "stats": self.stats.to_dict() if self.stats else None,
            "error": self.error,
            "pruned": self.pruned,
            "prune_reason": self.prune_reason,
            "raw_records": self.records,
        }


# --------------------------------------------------------------------------- #
# Space definition and constraint checking
# --------------------------------------------------------------------------- #


@dataclass
class SearchSpace:
    threads: list[int] = field(default_factory=list)
    blocks: list[int] = field(default_factory=list)
    shared_mem: list[int] = field(default_factory=lambda: [0])
    # When blocks is empty, derive it from an SM multiplier instead.
    sm_multiples: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])
    # Fraction of the SM limit above which a candidate is considered to be using
    # a full wave; used by the model-guided strategy.
    min_occupancy_pct: float = 0.0
    require_wave_multiple: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "threads": self.threads,
            "blocks": self.blocks,
            "shared_mem": self.shared_mem,
            "sm_multiples": self.sm_multiples,
            "min_occupancy_pct": self.min_occupancy_pct,
            "require_wave_multiple": self.require_wave_multiple,
        }


def default_thread_candidates(dev: DeviceInfo) -> list[int]:
    """Block sizes worth trying, restricted to the device's real limits.

    32-thread multiples only: a partial warp occupies a warp slot but cannot use
    all of its lanes, so it is never a preferred configuration and its inclusion
    would only enlarge the search.
    """
    warp = dev.warp_size or 32
    candidates = [warp * m for m in (1, 2, 4, 8, 6, 12, 16, 24, 32)]
    out = sorted({c for c in candidates if c <= dev.max_threads_per_block})
    # Always include the largest legal power-of-two and the hardware maximum.
    for extra in (256, 512, 1024, dev.max_threads_per_block):
        if 0 < extra <= dev.max_threads_per_block and extra % warp == 0:
            out.append(extra)
    return sorted(set(out))


def check_constraints(dev: DeviceInfo, cfg: LaunchConfig) -> str | None:
    """Return a human-readable reason the configuration cannot launch, or None."""
    warp = dev.warp_size or 32
    if cfg.threads <= 0:
        return "threads must be positive"
    if cfg.threads > dev.max_threads_per_block:
        return f"threads ({cfg.threads}) exceeds the device maximum ({dev.max_threads_per_block})"
    if cfg.threads % warp != 0:
        return (
            f"threads ({cfg.threads}) is not a multiple of the warp size ({warp}); the final "
            "partial warp wastes lanes"
        )
    if cfg.blocks <= 0:
        return "blocks must be positive"
    if cfg.blocks > 2**31 - 1:
        return "blocks exceeds the maximum grid dimension"
    if cfg.shared_mem < 0:
        return "shared memory cannot be negative"
    optin = dev.shared_mem_per_block_optin or dev.shared_mem_per_block
    if cfg.shared_mem > optin:
        return (
            f"dynamic shared memory ({cfg.shared_mem} B) exceeds the per-block opt-in limit "
            f"({optin} B)"
        )
    if cfg.shared_mem and cfg.shared_mem <= dev.shared_mem_per_block:
        return None  # fits within the default limit; no opt-in required
    return None


def generate_candidates(
    dev: DeviceInfo,
    space: SearchSpace,
    *,
    strategy: str = "model-guided",
    registers_per_thread: int | None = None,
    budget: int | None = None,
    seed_blocks: int | None = None,
    seed_threads: int | None = None,
) -> tuple[list[LaunchConfig], list[ConstraintViolation]]:
    """Produce a ranked, deduplicated list of legal configurations.

    Strategies differ in which configurations they consider first, not in which
    ones are legal; the legality filter is identical for all of them.

    * ``exhaustive``    — the full Cartesian product.
    * ``coarse-to-fine``— powers of two first, with intermediate sizes added
                          only when the budget allows.
    * ``stratified``    — interleaved by block size so an early budget cut still
                          covers a range of block sizes.
    * ``model-guided``  — ranked by predicted occupancy, then by how close the
                          grid is to an exact number of waves.
    """
    threads = space.threads or default_thread_candidates(dev)
    if space.blocks:
        blocks = list(space.blocks)
    else:
        blocks = [
            max(1, dev.multi_processor_count * m) for m in space.sm_multiples
        ] if space.sm_multiples else [dev.multi_processor_count * 8]
    if seed_threads and seed_threads not in threads and seed_threads <= dev.max_threads_per_block:
        threads = sorted(set(threads + [seed_threads]))
    if seed_blocks and seed_blocks not in blocks:
        blocks = sorted(set(blocks + [seed_blocks]))

    combos = [LaunchConfig(b, t, s) for b, t, s in itertools.product(blocks, threads, space.shared_mem)]

    legal: list[LaunchConfig] = []
    violations: list[ConstraintViolation] = []
    for c in combos:
        reason = check_constraints(dev, c)
        if reason:
            violations.append(ConstraintViolation(c, reason))
        else:
            legal.append(c)

    def occ_of(c: LaunchConfig) -> OccupancyModel:
        return occupancy_model(
            dev,
            threads_per_block=c.threads,
            registers_per_thread=registers_per_thread or 0,
            shared_mem_per_block=c.shared_mem,
        )

    if strategy == "exhaustive":
        ordered = legal
    elif strategy == "coarse-to-fine":
        pow2 = {32, 64, 128, 256, 512, 1024, 2048}
        first = [c for c in legal if c.threads in pow2]
        second = [c for c in legal if c.threads not in pow2]
        ordered = first + second
    elif strategy == "stratified":
        by_threads: dict[int, list[LaunchConfig]] = {}
        for c in legal:
            by_threads.setdefault(c.threads, []).append(c)
        ordered = []
        buckets = list(by_threads.values())
        for i in range(max((len(b) for b in buckets), default=0)):
            for b in buckets:
                if i < len(b):
                    ordered.append(b[i])
    else:  # model-guided
        def score(c: LaunchConfig) -> tuple[float, float, int]:
            occ = occ_of(c)
            w = wave_analysis(dev, occ, c.blocks)
            # Prefer high occupancy, then a grid that lands on whole waves, then
            # fewer blocks (less launch and scheduling overhead).
            wave_penalty = abs(w.waves - round(w.waves)) if w.waves >= 1 else 1.0
            return (-occ.occupancy_pct, wave_penalty, c.blocks)

        ordered = sorted(legal, key=score)

    # Apply the caller's occupancy floor and wave-multiple requirement.
    filtered: list[LaunchConfig] = []
    for c in ordered:
        occ = occ_of(c)
        if space.min_occupancy_pct and occ.occupancy_pct < space.min_occupancy_pct:
            violations.append(
                ConstraintViolation(
                    c,
                    f"predicted occupancy {occ.occupancy_pct:.1f}% is below the requested floor "
                    f"of {space.min_occupancy_pct:.1f}%",
                )
            )
            continue
        if space.require_wave_multiple and registers_per_thread:
            w = wave_analysis(dev, occ, c.blocks)
            if w.tail_ctas:
                violations.append(
                    ConstraintViolation(
                        c, f"grid is not a whole number of waves ({w.waves:.2f} waves, {w.tail_ctas} tail CTAs)"
                    )
                )
                continue
        filtered.append(c)

    # De-duplicate while preserving the strategy's order.
    seen: set[tuple[int, int, int]] = set()
    deduped: list[LaunchConfig] = []
    for c in filtered:
        k = (c.blocks, c.threads, c.shared_mem)
        if k not in seen:
            seen.add(k)
            deduped.append(c)

    if budget and len(deduped) > budget:
        deduped = deduped[:budget]
    return deduped, violations


# --------------------------------------------------------------------------- #
# Search execution
# --------------------------------------------------------------------------- #


@dataclass
class SearchResult:
    workload: str = ""
    problem_size: int = 0
    baseline_config: LaunchConfig | None = None
    baseline_us: float | None = None
    baseline_stats: SampleStats | None = None
    baseline_correctness: dict[str, Any] | None = None
    candidates: list[CandidateEvaluation] = field(default_factory=list)
    violations: list[ConstraintViolation] = field(default_factory=list)
    selected: CandidateEvaluation | None = None
    strategy: str = "model-guided"
    wall_time_s: float = 0.0
    iterations: int = 0
    warmup: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "workload": self.workload,
            "problem_size": self.problem_size,
            "baseline_config": self.baseline_config.to_dict() if self.baseline_config else None,
            "baseline_us": self.baseline_us,
            "baseline_stats": self.baseline_stats.to_dict() if self.baseline_stats else None,
            "baseline_correctness": self.baseline_correctness,
            "candidates": [c.to_dict() for c in self.candidates],
            "violations": [{"config": v.config.to_dict(), "reason": v.reason} for v in self.violations],
            "selected": self.selected.to_dict() if self.selected else None,
            "strategy": self.strategy,
            "wall_time_s": self.wall_time_s,
            "iterations": self.iterations,
            "warmup": self.warmup,
            "notes": self.notes,
        }


def _validated_records(records: dict[str, str], workload: str) -> tuple[bool, dict[str, Any]]:
    g: CorrectnessGate = gate_from_verification(verification_from_records(records, workload=workload))
    return g.validated, g.to_dict()


def run_search(
    exe: str | Path,
    *,
    dev: DeviceInfo,
    workload: str,
    problem_size_args: Sequence[str],
    problem_size: int = 0,
    candidates: Sequence[LaunchConfig],
    violations: Sequence[ConstraintViolation] = (),
    baseline_config: LaunchConfig | None = None,
    registers_per_thread: int | None = None,
    iterations: int = 30,
    warmup: int = 5,
    repeats: int = 1,
    verify: bool = True,
    budget_seconds: float | None = None,
    strategy: str = "model-guided",
    timeout: float = 900.0,
) -> SearchResult:
    """Run each candidate, gate correctness, and select the fastest legal one.

    ``problem_size_args`` is applied identically to every run, so all candidates
    perform the same work; a speedup reported here is a change in launch
    configuration only.
    """
    t0 = time.time()
    res = SearchResult(
        workload=workload, problem_size=problem_size, strategy=strategy,
        iterations=iterations, warmup=warmup,
    )
    res.violations = list(violations)
    res.baseline_config = baseline_config

    def measure(cfg: LaunchConfig, iters: int) -> tuple[SampleStats | None, dict[str, str], str | None]:
        args = list(problem_size_args) + cfg.args() + ["--iters", str(iters), "--warmup", str(warmup)]
        if verify:
            # Verification first, on a single fresh launch: a timed run may have
            # accumulated in-place updates, which would make the check fail for
            # reasons unrelated to correctness of the configuration.
            vargs = list(problem_size_args) + cfg.args() + ["--verify"]
            try:
                vres = run_harness(exe, vargs, timeout=timeout, repeats=1, log=False)
                if vres[0]:
                    verified = {**vres[0]}
            except Exception:  # noqa: BLE001 - a failed verify is a failed candidate
                verified = {}
        else:
            verified = {}
        try:
            records, _ = run_harness(exe, args, timeout=timeout, repeats=1, log=False)
        except Exception as exc:  # noqa: BLE001
            return None, verified, str(exc)
        if verified:
            records = {**records, **{k: v for k, v in verified.items() if k.startswith("verify.")}}
        samples: list[float] = []
        for key in ("timing.median_us",):
            if key in records:
                try:
                    samples.append(float(records[key]))
                except ValueError:
                    pass
        if not samples:
            return None, records, "harness produced no timing.median_us record"
        return SampleStats.from_values(samples), records, None

    # Baseline first so every candidate has a common reference point measured in
    # the same process/session.
    if baseline_config is not None:
        stats, records, err = measure(baseline_config, iterations)
        if err:
            res.notes.append(f"baseline measurement failed: {err}")
        if stats:
            res.baseline_us = stats.median
            res.baseline_stats = stats
        ok, gate = _validated_records(records, workload)
        res.baseline_correctness = gate
        if not ok:
            res.notes.append(
                "Baseline did not pass verification; candidate speedups are not comparable "
                "and no configuration may be selected."
            )

    for cfg in candidates:
        if budget_seconds is not None and (time.time() - t0) > budget_seconds:
            res.notes.append(
                f"Budget of {budget_seconds:.0f}s exhausted after {len(res.candidates)} candidates; "
                "remaining configurations were not evaluated."
            )
            break
        ev = CandidateEvaluation(config=cfg)
        if registers_per_thread:
            ev.occupancy = occupancy_model(
                dev, threads_per_block=cfg.threads,
                registers_per_thread=registers_per_thread, shared_mem_per_block=cfg.shared_mem,
            )
            ev.waves = wave_analysis(dev, ev.occupancy, cfg.blocks).to_dict()
        stats, records, err = measure(cfg, iterations)
        ev.records = records
        if err:
            ev.error = err
            res.candidates.append(ev)
            continue
        ev.stats = stats
        ev.latency_us = stats.median if stats else None
        if ev.latency_us and res.baseline_us:
            ev.baseline_us = res.baseline_us
            ev.speedup = res.baseline_us / ev.latency_us
        ok, gate = _validated_records(records, workload)
        ev.passed_correctness = ok
        ev.correctness = gate
        res.candidates.append(ev)

    # ---- selection ------------------------------------------------------- #
    eligible = [c for c in res.candidates if c.passed_correctness and c.latency_us]
    if not eligible:
        res.notes.append(
            "No candidate both passed verification and produced a timing. Nothing was selected."
        )
    else:
        # Empirical noise floor.  When the baseline configuration is also present
        # in the candidate set it is measured twice in the same session, and the
        # discrepancy between those two numbers is a direct measurement of the
        # run-to-run variation.  That value — not an assumed percentage — decides
        # whether any candidate's "speedup" is meaningful.
        noise_floor_pct: float | None = None
        if res.baseline_us and res.baseline_config is not None:
            dup = next(
                (
                    c for c in eligible
                    if c.config.blocks == res.baseline_config.blocks
                    and c.config.threads == res.baseline_config.threads
                    and c.config.shared_mem == res.baseline_config.shared_mem
                ),
                None,
            )
            if dup and dup.latency_us:
                noise_floor_pct = 100.0 * abs(dup.latency_us - res.baseline_us) / res.baseline_us
                res.notes.append(
                    f"Self-consistency check: the baseline configuration measured {res.baseline_us:.2f} us "
                    f"as the baseline and {dup.latency_us:.2f} us as a candidate — a difference of "
                    f"{noise_floor_pct:.2f}% for identical work. Differences at or below roughly this "
                    "level cannot be attributed to the configuration change."
                )

        best = min(eligible, key=lambda c: c.latency_us or float("inf"))
        ties = [
            c for c in eligible
            if c.latency_us and best.latency_us and c.latency_us <= best.latency_us * 1.01
        ]
        res.selected = min(ties, key=lambda c: (c.config.blocks, c.config.shared_mem))
        if len(ties) > 1 and res.selected is not best:
            res.notes.append(
                f"{len(ties)} configurations were within 1% of the fastest "
                f"({best.latency_us:.2f} us); selected {res.selected.config.label} as the "
                "lowest-risk member of that group. The difference is not resolved by this "
                "number of repeats."
            )
        # Do not let a sub-noise difference be reported as an improvement.
        if res.selected.speedup is not None and res.selected.speedup > 1.0:
            gain_pct = 100.0 * (res.selected.speedup - 1.0)
            if noise_floor_pct is not None and gain_pct <= noise_floor_pct:
                res.selected.speedup = None
                res.notes.append(
                    f"The apparent {gain_pct:.2f}% gain is within the measured run-to-run "
                    f"variation ({noise_floor_pct:.2f}%), so no improvement is claimed. "
                    "Increase repeats or reduce measurement noise before re-testing."
                )
            elif noise_floor_pct is None and gain_pct < 2.0:
                res.notes.append(
                    f"The apparent {gain_pct:.2f}% gain may be within measurement noise; the "
                    "self-consistency check could not be performed (the baseline configuration "
                    "was not among the candidates). Re-measure with repeats before relying on it."
                )
        if len(eligible) < len(res.candidates):
            failed = len(res.candidates) - len(eligible)
            res.notes.append(
                f"{failed} of {len(res.candidates)} evaluated candidate(s) were excluded from "
                "selection for failing verification or producing no timing."
            )

    res.wall_time_s = time.time() - t0
    return res


def revalidate(
    exe: str | Path,
    *,
    dev: DeviceInfo,
    workload: str,
    problem_size_args: Sequence[str],
    config: LaunchConfig,
    iterations: int = 50,
    warmup: int = 10,
    repeats: int = 3,
    timeout: float = 900.0,
) -> CandidateEvaluation:
    """Re-measure a selected configuration with more repeats.

    A selection made from single-shot measurements is provisional; this re-run
    establishes whether the difference survives repetition before it is reported.
    """
    ev = CandidateEvaluation(config=config)
    medians: list[float] = []
    for _ in range(max(1, repeats)):
        vargs = list(problem_size_args) + config.args() + ["--verify"]
        try:
            vrecords, _ = run_harness(exe, vargs, timeout=timeout, repeats=1, log=False)
        except Exception as exc:  # noqa: BLE001
            ev.error = str(exc)
            return ev
        ok, gate = _validated_records(vrecords, workload)
        ev.passed_correctness = ok
        ev.correctness = gate
        ev.records = vrecords
        if not ok:
            return ev
        args = list(problem_size_args) + config.args() + ["--iters", str(iterations), "--warmup", str(warmup)]
        try:
            records, _ = run_harness(exe, args, timeout=timeout, repeats=1, log=False)
        except Exception as exc:  # noqa: BLE001
            ev.error = str(exc)
            return ev
        try:
            medians.append(float(records.get("timing.median_us", "")))
        except ValueError:
            continue
    if medians:
        ev.stats = SampleStats.from_values(medians)
        ev.latency_us = ev.stats.median
    else:
        ev.error = "revalidation produced no timing"
    return ev


def summarise_search(result: SearchResult) -> list[str]:
    """Human-readable lines describing a search outcome, including failures."""
    lines: list[str] = []
    if result.baseline_us:
        lines.append(
            f"Baseline {result.baseline_config.label if result.baseline_config else '?'}: "
            f"{result.baseline_us:.2f} us"
        )
    if result.selected:
        s = result.selected
        lines.append(
            f"Selected {s.config.label}: {s.latency_us:.2f} us"
            + (f" ({s.speedup:.3f}x vs baseline)" if s.speedup else "")
        )
    failed = [c for c in result.candidates if not c.passed_correctness]
    if failed:
        lines.append(
            f"{len(failed)} candidate(s) failed verification and were excluded: "
            + ", ".join(c.config.label for c in failed[:8])
        )
    if result.violations:
        lines.append(f"{len(result.violations)} configuration(s) rejected by the hardware/occupancy model.")
    lines.extend(result.notes)
    return lines
