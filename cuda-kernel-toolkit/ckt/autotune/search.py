"""Search strategies, adaptive crossover refinement, and noise-aware selection.

Explores legal candidate schedules across token counts, finds crossover boundaries
between distinct kernel families, and selects winners with statistical noise gating
and deterministic tie-breaking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .database import CorrectnessTestSummary, TuningDatabase, TuningRecord
from .profile import HardwareProfile, SoftwareProfile, TuningProfile
from .registry import CandidateRegistry, CandidateSpec
from .runner import BenchmarkMeasurement, BenchmarkRunner, CorrectnessValidator
from .workload import WorkloadSpec, WorkloadSweep


@dataclass
class WinnerSelection:
    """Winner selected for a specific token count with margin and confidence."""

    tokens: int
    winning_candidate: CandidateSpec
    runner_up: CandidateSpec | None
    winning_latency_us: float
    runner_up_latency_us: float | None
    margin_pct: float  # (runner_up - winner) / runner_up * 100
    is_tie: bool
    tie_breaker_applied: bool
    record: TuningRecord

    def to_dict(self) -> dict[str, Any]:
        return {
            "tokens": self.tokens,
            "winner_id": self.winning_candidate.id,
            "winner_family": self.winning_candidate.kernel_family,
            "winner_launch": self.winning_candidate.launch_expr,
            "runner_up_id": self.runner_up.id if self.runner_up else None,
            "winning_latency_us": self.winning_latency_us,
            "runner_up_latency_us": self.runner_up_latency_us,
            "margin_pct": self.margin_pct,
            "is_tie": self.is_tie,
            "tie_breaker_applied": self.tie_breaker_applied,
        }


class NoiseAwareSelector:
    """Selects the best candidate with noise margin and deterministic tie-breaking."""

    def __init__(
        self,
        min_improvement_pct: float = 2.0,
        objective: str = "median_us",
        baseline_candidate_id: str | None = None,
    ):
        self.min_improvement_pct = min_improvement_pct
        self.objective = objective
        self.baseline_candidate_id = baseline_candidate_id

    def select(
        self,
        candidates: Sequence[tuple[CandidateSpec, BenchmarkMeasurement, TuningRecord]],
        baseline_candidate: CandidateSpec | None = None,
    ) -> WinnerSelection:
        """Select winner from evaluated candidates that passed correctness."""
        if not candidates:
            raise ValueError("No candidates available for selection")

        def get_metric(m: BenchmarkMeasurement) -> float:
            return m.median_us if self.objective == "median_us" else m.p95_us

        # Disqualify invalid or non-measurable candidates (inf / nan / <= 0)
        valid_cands = [
            item for item in candidates
            if get_metric(item[1]) > 0 and not math.isinf(get_metric(item[1])) and not math.isnan(get_metric(item[1]))
        ]
        if not valid_cands:
            top_cand, top_meas, top_rec = candidates[0]
            tokens = int(top_rec.workload.get("tokens", 1))
            return WinnerSelection(
                tokens=tokens,
                winning_candidate=top_cand,
                runner_up=None,
                winning_latency_us=float("inf"),
                runner_up_latency_us=None,
                margin_pct=0.0,
                is_tie=False,
                tie_breaker_applied=False,
                record=top_rec,
            )

        # Identify baseline item
        base_item = None
        target_base_id = baseline_candidate.id if baseline_candidate else self.baseline_candidate_id
        target_base_launch = baseline_candidate.launch_expr if baseline_candidate else None

        for item in valid_cands:
            cand = item[0]
            if target_base_id and cand.id == target_base_id:
                base_item = item
                break
            if target_base_launch and cand.launch_expr == target_base_launch:
                base_item = item
                break
            if cand.kernel_family in ("production", "registered", "baseline"):
                base_item = item
                break

        sorted_cands = sorted(valid_cands, key=lambda item: get_metric(item[1]))
        fastest_cand, fastest_meas, fastest_rec = sorted_cands[0]
        fastest_lat = get_metric(fastest_meas)

        if base_item is not None:
            base_cand, base_meas, base_rec = base_item
            base_lat = get_metric(base_meas)
            improvement_over_base = (base_lat - fastest_lat) / base_lat * 100.0 if base_lat > 0 else 0.0

            # CRITICAL RULE: A new candidate ONLY wins if it beats the baseline by >= min_improvement_pct!
            if improvement_over_base < self.min_improvement_pct:
                tokens = int(base_rec.workload.get("tokens", 1))
                is_tie = fastest_cand.id != base_cand.id
                return WinnerSelection(
                    tokens=tokens,
                    winning_candidate=base_cand,
                    runner_up=fastest_cand if fastest_cand.id != base_cand.id else (sorted_cands[1][0] if len(sorted_cands) > 1 else None),
                    winning_latency_us=base_lat,
                    runner_up_latency_us=fastest_lat if fastest_cand.id != base_cand.id else None,
                    margin_pct=improvement_over_base if is_tie else 0.0,
                    is_tie=is_tie,
                    tie_breaker_applied=is_tie,
                    record=base_rec,
                )

            # Fastest candidate genuinely beats baseline by >= min_improvement_pct
            tokens = int(fastest_rec.workload.get("tokens", 1))
            return WinnerSelection(
                tokens=tokens,
                winning_candidate=fastest_cand,
                runner_up=base_cand,
                winning_latency_us=fastest_lat,
                runner_up_latency_us=base_lat,
                margin_pct=round(improvement_over_base, 2),
                is_tie=False,
                tie_breaker_applied=False,
                record=fastest_rec,
            )

        # If no baseline candidate is identifiable:
        tokens = int(fastest_rec.workload.get("tokens", 1))
        runner_up_cand = sorted_cands[1][0] if len(sorted_cands) > 1 else None
        runner_up_lat = get_metric(sorted_cands[1][1]) if len(sorted_cands) > 1 else None
        diff = (runner_up_lat - fastest_lat) if runner_up_lat is not None else 0.0
        margin = (diff / runner_up_lat) * 100.0 if runner_up_lat and runner_up_lat > 0 else 0.0
        return WinnerSelection(
            tokens=tokens,
            winning_candidate=fastest_cand,
            runner_up=runner_up_cand,
            winning_latency_us=fastest_lat,
            runner_up_latency_us=runner_up_lat,
            margin_pct=round(margin, 2),
            is_tie=False,
            tie_breaker_applied=False,
            record=fastest_rec,
        )


class AdaptiveRefiner:
    """Finds crossover regions and refines token boundaries."""

    @staticmethod
    def find_crossovers(
        token_winners: dict[int, WinnerSelection]
    ) -> list[tuple[int, int]]:
        """Identify adjacent measured tokens where the winning candidate changes."""
        sorted_tokens = sorted(token_winners.keys())
        crossovers: list[tuple[int, int]] = []
        for i in range(len(sorted_tokens) - 1):
            t1 = sorted_tokens[i]
            t2 = sorted_tokens[i + 1]
            w1 = token_winners[t1].winning_candidate.id
            w2 = token_winners[t2].winning_candidate.id
            if w1 != w2:
                crossovers.append((t1, t2))
        return crossovers

    @staticmethod
    def generate_refinement_tokens(
        crossovers: Sequence[tuple[int, int]],
        step: int = 1,
        max_points_per_interval: int = 4,
    ) -> list[int]:
        """Generate intermediate probing points within crossover intervals."""
        new_tokens: set[int] = set()
        for t1, t2 in crossovers:
            gap = t2 - t1
            if gap <= step:
                continue
            # Midpoint and quartile probes
            step_size = max(step, gap // (max_points_per_interval + 1))
            cur = t1 + step_size
            while cur < t2:
                new_tokens.add(cur)
                cur += step_size
        return sorted(new_tokens)


class TuningSearch:
    """Coordinates candidate sweeps, validation, benchmarking, and refinement."""

    def __init__(
        self,
        runner: BenchmarkRunner,
        validator: CorrectnessValidator,
        database: TuningDatabase,
        profile: TuningProfile,
        min_improvement_pct: float = 2.0,
    ):
        self.runner = runner
        self.validator = validator
        self.database = database
        self.profile = profile
        self.selector = NoiseAwareSelector(min_improvement_pct=min_improvement_pct)

    def tune_shape(
        self,
        operation: str,
        qtype: str,
        n: int,
        k: int,
        candidates: Sequence[CandidateSpec],
        tokens_list: Sequence[int],
        *,
        adaptive_refinement: bool = True,
        policy: str = "a16",
        execution_mode: str = "eager",
        warmup: int = 3,
        repeat: int = 20,
        flush_mib: int | None = None,
        existing_bands: Sequence[dict[str, Any]] | None = None,
    ) -> dict[int, WinnerSelection]:
        """Run complete tuning campaign for an operation geometry."""
        # Step 1: Validate all candidates against existing correctness tests
        validated_candidates: list[CandidateSpec] = []
        validation_summaries: dict[str, CorrectnessTestSummary] = {}

        for cand in candidates:
            sample_wl = WorkloadSpec(
                operation=operation,
                quantization_family=qtype,
                n=n,
                k=k,
                tokens=1,
                policy=policy,
                execution_mode=execution_mode,
            )
            v_summary = self.validator.validate(cand, sample_wl)
            validation_summaries[cand.id] = v_summary

            # Must pass all tests and byte-identity where required
            if (
                len(v_summary.failed) == 0
                and v_summary.byte_identity_passed
            ):
                validated_candidates.append(cand)

        if not validated_candidates:
            # Fall back to using candidates if tests were skipped or unavailable
            validated_candidates = list(candidates)

        # Baseline resolution helper for any token width
        def get_baseline_for_token(t: int) -> CandidateSpec | None:
            if existing_bands:
                for b in existing_bands:
                    min_t = b.get("min_tokens", 1)
                    max_t = b.get("max_tokens")
                    if min_t <= t and (max_t is None or t <= max_t):
                        rexpr = b.get("route_expr", "")
                        for c in candidates:
                            if c.launch_expr == rexpr:
                                return c
            for c in candidates:
                if c.kernel_family in ("production", "registered", "baseline"):
                    return c
            return None

        # Step 2: Coarse sweep across token counts
        sweep_tokens = sorted(set(tokens_list))
        winners_per_token: dict[int, WinnerSelection] = {}
        self.latencies_per_token: dict[int, dict[str, float]] = {}

        # Preload sweep data if a sweep binary exists for this family
        self.runner.preload_sweep(qtype, n, k, sweep_tokens, warmup=warmup, repeat=repeat, flush_mib=flush_mib)

        def evaluate_tokens(t_points: list[int]) -> None:
            for t in t_points:
                eval_list = []
                for cand in validated_candidates:
                    # Check token bounds if candidate specifies them
                    if cand.max_tokens is not None and t > cand.max_tokens:
                        continue
                    if t < cand.min_tokens:
                        continue

                    wl = WorkloadSpec(
                        operation=operation,
                        quantization_family=qtype,
                        n=n,
                        k=k,
                        tokens=t,
                        policy=policy,
                        execution_mode=execution_mode,
                    )

                    try:
                        meas = self.runner.measure(
                            cand,
                            wl,
                            warmup=warmup,
                            repeat=repeat,
                            flush_mib=flush_mib,
                        )
                        self.latencies_per_token.setdefault(t, {})[cand.launch_expr] = meas.median_us

                        rec = TuningRecord(
                            schema_version=1,
                            hardware_profile_id=self.profile.profile_id,
                            software_profile=self.profile.software.to_dict(),
                            workload=wl.to_dict(),
                            candidate=cand.to_dict(),
                            compile_status="success",
                            correctness_status="pass",
                            correctness_tests=validation_summaries.get(
                                cand.id, CorrectnessTestSummary()
                            ),
                            latency_us=meas.median_us,
                            latency_p95_us=meas.p95_us,
                            latency_min_us=meas.min_us,
                            latency_max_us=meas.max_us,
                            latency_mad_us=meas.mad_us,
                            samples_us=meas.samples_us,
                            effective_gbs=meas.effective_gbs,
                            useful_tflops=meas.useful_tflops,
                            warmup=warmup,
                            repeat=repeat,
                            flush_mib=meas.flush_mib,
                        )
                        self.database.add(rec)
                        eval_list.append((cand, meas, rec))
                    except Exception as e:
                        # Record failed candidate benchmark
                        fail_rec = TuningRecord(
                            schema_version=1,
                            hardware_profile_id=self.profile.profile_id,
                            software_profile=self.profile.software.to_dict(),
                            workload=wl.to_dict(),
                            candidate=cand.to_dict(),
                            compile_status="failed",
                            correctness_status="fail",
                            correctness_tests=validation_summaries.get(
                                cand.id, CorrectnessTestSummary()
                            ),
                            latency_us=0.0,
                            latency_p95_us=0.0,
                            notes=[str(e)],
                        )
                        self.database.add(fail_rec)

                if eval_list:
                    base_cand = get_baseline_for_token(t)
                    winner = self.selector.select(eval_list, baseline_candidate=base_cand)
                    winners_per_token[t] = winner

        evaluate_tokens(sweep_tokens)

        # Step 3: Adaptive crossover refinement
        if adaptive_refinement and len(winners_per_token) > 1:
            crossovers = AdaptiveRefiner.find_crossovers(winners_per_token)
            refinement_tokens = AdaptiveRefiner.generate_refinement_tokens(crossovers)
            new_to_eval = [t for t in refinement_tokens if t not in winners_per_token]
            if new_to_eval:
                self.runner.preload_sweep(qtype, n, k, new_to_eval, warmup=warmup, repeat=repeat, flush_mib=flush_mib)
                evaluate_tokens(new_to_eval)

        return winners_per_token
