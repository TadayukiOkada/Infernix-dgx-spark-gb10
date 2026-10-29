"""recommendations — evidence-linked, ranked optimisation suggestions.

Every recommendation is a chain, not an assertion:

    observed symptom  ->  supporting metric (with its value and provenance)
                      ->  suspected cause
                      ->  proposed change
                      ->  expected effect and its trade-offs
                      ->  confidence
                      ->  the experiment that would confirm or refute it

The module separates recommendations that only change launch parameters from
those that require a source change, because the two have very different costs and
very different risks to correctness.  It also refuses to emit a recommendation
whose supporting metric is unavailable: "could not be determined" is a valid
outcome and is reported as such.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .environment import DeviceInfo
from .performance_model import (
    Classification,
    OccupancyModel,
    RooflineReport,
    WaveAnalysis,
)
from .profiler import KernelMetricResult
from .util import LOG


# Effort and risk are ordinal; impact is derived from measured headroom where a
# ceiling is known, and left as "unknown" otherwise rather than guessed.
EFFORT_ORDER = {"trivial": 0, "low": 1, "moderate": 2, "high": 3}
RISK_ORDER = {"none": 0, "low": 1, "moderate": 2, "high": 3}


@dataclass
class Recommendation:
    title: str = ""
    kind: str = "parameter"           # "parameter" | "source" | "measurement"
    symptom: str = ""
    evidence: list[str] = field(default_factory=list)
    metric_values: dict[str, Any] = field(default_factory=dict)
    suspected_cause: str = ""
    proposed_change: str = ""
    expected_effect: str = ""
    trade_offs: str = ""
    confidence: str = "low"           # low | medium | high
    effort: str = "moderate"
    risk: str = "low"
    validation: str = ""
    impact: str = "unknown"           # bounded estimate when a ceiling exists
    bounded_by: str = ""

    # ------------------------------------------------------------------ #
    @property
    def rank_key(self) -> tuple[int, int, int, int]:
        """Sort key: high confidence first, then low effort, low risk."""
        conf = {"high": 0, "medium": 1, "low": 2}.get(self.confidence, 3)
        return (conf, EFFORT_ORDER.get(self.effort, 2), RISK_ORDER.get(self.risk, 2), 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "kind": self.kind,
            "symptom": self.symptom,
            "evidence": self.evidence,
            "metric_values": self.metric_values,
            "suspected_cause": self.suspected_cause,
            "proposed_change": self.proposed_change,
            "expected_effect": self.expected_effect,
            "trade_offs": self.trade_offs,
            "confidence": self.confidence,
            "effort": self.effort,
            "risk": self.risk,
            "validation": self.validation,
            "impact": self.impact,
            "bounded_by": self.bounded_by,
        }


@dataclass
class RecommendationSet:
    kernel: str = ""
    classification: str = ""
    recommendations: list[Recommendation] = field(default_factory=list)
    unavailable_checks: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kernel": self.kernel,
            "classification": self.classification,
            "recommendations": [r.to_dict() for r in self.recommendations],
            "unavailable_checks": self.unavailable_checks,
            "notes": self.notes,
        }


def _mv(result: KernelMetricResult, *names: str) -> tuple[float | None, str]:
    """Return a metric value together with the name it was found under."""
    for n in names:
        if n in result.metrics:
            return result.metrics[n], n
    return None, ""


def _fmt(v: float | None, unit: str = "") -> str:
    if v is None:
        return "n/a"
    return f"{v:.2f}{unit}"


def recommend(
    *,
    result: KernelMetricResult,
    dev: DeviceInfo,
    classification: Classification,
    occupancy: OccupancyModel | None = None,
    roofline: RooflineReport | None = None,
    waves: WaveAnalysis | None = None,
    instruction_profile: Any | None = None,
    max_load_width_bytes: int | None = None,
) -> RecommendationSet:
    """Build the ranked recommendation set for one kernel."""
    rs = RecommendationSet(kernel=result.kernel, classification=classification.category)
    cat = classification.category

    # Expected sectors per request for a fully coalesced warp.  A warp moving
    # `width` bytes per lane covers 32*width bytes, which is `width` sectors of
    # 32 B.  Using a fixed constant here mis-flags wide vector loads as
    # uncoalesced, so the bound must follow the access width actually emitted.
    if max_load_width_bytes is None and instruction_profile is not None:
        max_load_width_bytes = getattr(instruction_profile, "max_load_width", 0) or None
    expected_spr = max(1.0, float(max_load_width_bytes)) if max_load_width_bytes else 4.0

    comp, comp_name = _mv(result, "sm__throughput.avg.pct_of_peak_sustained_elapsed")
    mem, mem_name = _mv(result, "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed")
    l2, _ = _mv(result, "lts__throughput.avg.pct_of_peak_sustained_elapsed")
    warps, _ = _mv(result, "sm__warps_active.avg.pct_of_peak_sustained_active")
    issue, _ = _mv(result, "smsp__issue_active.avg.pct_of_peak_sustained_active")
    spr, _ = _mv(result, "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio")
    hit, _ = _mv(result, "lts__t_sector_hit_rate.pct")
    regs, _ = _mv(result, "launch__registers_per_thread")
    # Fall back to the compiler's report when NCU did not supply the value; the
    # provenance of the number matters, so it is recorded rather than blended.
    if regs is None and occupancy is not None and occupancy.registers_per_thread:
        regs = float(occupancy.registers_per_thread)

    # ---------------------------------------------------------------- #
    # Category-driven recommendations.
    # ---------------------------------------------------------------- #
    if cat == "memory-bandwidth-bound":
        headroom = ""
        impact = "unknown"
        if roofline and roofline.bandwidth_ceiling_gbps and roofline.effective_bandwidth_gbps:
            impact = "none" if roofline.memory_utilization_pct and roofline.memory_utilization_pct >= 95 else "moderate"
            headroom = (
                f"Effective bandwidth {roofline.effective_bandwidth_gbps:.1f} GB/s against a measured "
                f"ceiling of {roofline.bandwidth_ceiling_gbps:.1f} GB/s"
            )
        ev = [f"Memory throughput {_fmt(mem, '%')} of peak ({mem_name})"]
        if headroom:
            ev.append(headroom)
        if spr is not None:
            ev.append(f"Loads average {spr:.2f} sectors per request")
        if hit is not None:
            ev.append(f"L2 hit rate {hit:.1f}%")
        rs.recommendations.append(
            Recommendation(
                title="Reduce bytes moved or improve reuse",
                kind="source",
                symptom="The memory system is saturated while execution pipes are idle",
                evidence=ev,
                metric_values={
                    "memory_throughput_pct": mem,
                    "l2_throughput_pct": l2,
                    "l2_hit_rate_pct": hit,
                    "sectors_per_request": spr,
                },
                suspected_cause=(
                    "Total traffic exceeds what the memory system can deliver in the kernel's "
                    "runtime. Either the byte volume is inherently too high, or part of the "
                    "traffic is wasted on partially-used cache lines."
                ),
                proposed_change=(
                    "In priority order: (1) fuse adjacent elementwise passes so intermediate "
                    "results stay on chip; (2) use a narrower dtype where precision allows; "
                    "(3) tile so that data reused across threads is loaded once into shared "
                    "memory or registers; (4) if sectors-per-request is high, restore coalescing "
                    "by reordering the thread-to-data mapping."
                ),
                expected_effect=(
                    "A bandwidth-bound kernel's runtime scales with byte volume, so a reduction in "
                    "traffic transfers almost directly into time."
                ),
                trade_offs=(
                    "Fusion couples previously independent kernels and complicates correctness "
                    "verification; narrower dtypes change numerical results and must be re-verified."
                ),
                confidence="high" if (mem is not None and mem >= 80.0) else "medium",
                effort="moderate",
                risk="moderate",
                validation=(
                    "Re-measure both latency and bytes moved; the L2/DRAM traffic ratio must fall "
                    "for the change to be the reason the kernel got faster."
                ),
                impact=impact,
                bounded_by=(
                    "hard: the measured DRAM ceiling" if headroom else "unknown: no measured bandwidth ceiling"
                ),
            )
        )

    elif cat == "memory-latency-bound":
        rs.recommendations.append(
            Recommendation(
                title="Increase memory-level parallelism per thread",
                kind="source",
                symptom="Warps stall on long-scoreboard waits while neither compute nor memory is saturated",
                evidence=[
                    f"Compute utilisation {_fmt(comp, '%')} and memory utilisation {_fmt(mem, '%')} are both low",
                    "Dominant stall reason is long_scoreboard",
                    f"Registers per thread {_fmt(regs)}",
                ],
                metric_values={"compute_pct": comp, "memory_pct": mem, "registers": regs},
                suspected_cause=(
                    "Each thread waits for its own load to return before issuing the dependent "
                    "arithmetic. With insufficient independent work in flight, memory latency is "
                    "not hidden."
                ),
                proposed_change=(
                    "Issue several independent loads before consuming any of them: unroll the "
                    "elementwise loop by 2-4 and keep separate accumulator chains, or prefetch the "
                    "next tile while computing the current one. Verify that the register count "
                    "does not rise far enough to reduce resident blocks."
                ),
                expected_effect=(
                    "Improves latency hiding without changing byte volume; the long-scoreboard "
                    "stall should fall while memory throughput stays constant."
                ),
                trade_offs=(
                    "More live values raise register pressure; on a register-limited kernel this "
                    "can reduce occupancy enough to cancel the benefit."
                ),
                confidence="high" if classification.confidence == "high" else "medium",
                effort="low",
                risk="low",
                validation=(
                    "The long_scoreboard stall ratio must drop, and latency must fall while "
                    "L2 bytes moved stays the same. If bytes moved also fell, something else changed."
                ),
                impact="moderate",
                bounded_by="unknown: limited by memory latency, not by a bandwidth ceiling",
            )
        )
        if occupancy and occupancy.occupancy_pct < 50.0:
            rs.recommendations.append(
                _occupancy_recommendation(occupancy, warps, instruction_profile)
            )

    elif cat == "occupancy-limited":
        rs.recommendations.append(_occupancy_recommendation(occupancy, warps, instruction_profile))

    elif cat == "compute-bound":
        rs.recommendations.append(
            Recommendation(
                title="Reduce arithmetic work or widen the datapath",
                kind="source",
                symptom="Execution pipes are saturated",
                evidence=[
                    f"SM throughput {_fmt(comp, '%')} of peak",
                    f"Issue active {_fmt(issue, '%')}",
                ],
                metric_values={"sm_throughput_pct": comp, "issue_active_pct": issue},
                suspected_cause=(
                    "The kernel issues more arithmetic and address-generation instructions than "
                    "the pipes can retire in the available time."
                ),
                proposed_change=(
                    "Check the instruction mix first: if integer address arithmetic dominates, "
                    "hoist loop-invariant offset computation and use pointer increments. If "
                    "floating-point work dominates, look for redundant computation and consider "
                    "vectorised or lower-precision forms where the numerics allow."
                ),
                expected_effect="Arithmetic-bound kernels scale with the reduction in retired instructions.",
                trade_offs="Lower precision changes results and must be re-verified; vectorisation may raise register pressure.",
                confidence="high" if (comp is not None and comp >= 80.0) else "medium",
                effort="moderate",
                risk="moderate",
                validation="Instruction count per unit of work and SM throughput should both fall.",
                impact="moderate",
                bounded_by="hard: the measured compute ceiling for the dtype",
            )
        )

    elif cat == "synchronization-bound":
        bar_ratio = classification.counter_summary.get("stall_barrier")
        bar_pct = getattr(classification, "stall_breakdown_pct", {}).get("barrier")
        ev = [
            f"Dominant stall reason is barrier ({_fmt(bar_ratio)} warps/issue"
            + (f", {bar_pct:.1f}% of all stalls)" if bar_pct is not None else ")")
        ]
        if occupancy and occupancy.active_blocks_per_sm == 1:
            ev.append(
                f"Only 1 block/SM is resident ({occupancy.occupancy_pct:.1f}% occupancy, "
                f"{occupancy.registers_per_thread} regs/thread), so the entire SM stalls whenever "
                "the single CTA hits __syncthreads()"
            )
        rs.recommendations.append(
            Recommendation(
                title="Reduce barriers or rebalance work between them",
                kind="source",
                symptom="Warps idle at __syncthreads()",
                evidence=ev,
                metric_values={
                    "barrier_stall_ratio": bar_ratio,
                    "barrier_stall_pct": bar_pct,
                    "active_blocks_per_sm": getattr(occupancy, "active_blocks_per_sm", None),
                },
                suspected_cause=(
                    "Work is unevenly distributed across warps between barriers, or a single resident "
                    "CTA per SM executes frequent stage barriers with no second CTA to overlap."
                ),
                proposed_change=(
                    "Restructure so each warp's share of the work between barriers is equal, or "
                    "reduce the number of barrier-delimited stages. A warp-synchronous formulation "
                    "with shuffles can remove barriers entirely where the algorithm permits."
                ),
                expected_effect="Fewer idle warp-cycles at synchronization points.",
                trade_offs="Warp-synchronous code is more fragile: correctness depends on the execution model and must be re-verified.",
                confidence="high" if (bar_pct is not None and bar_pct >= 35.0) else "medium",
                effort="moderate",
                risk="moderate",
                validation="The barrier stall ratio should fall and latency with it, with the same result.",
                impact="moderate" if (bar_pct is not None and bar_pct >= 35.0) else "unknown",
                bounded_by=f"hard: barrier stall share ({bar_pct:.1f}% of stalls)" if bar_pct else "unknown",
            )
        )
        if occupancy and occupancy.occupancy_pct < 50.0:
            rs.recommendations.append(
                _occupancy_recommendation(occupancy, warps, instruction_profile)
            )

    elif cat == "control-flow-bound":
        rs.recommendations.append(
            Recommendation(
                title="Reduce divergence and shrink the hot instruction footprint",
                kind="source",
                symptom="Warps stall on instruction fetch or branch resolution",
                evidence=["Dominant stall reason is no_instruction or branch_resolving"],
                metric_values={"branch_stall_ratio": classification.counter_summary.get("stall_branch_resolving")},
                suspected_cause=(
                    "The instruction working set exceeds the cache, or branches diverge so that "
                    "predicated paths execute serially."
                ),
                proposed_change=(
                    "Replace data-dependent branches in the inner loop with branch-free select or "
                    "predication; move rare paths out of the loop; reduce aggressive unrolling if "
                    "the unrolled body no longer fits the instruction cache."
                ),
                expected_effect="Fewer instruction-fetch stalls and less serialised divergent execution.",
                trade_offs="Predication executes both paths' instructions; it only helps when divergence is high.",
                confidence="medium",
                effort="moderate",
                risk="low",
                validation="Divergence and the affected stall ratio should fall together.",
                impact="unknown",
                bounded_by="unknown",
            )
        )

    elif cat == "instruction-latency-bound":
        rs.recommendations.append(
            Recommendation(
                title="Expose more instruction-level parallelism",
                kind="source",
                symptom="Warps stall on fixed-latency dependencies",
                evidence=["Dominant stall reason is wait"],
                metric_values={"wait_stall_ratio": classification.counter_summary.get("stall_wait")},
                suspected_cause=(
                    "Dependency chains are longer than the machine can overlap, so each thread "
                    "serialises on its own arithmetic latency."
                ),
                proposed_change=(
                    "Split the accumulation into multiple independent chains and interleave them, "
                    "so several operations are in flight per thread."
                ),
                expected_effect="Higher per-thread throughput at the same occupancy.",
                trade_offs="More accumulators consume registers, which can reduce occupancy.",
                confidence="medium",
                effort="low",
                risk="low",
                validation="The wait stall ratio should fall and throughput rise.",
                impact="moderate",
                bounded_by="unknown",
            )
        )

    elif cat == "launch-overhead":
        rs.recommendations.append(
            Recommendation(
                title="Amortise launch overhead across more work per launch",
                kind="source",
                symptom="Kernel duration is comparable to the launch overhead",
                evidence=["Duration is within the same order as the measured empty-kernel cost"],
                metric_values={},
                suspected_cause="The grid is too small for the launch and teardown cost to be amortised.",
                proposed_change=(
                    "Batch several logical iterations into one launch, or combine adjacent kernels "
                    "so one launch covers more work."
                ),
                expected_effect="Per-unit-of-work time falls even though kernel time is unchanged.",
                trade_offs="Batching increases per-launch memory footprint and complicates error attribution.",
                confidence="medium",
                effort="moderate",
                risk="low",
                validation="Total time for N units of work should fall while per-launch time stays flat.",
                impact="unknown",
                bounded_by="hard: the measured launch overhead",
            )
        )

    # ---------------------------------------------------------------- #
    # Structural checks that apply regardless of classification.
    # ---------------------------------------------------------------- #
    if waves and waves.problem == "under-fill":
        rs.recommendations.append(
            Recommendation(
                title="Increase grid parallelism to fill the machine",
                kind="parameter",
                symptom=f"Only {waves.waves:.2f} waves of work for {dev.multi_processor_count} SMs",
                evidence=[
                    f"{waves.total_ctas} CTAs against a concurrent capacity of {waves.blocks_per_wave}",
                    f"At most {100.0 * waves.waves:.1f}% of the device is engaged at any instant",
                ],
                metric_values=waves.to_dict(),
                suspected_cause=(
                    "The problem is decomposed into fewer CTAs than the device can host "
                    "simultaneously, so execution pipes are idle for lack of work rather than "
                    "for lack of parallelism within a CTA."
                ),
                proposed_change=(
                    "Split the work more finely — for example give each CTA a smaller slice, or "
                    "split the reduction dimension so that multiple CTAs cooperate — until the "
                    "CTA count comfortably exceeds the concurrent capacity."
                ),
                expected_effect="More resident work means better latency hiding and higher utilisation.",
                trade_offs="Finer decomposition adds per-CTA overhead and, for reductions, an extra combining step.",
                confidence="high",
                effort="low",
                risk="low",
                validation="Measure the same problem size at a larger grid; latency should fall.",
                impact=f"bounded: up to {100.0 - 100.0 * waves.waves:.0f}% of the device is unused",
                bounded_by="hard: concurrent CTA capacity from the occupancy model",
            )
        )
    elif waves and waves.problem == "tail":
        rs.recommendations.append(
            Recommendation(
                title="Remove the partial-wave tail",
                kind="parameter",
                symptom=f"{waves.waves:.2f} waves, with a tail of {waves.tail_ctas} CTAs",
                evidence=[
                    f"{waves.total_ctas} CTAs, {waves.full_waves} full wave(s) of {waves.blocks_per_wave}",
                    f"Tail occupancy {waves.tail_efficiency_pct:.1f}% of one wave",
                ],
                metric_values=waves.to_dict(),
                suspected_cause=(
                    "The final wave has fewer CTAs than the device can run concurrently, so part "
                    "of the machine is idle while the tail executes."
                ),
                proposed_change=(
                    "Reshape the grid so the CTA count is an exact multiple of the wave size, or "
                    "convert to a persistent grid-stride loop with a CTA count fixed at one or a "
                    "few full waves."
                ),
                expected_effect=(
                    "Recovers the idle fraction of the tail. The benefit is bounded by the tail's "
                    "share of total runtime, so it is only material when the wave count is small."
                ),
                trade_offs="A grid-stride loop adds loop overhead and index arithmetic per iteration.",
                confidence="medium" if waves.waves < 4 else "low",
                effort="low",
                risk="low",
                validation="Compare total time for the same work at a wave-multiple grid size.",
                impact=f"bounded: at most ~{waves.quantization_loss_pct:.1f}% of runtime",
                bounded_by="hard: the tail fraction",
            )
        )

    if instruction_profile is not None and getattr(instruction_profile, "has_local_memory", False):
        rs.recommendations.append(
            Recommendation(
                title="Eliminate register spills",
                kind="source",
                symptom="SASS contains local-memory loads and stores",
                evidence=[
                    f"Local loads {getattr(instruction_profile, 'local_loads', 0)}, "
                    f"local stores {getattr(instruction_profile, 'local_stores', 0)} in the disassembly",
                    f"Registers per thread {_fmt(regs)}",
                ],
                metric_values={
                    "local_loads": getattr(instruction_profile, "local_loads", 0),
                    "local_stores": getattr(instruction_profile, "local_stores", 0),
                    "registers": regs,
                },
                suspected_cause=(
                    "Live-value count exceeds the register file allocation for the chosen block "
                    "size, so the compiler spills to local memory, which is served from L1/L2 "
                    "and DRAM rather than registers."
                ),
                proposed_change=(
                    "Reduce live values: shrink the unroll factor, recompute rather than retain, "
                    "or restructure so fewer values are simultaneously live. Alternatively, relax "
                    "the register cap if one was set, or reduce the block size so that the same "
                    "register count admits more blocks."
                ),
                expected_effect="Removes memory traffic that exists only to compensate for register pressure.",
                trade_offs="Lower register usage may require less unrolling, which reduces instruction-level parallelism.",
                confidence="high",
                effort="moderate",
                risk="low",
                validation="Spill counts in the ptxas report and the STL/LDL instructions in SASS must both reach zero.",
                impact="moderate",
                bounded_by="unknown: the spilled traffic's cost is not measured here",
            )
        )

    if spr is not None and spr > expected_spr * 1.3 and cat != "memory-bandwidth-bound":
        rs.recommendations.append(
            Recommendation(
                title="Restore coalescing on global loads",
                kind="source",
                symptom=f"Loads average {spr:.2f} sectors per request",
                evidence=[
                    f"Sectors per request {spr:.2f} against an expected {expected_spr:.0f} for "
                    f"{max_load_width_bytes or 4}-byte accesses by a full warp",
                    f"L2 hit rate {_fmt(hit, '%')}",
                ],
                metric_values={"sectors_per_request": spr, "l2_hit_rate_pct": hit},
                suspected_cause=(
                    "Adjacent lanes in a warp are not accessing adjacent addresses, so each request "
                    "touches more cache lines than the bytes actually consumed."
                ),
                proposed_change=(
                    "Reorganise the thread-to-data mapping so that lane i reads element base+i, or "
                    "have each thread load a wider vector type when the layout is already contiguous."
                ),
                expected_effect="Fewer sectors per request means the same useful bytes cost fewer transactions.",
                trade_offs="A vectorised load path requires alignment and a length divisible by the vector width.",
                confidence="medium",
                effort="low",
                risk="low",
                validation=(
                    f"Sectors per request should approach {expected_spr:.0f} and effective bandwidth "
                    "should rise."
                ),
                impact="moderate",
                bounded_by="hard: the memory system's transaction rate",
            )
        )

    if instruction_profile is not None and getattr(instruction_profile, "fp64_ops", 0) > 0:
        fp64_cnt = int(getattr(instruction_profile, "fp64_ops", 0))
        rs.recommendations.append(
            Recommendation(
                title="Eliminate unintentional FP64 instructions (67:1 penalty on GB10)",
                kind="source",
                symptom=f"SASS contains {fp64_cnt} FP64 instruction(s)",
                evidence=[
                    f"Disassembly contains {fp64_cnt} double-precision (DFMA/DMUL/DADD/.F64) instructions",
                    "On GB10, FP64 throughput (~0.42 TFLOP/s) is ~67x slower than FP32 (~27.8 TFLOP/s)",
                ],
                metric_values={"fp64_ops": fp64_cnt},
                suspected_cause=(
                    "Un-suffixed floating-point literals (e.g. `1.0` instead of `1.0f`) or math calls "
                    "(`exp`, `sqrt`, `fma` on double) caused the compiler to emit FP64 instructions."
                ),
                proposed_change=(
                    "Suffix all float literals with `f` and use float intrinsics (`fmaf`, `expf`, `rsqrtf`)."
                ),
                expected_effect="Eliminates 67:1 rate-limited FP64 pipe stalls and frees pair-allocated registers.",
                trade_offs="None when single precision is the intended contract.",
                confidence="high",
                effort="trivial",
                risk="low",
                validation="Verify with `ckt static` or `ckt compare` that `fp64_ops` drops to 0.",
                impact="high",
                bounded_by="hard: 67:1 FP64-to-FP32 throughput ratio on GB10",
            )
        )

    if (
        roofline is not None
        and getattr(roofline, "working_set_vs_l2_ratio", None) is not None
        and getattr(roofline, "working_set_vs_l2_ratio", 0.0) > 0.75
        and hit is not None
        and hit < 35.0
    ):
        ws_ratio = float(getattr(roofline, "working_set_vs_l2_ratio", 0.0))
        l2_mb = (dev.l2_cache_size or 0) / (1024.0 * 1024.0)
        rs.recommendations.append(
            Recommendation(
                title="Tile across columns/batches to exploit the L2 cache and shared memory",
                kind="source",
                symptom=f"L2 hit rate is {hit:.1f}% with a working set {ws_ratio:.2f}x the {l2_mb:.0f} MB L2 cache",
                evidence=[
                    f"Declared bytes ({(roofline.bytes_moved or 0) / 1e6:.2f} MB) = {ws_ratio:.2f}x L2 capacity ({l2_mb:.0f} MB)",
                    f"Measured L2 sector hit rate is {hit:.1f}%",
                ],
                metric_values={"working_set_vs_l2_ratio": ws_ratio, "l2_hit_rate_pct": hit},
                suspected_cause=(
                    "The working set exceeds L2 capacity and streams cold from unified LPDDR5X memory "
                    "on every pass."
                ),
                proposed_change=(
                    "Process multiple output columns/batch rows per weight load (register/SMEM tiling) "
                    "or reorder CTA scheduling (L2 swizzling) so shared inputs are reused while resident in L2/SMEM."
                ),
                expected_effect="Raises L2 hit rate and multiplies effective per-weight arithmetic intensity.",
                trade_offs="Processing more columns per CTA increases accumulator register usage; check `register_step_table` in `OccupancyModel` (`free_reg_headroom_per_thread`) to stay within the current block-residency step.",
                confidence="medium",
                effort="moderate",
                risk="low",
                validation="L2 hit rate (`lts__t_sector_hit_rate.pct`) should increase and DRAM bytes read should fall.",
                impact="moderate",
                bounded_by="hard: L2 cache capacity and free register headroom per thread",
            )
        )

    # ---------------------------------------------------------------- #
    # Explicitly report checks that could not be performed.
    # ---------------------------------------------------------------- #
    if roofline and roofline.bandwidth_ceiling_gbps is None:
        rs.unavailable_checks.append(
            "Bandwidth utilisation could not be computed: no measured bandwidth ceiling is "
            "available. Run the bandwidth probe before acting on any memory-related "
            "recommendation."
        )
    if roofline and roofline.compute_ceiling_tflops is None:
        rs.unavailable_checks.append(
            "Compute utilisation could not be computed: no measured arithmetic ceiling is "
            "available for this dtype. Run the compute probe."
        )
    if not any("issue_stalled" in k for k in result.metrics):
        rs.unavailable_checks.append(
            "No stall-reason metrics were collected, so the latency-bound categories cannot be "
            "distinguished. Run a Stage C pass."
        )
    if regs is None and occupancy is None:
        rs.unavailable_checks.append(
            "Registers per thread are unavailable, so occupancy could not be predicted. Check "
            "the ptxas verbose report for this kernel."
        )

    if not rs.recommendations:
        rs.notes.append(
            "No recommendation was produced for this kernel. This means no diagnostic signature "
            "was recognised, not that the kernel is optimal: "
            + "; ".join(rs.unavailable_checks)
            if rs.unavailable_checks
            else "No diagnostic signature was recognised and all inputs were available; "
                 "the kernel may genuinely be at a hardware limit. Confirm by comparing measured "
                 "throughput against the measured ceilings."
        )

    rs.recommendations.sort(key=lambda r: r.rank_key)
    return rs


def _occupancy_recommendation(
    occupancy: OccupancyModel | None,
    warps: float | None,
    instruction_profile: Any | None,
) -> Recommendation:
    ev: list[str] = []
    mv: dict[str, Any] = {}
    if occupancy:
        ev.append(
            f"Predicted occupancy {occupancy.occupancy_pct:.1f}% "
            f"({occupancy.active_blocks_per_sm} blocks/SM), limited by {occupancy.limiting_factor}"
        )
        ev.append(
            f"Allocation is {occupancy.registers_allocated_per_block} registers and "
            f"{occupancy.smem_allocated_per_block} B shared memory per block"
        )
        mv = occupancy.to_dict()
    if warps is not None:
        ev.append(f"NCU reports {warps:.1f}% of peak active warps")
    if instruction_profile is not None and getattr(instruction_profile, "has_local_memory", False):
        ev.append(
            "SASS shows register spills, so register pressure is already degrading this kernel"
        )

    change = (
        "Reduce the per-thread resource footprint so more blocks fit: lower the register count "
        "(less unrolling or fewer live values), or use a smaller block size so the same register "
        "allocation admits more blocks. Check the predicted occupancy for the alternative "
        "configuration before running it."
    )
    trade_offs = (
        "Lower occupancy is not automatically bad: if the kernel already hides latency, adding "
        "resident blocks can reduce per-thread resources without improving throughput."
    )
    bounded = "unknown"
    if occupancy and occupancy.limiting_factor == "registers" and occupancy.max_regs_per_thread_to_gain_one_block:
        bounded = (
            f"bounded: registers must reach {occupancy.max_regs_per_thread_to_gain_one_block} "
            "per thread for another resident block"
        )
        step_summary = ""
        if getattr(occupancy, "register_step_table", None):
            steps = [
                f"{r['blocks_per_sm']} blk ({r['occupancy_pct']:.0f}% occ) <= {r['max_regs_per_thread']} regs"
                for r in occupancy.register_step_table[:6]
            ]
            step_summary = " Exact register step thresholds: " + "; ".join(steps) + "."
        change = (
            f"Reduce registers per thread from {occupancy.registers_per_thread} to at most "
            f"{occupancy.max_regs_per_thread_to_gain_one_block} so one more block becomes resident "
            f"per SM.{step_summary}"
        )
    return Recommendation(
        title="Raise occupancy if the kernel is latency-limited",
        kind="parameter",
        symptom="Fewer warps are resident than the hardware can hold",
        evidence=ev,
        metric_values=mv,
        suspected_cause=(
            "Resident warp count is constrained by per-thread registers or per-block shared "
            "memory, leaving too few warps to hide memory and arithmetic latency."
        ),
        proposed_change=change,
        expected_effect=(
            "More resident warps improve latency hiding, but only if the kernel is latency-limited "
            "rather than throughput-limited."
        ),
        trade_offs=trade_offs,
        confidence="medium",
        effort="low",
        risk="low",
        validation=(
            "Sweep block size at constant work and re-measure; occupancy should rise and latency "
            "should fall. If latency does not improve, occupancy was not the limiter and this "
            "recommendation should be discarded."
        ),
        impact="moderate",
        bounded_by=bounded,
    )


def rank_across_kernels(sets: Sequence[RecommendationSet]) -> list[RecommendationSet]:
    """Order kernels by the rank of their top recommendation."""
    def key(rs: RecommendationSet) -> tuple[int, str]:
        if not rs.recommendations:
            return (9, rs.kernel)
        return (rs.recommendations[0].rank_key[0], rs.kernel)

    return sorted(sets, key=key)
