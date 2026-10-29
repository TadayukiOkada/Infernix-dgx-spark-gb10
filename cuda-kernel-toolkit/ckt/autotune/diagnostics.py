"""Custom-kernel investigation workflow and hardware bottleneck diagnosis.

Analyzes workloads where existing candidates perform poorly against roofline or
hardware ceilings and classifies outcomes into reuse_existing_schedule,
change_dispatch, investigate_bottleneck, or custom_kernel_candidate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .database import TuningRecord
from .profile import HardwareProfile
from .workload import WorkloadSpec
from ..kernel_math import calculate_wave_tables, calculate_workload_math
from ..util import to_plain


@dataclass
class DiagnosticOutcome:
    """Diagnostic verdict and evidence for a specific workload."""

    workload_key: str
    outcome: str  # "reuse_existing_schedule", "change_dispatch", "investigate_bottleneck", "custom_kernel_candidate"
    best_candidate_id: str
    measured_latency_us: float
    memory_floor_us: float
    roofline_headroom_ratio: float  # measured / memory_floor
    effective_gbs: float
    bandwidth_utilization_pct: float
    cta_count: int
    wave_efficiency_pct: float
    identified_bottlenecks: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


class CustomKernelDiagnostician:
    """Diagnoses performance gaps and classifies whether a custom kernel is justified."""

    def __init__(self, hw_profile: HardwareProfile):
        self.hw_profile = hw_profile

    def diagnose(
        self,
        record: TuningRecord,
        runner_up_record: TuningRecord | None = None,
    ) -> DiagnosticOutcome:
        """Analyze a workload record and determine diagnostic outcome."""
        wl = record.workload
        cand = record.candidate

        n = int(wl.get("n", 0))
        k = int(wl.get("k", 0))
        t = int(wl.get("tokens", 1))
        qtype = str(wl.get("quantization_family", "bf16"))

        measured_us = record.latency_us
        gbs = record.effective_gbs
        peak_bw = self.hw_profile.sustained_read_gbps or 240.0
        bw_util = (gbs / peak_bw) * 100.0 if peak_bw > 0 else 0.0

        # Estimate theoretical memory floor
        # Bytes = model weights + activations + outputs
        # Approximate weight bytes
        bits_per_weight = {
            "nvfp4": 4.5,  # with scale overhead
            "fp8": 8.5,
            "q4": 4.5,
            "q5": 5.5,
            "q6": 6.5,
            "q8": 8.5,
            "bf16": 16.0,
        }.get(qtype.lower(), 16.0)

        w_bytes = (n * k * bits_per_weight) / 8.0
        x_bytes = k * t * 2.0  # BF16 input
        y_bytes = n * t * 2.0  # BF16 output
        total_bytes = w_bytes + x_bytes + y_bytes

        mem_floor_us = (total_bytes / (peak_bw * 1e9)) * 1e6
        headroom_ratio = measured_us / mem_floor_us if mem_floor_us > 0 else 1.0

        # Wave efficiency on target SMs
        tile_rows = cand.get("tile_rows") or 64
        tile_tokens = cand.get("tile_tokens") or 16
        ctas_n = math.ceil(n / tile_rows)
        ctas_t = math.ceil(t / tile_tokens)
        total_ctas = ctas_n * ctas_t

        sms = self.hw_profile.sm_count or 48
        full_waves = total_ctas // sms
        tail_ctas = total_ctas % sms
        wave_eff = (total_ctas / ((full_waves + (1 if tail_ctas > 0 else 0)) * sms)) * 100.0 if sms > 0 else 100.0

        bottlenecks = []
        recommendations = []

        # Analyze potential root causes
        if bw_util >= 75.0:
            outcome = "reuse_existing_schedule"
            recommendations.append(
                f"Candidate operates at {bw_util:.1f}% of DRAM sustained read ceiling; current schedule is near-optimal."
            )
        elif wave_eff < 60.0 and t > 4:
            bottlenecks.append(
                f"Low CTA wave efficiency ({wave_eff:.1f}% on {sms} SMs); partial wave leaves many SMs idle."
            )
            outcome = "change_dispatch"
            recommendations.append("Switch to a tile with smaller row extent or different token blocking to fill SM waves.")
        elif t <= 2 and cand.get("kernel_family") == "mma":
            bottlenecks.append("Using Tensor Core MMA for narrow decode (T <= 2) incurs tile waste.")
            outcome = "change_dispatch"
            recommendations.append("Switch to specialized GEMV or SIMT vector path for single-token decode.")
        elif t >= 64 and cand.get("kernel_family") in ("gemv", "simt"):
            bottlenecks.append(f"Vector kernel family '{cand.get('kernel_family')}' saturates instruction pipe at T={t}.")
            outcome = "change_dispatch"
            recommendations.append("Switch to Tensor Core MMA or TMA kernel for wide batch / prefill.")
        elif headroom_ratio > 3.0:
            bottlenecks.append(
                f"Large roofline gap: measured latency is {headroom_ratio:.2f}x the DRAM floor with low utilization ({bw_util:.1f}%)."
            )
            # Check if this is an structural issue requiring custom kernel
            if t in (3, 5, 7) and cand.get("kernel_family") == "mma":
                bottlenecks.append("Irregular token width causes unaligned MMA padding waste.")
                outcome = "custom_kernel_candidate"
                recommendations.append(
                    "Structural limitation: existing tiles require power-of-two token padding; custom narrow-tile kernel may improve latency."
                )
            else:
                outcome = "investigate_bottleneck"
                recommendations.append(
                    "Profile with NCU to inspect pipeline stalls, register spills, or shared memory bank conflicts."
                )
        else:
            outcome = "reuse_existing_schedule"
            recommendations.append("Candidate latency is acceptable given workload dimensions.")

        workload_key = f"{wl.get('operation')}_{qtype}_n{n}_k{k}_t{t}"

        return DiagnosticOutcome(
            workload_key=workload_key,
            outcome=outcome,
            best_candidate_id=cand.get("id", "unknown"),
            measured_latency_us=measured_us,
            memory_floor_us=round(mem_floor_us, 3),
            roofline_headroom_ratio=round(headroom_ratio, 2),
            effective_gbs=round(gbs, 1),
            bandwidth_utilization_pct=round(bw_util, 1),
            cta_count=total_ctas,
            wave_efficiency_pct=round(wave_eff, 1),
            identified_bottlenecks=bottlenecks,
            recommendations=recommendations,
            evidence={
                "w_bytes": int(w_bytes),
                "total_bytes": int(total_bytes),
                "tile_rows": tile_rows,
                "tile_tokens": tile_tokens,
                "sm_count": sms,
            },
        )
