"""Unit and integration tests for ckt autotune offline autotuning and dispatch generation.

Tests discovery, candidate registry, legality pruning, hardware profiling,
noise-aware selection, adaptive crossover refinement, policy compression,
deterministic C++ code generation, diagnostics, and results database.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from ckt.autotune.database import (
    CorrectnessTestSummary,
    TuningDatabase,
    TuningRecord,
)
from ckt.autotune.diagnostics import CustomKernelDiagnostician
from ckt.autotune.discovery import RepositoryDiscovery
from ckt.autotune.generator import DispatchGenerator, DispatchInterval, PolicyCompressor
from ckt.autotune.profile import HardwareProfile, SoftwareProfile, TuningProfile
from ckt.autotune.registry import CandidateLegality, CandidateRegistry, CandidateSpec
from ckt.autotune.search import (
    AdaptiveRefiner,
    NoiseAwareSelector,
    WinnerSelection,
)
from ckt.autotune.workload import WorkloadSpec, WorkloadSweep


def test_repository_discovery_surface_area():
    disco = RepositoryDiscovery()
    report = disco.discover()

    # Must be quantization-family agnostic: discover all supported families
    assert len(report.families) >= 7
    expected_families = {"nvfp4", "fp8", "q4", "q5", "q6", "q8", "bf16"}
    for fam in expected_families:
        assert fam in report.families, f"Family {fam} missing from discovery"

    # Shapes count must be substantial (>90 registered shapes)
    assert report.total_shapes >= 90

    # Test associations and byte-identity tests must be mapped
    nvfp4_fam = report.families["nvfp4"]
    assert any("linear" in op for op in nvfp4_fam.operations)
    assert any("gemv" in kf for kf in nvfp4_fam.kernel_families)
    assert any("mma" in kf for kf in nvfp4_fam.kernel_families)

    # Check Markdown report generation
    md = report.render_markdown()
    assert "# Infernix CUDA Repository Discovery Report" in md
    assert "Discovered Quantization Families" in md


def test_hardware_profile_and_fingerprint():
    hw = HardwareProfile(
        gpu_name="NVIDIA GB10",
        fingerprint="test_fingerprint_123",
        compute_capability="12.1",
        sm_count=48,
        max_threads_per_sm=1536,
        max_threads_per_block=1024,
        regs_per_sm=65536,
        shared_mem_per_block_default=49152,
        shared_mem_per_block_opt_in=101376,
        shared_mem_per_sm=102400,
        l2_cache_size=25165824,
        bandwidth_gbps=243.7,
        sustained_read_gbps=243.7,
        features=["tma", "nvfp4", "fp8", "bf16"],
        arch_name="sm_121",
    )
    sw = SoftwareProfile(
        cuda_runtime_version="13.0",
        cuda_driver_version="system",
        compiler="nvcc",
        compiler_version="V13.0.0",
        framework_revision="abcdef123456",
        framework_clean=True,
    )
    tp = TuningProfile(profile_id="gb10_sm121_test", hardware=hw, software=sw)

    d = tp.to_dict()
    assert d["profile_id"] == "gb10_sm121_test"
    assert d["hardware"]["sm_count"] == 48

    tp_round = TuningProfile.from_dict(d)
    assert tp_round.profile_id == tp.profile_id
    assert tp_round.hardware.compute_capability == "12.1"


def test_workload_and_sweep_parsing():
    wl = WorkloadSpec(
        operation="linear",
        quantization_family="nvfp4",
        n=128,
        k=2560,
        tokens=16,
        policy="a16",
    )
    assert wl.key == "linear_nvfp4_n128_k2560_t16_a16_exact"
    assert wl.geometry_key == "linear_nvfp4_n128_k2560"

    # Sweep from string: comma separated
    t_list = WorkloadSweep.from_string("1, 2, 4, 8, 16")
    assert t_list == [1, 2, 4, 8, 16]

    # Sweep from string: range with step
    t_range = WorkloadSweep.from_string("1:16:4")
    assert t_range == [1, 5, 9, 13]

    # Boundary probe generator
    b_pts = WorkloadSweep.boundary_probe_tokens([8, 16, 32], step=1, max_tokens=64)
    assert 8 in b_pts
    assert 9 in b_pts
    assert 16 in b_pts
    assert 17 in b_pts


def test_candidate_registry_and_legality_pruning():
    hw = HardwareProfile(
        gpu_name="NVIDIA GB10",
        fingerprint="fp123",
        compute_capability="12.1",
        sm_count=48,
        max_threads_per_sm=1536,
        max_threads_per_block=1024,
        regs_per_sm=65536,
        shared_mem_per_block_default=49152,
        shared_mem_per_block_opt_in=101376,
        shared_mem_per_sm=102400,
        l2_cache_size=25165824,
        bandwidth_gbps=243.7,
        sustained_read_gbps=243.7,
        features=["nvfp4", "fp8"],
        arch_name="sm_121",
    )
    reg = CandidateRegistry(hw_profile=hw)

    # Legal candidate
    cand_legal = CandidateSpec(
        id="cand_valid",
        operation="linear",
        quantization_family="nvfp4",
        kernel_family="mma",
        n=128,
        k=2560,
        template_name="Nvfp4A16MmaSchedule",
        launch_expr="launch_valid",
        threads_per_block=256,
        shared_mem_bytes=32768,
        requires_features=["nvfp4"],
    )
    verdict1 = reg.check_legality(cand_legal)
    assert verdict1.legal
    assert len(verdict1.reasons) == 0

    # Illegal candidate: excessive shared memory
    cand_smem_overflow = CandidateSpec(
        id="cand_overflow",
        operation="linear",
        quantization_family="nvfp4",
        kernel_family="mma",
        n=128,
        k=2560,
        template_name="Nvfp4A16MmaSchedule",
        launch_expr="launch_overflow",
        threads_per_block=256,
        shared_mem_bytes=200000,  # exceeds 101376 B
    )
    verdict2 = reg.check_legality(cand_smem_overflow)
    assert not verdict2.legal
    assert any("Shared memory" in r for r in verdict2.reasons)

    # Illegal candidate: unsupported feature (TMA when missing from features)
    cand_missing_feat = CandidateSpec(
        id="cand_no_tma",
        operation="linear",
        quantization_family="nvfp4",
        kernel_family="tma",
        n=128,
        k=2560,
        template_name="Nvfp4A4TmaSchedule",
        launch_expr="launch_tma",
        threads_per_block=256,
        requires_features=["tma"],  # not in hw.features
    )
    verdict3 = reg.check_legality(cand_missing_feat)
    assert not verdict3.legal
    assert any("feature 'tma'" in r for r in verdict3.reasons)


def test_noise_aware_selector_and_tie_breaking():
    selector = NoiseAwareSelector(min_improvement_pct=2.0, baseline_candidate_id="cand_baseline")

    cand_base = CandidateSpec(
        id="cand_baseline",
        operation="linear",
        quantization_family="nvfp4",
        kernel_family="simt",
        n=128,
        k=2560,
        template_name="C2",
        launch_expr="launch_simt",
        threads_per_block=128,
    )
    cand_alt = CandidateSpec(
        id="cand_alt",
        operation="linear",
        quantization_family="nvfp4",
        kernel_family="sliced_k",
        n=128,
        k=2560,
        template_name="SlicedK",
        launch_expr="launch_sliced",
        threads_per_block=256,
    )

    from ckt.autotune.runner import BenchmarkMeasurement

    rec_base = TuningRecord(
        schema_version=1,
        hardware_profile_id="hw",
        software_profile={},
        workload={"tokens": 2},
        candidate=cand_base.to_dict(),
        compile_status="success",
        correctness_status="pass",
        correctness_tests=CorrectnessTestSummary(),
        latency_us=10.0,
        latency_p95_us=10.2,
    )
    meas_base = BenchmarkMeasurement(
        median_us=10.0,
        min_us=9.8,
        p95_us=10.2,
        max_us=10.3,
        mad_us=0.1,
        samples_us=[9.8, 10.0, 10.2],
        effective_gbs=100.0,
        useful_tflops=1.0,
        warmup=2,
        repeat=10,
        flush_mib=32,
    )

    # Case 1: Alt is slightly faster (0.8% faster) — below 2% noise margin
    rec_alt_close = TuningRecord(
        schema_version=1,
        hardware_profile_id="hw",
        software_profile={},
        workload={"tokens": 2},
        candidate=cand_alt.to_dict(),
        compile_status="success",
        correctness_status="pass",
        correctness_tests=CorrectnessTestSummary(),
        latency_us=9.92,  # 0.8% faster
        latency_p95_us=10.1,
    )
    meas_alt_close = BenchmarkMeasurement(
        median_us=9.92,
        min_us=9.7,
        p95_us=10.1,
        max_us=10.2,
        mad_us=0.1,
        samples_us=[9.7, 9.92, 10.1],
        effective_gbs=101.0,
        useful_tflops=1.0,
        warmup=2,
        repeat=10,
        flush_mib=32,
    )

    # Tie-breaking should prefer the baseline
    winner1 = selector.select([
        (cand_alt, meas_alt_close, rec_alt_close),
        (cand_base, meas_base, rec_base),
    ])
    assert winner1.winning_candidate.id == "cand_baseline"
    assert winner1.tie_breaker_applied

    # Case 2: Alt is significantly faster (15% faster) — well above 2% margin
    rec_alt_fast = TuningRecord(
        schema_version=1,
        hardware_profile_id="hw",
        software_profile={},
        workload={"tokens": 2},
        candidate=cand_alt.to_dict(),
        compile_status="success",
        correctness_status="pass",
        correctness_tests=CorrectnessTestSummary(),
        latency_us=8.5,
        latency_p95_us=8.7,
    )
    meas_alt_fast = BenchmarkMeasurement(
        median_us=8.5,
        min_us=8.3,
        p95_us=8.7,
        max_us=8.8,
        mad_us=0.1,
        samples_us=[8.3, 8.5, 8.7],
        effective_gbs=120.0,
        useful_tflops=1.2,
        warmup=2,
        repeat=10,
        flush_mib=32,
    )

    winner2 = selector.select([
        (cand_base, meas_base, rec_base),
        (cand_alt, meas_alt_fast, rec_alt_fast),
    ])
    assert winner2.winning_candidate.id == "cand_alt"
    assert not winner2.tie_breaker_applied
    assert winner2.margin_pct > 10.0


def test_adaptive_crossover_refinement():
    cand1 = CandidateSpec("c1", "linear", "q8", "simt", 1024, 2048, "S1", "launch_s1")
    cand2 = CandidateSpec("c2", "linear", "q8", "mma", 1024, 2048, "S2", "launch_s2")

    rec1 = TuningRecord(1, "hw", {}, {}, cand1.to_dict(), "success", "pass", CorrectnessTestSummary(), 5.0, 5.2)
    rec2 = TuningRecord(1, "hw", {}, {}, cand2.to_dict(), "success", "pass", CorrectnessTestSummary(), 4.0, 4.2)

    winners = {
        1: WinnerSelection(1, cand1, None, 5.0, None, 0.0, False, False, rec1),
        8: WinnerSelection(8, cand1, None, 7.0, None, 0.0, False, False, rec1),
        32: WinnerSelection(32, cand2, None, 10.0, None, 0.0, False, False, rec2),
        64: WinnerSelection(64, cand2, None, 14.0, None, 0.0, False, False, rec2),
    }

    crossovers = AdaptiveRefiner.find_crossovers(winners)
    assert crossovers == [(8, 32)]

    ref_tokens = AdaptiveRefiner.generate_refinement_tokens(crossovers, step=1, max_points_per_interval=3)
    assert len(ref_tokens) > 0
    for t in ref_tokens:
        assert 8 < t < 32


def test_policy_compressor_and_interval_ladder():
    cand_gemv = CandidateSpec("c_gemv", "linear", "nvfp4", "gemv", 128, 2560, "Gemv", "launch_gemv")
    cand_simt = CandidateSpec("c_simt", "linear", "nvfp4", "simt", 128, 2560, "Simt", "launch_simt")
    cand_mma = CandidateSpec("c_mma", "linear", "nvfp4", "mma", 128, 2560, "Mma", "launch_mma")

    r_gemv = TuningRecord(1, "hw", {}, {}, cand_gemv.to_dict(), "success", "pass", CorrectnessTestSummary(), 1.0, 1.1)
    r_simt = TuningRecord(1, "hw", {}, {}, cand_simt.to_dict(), "success", "pass", CorrectnessTestSummary(), 2.0, 2.1)
    r_mma = TuningRecord(1, "hw", {}, {}, cand_mma.to_dict(), "success", "pass", CorrectnessTestSummary(), 3.0, 3.1)

    winners = {
        1: WinnerSelection(1, cand_gemv, None, 1.0, None, 0.0, False, False, r_gemv),
        2: WinnerSelection(2, cand_simt, None, 2.0, None, 0.0, False, False, r_simt),
        3: WinnerSelection(3, cand_simt, None, 2.5, None, 0.0, False, False, r_simt),
        4: WinnerSelection(4, cand_simt, None, 3.0, None, 0.0, False, False, r_simt),
        8: WinnerSelection(8, cand_mma, None, 4.0, None, 0.0, False, False, r_mma),
        16: WinnerSelection(16, cand_mma, None, 5.0, None, 0.0, False, False, r_mma),
    }

    intervals = PolicyCompressor.compress(winners, token_step=1)
    assert len(intervals) == 3

    assert intervals[0].min_tokens == 1 and intervals[0].max_tokens == 1
    assert intervals[0].candidate.id == "c_gemv"
    assert intervals[0].condition == "tokens == 1"

    assert intervals[1].min_tokens == 2 and intervals[1].max_tokens == 4
    assert intervals[1].candidate.id == "c_simt"
    assert intervals[1].condition == "tokens <= 4"

    assert intervals[2].min_tokens == 8 and intervals[2].max_tokens is None
    assert intervals[2].candidate.id == "c_mma"
    assert intervals[2].condition == "default"


def test_deterministic_cpp_generation():
    hw = HardwareProfile(
        gpu_name="NVIDIA GB10",
        fingerprint="fp12345",
        compute_capability="12.1",
        sm_count=48,
        max_threads_per_sm=1536,
        max_threads_per_block=1024,
        regs_per_sm=65536,
        shared_mem_per_block_default=49152,
        shared_mem_per_block_opt_in=101376,
        shared_mem_per_sm=102400,
        l2_cache_size=25165824,
        bandwidth_gbps=243.7,
        sustained_read_gbps=243.7,
        arch_name="sm_121",
    )
    sw = SoftwareProfile("13.0", "system", "nvcc", "V13.0.0", "1234567890ab", True)
    tp = TuningProfile("gb10_sm121_fp12345", hw, sw)
    generator = DispatchGenerator(tp)

    cand1 = CandidateSpec("c1", "linear", "nvfp4", "gemv", 128, 2560, "Gemv", "nvfp4_linear_a16_gemv<Geometry, Gemv>")
    cand2 = CandidateSpec("c2", "linear", "nvfp4", "mma", 128, 2560, "Mma", "nvfp4_linear_a16_mma<Geometry, Mma>")

    inter1 = DispatchInterval(1, 1, cand1, cand1.launch_expr, "tokens == 1")
    inter2 = DispatchInterval(2, None, cand2, cand2.launch_expr, "default")

    cpp = generator.generate_cpp(
        operation="linear",
        qtype="nvfp4",
        n=128,
        k=2560,
        intervals=[inter1, inter2],
        disabled_routes=["a4_route_disabled"],
        results_hash="hash_repro_123",
    )

    # Verify deterministic elements
    assert "Autotuned Dispatch Policy generated by cuda-kernel-toolkit" in cpp
    assert "Target GPU: NVIDIA GB10 (sm_121)" in cpp
    assert "Results Hash: hash_repro_123" in cpp
    assert "Nvfp4Launch select_a16(int tokens)" in cpp
    assert "if (tokens == 1) return nvfp4_linear_a16_gemv<Geometry, Gemv>;" in cpp
    assert "return nvfp4_linear_a16_mma<Geometry, Mma>;" in cpp
    assert "const Nvfp4LinearShape kNvfp4N128K2560{" in cpp
    assert "throw std::logic_error" in cpp


def test_tuning_database_persistence():
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "test_db.json"
        csv_path = Path(tmpdir) / "test_db.csv"

        db = TuningDatabase()
        cand = CandidateSpec("cand_1", "linear", "q8", "mma", 1024, 2048, "S1", "launch_s1")
        rec = TuningRecord(
            schema_version=1,
            hardware_profile_id="prof_1",
            software_profile={"compiler": "nvcc"},
            workload={"operation": "linear", "quantization_family": "q8", "n": 1024, "k": 2048, "tokens": 8},
            candidate=cand.to_dict(),
            compile_status="success",
            correctness_status="pass",
            correctness_tests=CorrectnessTestSummary(passed=["test_1"], byte_identity_passed=True),
            latency_us=12.5,
            latency_p95_us=12.8,
            samples_us=[12.3, 12.5, 12.8],
            effective_gbs=150.0,
            useful_tflops=2.0,
        )
        db.add(rec)
        assert db.count() == 1

        db.save_json(json_path)
        assert json_path.exists()

        db_loaded = TuningDatabase()
        db_loaded.load(json_path)
        assert db_loaded.count() == 1
        loaded_rec = db_loaded.records[0]
        assert loaded_rec.latency_us == 12.5
        assert loaded_rec.candidate.get("id") == "cand_1"

        db.export_csv(csv_path)
        assert csv_path.exists()
        content = csv_path.read_text()
        assert "cand_1" in content
        assert "12.5" in content


def test_custom_kernel_diagnostics():
    hw = HardwareProfile(
        gpu_name="NVIDIA GB10",
        fingerprint="fp123",
        compute_capability="12.1",
        sm_count=48,
        max_threads_per_sm=1536,
        max_threads_per_block=1024,
        regs_per_sm=65536,
        shared_mem_per_block_default=49152,
        shared_mem_per_block_opt_in=101376,
        shared_mem_per_sm=102400,
        l2_cache_size=25165824,
        bandwidth_gbps=243.7,
        sustained_read_gbps=243.7,
        arch_name="sm_121",
    )
    diagnostician = CustomKernelDiagnostician(hw)

    cand = CandidateSpec("c_mma", "linear", "nvfp4", "mma", 128, 2560, "Mma", "launch_mma", tile_rows=64, tile_tokens=16)

    # Workload with near-roofline performance
    rec_good = TuningRecord(
        schema_version=1,
        hardware_profile_id="hw",
        software_profile={},
        workload={"operation": "linear", "quantization_family": "nvfp4", "n": 128, "k": 2560, "tokens": 16},
        candidate=cand.to_dict(),
        compile_status="success",
        correctness_status="pass",
        correctness_tests=CorrectnessTestSummary(),
        latency_us=1.2,
        latency_p95_us=1.3,
        effective_gbs=195.0,  # 80% of bandwidth
    )
    diag_good = diagnostician.diagnose(rec_good)
    assert diag_good.outcome == "reuse_existing_schedule"

    # Workload with severe gap on single-token decode using MMA
    rec_bad_decode = TuningRecord(
        schema_version=1,
        hardware_profile_id="hw",
        software_profile={},
        workload={"operation": "linear", "quantization_family": "nvfp4", "n": 128, "k": 2560, "tokens": 1},
        candidate=cand.to_dict(),
        compile_status="success",
        correctness_status="pass",
        correctness_tests=CorrectnessTestSummary(),
        latency_us=25.0,
        latency_p95_us=26.0,
        effective_gbs=20.0,
    )
    diag_decode = diagnostician.diagnose(rec_bad_decode)
    assert diag_decode.outcome == "change_dispatch"
    assert any("narrow decode" in b for b in diag_decode.identified_bottlenecks)
