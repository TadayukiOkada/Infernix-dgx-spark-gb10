from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from ckt import spec
from ckt.binary_inspector import compare_profiles, parse_sass, profile_instructions
from ckt.cli import _env_from_dict
from ckt.environment import DeviceInfo
from ckt.hotspots import analyze_nsys_sqlite
from ckt.kernel_math import (
    calculate_ab_stats,
    calculate_mlp,
    calculate_occupancy_tables,
    calculate_smem_layout,
    calculate_wave_tables,
    calculate_workload_math,
)
from ckt.performance_model import classify_bottleneck, occupancy_model, roofline_report
from ckt.profiler import KernelMetricResult
from ckt.recommendations import recommend


def gb10() -> DeviceInfo:
    d = DeviceInfo(
        name="NVIDIA GB10",
        compute_capability="12.1",
        cc_major=12,
        cc_minor=1,
        arch="sm_121a",
        multi_processor_count=48,
        warp_size=32,
        max_threads_per_block=1024,
        max_threads_per_multiprocessor=1536,
        max_blocks_per_multiprocessor=32,
        regs_per_multiprocessor=65536,
        shared_mem_per_block=49152,
        shared_mem_per_multiprocessor=102400,
        shared_mem_per_block_optin=101376,
        reserved_shared_mem_per_block=1024,
        l2_cache_size=25165824,
        integrated_uma=True,
        measured_bandwidth_gbps=241.4,
        bandwidth_source="measured",
    )
    d.arch_facts = spec.arch_facts(12, 1)
    d.apply_compute_probe(
        {
            "fp32_fma": {"tflops": 27.99},
            "fp64_fma": {"tflops": 0.42},
            "fp16_mma": {"tflops": 124.71},
            "int8_mma": {"tops": 249.16},
            "int_add": {"gopp_per_s": 13900.0},
            "smem_load": {"gbps": 12265.0},
        }
    )
    return d


class TestKernelMathAndGB10(unittest.TestCase):
    def test_compute_probe_populates_all_ceilings_and_roundtrips_env(self) -> None:
        dev = gb10()
        self.assertAlmostEqual(dev.dtype_peaks["fp32"].tflops, 27.99)
        self.assertAlmostEqual(dev.dtype_peaks["fp64"].tflops, 0.42)
        self.assertAlmostEqual(dev.dtype_peaks["fp16"].tflops, 124.71)
        self.assertAlmostEqual(dev.dtype_peaks["bf16"].tflops, 124.71)
        self.assertAlmostEqual(dev.dtype_peaks["int8"].tflops, 249.16)
        self.assertAlmostEqual(dev.dtype_peaks["int32"].tflops, 13.9)
        self.assertAlmostEqual(dev.measured_smem_bandwidth_gbps or 0.0, 12265.0)

        raw = {
            "hostname": "spark",
            "platform": "Linux",
            "python_version": "3.11.0",
            "device": dev.to_dict(),
            "tools": {"nvcc": {"name": "nvcc", "path": "/usr/local/cuda/bin/nvcc", "version": "13.0", "available": True}},
        }
        rehydrated = _env_from_dict(raw)
        ct, src = rehydrated.device.compute_ceiling_tflops
        self.assertAlmostEqual(ct or 0.0, 27.99)
        self.assertEqual(src, "measured")
        self.assertTrue(rehydrated.tools["nvcc"].available)

    def test_occupancy_step_tables_and_headroom(self) -> None:
        dev = gb10()
        # At 256 threads (8 warps) and 128 regs/thread: 128*32 = 4096 regs/warp * 8 = 32768 regs/block -> 2 blocks/SM (66.67% occ)
        res = calculate_occupancy_tables(dev, threads_per_block=256, registers_per_thread=128, shared_mem_per_block=0)
        cur = res["current"]
        self.assertEqual(cur["active_blocks_per_sm"], 2)
        self.assertAlmostEqual(cur["occupancy_pct"], 33.33, places=1)
        self.assertEqual(cur["max_regs_per_thread_to_gain_one_block"], 80)
        self.assertEqual(cur["free_reg_headroom_per_thread"], 0)
        # At 129 regs/thread -> rounds up to 4352 regs/warp * 8 = 34816 -> 1 block/SM (16.67% occ)!
        res129 = calculate_occupancy_tables(dev, threads_per_block=256, registers_per_thread=129, shared_mem_per_block=0)
        self.assertEqual(res129["current"]["active_blocks_per_sm"], 1)
        self.assertEqual(res129["current"]["max_regs_per_thread_to_gain_one_block"], 128)
        self.assertEqual(res129["current"]["free_reg_headroom_per_thread"], 126)

    def test_roofline_auto_selects_int8_tensor_core_and_l2_residency(self) -> None:
        dev = gb10()
        sass = parse_sass(
            "/*0000*/ IMMA.16832.S8.S8 R4, R8, R12, R4 ;\n"
            "/*0010*/ LDG.E.128 R0, [R2.64] ;\n"
        )
        prof = profile_instructions(sass)
        self.assertEqual(prof.int_tensor_core_ops, 1)
        rof = roofline_report(
            bytes_moved=44269568.0,
            flops=536870912.0,
            latency_us=613.5,
            dev=dev,
            instruction_profile=prof,
        )
        self.assertEqual(rof.compute_ceiling_dtype, "int8")
        self.assertAlmostEqual(rof.compute_ceiling_tflops or 0.0, 249.16)
        self.assertEqual(rof.l2_residency_regime, " dram-streaming".strip())
        self.assertGreater(rof.working_set_vs_l2_ratio or 0.0, 1.5)

    def test_gb10_classifier_fixes_40_to_70_dead_zone(self) -> None:
        dev = gb10()
        # Case 1: rows_kernel (mem=46.1% NCU, but eff_bw = 176.7 GB/s = 73.2% of measured 241.4 GB/s ceiling)
        rof_rows = roofline_report(
            bytes_moved=44269568.0,
            flops=536870912.0,
            latency_us=250.56,
            dev=dev,
        )
        res_rows = KernelMetricResult(
            kernel="rows_kernel",
            metrics={
                "sm__throughput.avg.pct_of_peak_sustained_elapsed": 18.02,
                "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": 46.10,
                "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio": 12.77,
                "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio": 1.66,
            },
        )
        cl_rows = classify_bottleneck(res_rows, dev=dev, roofline=rof_rows)
        self.assertEqual(cl_rows.category, "memory-bandwidth-bound")
        self.assertIn("long_scoreboard", cl_rows.stall_breakdown_pct)

        # Case 2: mma_rows_kernel (mem=45.95% NCU, eff_bw = 72.12 GB/s = 29.9% of measured ceiling, long_scoreboard=9.82)
        rof_mma = roofline_report(
            bytes_moved=44269568.0,
            flops=536870912.0,
            latency_us=613.5,
            dev=dev,
        )
        occ_mma = occupancy_model(dev, threads_per_block=256, registers_per_thread=128, shared_mem_per_block=38912)
        res_mma = KernelMetricResult(
            kernel="mma_rows_kernel",
            metrics={
                "sm__throughput.avg.pct_of_peak_sustained_elapsed": 24.16,
                "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": 45.95,
                "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio": 9.82,
                "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio": 2.42,
                "lts__t_sector_hit_rate.pct": 18.78,
            },
        )
        cl_mma = classify_bottleneck(res_mma, dev=dev, occupancy=occ_mma, roofline=rof_mma)
        self.assertEqual(cl_mma.category, "memory-latency-bound")
        recs_mma = recommend(
            result=res_mma,
            dev=dev,
            classification=cl_mma,
            occupancy=occ_mma,
            roofline=rof_mma,
        )
        self.assertGreaterEqual(len(recs_mma.recommendations), 2)

    def test_workload_smem_mlp_and_ab_calculators(self) -> None:
        dev = gb10()
        wl = calculate_workload_math(
            dev,
            m=2,
            n=4096,
            k=16384,
            weight_dtype="iq3_s",
            act_dtype="q8_1",
            out_dtype="fp32",
            latency_us=613.5,
            compute_dtype="int8",
        )
        self.assertEqual(wl["flops"], 2 * 2 * 4096 * 16384)
        self.assertEqual(wl["weights"]["bytes"], (4096 * 16384 // 256) * 110)

        sm = calculate_smem_layout(dev, rows=32, cols=32, elem_bytes=4, stages=1, threads_per_block=256, registers_per_thread=64)
        self.assertEqual(sm["unpadded"]["column_access_bank_conflict_way"], 32)
        self.assertEqual(sm["scalar_4b_conflict_free_padding"]["pad_elements"], 1)
        self.assertEqual(sm["scalar_4b_conflict_free_padding"]["bank_conflict_way"], 1)

        mlp = calculate_mlp(dev, active_warps_per_sm=16, load_width_bytes=16, mem_latency_ns=250.0)
        self.assertGreaterEqual(mlp["independent_loads_per_thread_needed"], 1)

        ab = calculate_ab_stats(
            [613.5, 614.0, 613.1, 613.8],
            [480.2, 480.9, 479.8, 480.5],
            launches_per_token=36,
            baseline_token_ms=45.0,
        )
        self.assertEqual(ab["verdict"], "MEANINGFUL_GAIN")
        self.assertGreater(ab["end_to_end_projection"]["projected_candidate_tok_per_s"], ab["end_to_end_projection"]["baseline_tok_per_s"])

    def test_compare_profiles_and_hotspots_sqlite(self) -> None:
        p1 = profile_instructions(parse_sass("/*0000*/ DFMA R0, R2, R4, R0 ;\n/*0010*/ LDG.E R4, [R2] ;\n"))
        p2 = profile_instructions(parse_sass("/*0000*/ FFMA R0, R2, R4, R0 ;\n/*0010*/ LDG.E.128 R4, [R2] ;\n"))
        diff = compare_profiles(p1, p2, regs_a=128, regs_b=80)
        self.assertEqual(diff["instructions"]["fp64_ops"]["delta"], -1)
        self.assertEqual(diff["instructions"]["max_load_width"]["after"], 16)
        self.assertEqual(diff["resources"]["registers"]["delta"], -48)

        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "trace.sqlite"
            con = sqlite3.connect(str(db_path))
            con.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
            con.execute(
                "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL ("
                "demangledName INTEGER, start INTEGER, end INTEGER, "
                "gridX INTEGER, gridY INTEGER, gridZ INTEGER, "
                "blockX INTEGER, blockY INTEGER, blockZ INTEGER, "
                "registersPerThread INTEGER, dynamicSharedMemory INTEGER, staticSharedMemory INTEGER)"
            )
            con.execute("INSERT INTO StringIds VALUES (1, 'mma_rows_kernel<iq3_s>')")
            con.execute("INSERT INTO StringIds VALUES (2, 'cudaLaunchKernel')")
            con.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (1, 1000, 614500, 200, 1, 1, 256, 1, 1, 128, 38912, 0)")
            con.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (nameId INTEGER, start INTEGER, end INTEGER)")
            con.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (2, 500, 1500)")
            con.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_SYNCHRONIZATION (syncType INTEGER, start INTEGER, end INTEGER)")
            con.execute("INSERT INTO CUPTI_ACTIVITY_KIND_SYNCHRONIZATION VALUES (1, 2000, 3000)")
            con.commit()
            con.close()

            hs = analyze_nsys_sqlite(db_path, gb10())
            self.assertEqual(hs["total_calls"], 1)
            self.assertEqual(hs["by_geometry"][0]["functional_block"], "dense_ggml")
            self.assertEqual(hs["by_geometry"][0]["active_blocks_per_sm"], 2)
            self.assertAlmostEqual(hs["by_geometry"][0]["occupancy_pct"], 33.33, places=1)
            self.assertEqual(len(hs["cuda_runtime_hotspots"]), 1)
            self.assertEqual(hs["cuda_runtime_hotspots"][0]["name"], "cudaLaunchKernel")
            self.assertEqual(len(hs["sync_breakdown"]), 1)

    def test_profiling_capabilities_detection(self) -> None:
        from ckt.environment import detect_profiling_capabilities

        prof = detect_profiling_capabilities()
        self.assertIn("perf_event_paranoid", prof)
        self.assertIn("kptr_restrict", prof)
        self.assertIn("cpu_sampling_available", prof)
        self.assertIn("pmu_events_available", prof)

    def test_shapes_parse_and_ab_csv(self) -> None:
        from ckt.ab_runner import compare_shape_csvs
        from ckt.shapes import boundary_tokens_for_shape, parse_shape_file

        dev = gb10()
        dev.apply_compute_probe(
            {
                "fp32_fma": {"tflops": 27.76},
                "fp64_fma": {"tflops": 0.41},
                "fp16_mma": {"tflops": 123.76},
                "fp16_f16acc_mma": {"tflops": 123.77},
                "bf16_mma": {"tflops": 123.76},
                "int8_mma": {"tops": 247.61},
                "int4_mma": {"tops": 34.95},
                "fp8_e4m3_mma": {"tflops": 244.30},
                "mxfp8_mma": {"tflops": 244.05},
                "mxfp6_mma": {"tflops": 244.06},
                "mxfp4_mma": {"tflops": 488.40},
                "nvfp4_mma": {"tflops": 494.56},
            }
        )
        self.assertIn("fp8", dev.dtype_peaks)
        self.assertIn("nvfp4", dev.dtype_peaks)
        self.assertIn("fp4", dev.dtype_peaks)
        self.assertTrue(any("Legacy INT4" in n for n in dev.measured_ratio_notes))

        with tempfile.TemporaryDirectory() as tmp:
            q4_shapes_dir = Path(tmp) / "q4" / "shapes"
            q4_shapes_dir.mkdir(parents=True)
            cu_file = q4_shapes_dir / "n4096_k5120.cu"
            cu_file.write_text(
                "// Selected on RTX 5090.\n"
                "Q4Launch select_q4_n4096_k5120(int tokens) {\n"
                "    if (tokens <= 1) return launch_q4_a16_gemv_r1_w8_direct;\n"
                "    if (tokens <= 8) return launch_q4_a16_sliced_r16_t8_w4_s2;\n"
                "    return launch_q4_a16_mma_r64_t128;\n"
                "}\n"
            )
            spec = parse_shape_file(cu_file, dev=dev)
            self.assertIsNotNone(spec)
            assert spec is not None
            self.assertEqual(spec.qtype, "q4")
            self.assertEqual(spec.n, 4096)
            self.assertEqual(spec.k, 5120)
            self.assertEqual(spec.tuning_origin, "RTX 5090")
            self.assertEqual(len(spec.bands), 3)
            self.assertEqual(boundary_tokens_for_shape(spec), [1, 2, 8, 9])

            csv_a = Path(tmp) / "a.csv"
            csv_b = Path(tmp) / "b.csv"
            csv_a.write_text("qtype,policy,N,K,T,median_us,min_us,p95_us,effective_gbs,useful_tflops\nQ4,A16,4096,5120,1,50.0,49.0,51.0,220.0,0.8\nQ4,A16,4096,5120,8,60.0,59.0,61.0,185.0,5.5\n")
            csv_b.write_text("qtype,policy,N,K,T,median_us,min_us,p95_us,effective_gbs,useful_tflops\nQ4,A16,4096,5120,1,45.0,44.0,46.0,244.0,0.9\nQ4,A16,4096,5120,8,54.0,53.0,55.0,205.0,6.1\n")
            ab = compare_shape_csvs(csv_a, csv_b)
            self.assertEqual(ab["wins"], 2)
            self.assertEqual(ab["losses"], 0)
            self.assertEqual(ab["verdict"], "PASS_STRICT_IMPROVEMENT")
            self.assertGreater(ab["geomean_speedup_x"], 1.10)

    def test_bandwidth_breakdown_hydration(self):
        raw = {
            "device": {
                "name": "NVIDIA GB10",
                "compute_capability_major": 12,
                "compute_capability_minor": 1,
                "measured_bandwidth_gbps": 252.0,
            },
            "bandwidth": {
                "read": {"gbps": 252.0},
                "write": {"gbps": 199.3},
                "copy": {"gbps": 228.5},
                "sustained_read_gbps": 252.0,
            },
        }
        env = _env_from_dict(raw)
        self.assertEqual(env.device.measured_bandwidth_gbps, 252.0)
        self.assertEqual(env.device.measured_read_bandwidth_gbps, 252.0)
        self.assertEqual(env.device.measured_write_bandwidth_gbps, 199.3)
        self.assertEqual(env.device.measured_copy_bandwidth_gbps, 228.5)


if __name__ == "__main__":
    unittest.main()
