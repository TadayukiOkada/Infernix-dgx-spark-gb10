"""Unit tests for ``ckt.cliffs`` cross-family cliff detection, shape parsing, and merging."""

from __future__ import annotations

from pathlib import Path
import unittest

from ckt.cliffs import (
    CaseResult,
    CliffsReport,
    FamilySummary,
    detect_cliffs,
    merge_reports,
    parse_bench_csv_text,
    parse_causal_conv1d_stdout,
    parse_family_list,
    parse_fp8_linear_swiglu_stdout,
    parse_ggml_linear_stdout,
    parse_hc_mix_fused_stdout,
    parse_hyper_connection_stdout,
    parse_offloaded_moe_wide_stdout,
    parse_qsa_attention_stdout,
    parse_resident_moe_stdout,
    parse_rmsnorm_rope_stdout,
    render_markdown_report,
)
from ckt.shapes import boundary_tokens_for_shape, parse_shape_file


REPO_ROOT = Path(__file__).resolve().parents[2]


class TestCliffsDetection(unittest.TestCase):
    def test_detects_upward_and_inverted_cliffs_and_filters_noise(self) -> None:
        pts = [
            {"T": 1, "median_us": 10.0, "active_route": "gemv"},
            {"T": 4, "median_us": 10.2, "active_route": "gemv"},
            {"T": 8, "median_us": 20.0, "active_route": "sliced_a", "shape_line": 22},
            {"T": 9, "median_us": 15.0, "active_route": "sliced_b", "shape_line": 23},
            {"T": 16, "median_us": 15.3, "active_route": "sliced_b"},
            {"T": 17, "median_us": 22.0, "active_route": "mma_c", "shape_line": 24},
        ]
        cliffs = detect_cliffs(
            pts,
            family="linear:q8",
            case_id="q8:a16:n2560_k2560",
            source_file="src/ops/linear/q8/shapes/n2560_k2560.cu",
        )
        self.assertEqual(len(cliffs), 3)
        self.assertEqual((cliffs[0].kind, cliffs[0].from_T, cliffs[0].to_T, cliffs[0].severity), ("upward_cliff", 4, 8, "critical"))
        self.assertEqual((cliffs[1].kind, cliffs[1].from_T, cliffs[1].to_T, cliffs[1].severity), ("inverted_cliff", 8, 9, "critical"))
        self.assertEqual((cliffs[2].kind, cliffs[2].from_T, cliffs[2].to_T, cliffs[2].route_changed), ("upward_cliff", 16, 17, True))

    def test_min_delta_us_suppresses_tiny_pct_jitter(self) -> None:
        pts = [
            {"T": 1, "median_us": 1.0, "active_route": "r1"},
            {"T": 2, "median_us": 1.3, "active_route": "r2"},
        ]
        cliffs = detect_cliffs(pts, family="linear:bf16", case_id="tiny", min_delta_us=0.5)
        self.assertEqual(cliffs, [])

    def test_parse_family_list_expands_linear_and_validates(self) -> None:
        expanded = parse_family_list("linear,gated_delta_net,qsa_attention")
        self.assertIn("linear:bf16", expanded)
        self.assertIn("linear:q8", expanded)
        self.assertEqual(expanded[-2:], ["gated_delta_net", "qsa_attention"])

    def test_shape_parser_handles_launch_routes_and_early_return_helpers(self) -> None:
        q8_16384 = parse_shape_file(REPO_ROOT / "src/ops/linear/q8/shapes/n16384_k2560.cu")
        self.assertIsNotNone(q8_16384)
        bt_16384 = boundary_tokens_for_shape(q8_16384, policy="a16", max_t=512)
        for expected in (2, 4, 5, 6, 8, 16, 32, 64, 96, 128, 192, 384, 511):
            self.assertIn(expected, bt_16384)

        q8_1280 = parse_shape_file(REPO_ROOT / "src/ops/linear/q8/shapes/n1280_k2560.cu")
        self.assertIsNotNone(q8_1280)
        bt_1280 = boundary_tokens_for_shape(q8_1280, policy="a16", max_t=512)
        for expected in (48, 49, 96, 97, 128, 129, 192, 193, 512):
            self.assertIn(expected, bt_1280)

        bf16_1280 = parse_shape_file(REPO_ROOT / "src/ops/linear/bf16/shapes/n1280_k2560.cu")
        self.assertIsNotNone(bf16_1280)
        bt_bf16 = boundary_tokens_for_shape(bf16_1280, policy="a16", max_t=512)
        for expected in (8, 9, 16, 17, 64, 65, 96, 97, 128, 129, 255, 256, 384, 385, 512):
            self.assertIn(expected, bt_bf16)

    def test_parsers_cover_all_standalone_and_projection_schemas(self) -> None:
        fp8_out = (
            "pol       T   median_us      min_us      p95_us   eff_GB/s    TFLOP/s     TC_%\n"
            "A16       1     688.000     688.000     688.000      259.3       0.52        -\n"
            "A16       8     697.216     697.216     697.216      256.3       4.09        -\n"
        )
        self.assertEqual([(p["T"], p["median_us"]) for p in parse_fp8_linear_swiglu_stdout(fp8_out)], [(1, 688.0), (8, 697.216)])

        ggml_out = (
            "iq3s_6144      N=  6144 K= 2560 T=    1 route=exact median       40.8 us (min       40.8, p95       40.8) |     0.8 TOPS\n"
            "iq3s_6144      N=  6144 K= 2560 T=    8 route=exact median       48.9 us (min       48.9, p95       48.9) |     5.1 TOPS\n"
        )
        self.assertEqual(parse_ggml_linear_stdout(ggml_out)[("iq3s_6144", "exact")][1]["median_us"], 48.9)

        hc_out = "T=4 | per call median   35.31 us (min   35.31, p95   35.31) |    218 GB/s\n"
        self.assertEqual(parse_hyper_connection_stdout(hc_out)[0]["median_us"], 35.31)

        hc_fused_out = "q8_g32_fp16\t4\tcomposed_cold\t45.25\t42.85\t94.78\t172.5\t0.008\nq8_g32_fp16\t5\tcomposed_cold\t55.14\t54.08\t56.16\t142.0\t0.008\n"
        self.assertEqual(len(parse_hc_mix_fused_stdout(hc_fused_out)["composed_cold"]), 2)

        qsa_out = "int8       8192     1  1      40.80      14.34         26.46\nint8       8192     4  1      63.20      18.24         44.96\n"
        self.assertEqual(parse_qsa_attention_stdout(qsa_out)[("int8", 8192)][1]["median_us"], 63.2)

        rr_out = (
            "form=text route=fused Q=24 K=4 T=1 execution=graph median=3.168 us min=3.168 us p95=3.168 us useful=9.1 GB/s\n"
            "form=pair W=2 B=1 T=2 heads_per_cta=8 execution=graph median=3.296 us min=3.296 us p95=3.296 us useful=12.4 GB/s\n"
        )
        rr_grouped = parse_rmsnorm_rope_stdout(rr_out)
        self.assertIn("text:fused:Q24_K4", rr_grouped)
        self.assertIn("pair:B1", rr_grouped)

        wide_out = "T=    8 placement=frames experts used  73 (wide   0, narrow  73, largest    3) wide entries      0 of     80 | moe_experts median     895.2 us (min     895.2, p95     895.2) |\n"
        self.assertEqual(parse_offloaded_moe_wide_stdout(wide_out)["frames"][0]["median_us"], 895.2)

        res_moe_out = "T=1 entries= 10 | median   121.7 us (min   121.7, p95   121.7) |    215 GB/s\n"
        self.assertEqual(parse_resident_moe_stdout(res_moe_out)[0]["T"], 1)

        conv_out = "causal_conv1d prefill-cold [8192,8,1] median=    4.13 us  min=    4.03 us  p95=    5.98 us\n"
        self.assertEqual(parse_causal_conv1d_stdout(conv_out)["prefill-cold"][0]["T"], 8)

        gdn_csv = (
            "# comment line\n"
            "state_form,normalization,implementation,tokens,median_us,min_us,p95_us\n"
            "running,fused,public.total,15,46.208,46.208,46.208\n"
            "running,fused,public.total,16,25.248,25.248,25.248\n"
        )
        gdn_pts = parse_bench_csv_text(gdn_csv, token_cols=("tokens", "T"), route_cols=("implementation",))
        self.assertEqual([(p["T"], p["median_us"]) for p in gdn_pts], [(15, 46.208), (16, 25.248)])

    def test_merge_reports_updates_families_and_reranks_top_cliffs(self) -> None:
        ev_old = detect_cliffs(
            [{"T": 15, "median_us": 46.2, "active_route": "rec"}, {"T": 16, "median_us": 25.2, "active_route": "chk"}],
            family="gated_delta_net",
            case_id="running:sweep",
        )[0]
        base = CliffsReport(
            device_name="NVIDIA GB10",
            cuda_arch="12.1",
            l2_mib=24,
            flush_mib=32,
            max_tokens=512,
            warmup=2,
            repeat=8,
            upward_threshold_pct=15.0,
            inverted_threshold_pct=-8.0,
            min_delta_us=0.5,
            families_requested=["linear:q8", "gated_delta_net"],
            family_summaries=[
                FamilySummary(family="linear:q8", cases_total=1, cases_ok=1, points_measured=1, elapsed_s=1.0),
                FamilySummary(family="gated_delta_net", cases_total=1, cases_ok=1, cliffs_total=1, inverted_cliffs=1, critical_cliffs=1, worst_inverted_pct=ev_old.step_delta_pct, elapsed_s=1.0),
            ],
            cases=[
                CaseResult(family="linear:q8", case_id="q8:old", status="ok", elapsed_s=1.0),
                CaseResult(family="gated_delta_net", case_id="running:sweep", status="ok", cliffs=[ev_old], elapsed_s=1.0),
            ],
            top_cliffs=[ev_old],
            total_elapsed_s=2.0,
        ).to_dict()

        ev_new = detect_cliffs(
            [{"T": 1, "median_us": 10.0, "active_route": "a"}, {"T": 2, "median_us": 20.0, "active_route": "b"}],
            family="linear:q8",
            case_id="q8:n16384_k2560",
        )[0]
        patch = CliffsReport(
            device_name="NVIDIA GB10",
            cuda_arch="12.1",
            l2_mib=24,
            flush_mib=32,
            max_tokens=512,
            warmup=2,
            repeat=8,
            upward_threshold_pct=15.0,
            inverted_threshold_pct=-8.0,
            min_delta_us=0.5,
            families_requested=["linear:q8", "candidate_selector"],
            family_summaries=[
                FamilySummary(family="linear:q8", cases_total=1, cases_ok=1, points_measured=2, cliffs_total=1, upward_cliffs=1, critical_cliffs=1, worst_upward_pct=ev_new.step_delta_pct, elapsed_s=2.0),
                FamilySummary(family="candidate_selector", cases_total=1, cases_ok=1, points_measured=2, elapsed_s=0.5),
            ],
            cases=[
                CaseResult(family="linear:q8", case_id="q8:n16384_k2560", status="ok", points=[{"T": 1}, {"T": 2}], cliffs=[ev_new], elapsed_s=2.0),
                CaseResult(family="candidate_selector", case_id="B1:greedy", status="ok", points=[{"T": 1}, {"T": 2}], elapsed_s=0.5),
            ],
            top_cliffs=[ev_new],
            total_elapsed_s=2.5,
        )
        merged = merge_reports(base, patch)
        self.assertEqual([s.family for s in merged.family_summaries], ["linear:q8", "gated_delta_net", "candidate_selector"])
        self.assertEqual([c.case_id for c in merged.cases], ["q8:n16384_k2560", "running:sweep", "B1:greedy"])
        self.assertEqual(len(merged.top_cliffs), 2)
        self.assertIn("candidate_selector", render_markdown_report(merged))


if __name__ == "__main__":
    unittest.main()
